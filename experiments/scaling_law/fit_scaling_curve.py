# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
fit_scaling_curve.py — Chinchilla L(N, D) fitting for heterogeneous cluster
============================================================================
Fits the Chinchilla scaling law to empirical loss measurements from the
Neuron_SP 2×A6000+1×H100 cluster experiments.

Chinchilla functional form (Hoffmann et al., 2022):
    L(N, D) = E + A / N^α + B / D^β

Where:
  N = number of parameters
  D = number of training tokens
  E = irreducible (entropy) loss
  A, α = parameter-efficiency constants
  B, β = data-efficiency constants

Heterogeneous-specific overhead terms (additions to standard Chinchilla):
  L_hetero(N, D) = L(N, D) + Φ_bubble(pipeline_stages, n_microbatch) + Φ_numa(N)

Where:
  Φ_bubble = bubble_ratio × grad_penalty_factor
  Φ_numa   = cross_NUMA_latency × sync_frequency / tokens_per_step

Usage:
    python experiments/scaling_law/fit_scaling_curve.py --data experiments/scaling_law/measurements.csv
    python experiments/scaling_law/fit_scaling_curve.py --demo   # run with synthetic data

Output:
    experiments/scaling_law/scaling_fit_results.json
    experiments/scaling_law/scaling_fit_plot.png  (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ScalingPoint:
    """A single (N, D, L) measurement from the heterogeneous cluster."""
    model_name: str  # e.g. "tiny_70m"
    N: int  # parameter count
    D: int  # tokens trained on at this checkpoint
    L: float  # cross-entropy loss (nats or bits, consistent)
    # Optional overhead measurements
    pipeline_bubble_ratio: float = 0.0
    cross_numa_latency_ms: float = 0.0
    # Cluster config at time of measurement
    pipeline_stages: int = 1


@dataclass
class ChinchillaFitResult:
    """Fitted Chinchilla parameters and heterogeneous overhead terms."""
    E: float  # irreducible loss
    A: float  # parameter coefficient
    alpha: float  # parameter exponent
    B: float  # data coefficient
    beta: float  # data exponent
    # Heterogeneous overhead fit
    phi_bubble: float  # bubble overhead coefficient
    phi_numa: float  # cross-NUMA overhead coefficient
    # Goodness of fit
    rmse: float
    r_squared: float
    n_points: int


# ---------------------------------------------------------------------------
# Chinchilla loss function
# ---------------------------------------------------------------------------
def chinchilla_loss(
    N: float,
    D: float,
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
) -> float:
    """
    Standard Chinchilla loss formula.
        L(N, D) = E + A/N^alpha + B/D^beta
    """
    return E + A / (N**alpha) + B / (D**beta)


def hetero_loss(
    N: float,
    D: float,
    pipeline_bubble_ratio: float,
    cross_numa_latency_ms: float,
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    phi_bubble: float,
    phi_numa: float,
) -> float:
    """
    Heterogeneous cluster loss: Chinchilla + overhead terms.

    Overhead interpretation:
      phi_bubble × pipeline_bubble_ratio: effective compute wasted per step
        → equivalent to training on fewer tokens, increasing L
      phi_numa × cross_numa_latency_ms: straggler penalty from cross-NUMA sync
        → adds a constant bias to observed loss at fixed N, D
    """
    l_base = chinchilla_loss(N, D, E, A, alpha, B, beta)
    l_bubble = phi_bubble * pipeline_bubble_ratio
    l_numa = phi_numa * cross_numa_latency_ms
    return l_base + l_bubble + l_numa


# ---------------------------------------------------------------------------
# Optimizer (pure-Python Nelder-Mead since scipy may not be available)
# ---------------------------------------------------------------------------
class NelderMead:
    """
    Minimal Nelder-Mead simplex optimizer.
    Suitable for fitting ~7 parameters to O(10) data points.
    """

    def __init__(
        self,
        objective,
        x0: List[float],
        max_iter: int = 5000,
        tol: float = 1e-9,
        alpha: float = 1.0,
        gamma: float = 2.0,
        rho: float = 0.5,
        sigma: float = 0.5,
    ) -> None:
        self.f = objective
        self.x0 = list(x0)
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

    def minimize(self) -> Tuple[List[float], float]:
        n = len(self.x0)
        # Build initial simplex
        simplex = [list(self.x0)]
        for i in range(n):
            point = list(self.x0)
            point[i] *= 1.05 if abs(point[i]) > 1e-8 else 0.00025
            simplex.append(point)

        f_vals = [self.f(x) for x in simplex]

        for _ in range(self.max_iter):
            # Sort
            order = sorted(range(n + 1), key=lambda i: f_vals[i])
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]

            # Convergence check
            if max(abs(f_vals[i] - f_vals[0]) for i in range(1, n + 1)) < self.tol:
                break

            # Centroid (exclude worst)
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

            # Reflect
            xr = [centroid[j] + self.alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = self.f(xr)

            if f_vals[0] <= fr < f_vals[-2]:
                simplex[-1] = xr
                f_vals[-1] = fr
            elif fr < f_vals[0]:
                # Expand
                xe = [centroid[j] + self.gamma * (xr[j] - centroid[j]) for j in range(n)]
                fe = self.f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    f_vals[-1] = fr
            else:
                # Contract
                xc = [centroid[j] + self.rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fc = self.f(xc)
                if fc < f_vals[-1]:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + self.sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    f_vals = [self.f(x) for x in simplex]

        return simplex[0], f_vals[0]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_chinchilla(points: List[ScalingPoint]) -> ChinchillaFitResult:
    """
    Fit Chinchilla + heterogeneous overhead parameters to empirical data.

    Parameters (7 total):
        [E, log_A, alpha, log_B, beta, phi_bubble, phi_numa]

    Using log_A and log_B to ensure positivity during optimization.
    """

    def objective(params: List[float]) -> float:
        E, log_A, alpha, log_B, beta, phi_bubble, phi_numa = params
        A = math.exp(log_A)
        B = math.exp(log_B)
        # Clamp exponents to valid range
        alpha_c = max(0.01, min(alpha, 2.0))
        beta_c = max(0.01, min(beta, 2.0))
        E_c = max(0.0, E)
        phi_bubble_c = max(0.0, phi_bubble)
        phi_numa_c = max(0.0, phi_numa)

        total_sq_err = 0.0
        for p in points:
            try:
                L_pred = hetero_loss(
                    p.N,
                    p.D,
                    p.pipeline_bubble_ratio,
                    p.cross_numa_latency_ms,
                    E_c,
                    A,
                    alpha_c,
                    B,
                    beta_c,
                    phi_bubble_c,
                    phi_numa_c,
                )
                total_sq_err += (L_pred - p.L)**2
            except (OverflowError, ZeroDivisionError, ValueError):
                total_sq_err += 1e6
        return total_sq_err

    # Initial guess from Hoffmann et al. Table A1 (adjusted for nats)
    x0 = [
        1.69,  # E  (~entropy of natural language in nats)
        math.log(406.4),  # log_A
        0.34,  # alpha
        math.log(410.7),  # log_B
        0.28,  # beta
        0.01,  # phi_bubble
        0.001,  # phi_numa
    ]

    optimizer = NelderMead(objective, x0, max_iter=10000, tol=1e-12)
    best_params, best_loss = optimizer.minimize()

    E, log_A, alpha, log_B, beta, phi_bubble, phi_numa = best_params
    A = math.exp(log_A)
    B = math.exp(log_B)
    alpha = max(0.01, min(alpha, 2.0))
    beta = max(0.01, min(beta, 2.0))
    phi_bubble = max(0.0, phi_bubble)
    phi_numa = max(0.0, phi_numa)

    # Compute RMSE and R²
    L_vals = [p.L for p in points]
    L_mean = sum(L_vals) / len(L_vals)
    ss_tot = sum((l - L_mean)**2 for l in L_vals)
    L_preds = [
        hetero_loss(
            p.N,
            p.D,
            p.pipeline_bubble_ratio,
            p.cross_numa_latency_ms,
            E,
            A,
            alpha,
            B,
            beta,
            phi_bubble,
            phi_numa,
        ) for p in points
    ]
    ss_res = sum((l - lp)**2 for l, lp in zip(L_vals, L_preds))
    rmse = math.sqrt(ss_res / len(points))
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return ChinchillaFitResult(
        E=E,
        A=A,
        alpha=alpha,
        B=B,
        beta=beta,
        phi_bubble=phi_bubble,
        phi_numa=phi_numa,
        rmse=rmse,
        r_squared=r_sq,
        n_points=len(points),
    )


def predict_optimal_allocation(
    fit: ChinchillaFitResult,
    compute_budget_flops: float,
) -> Tuple[int, int, float]:
    """
    Given a FLOPs budget C, find optimal (N*, D*) that minimizes L(N, D)
    subject to C = 6 × N × D (Chinchilla FLOPs formula).

    Returns (N_opt, D_opt, L_opt).

    Analytical solution from Hoffmann et al.:
        N* = (A × alpha / (B × beta))^(1/(alpha + beta)) × (C / 6)^(beta/(alpha+beta))
        D* = C / (6 × N*)
    """
    A, alpha, B, beta, E = fit.A, fit.alpha, fit.B, fit.beta, fit.E
    exponent_ratio = beta / (alpha + beta)
    scale = (A * alpha / (B * beta))**(1.0 / (alpha + beta))
    N_opt = scale * (compute_budget_flops / 6.0)**exponent_ratio
    D_opt = compute_budget_flops / (6.0 * N_opt)
    L_opt = chinchilla_loss(N_opt, D_opt, E, A, alpha, B, beta)
    return int(N_opt), int(D_opt), L_opt


# ---------------------------------------------------------------------------
# Demo synthetic data (matches our experiment matrix)
# ---------------------------------------------------------------------------
def _make_demo_data() -> List[ScalingPoint]:
    """
    Generate synthetic scaling points matching our experiment_matrix.yaml,
    using Chinchilla ground-truth + heterogeneous noise.
    Based on Hoffmann et al. 2022 Table A1 values.
    """
    E_true, A_true, alpha_true = 1.69, 406.4, 0.34
    B_true, beta_true = 410.7, 0.28
    phi_bubble_true, phi_numa_true = 0.012, 0.0008

    configs = [
        # (model_name, N, D_chinchilla, bubble, numa_ms, stages)
        ("tiny_70m", 70_000_000, 1_400_000_000, 0.0, 8.5, 1),
        ("small_160m", 160_000_000, 3_200_000_000, 0.0, 9.1, 1),
        ("medium_410m", 410_000_000, 8_200_000_000, 0.083, 10.3, 2),
        ("large_1b", 1_000_000_000, 20_000_000_000, 0.111, 12.7, 3),
        # Sub-optimal D points (for fitting β)
        ("tiny_70m_half", 70_000_000, 700_000_000, 0.0, 8.5, 1),
        ("small_160m_half", 160_000_000, 1_600_000_000, 0.0, 9.1, 1),
        ("medium_410m_half", 410_000_000, 4_100_000_000, 0.083, 10.3, 2),
    ]

    points = []
    import random
    random.seed(42)
    for name, N, D, bubble, numa, stages in configs:
        L_true = hetero_loss(
            N,
            D,
            bubble,
            numa,
            E_true,
            A_true,
            alpha_true,
            B_true,
            beta_true,
            phi_bubble_true,
            phi_numa_true,
        )
        # Add small Gaussian noise (σ=0.01) to simulate measurement variance
        L_obs = L_true + random.gauss(0, 0.01)
        points.append(
            ScalingPoint(
                model_name=name,
                N=N,
                D=D,
                L=L_obs,
                pipeline_bubble_ratio=bubble,
                cross_numa_latency_ms=numa,
                pipeline_stages=stages,
            ))
    return points


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def load_csv(path: str) -> List[ScalingPoint]:
    """
    Load measurements from CSV.
    Expected columns: model_name,N,D,L,pipeline_bubble_ratio,cross_numa_latency_ms,pipeline_stages
    """
    import csv
    points = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append(
                ScalingPoint(
                    model_name=row["model_name"],
                    N=int(float(row["N"])),
                    D=int(float(row["D"])),
                    L=float(row["L"]),
                    pipeline_bubble_ratio=float(row.get("pipeline_bubble_ratio", 0)),
                    cross_numa_latency_ms=float(row.get("cross_numa_latency_ms", 0)),
                    pipeline_stages=int(row.get("pipeline_stages", 1)),
                ))
    return points


def save_csv_template(path: str) -> None:
    """Write an empty CSV template for recording measurements."""
    header = "model_name,N,D,L,pipeline_bubble_ratio,cross_numa_latency_ms,pipeline_stages\n"
    examples = [
        "tiny_70m,70000000,1400000000,,0.0,8.5,1\n",
        "small_160m,160000000,3200000000,,0.0,9.1,1\n",
        "medium_410m,410000000,8200000000,,0.083,10.3,2\n",
        "large_1b,1000000000,20000000000,,0.111,12.7,3\n",
    ]
    with open(path, "w") as f:
        f.write(header)
        f.writelines(examples)
    print(f"[fit_scaling_curve] Template saved: {path}")


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------
def _try_plot(
    points: List[ScalingPoint],
    fit: ChinchillaFitResult,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[fit_scaling_curve] matplotlib not available; skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Heterogeneous Chinchilla Scaling: 2×A6000+H100", fontsize=13)

    # -- Left: L vs N at Chinchilla-optimal D --
    ax = axes[0]
    N_range = np.logspace(math.log10(50e6), math.log10(2e9), 200)
    D_opt = N_range * 20
    L_pred = [chinchilla_loss(n, d, fit.E, fit.A, fit.alpha, fit.B, fit.beta) for n, d in zip(N_range, D_opt)]
    ax.loglog(N_range, L_pred, "b-", label="Chinchilla fit", linewidth=2)
    for p in points:
        if p.D >= p.N * 18:  # near-optimal D points only
            ax.scatter(p.N, p.L, color="red", zorder=5, s=60)
            ax.annotate(
                p.model_name.split("_")[0],
                (p.N, p.L),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
            )
    ax.set_xlabel("Parameters N")
    ax.set_ylabel("Loss L(N, D_opt)")
    ax.set_title("Loss vs. Scale (Chinchilla-optimal D)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # -- Right: L vs D at fixed N (each model) --
    ax = axes[1]
    colors = {"tiny_70m": "green", "small_160m": "orange", "medium_410m": "purple", "large_1b": "red"}
    for name, N_fixed, color in [
        ("Tiny 70M", 70e6, "green"),
        ("Small 160M", 160e6, "orange"),
        ("Medium 410M", 410e6, "purple"),
        ("Large 1B", 1e9, "red"),
    ]:
        D_range = np.logspace(math.log10(N_fixed * 5), math.log10(N_fixed * 50), 100)
        L_curve = [chinchilla_loss(N_fixed, d, fit.E, fit.A, fit.alpha, fit.B, fit.beta) for d in D_range]
        ax.semilogx(D_range, L_curve, label=name, color=color, linewidth=1.5)
    # Scatter measured points
    for p in points:
        ax.scatter(p.D, p.L, color="black", zorder=5, s=40, alpha=0.7)
    ax.set_xlabel("Training Tokens D")
    ax.set_ylabel("Loss L(N, D)")
    ax.set_title("Loss vs. Tokens (per model size)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[fit_scaling_curve] Plot saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Chinchilla Scaling Law Fitter")
    p.add_argument("--data", default=None, help="Path to measurements CSV")
    p.add_argument("--demo", action="store_true", help="Run with synthetic demo data")
    p.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Output directory for results JSON and plot",
    )
    p.add_argument(
        "--compute_budget",
        type=float,
        default=None,
        help="FLOPs budget for optimal N*,D* prediction (e.g. 1e21)",
    )
    p.add_argument(
        "--save_template",
        action="store_true",
        help="Save empty CSV template for recording measurements",
    )
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.save_template:
        save_csv_template(os.path.join(out_dir, "measurements.csv"))
        return

    if args.demo:
        print("[fit_scaling_curve] Running with synthetic demo data...")
        points = _make_demo_data()
    elif args.data:
        print(f"[fit_scaling_curve] Loading: {args.data}")
        points = load_csv(args.data)
    else:
        print("[fit_scaling_curve] No data provided. Use --demo or --data <csv>.\n"
              "  To create a template: --save_template")
        sys.exit(1)

    print(f"[fit_scaling_curve] Fitting {len(points)} measurement points...")
    fit = fit_chinchilla(points)

    print("\n" + "=" * 60)
    print("  Chinchilla Fit Results — Heterogeneous Cluster")
    print("=" * 60)
    print(f"  L(N,D) = {fit.E:.4f} + {fit.A:.2f}/N^{fit.alpha:.4f} + {fit.B:.2f}/D^{fit.beta:.4f}")
    print(f"  Hetero overhead: φ_bubble={fit.phi_bubble:.5f}  φ_numa={fit.phi_numa:.6f}")
    print(f"  RMSE = {fit.rmse:.5f}   R² = {fit.r_squared:.5f}   N_points = {fit.n_points}")
    print("=" * 60)

    # Predict optimal allocation if compute budget provided
    if args.compute_budget is not None:
        N_opt, D_opt, L_opt = predict_optimal_allocation(fit, args.compute_budget)
        print(f"\n  Optimal allocation for C = {args.compute_budget:.2e} FLOPs:")
        print(f"    N* = {N_opt:,}  ({N_opt/1e6:.1f}M params)")
        print(f"    D* = {D_opt:,}  ({D_opt/1e9:.1f}B tokens)")
        print(f"    L* = {L_opt:.4f}")

    # Save JSON results
    result_dict = asdict(fit)
    result_dict["measured_points"] = [{
        "model_name":
        pt.model_name,
        "N":
        pt.N,
        "D":
        pt.D,
        "L_measured":
        pt.L,
        "L_predicted":
        hetero_loss(
            pt.N,
            pt.D,
            pt.pipeline_bubble_ratio,
            pt.cross_numa_latency_ms,
            fit.E,
            fit.A,
            fit.alpha,
            fit.B,
            fit.beta,
            fit.phi_bubble,
            fit.phi_numa,
        ),
        "residual":
        pt.L - hetero_loss(
            pt.N,
            pt.D,
            pt.pipeline_bubble_ratio,
            pt.cross_numa_latency_ms,
            fit.E,
            fit.A,
            fit.alpha,
            fit.B,
            fit.beta,
            fit.phi_bubble,
            fit.phi_numa,
        ),
    } for pt in points]
    out_json = os.path.join(out_dir, "scaling_fit_results.json")
    with open(out_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n[fit_scaling_curve] Results saved: {out_json}")

    # Plot
    _try_plot(points, fit, os.path.join(out_dir, "scaling_fit_plot.png"))


def predict_7b_training(results_json: str = None) -> dict:
    """Predict loss for a 7B model at various token budgets.

    Uses the Chinchilla-style scaling law: L(N, D) = E + A/N^alpha + B/D^beta
    with parameters fitted from prior experiments on the heterogeneous cluster.

    Returns a dict with predictions and writes to scaling_7b_predictions.json.
    """
    if results_json is None:
        results_json = os.path.join(
            os.path.dirname(__file__), "scaling_fit_results.json"
        )
    with open(results_json) as f:
        params = json.load(f)

    E = params["E"]
    A = params["A"]
    alpha = params["alpha"]
    B = params["B"]
    beta = params["beta"]

    N = 7e9  # 7B parameters
    token_budgets = [20e9, 50e9, 100e9, 140e9, 300e9]

    predictions = []
    for D in token_budgets:
        L = E + A / (N ** alpha) + B / (D ** beta)
        total_flops = 6 * N * D
        predictions.append({
            "tokens_B": D / 1e9,
            "predicted_loss": round(L, 4),
            "total_pflops": round(total_flops / 1e15, 1),
            "chinchilla_ratio": round(D / N, 1),
        })

    result = {
        "model_params": "7B",
        "scaling_law": f"L = {E:.4f} + {A:.2f}/N^{alpha:.4f} + {B:.2f}/D^{beta:.4f}",
        "predictions": predictions,
    }

    out_path = os.path.join(os.path.dirname(__file__), "scaling_7b_predictions.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[predict_7b] Saved to {out_path}")
    for p in predictions:
        print(f"  {p['tokens_B']:.0f}B tokens -> loss={p['predicted_loss']:.4f}")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--predict-7b":
        predict_7b_training()
    else:
        main()
