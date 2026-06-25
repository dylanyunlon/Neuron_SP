# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
convergence_analysis.py — DES-LOC convergence analysis from training logs
=========================================================================
Reads benchmark_results_*.json logs from one or more directories and:
  1. Extracts per-step loss curves for DDP, DESLOC, and LocalAdam runs
  2. Fits a power-law decay L(t) = a * t^(-b) + c to each loss curve
  3. Computes the effective Chinchilla ratio (tokens/params) per run
  4. Compares DES-LOC vs DDP convergence speed (steps to target loss)
  5. Cross-references with scaling_7b_predictions.json expected losses
  6. Outputs a convergence_report.json with all analysis results

Usage:
    python experiments/convergence_analysis.py --log-dir desloc_results
    python experiments/convergence_analysis.py --log-dir desloc_results --output report.json
    python experiments/convergence_analysis.py --log-dir desloc_results desloc_results_0422

The power-law fit uses the form L(t) = a * t^(-b) + c, where t is the step
index (starting from 1). This captures the diminishing-returns shape of
typical loss curves and lets us extrapolate convergence rates without
requiring the full Chinchilla parameterization (which needs N and D).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RunInfo:
    """Parsed information from a single benchmark log file."""
    filepath: str
    timestamp: str
    model_size: str
    method: str  # 'DDP', 'DESLOC', 'LocalAdam'
    max_steps: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    Kx: int
    Ku: int
    Kv: int
    final_loss: float
    avg_loss: float
    total_time_seconds: float
    tokens_per_second_per_gpu: float
    tokens_per_second_cluster: float
    peak_memory_gb: float
    mfu: float
    sync_counts: Dict[str, int]
    losses: List[float]
    max_seq_len: int = 1024


@dataclass
class PowerLawFit:
    """Result of fitting L(t) = a * t^(-b) + c to a loss curve."""
    a: float  # scale coefficient
    b: float  # decay exponent (higher = faster convergence)
    c: float  # asymptotic loss floor
    rmse: float
    r_squared: float
    n_points: int


@dataclass
class ConvergenceComparison:
    """Comparison of convergence between two methods on the same model size."""
    model_size: str
    method_a: str
    method_b: str
    # Power-law exponents
    decay_rate_a: float
    decay_rate_b: float
    # Steps to reach various loss thresholds (None if never reached)
    steps_to_target: Dict[str, Dict[str, Optional[int]]]
    # Convergence speed ratio: >1 means method_a converges faster
    convergence_speed_ratio: float
    # Final loss comparison
    final_loss_a: float
    final_loss_b: float
    loss_gap: float  # final_loss_b - final_loss_a (positive = A is better)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def parse_log_file(filepath: str) -> List[RunInfo]:
    """Parse a benchmark_results JSON file into RunInfo objects.

    A single file may contain one result key (DDP, DESLOC, or LocalAdam).
    Returns a list since the format allows multiple results per file.
    """
    with open(filepath) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    timestamp = data.get("timestamp", "")
    runs = []

    for method, result in data.get("results", {}).items():
        losses = result.get("losses", [])
        if not losses:
            continue

        runs.append(RunInfo(
            filepath=filepath,
            timestamp=timestamp,
            model_size=cfg.get("model_size", "unknown"),
            method=method,
            max_steps=cfg.get("max_steps", len(losses)),
            batch_size=cfg.get("batch_size", 0),
            gradient_accumulation=cfg.get("gradient_accumulation", 1),
            learning_rate=cfg.get("learning_rate", 0.0),
            Kx=cfg.get("Kx", 1),
            Ku=cfg.get("Ku", 1),
            Kv=cfg.get("Kv", 1),
            final_loss=result.get("final_loss", losses[-1]),
            avg_loss=result.get("avg_loss", sum(losses) / len(losses)),
            total_time_seconds=result.get("total_time_seconds", 0.0),
            tokens_per_second_per_gpu=result.get("tokens_per_second_per_gpu", 0.0),
            tokens_per_second_cluster=result.get("tokens_per_second_cluster", 0.0),
            peak_memory_gb=result.get("peak_memory_gb", 0.0),
            mfu=result.get("mfu", 0.0),
            sync_counts=result.get("sync_counts", {}),
            losses=losses,
            max_seq_len=cfg.get("max_seq_len", 1024),
        ))

    return runs


def scan_log_directories(log_dirs: List[str]) -> List[RunInfo]:
    """Scan one or more directories for benchmark JSON files."""
    all_runs = []
    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            print(f"[convergence] Warning: {log_dir} is not a directory, skipping.")
            continue
        for fname in sorted(os.listdir(log_dir)):
            if not fname.startswith("benchmark_results_") or not fname.endswith(".json"):
                continue
            fpath = os.path.join(log_dir, fname)
            try:
                runs = parse_log_file(fpath)
                all_runs.extend(runs)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[convergence] Warning: failed to parse {fpath}: {e}")
    return all_runs


# ---------------------------------------------------------------------------
# Power-law fitting via Nelder-Mead (no scipy dependency)
# ---------------------------------------------------------------------------
def _nelder_mead_minimize(
    objective,
    x0: List[float],
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> Tuple[List[float], float]:
    """Minimal Nelder-Mead optimizer for small-dimensional problems."""
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] *= 1.05 if abs(point[i]) > 1e-8 else 0.00025
        simplex.append(point)

    f_vals = [objective(x) for x in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: f_vals[i])
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        if max(abs(f_vals[i] - f_vals[0]) for i in range(1, n + 1)) < tol:
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]

        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = objective(xr)

        if f_vals[0] <= fr < f_vals[-2]:
            simplex[-1] = xr
            f_vals[-1] = fr
        elif fr < f_vals[0]:
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
        else:
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = objective(xc)
            if fc < f_vals[-1]:
                simplex[-1] = xc
                f_vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                f_vals = [objective(x) for x in simplex]

    return simplex[0], f_vals[0]


def fit_power_law(losses: List[float]) -> PowerLawFit:
    """Fit L(t) = a * t^(-b) + c to a loss curve.

    We skip the first step (t=1) to avoid fitting to random initialization
    noise, then use steps t=2..T. The exponent b captures convergence rate:
    larger b means faster initial convergence.
    """
    if len(losses) < 5:
        return PowerLawFit(a=0.0, b=0.0, c=losses[-1] if losses else 0.0, rmse=0.0, r_squared=0.0, n_points=0)

    # Skip first point (random init), subsample if >200 points for speed
    skip = 1
    steps = list(range(skip + 1, len(losses) + 1))
    vals = losses[skip:]
    if len(steps) > 200:
        stride = len(steps) // 200
        steps = steps[::stride]
        vals = vals[::stride]

    def objective(params: List[float]) -> float:
        a, b, c = params
        if b < 0.001 or a < 0.0:
            return 1e12
        total = 0.0
        for t, L in zip(steps, vals):
            try:
                L_pred = a * (t ** (-b)) + c
                total += (L_pred - L) ** 2
            except (OverflowError, ValueError):
                total += 1e6
        return total

    # Initial guess: a ≈ range of loss, b ≈ 0.5 (typical), c ≈ final loss
    a0 = max(vals[0] - vals[-1], 0.1)
    c0 = vals[-1]
    x0 = [a0, 0.5, c0]

    best_params, _ = _nelder_mead_minimize(objective, x0, max_iter=8000)
    a, b, c = best_params
    b = max(b, 0.0)

    # Compute RMSE and R²
    L_mean = sum(vals) / len(vals)
    ss_tot = sum((v - L_mean) ** 2 for v in vals)
    ss_res = 0.0
    for t, L in zip(steps, vals):
        L_pred = a * (t ** (-b)) + c
        ss_res += (L_pred - L) ** 2
    rmse = math.sqrt(ss_res / len(vals)) if vals else 0.0
    r_sq = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return PowerLawFit(a=a, b=b, c=c, rmse=rmse, r_squared=r_sq, n_points=len(vals))


# ---------------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------------
def _steps_to_loss(losses: List[float], target: float) -> Optional[int]:
    """Find first step where loss drops to or below the target."""
    for i, L in enumerate(losses):
        if L <= target:
            return i + 1
    return None


def _compute_chinchilla_ratio(run: RunInfo) -> float:
    """Compute effective tokens/params ratio (Chinchilla ratio).

    tokens_trained = steps × batch_size × grad_accum × seq_len × n_gpus
    For simplicity, use tokens_per_second_cluster × total_time as a cross-check.
    """
    model_params = _parse_model_params(run.model_size)
    if model_params == 0:
        return 0.0
    # Estimate total tokens from throughput × time
    total_tokens = run.tokens_per_second_cluster * run.total_time_seconds
    if total_tokens == 0:
        # Fallback: steps × batch × grad_accum × seq_len (single-GPU view)
        total_tokens = run.max_steps * run.batch_size * run.gradient_accumulation * run.max_seq_len
    return total_tokens / model_params


def _parse_model_params(model_size_str: str) -> float:
    """Parse model size string like '125M', '700M', '1.3B', '7B', '13B'."""
    s = model_size_str.strip().upper()
    if s.endswith("B"):
        return float(s[:-1]) * 1e9
    elif s.endswith("M"):
        return float(s[:-1]) * 1e6
    return 0.0


def _comm_reduction_ratio(run: RunInfo) -> float:
    """Compute communication reduction ratio from sync periods.

    For DES-LOC: ratio = 1 - (1/Kx + 1/Ku + 1/Kv)
    For DDP (K=1): ratio = 0 (full sync every step)
    """
    if run.Kx <= 1 and run.Ku <= 1 and run.Kv <= 1:
        return 0.0
    return 1.0 - (1.0 / max(run.Kx, 1) + 1.0 / max(run.Ku, 1) + 1.0 / max(run.Kv, 1))


def _build_best_config(groups: Dict[str, Dict[str, List[RunInfo]]]) -> dict:
    """Find the best (lowest final_loss) DESLOC config per model_size and overall."""
    best_per_size: Dict[str, dict] = {}
    global_best: Optional[dict] = None

    for model_size, methods in sorted(groups.items()):
        desloc_runs = [r for r in methods.get("DESLOC", []) if len(r.losses) >= 5]
        if not desloc_runs:
            continue
        best_run = min(desloc_runs, key=lambda r: r.final_loss)
        entry = {
            "model_size": best_run.model_size,
            "Kx": best_run.Kx,
            "Ku": best_run.Ku,
            "Kv": best_run.Kv,
            "final_loss": round(best_run.final_loss, 6),
            "avg_loss": round(best_run.avg_loss, 6),
            "comm_reduction_ratio": round(_comm_reduction_ratio(best_run), 4),
            "mfu": round(best_run.mfu, 6),
            "tokens_per_second_cluster": round(best_run.tokens_per_second_cluster, 1),
            "timestamp": best_run.timestamp,
        }
        best_per_size[model_size] = entry
        if global_best is None or best_run.final_loss < global_best["final_loss"]:
            global_best = entry

    return {
        "overall": global_best,
        "per_model_size": best_per_size,
        "note": (
            "best_config shows the DESLOC K-configuration achieving the lowest "
            "final training loss for each model size across all benchmark runs"
        ),
    }


def _build_throughput_comparison(groups: Dict[str, Dict[str, List[RunInfo]]]) -> dict:
    """Compare throughput between DESLOC and DDP baseline per model_size."""
    comparison: Dict[str, dict] = {}

    for model_size in sorted(groups.keys()):
        methods = groups[model_size]
        ddp_runs = [r for r in methods.get("DDP", []) if r.tokens_per_second_cluster > 0]
        desloc_runs = [r for r in methods.get("DESLOC", []) if r.tokens_per_second_cluster > 0]
        localadam_runs = [r for r in methods.get("LocalAdam", []) if r.tokens_per_second_cluster > 0]

        if not ddp_runs:
            continue

        def _agg(runs: List[RunInfo]) -> dict:
            if not runs:
                return {}
            best = max(runs, key=lambda r: r.tokens_per_second_cluster)
            return {
                "max_tokens_per_second_cluster": round(best.tokens_per_second_cluster, 1),
                "max_tokens_per_second_per_gpu": round(best.tokens_per_second_per_gpu, 1),
                "max_mfu": round(best.mfu, 6),
                "peak_memory_gb": round(best.peak_memory_gb, 3),
                "n_runs": len(runs),
            }

        ddp_best_tps = max(r.tokens_per_second_cluster for r in ddp_runs)
        desloc_ratio = (
            round(max(r.tokens_per_second_cluster for r in desloc_runs) / ddp_best_tps, 4)
            if desloc_runs else None
        )
        localadam_ratio = (
            round(max(r.tokens_per_second_cluster for r in localadam_runs) / ddp_best_tps, 4)
            if localadam_runs else None
        )

        comparison[model_size] = {
            "DDP": _agg(ddp_runs),
            "DESLOC": _agg(desloc_runs) if desloc_runs else None,
            "LocalAdam": _agg(localadam_runs) if localadam_runs else None,
            "desloc_vs_ddp_throughput_ratio": desloc_ratio,
            "localadam_vs_ddp_throughput_ratio": localadam_ratio,
        }

    return {
        "by_model_size": comparison,
        "note": (
            "throughput_comparison reports peak tokens/second for each method. "
            "desloc_vs_ddp_throughput_ratio > 1 means DESLOC is faster than DDP baseline."
        ),
    }


def _build_comm_reduction(runs: List[RunInfo]) -> dict:
    """Compute weighted communication reduction across all DESLOC runs."""
    config_stats: Dict[str, dict] = {}

    for run in runs:
        key = f"Kx={run.Kx}_Ku={run.Ku}_Kv={run.Kv}"
        if key not in config_stats:
            config_stats[key] = {
                "Kx": run.Kx,
                "Ku": run.Ku,
                "Kv": run.Kv,
                "comm_reduction_ratio": round(_comm_reduction_ratio(run), 4),
                "n_runs": 0,
                "model_sizes": set(),
                "total_steps": 0,
                "weighted_comm_reduction": 0.0,
            }
        s = config_stats[key]
        s["n_runs"] += 1
        s["model_sizes"].add(run.model_size)
        steps = len(run.losses)
        s["total_steps"] += steps
        s["weighted_comm_reduction"] += _comm_reduction_ratio(run) * steps

    # Compute overall weighted comm reduction (weighted by total training steps)
    total_steps_all = sum(s["total_steps"] for s in config_stats.values())
    weighted_avg = (
        sum(s["weighted_comm_reduction"] for s in config_stats.values()) / total_steps_all
        if total_steps_all > 0 else 0.0
    )

    # Serialize (convert sets to sorted lists)
    serializable = {}
    for k, s in sorted(config_stats.items()):
        serializable[k] = {
            "Kx": s["Kx"],
            "Ku": s["Ku"],
            "Kv": s["Kv"],
            "comm_reduction_ratio": s["comm_reduction_ratio"],
            "n_runs": s["n_runs"],
            "model_sizes": sorted(s["model_sizes"]),
            "total_training_steps": s["total_steps"],
        }

    baseline_key = "Kx=1_Ku=1_Kv=1"
    return {
        "formula": "comm_reduction = 1 - (1/Kx + 1/Ku + 1/Kv)",
        "baseline": baseline_key,
        "configs": serializable,
        "overall_weighted_comm_reduction": round(weighted_avg, 4),
        "note": (
            "comm_reduction measures the fraction of communication rounds eliminated vs full sync. "
            "overall_weighted_comm_reduction is weighted by total training steps across all runs."
        ),
    }


def _build_loss_trajectories(groups: Dict[str, Dict[str, List[RunInfo]]]) -> dict:
    """Build per-config grouped loss curves for plotting and analysis."""
    trajectories: Dict[str, dict] = {}

    for model_size in sorted(groups.keys()):
        methods = groups[model_size]
        trajectories[model_size] = {}

        for method in sorted(methods.keys()):
            method_runs = [r for r in methods[method] if len(r.losses) >= 5]
            if not method_runs:
                continue

            # Group by K-config
            k_groups: Dict[str, List[RunInfo]] = {}
            for run in method_runs:
                k_key = f"Kx={run.Kx}_Ku={run.Ku}_Kv={run.Kv}"
                if k_key not in k_groups:
                    k_groups[k_key] = []
                k_groups[k_key].append(run)

            method_entry = {}
            for k_key, k_runs in sorted(k_groups.items()):
                # Pick the longest run as the representative trajectory
                rep = max(k_runs, key=lambda r: len(r.losses))
                # Subsample losses to max 100 points for report size
                losses = rep.losses
                if len(losses) > 100:
                    stride = len(losses) // 100
                    losses = losses[::stride]

                method_entry[k_key] = {
                    "Kx": rep.Kx,
                    "Ku": rep.Ku,
                    "Kv": rep.Kv,
                    "n_steps_total": len(rep.losses),
                    "n_steps_sampled": len(losses),
                    "final_loss": round(rep.final_loss, 6),
                    "comm_reduction_ratio": round(_comm_reduction_ratio(rep), 4),
                    "representative_timestamp": rep.timestamp,
                    "n_runs_in_group": len(k_runs),
                    "losses": [round(v, 6) for v in losses],
                }

            trajectories[model_size][method] = method_entry

    return {
        "by_model_size": trajectories,
        "note": (
            "loss_trajectories groups training loss curves by (model_size, method, K-config). "
            "Losses are subsampled to ≤100 points using the longest representative run per group."
        ),
    }


def compare_convergence(
    runs_a: List[RunInfo],
    runs_b: List[RunInfo],
    method_a: str,
    method_b: str,
    model_size: str,
) -> Optional[ConvergenceComparison]:
    """Compare convergence between two methods on the same model size.

    Uses the run with the most steps from each group. Returns None if either
    group is empty.
    """
    if not runs_a or not runs_b:
        return None

    # Pick the longest run from each group
    best_a = max(runs_a, key=lambda r: len(r.losses))
    best_b = max(runs_b, key=lambda r: len(r.losses))

    fit_a = fit_power_law(best_a.losses)
    fit_b = fit_power_law(best_b.losses)

    # Steps-to-target at various loss thresholds
    min_loss = min(best_a.final_loss, best_b.final_loss)
    max_loss = max(best_a.losses[0], best_b.losses[0])
    targets = {}
    for frac in [0.9, 0.8, 0.7, 0.6]:
        threshold = min_loss + frac * (max_loss - min_loss)
        label = f"L={threshold:.2f}"
        s_a = _steps_to_loss(best_a.losses, threshold)
        s_b = _steps_to_loss(best_b.losses, threshold)
        targets[label] = {method_a: s_a, method_b: s_b}

    # Convergence speed ratio based on decay exponents
    speed_ratio = fit_a.b / fit_b.b if fit_b.b > 1e-6 else float("inf")

    return ConvergenceComparison(
        model_size=model_size,
        method_a=method_a,
        method_b=method_b,
        decay_rate_a=fit_a.b,
        decay_rate_b=fit_b.b,
        steps_to_target=targets,
        convergence_speed_ratio=speed_ratio,
        final_loss_a=best_a.final_loss,
        final_loss_b=best_b.final_loss,
        loss_gap=best_b.final_loss - best_a.final_loss,
    )


# ---------------------------------------------------------------------------
# Scaling law cross-reference
# ---------------------------------------------------------------------------
def load_scaling_predictions(path: str) -> Optional[dict]:
    """Load scaling_7b_predictions.json for cross-referencing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cross_reference_scaling(
    runs: List[RunInfo],
    predictions: dict,
) -> List[dict]:
    """Compare observed 7B losses against Chinchilla-predicted losses."""
    results = []
    runs_7b = [r for r in runs if r.model_size.upper() in ("7B", "7.0B")]
    if not runs_7b or not predictions:
        return results

    pred_map = {p["tokens_B"]: p for p in predictions.get("predictions", [])}

    for run in runs_7b:
        chinchilla_ratio = _compute_chinchilla_ratio(run)
        model_params = _parse_model_params(run.model_size)
        total_tokens_b = (chinchilla_ratio * model_params) / 1e9 if model_params > 0 else 0.0

        # Find nearest prediction bucket
        nearest_key = None
        min_dist = float("inf")
        for tb in pred_map:
            dist = abs(tb - total_tokens_b)
            if dist < min_dist:
                min_dist = dist
                nearest_key = tb

        predicted_loss = pred_map[nearest_key]["predicted_loss"] if nearest_key is not None else None

        results.append({
            "method": run.method,
            "timestamp": run.timestamp,
            "total_tokens_B": round(total_tokens_b, 2),
            "chinchilla_ratio": round(chinchilla_ratio, 2),
            "observed_final_loss": run.final_loss,
            "nearest_predicted_tokens_B": nearest_key,
            "predicted_loss": predicted_loss,
            "loss_delta": round(run.final_loss - predicted_loss, 4) if predicted_loss is not None else None,
        })

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(
    runs: List[RunInfo],
    scaling_predictions_path: str,
    output_path: str,
) -> dict:
    """Generate the full convergence report."""
    # Group runs by (model_size, method)
    groups: Dict[str, Dict[str, List[RunInfo]]] = {}
    for run in runs:
        if run.model_size not in groups:
            groups[run.model_size] = {}
        if run.method not in groups[run.model_size]:
            groups[run.model_size][run.method] = []
        groups[run.model_size][run.method].append(run)

    # Per-run summaries with power-law fits
    run_summaries = []
    for run in runs:
        if len(run.losses) < 5:
            continue
        fit = fit_power_law(run.losses)
        run_summaries.append({
            "timestamp": run.timestamp,
            "model_size": run.model_size,
            "method": run.method,
            "max_steps": run.max_steps,
            "num_losses": len(run.losses),
            "K": {"Kx": run.Kx, "Ku": run.Ku, "Kv": run.Kv},
            "comm_reduction_ratio": round(_comm_reduction_ratio(run), 4),
            "chinchilla_ratio": round(_compute_chinchilla_ratio(run), 2),
            "final_loss": run.final_loss,
            "avg_loss": round(run.avg_loss, 4),
            "power_law_fit": {
                "a": round(fit.a, 6),
                "b": round(fit.b, 6),
                "c": round(fit.c, 6),
                "rmse": round(fit.rmse, 6),
                "r_squared": round(fit.r_squared, 6),
            },
            "tokens_per_second_cluster": round(run.tokens_per_second_cluster, 1),
            "mfu": round(run.mfu, 6),
            "peak_memory_gb": round(run.peak_memory_gb, 3),
        })

    # Convergence comparisons: DES-LOC vs DDP for each model size
    comparisons = []
    for model_size, methods in sorted(groups.items()):
        ddp_runs = methods.get("DDP", [])
        desloc_runs = methods.get("DESLOC", [])
        localadam_runs = methods.get("LocalAdam", [])

        # Only compare runs with enough steps (>=50) to be meaningful
        ddp_long = [r for r in ddp_runs if len(r.losses) >= 50]
        desloc_long = [r for r in desloc_runs if len(r.losses) >= 50]
        localadam_long = [r for r in localadam_runs if len(r.losses) >= 50]

        cmp = compare_convergence(desloc_long, ddp_long, "DESLOC", "DDP", model_size)
        if cmp is not None:
            comparisons.append({
                "model_size": cmp.model_size,
                "method_a": cmp.method_a,
                "method_b": cmp.method_b,
                "decay_rate_a": round(cmp.decay_rate_a, 6),
                "decay_rate_b": round(cmp.decay_rate_b, 6),
                "convergence_speed_ratio": round(cmp.convergence_speed_ratio, 4),
                "final_loss_a": round(cmp.final_loss_a, 6),
                "final_loss_b": round(cmp.final_loss_b, 6),
                "loss_gap": round(cmp.loss_gap, 6),
                "steps_to_target": cmp.steps_to_target,
            })

        cmp_la = compare_convergence(desloc_long, localadam_long, "DESLOC", "LocalAdam", model_size)
        if cmp_la is not None:
            comparisons.append({
                "model_size": cmp_la.model_size,
                "method_a": cmp_la.method_a,
                "method_b": cmp_la.method_b,
                "decay_rate_a": round(cmp_la.decay_rate_a, 6),
                "decay_rate_b": round(cmp_la.decay_rate_b, 6),
                "convergence_speed_ratio": round(cmp_la.convergence_speed_ratio, 4),
                "final_loss_a": round(cmp_la.final_loss_a, 6),
                "final_loss_b": round(cmp_la.final_loss_b, 6),
                "loss_gap": round(cmp_la.loss_gap, 6),
                "steps_to_target": cmp_la.steps_to_target,
            })

    # Scaling law cross-reference
    predictions = load_scaling_predictions(scaling_predictions_path)
    scaling_xref = cross_reference_scaling(runs, predictions) if predictions else []

    # Summary statistics per model_size × method
    summary_table = []
    for model_size in sorted(groups.keys()):
        for method in sorted(groups[model_size].keys()):
            method_runs = [r for r in groups[model_size][method] if len(r.losses) >= 5]
            if not method_runs:
                continue
            final_losses = [r.final_loss for r in method_runs]
            summary_table.append({
                "model_size": model_size,
                "method": method,
                "n_runs": len(method_runs),
                "best_final_loss": round(min(final_losses), 6),
                "worst_final_loss": round(max(final_losses), 6),
                "mean_final_loss": round(sum(final_losses) / len(final_losses), 6),
                "max_steps": max(len(r.losses) for r in method_runs),
            })

    # New required sections
    best_config = _build_best_config(groups)
    throughput_comparison = _build_throughput_comparison(groups)
    comm_reduction = _build_comm_reduction(runs)
    loss_trajectories = _build_loss_trajectories(groups)

    report = {
        "analysis": "DES-LOC Convergence Analysis",
        "log_directories_scanned": len(set(os.path.dirname(r.filepath) for r in runs)),
        "total_runs_parsed": len(runs),
        "total_runs_analyzed": len(run_summaries),
        "best_config": best_config,
        "throughput_comparison": throughput_comparison,
        "comm_reduction": comm_reduction,
        "loss_trajectories": loss_trajectories,
        "summary_table": summary_table,
        "convergence_comparisons": comparisons,
        "scaling_law_cross_reference": scaling_xref,
        "run_details": run_summaries,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze DES-LOC convergence from training logs",
    )
    parser.add_argument(
        "--log-dir",
        nargs="+",
        default=["desloc_results"],
        help="One or more directories containing benchmark_results_*.json files",
    )
    parser.add_argument(
        "--output",
        default="experiments/convergence_report.json",
        help="Path to write the convergence report JSON (default: experiments/convergence_report.json)",
    )
    parser.add_argument(
        "--scaling-predictions",
        default=None,
        help="Path to scaling_7b_predictions.json (auto-detected if not set)",
    )
    args = parser.parse_args()

    # Auto-detect scaling predictions path
    scaling_path = args.scaling_predictions
    if scaling_path is None:
        candidates = [
            os.path.join("experiments", "scaling_law", "scaling_7b_predictions.json"),
            os.path.join(os.path.dirname(__file__), "scaling_law", "scaling_7b_predictions.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                scaling_path = c
                break

    print(f"[convergence] Scanning log directories: {args.log_dir}")
    runs = scan_log_directories(args.log_dir)
    print(f"[convergence] Parsed {len(runs)} runs")

    if not runs:
        print("[convergence] No runs found. Check --log-dir paths.")
        sys.exit(1)

    # Group summary
    model_counts: Dict[str, Dict[str, int]] = {}
    for r in runs:
        if r.model_size not in model_counts:
            model_counts[r.model_size] = {}
        model_counts[r.model_size][r.method] = model_counts[r.model_size].get(r.method, 0) + 1
    for ms in sorted(model_counts.keys()):
        methods_str = ", ".join(f"{m}={c}" for m, c in sorted(model_counts[ms].items()))
        print(f"  {ms}: {methods_str}")

    if scaling_path and os.path.exists(scaling_path):
        print(f"[convergence] Using scaling predictions: {scaling_path}")
    else:
        print("[convergence] No scaling predictions found; skipping cross-reference.")

    report = generate_report(runs, scaling_path or "", args.output)

    n_cmp = len(report["convergence_comparisons"])
    print(f"\n[convergence] Report written: {args.output}")
    print(f"  Runs analyzed: {report['total_runs_analyzed']}")
    print(f"  Convergence comparisons: {n_cmp}")
    if report["scaling_law_cross_reference"]:
        print(f"  Scaling law cross-references: {len(report['scaling_law_cross_reference'])}")

    # Print key findings
    for cmp in report["convergence_comparisons"]:
        gap = cmp["loss_gap"]
        direction = "better" if gap > 0 else "worse"
        print(
            f"  {cmp['model_size']}: {cmp['method_a']} vs {cmp['method_b']}"
            f" — speed ratio={cmp['convergence_speed_ratio']:.3f},"
            f" loss gap={abs(gap):.4f} ({cmp['method_a']} {direction})"
        )


if __name__ == "__main__":
    main()
