# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
run_ablation.py — Automated DES-LOC ablation over Kx / Ku / Kv sync periods
=============================================================================
Reads the experiment matrix from experiments/scaling_law/experiment_matrix.yaml,
then sweeps every (Kx, Ku, Kv) combination from the configured grids for a
configurable number of training steps (default 1000).

For each configuration the script records:
  - final training loss
  - tokens / second (cluster-level)
  - total communication bytes transferred

Results are written as JSON files under experiments/ablation_results/.

Because the full GPU run requires real hardware, this script supports two modes:
  --mode simulate   (default) deterministic synthetic benchmark based on the
                    scaling-law fit + α-β communication model
  --mode live       actually invoke deepspeed training (requires GPUs)

Usage:
    python experiments/run_ablation.py                          # simulate
    python experiments/run_ablation.py --mode live --steps 500  # real GPUs
    python experiments/run_ablation.py --model medium_410m      # single model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_MATRIX_PATH = _SCRIPT_DIR / 'scaling_law' / 'experiment_matrix.yaml'
_RESULTS_DIR = _SCRIPT_DIR / 'ablation_results'
_SCALING_FIT = _SCRIPT_DIR / 'scaling_law' / 'scaling_fit_results.json'

# ---------------------------------------------------------------------------
# Sweep grids (from task spec)
# ---------------------------------------------------------------------------
KX_GRID = [1, 2, 4, 8, 16, 32]
KU_GRID = [1, 2, 4, 8]
KV_GRID = [4, 8, 16, 32, 64]

DEFAULT_STEPS = 1000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AblationConfig:
    """Single ablation point: model + (Kx, Ku, Kv) + training hyper-params."""
    model_name: str
    Kx: int
    Ku: int
    Kv: int
    steps: int
    hidden_size: int
    num_layers: int
    num_heads: int
    vocab_size: int
    max_seq_len: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    # Cluster info for communication modelling
    num_gpus: int = 3
    pcie_bw_gbs: float = 25.0  # conservative: A6000 bottleneck
    params_approx: float = 0.0


@dataclass
class AblationResult:
    """Metrics recorded for a single ablation point."""
    config: Dict[str, Any]
    final_loss: float
    tokens_per_second: float
    communication_bytes: int
    loss_curve: List[float]
    wall_time_seconds: float
    comm_reduction_ratio: float


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------
def load_experiment_matrix(path: str = str(_MATRIX_PATH)) -> Dict[str, Any]:
    """Parse experiments/scaling_law/experiment_matrix.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_scaling_fit(path: str = str(_SCALING_FIT)) -> Optional[Dict[str, Any]]:
    """Load Chinchilla fit parameters for simulation mode."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Communication model — α + βN (latency + bandwidth)
# ---------------------------------------------------------------------------
def estimate_communication_bytes(
    params_approx: float,
    steps: int,
    Kx: int,
    Ku: int,
    Kv: int,
    num_gpus: int,
) -> int:
    """Estimate total bytes transferred during training.

    DES-LOC decomposes allreduce into three independent sync channels:
      x (parameters):  synced every Kx steps  → 4 bytes × params per sync
      u (momentum):    synced every Ku steps  → 4 bytes × params per sync
      v (variance):    synced every Kv steps  → 4 bytes × params per sync

    Standard DDP (K=1 for all) transfers 4 × params × (num_gpus-1)/num_gpus
    per step for the gradient allreduce. DES-LOC replaces that with the three
    independent channels at lower frequency.
    """
    bytes_per_param = 4  # fp32 allreduce
    # ring-allreduce transfers (n-1)/n × total_bytes per collective
    ring_factor = (num_gpus - 1) / num_gpus if num_gpus > 1 else 1.0
    param_bytes = int(params_approx) * bytes_per_param

    n_sync_x = steps // Kx if Kx > 0 else steps
    n_sync_u = steps // Ku if Ku > 0 else steps
    n_sync_v = steps // Kv if Kv > 0 else steps

    total = int(ring_factor * param_bytes * (n_sync_x + n_sync_u + n_sync_v))
    return total


def comm_reduction_ratio(Kx: int, Ku: int, Kv: int) -> float:
    """Communication reduction vs. DDP baseline (K=1 everywhere).

    DDP syncs 3 streams every step → ratio = 1 - (1/Kx + 1/Ku + 1/Kv) / 3
    """
    baseline = 3.0  # three full syncs per step
    reduced = 1.0 / max(Kx, 1) + 1.0 / max(Ku, 1) + 1.0 / max(Kv, 1)
    return 1.0 - reduced / baseline


# ---------------------------------------------------------------------------
# Simulation mode — deterministic synthetic benchmark
# ---------------------------------------------------------------------------
def _chinchilla_loss(N: float, D: float, E: float, A: float, alpha: float,
                     B: float, beta: float) -> float:
    return E + A / (N**alpha) + B / (D**beta)


def simulate_run(cfg: AblationConfig, scaling_fit: Optional[Dict]) -> AblationResult:
    """Produce deterministic ablation metrics without real GPUs.

    Loss model: Chinchilla L(N, D) + convergence penalty from delayed sync.
    The penalty grows with sync period K — larger K means staler gradients,
    which slows convergence. We model this as:
        L_penalty = γ × log(1 + Kx) × log(1 + Ku) × log(1 + Kv)
    where γ is calibrated so that K=(32,8,64) gives ~8% loss degradation
    relative to K=(1,1,4).

    Throughput model: baseline tokens/s improved by overlap fraction from
    reduced communication. Each sync-skip lets the GPU compute instead of
    waiting on the collective.
    """
    t0 = time.monotonic()

    # Scaling law parameters
    if scaling_fit is not None:
        E = scaling_fit['E']
        A = scaling_fit['A']
        alpha = scaling_fit['alpha']
        B = scaling_fit['B']
        beta = scaling_fit['beta']
    else:
        # Hoffmann et al. defaults
        E, A, alpha = 1.69, 406.4, 0.34
        B, beta = 410.7, 0.28

    tokens_per_step = (cfg.micro_batch_size * cfg.gradient_accumulation_steps
                       * cfg.max_seq_len * cfg.num_gpus)

    # Deterministic seed based on config for reproducibility
    seed_str = f'{cfg.model_name}_{cfg.Kx}_{cfg.Ku}_{cfg.Kv}_{cfg.steps}'
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Convergence penalty from delayed synchronisation
    gamma = 0.012  # calibrated so worst-case gives ~8% degradation
    sync_penalty = gamma * math.log(1 + cfg.Kx) * math.log(1 + cfg.Ku) * math.log(1 + cfg.Kv)

    loss_curve = []
    for step in range(1, cfg.steps + 1):
        D_so_far = step * tokens_per_step
        base_loss = _chinchilla_loss(cfg.params_approx, max(D_so_far, 1), E, A, alpha, B, beta)
        # Stochastic gradient noise decays as training proceeds
        noise = rng.gauss(0, 0.02 / math.sqrt(step))
        loss = base_loss + sync_penalty + noise
        loss_curve.append(round(loss, 6))

    final_loss = loss_curve[-1]

    # Throughput: baseline + overlap savings from reduced sync frequency
    cr = comm_reduction_ratio(cfg.Kx, cfg.Ku, cfg.Kv)
    # On PCIe clusters, communication can consume 15-30% of step time
    comm_fraction = 0.22
    speedup = 1.0 + cr * comm_fraction
    # Base tokens/s scales with model size (smaller models are faster)
    base_tps = 50000 * (70e6 / max(cfg.params_approx, 1))**0.5
    tokens_per_second = round(base_tps * speedup, 1)

    comm_bytes = estimate_communication_bytes(
        cfg.params_approx, cfg.steps, cfg.Kx, cfg.Ku, cfg.Kv, cfg.num_gpus)

    wall_time = time.monotonic() - t0

    return AblationResult(
        config={
            'model_name': cfg.model_name,
            'Kx': cfg.Kx,
            'Ku': cfg.Ku,
            'Kv': cfg.Kv,
            'steps': cfg.steps,
            'params_approx': cfg.params_approx,
            'hidden_size': cfg.hidden_size,
            'num_layers': cfg.num_layers,
        },
        final_loss=final_loss,
        tokens_per_second=tokens_per_second,
        communication_bytes=comm_bytes,
        loss_curve=loss_curve,
        wall_time_seconds=round(wall_time, 4),
        comm_reduction_ratio=round(cr, 4),
    )


# ---------------------------------------------------------------------------
# Live mode — placeholder for real GPU training
# ---------------------------------------------------------------------------
def live_run(cfg: AblationConfig) -> AblationResult:
    """Execute a real training run via deepspeed.

    This is a placeholder that documents the expected integration point.
    When real GPUs are available, this function should:
      1. Build the deepspeed config with DES-LOC sync periods
      2. Launch run_pretrain.py with the appropriate arguments
      3. Parse the resulting log for loss / throughput / comm metrics
    """
    raise NotImplementedError(
        'Live mode requires GPU hardware. Use --mode simulate for offline ablation, '
        'or implement the deepspeed launch integration here.'
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_configs(
    matrix: Dict[str, Any],
    model_filter: Optional[str],
    steps: int,
) -> List[AblationConfig]:
    """Generate all (model × Kx × Ku × Kv) ablation configurations."""
    cluster = matrix['cluster']
    num_gpus = len(cluster['gpus'])
    # Use the slowest PCIe link as the bottleneck
    pcie_bw = min(g['pcie_bw_gbs'] for g in cluster['gpus'])

    configs = []
    for exp in matrix['experiments']:
        name = exp['name']
        if model_filter is not None and name != model_filter:
            continue

        for kx in KX_GRID:
            for ku in KU_GRID:
                for kv in KV_GRID:
                    configs.append(AblationConfig(
                        model_name=name,
                        Kx=kx,
                        Ku=ku,
                        Kv=kv,
                        steps=steps,
                        hidden_size=exp['hidden_size'],
                        num_layers=exp['num_layers'],
                        num_heads=exp['num_heads'],
                        vocab_size=exp['vocab_size'],
                        max_seq_len=exp['max_seq_len'],
                        micro_batch_size=exp['micro_batch_size'],
                        gradient_accumulation_steps=exp['gradient_accumulation_steps'],
                        learning_rate=exp['learning_rate'],
                        num_gpus=num_gpus,
                        pcie_bw_gbs=pcie_bw,
                        params_approx=float(exp['params_approx']),
                    ))
    return configs


def run_ablation(
    mode: str,
    model_filter: Optional[str] = None,
    steps: int = DEFAULT_STEPS,
    matrix_path: Optional[pathlib.Path] = None,
) -> List[AblationResult]:
    """Run the full ablation sweep and persist results."""
    path = str(matrix_path) if matrix_path is not None else str(_MATRIX_PATH)
    matrix = load_experiment_matrix(path)
    scaling_fit = load_scaling_fit()
    configs = build_configs(matrix, model_filter, steps)

    print(f'[ablation] Mode: {mode}')
    print(f'[ablation] Sweep: Kx={KX_GRID}, Ku={KU_GRID}, Kv={KV_GRID}')
    print(f'[ablation] Configs to run: {len(configs)}')
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: List[AblationResult] = []
    for i, cfg in enumerate(configs, 1):
        tag = f'{cfg.model_name}_Kx{cfg.Kx}_Ku{cfg.Ku}_Kv{cfg.Kv}'
        print(f'  [{i}/{len(configs)}] {tag} ...', end=' ', flush=True)

        if mode == 'simulate':
            result = simulate_run(cfg, scaling_fit)
        else:
            result = live_run(cfg)

        results.append(result)
        print(f'loss={result.final_loss:.4f}  tps={result.tokens_per_second:.0f}  '
              f'comm={result.communication_bytes / 1e9:.2f}GB')

    # Write per-model summary files
    model_groups: Dict[str, List[AblationResult]] = {}
    for r in results:
        mn = r.config['model_name']
        model_groups.setdefault(mn, []).append(r)

    for model_name, model_results in model_groups.items():
        out_path = _RESULTS_DIR / f'ablation_{model_name}.json'
        payload = {
            'model_name': model_name,
            'sweep': {'Kx': KX_GRID, 'Ku': KU_GRID, 'Kv': KV_GRID},
            'steps': steps,
            'n_configs': len(model_results),
            'results': [_result_to_dict(r) for r in model_results],
        }
        with open(out_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'[ablation] Saved {out_path}')

    # Write combined summary
    summary_path = _RESULTS_DIR / 'ablation_summary.json'
    summary = {
        'total_configs': len(results),
        'steps_per_config': steps,
        'mode': mode,
        'models': list(model_groups.keys()),
        'sweep': {'Kx': KX_GRID, 'Ku': KU_GRID, 'Kv': KV_GRID},
        'best_per_model': {},
    }
    for mn, mr in model_groups.items():
        best = min(mr, key=lambda r: r.final_loss)
        summary['best_per_model'][mn] = {
            'Kx': best.config['Kx'],
            'Ku': best.config['Ku'],
            'Kv': best.config['Kv'],
            'final_loss': best.final_loss,
            'tokens_per_second': best.tokens_per_second,
            'comm_reduction_ratio': best.comm_reduction_ratio,
        }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[ablation] Summary saved {summary_path}')

    return results


def _result_to_dict(r: AblationResult) -> Dict[str, Any]:
    """Serialise an AblationResult without the full loss curve to save space."""
    return {
        'Kx': r.config['Kx'],
        'Ku': r.config['Ku'],
        'Kv': r.config['Kv'],
        'final_loss': r.final_loss,
        'tokens_per_second': r.tokens_per_second,
        'communication_bytes': r.communication_bytes,
        'comm_reduction_ratio': r.comm_reduction_ratio,
        'wall_time_seconds': r.wall_time_seconds,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='DES-LOC ablation sweep over Kx / Ku / Kv sync periods')
    parser.add_argument(
        '--mode', choices=['simulate', 'live'], default='simulate',
        help='simulate = deterministic synthetic run; live = real GPU training')
    parser.add_argument(
        '--model', default=None,
        help='Run ablation for a single model (e.g. tiny_70m). Default: all.')
    parser.add_argument(
        '--steps', type=int, default=DEFAULT_STEPS,
        help=f'Training steps per configuration (default {DEFAULT_STEPS})')
    parser.add_argument(
        '--matrix', default=str(_MATRIX_PATH),
        help='Path to experiment_matrix.yaml')
    args = parser.parse_args()

    matrix_path = pathlib.Path(args.matrix)
    run_ablation(mode=args.mode, model_filter=args.model, steps=args.steps,
                 matrix_path=matrix_path)


if __name__ == '__main__':
    main()
