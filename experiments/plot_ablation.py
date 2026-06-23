# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
plot_ablation.py — Heatmap visualisation of DES-LOC Kx / Ku / Kv ablation
===========================================================================
Reads JSON results from experiments/ablation_results/ (produced by
run_ablation.py) and generates publication-quality heatmaps showing the
interaction between sync periods and three key metrics:

  1. Final training loss          (lower is better)
  2. Cluster throughput (tok/s)   (higher is better)
  3. Communication volume (GB)    (lower is better)

For each model size the script produces a 3-panel figure (one panel per
metric) with Kx on the y-axis and Kv on the x-axis, marginalised over Ku
(or sliced at a fixed Ku if --ku-slice is given).

Additionally a combined "Pareto-front" scatter (loss vs comm_reduction) is
generated across all model sizes to highlight the optimal operating points.

Output directory: experiments/figures/

Usage:
    python experiments/plot_ablation.py                     # all models
    python experiments/plot_ablation.py --model tiny_70m    # one model
    python experiments/plot_ablation.py --ku-slice 2        # fix Ku=2
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_RESULTS_DIR = _SCRIPT_DIR / 'ablation_results'
_FIGURES_DIR = _SCRIPT_DIR / 'figures'

# ---------------------------------------------------------------------------
# Style — matches plot_figures.py for visual consistency
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ablation_results(
    model_name: Optional[str] = None,
    results_dir: Optional[pathlib.Path] = None,
) -> Dict[str, List[Dict]]:
    """Load ablation result JSON files, grouped by model name."""
    rdir = results_dir if results_dir is not None else _RESULTS_DIR
    model_results: Dict[str, List[Dict]] = {}
    if not rdir.exists():
        raise FileNotFoundError(f'No results directory: {rdir}')

    for fname in sorted(rdir.iterdir()):
        if not fname.name.startswith('ablation_') or not fname.name.endswith('.json'):
            continue
        if fname.name == 'ablation_summary.json':
            continue

        with open(fname) as f:
            data = json.load(f)

        mn = data.get('model_name', fname.stem.replace('ablation_', ''))
        if model_name is not None and mn != model_name:
            continue
        model_results[mn] = data.get('results', [])

    return model_results


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------
def _pivot_metric(
    results: List[Dict],
    metric: str,
    kx_vals: List[int],
    kv_vals: List[int],
    ku_slice: Optional[int] = None,
    aggregation: str = 'mean',
) -> np.ndarray:
    """Build a 2D array [Kx × Kv] for the given metric.

    If ku_slice is None, aggregate over all Ku values; otherwise filter to
    the specified Ku and take the single value.
    """
    grid = np.full((len(kx_vals), len(kv_vals)), np.nan)
    kx_idx = {v: i for i, v in enumerate(kx_vals)}
    kv_idx = {v: i for i, v in enumerate(kv_vals)}

    # Accumulate values per (Kx, Kv) cell
    accum: Dict[Tuple[int, int], List[float]] = {}
    for r in results:
        kx, ku, kv = r['Kx'], r['Ku'], r['Kv']
        if ku_slice is not None and ku != ku_slice:
            continue
        if kx not in kx_idx or kv not in kv_idx:
            continue
        key = (kx, kv)
        accum.setdefault(key, []).append(r[metric])

    for (kx, kv), vals in accum.items():
        if aggregation == 'mean':
            grid[kx_idx[kx], kv_idx[kv]] = sum(vals) / len(vals)
        elif aggregation == 'min':
            grid[kx_idx[kx], kv_idx[kv]] = min(vals)
        elif aggregation == 'max':
            grid[kx_idx[kx], kv_idx[kv]] = max(vals)

    return grid


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------
def plot_model_heatmaps(
    model_name: str,
    results: List[Dict],
    ku_slice: Optional[int] = None,
) -> None:
    """Generate a 3-panel heatmap figure for one model size."""
    kx_vals = sorted(set(r['Kx'] for r in results))
    kv_vals = sorted(set(r['Kv'] for r in results))

    metrics = [
        ('final_loss', 'Final Loss', 'YlOrRd', 'lower is better'),
        ('tokens_per_second', 'Tokens / s', 'YlGn', 'higher is better'),
        ('communication_bytes', 'Comm (bytes)', 'YlOrRd_r', 'lower is better'),
    ]

    ku_label = f'Ku={ku_slice}' if ku_slice is not None else 'mean over Ku'

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        f'DES-LOC Ablation: {model_name} ({ku_label})',
        fontsize=14, fontweight='bold')

    for ax, (metric, title, cmap, note) in zip(axes, metrics):
        grid = _pivot_metric(results, metric, kx_vals, kv_vals, ku_slice)

        # Normalise communication bytes to GB for readability
        if metric == 'communication_bytes':
            grid = grid / 1e9
            title = 'Comm (GB)'

        im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='lower')
        ax.set_xticks(range(len(kv_vals)))
        ax.set_xticklabels([str(v) for v in kv_vals])
        ax.set_yticks(range(len(kx_vals)))
        ax.set_yticklabels([str(v) for v in kx_vals])
        ax.set_xlabel('$K_v$')
        ax.set_ylabel('$K_x$')
        ax.set_title(f'{title}\n({note})')

        # Annotate cells
        for i in range(len(kx_vals)):
            for j in range(len(kv_vals)):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                if metric == 'communication_bytes':
                    txt = f'{val:.1f}'
                elif metric == 'tokens_per_second':
                    txt = f'{val:.0f}'
                else:
                    txt = f'{val:.3f}'
                # Choose text colour based on background brightness
                norm_val = (val - np.nanmin(grid)) / (np.nanmax(grid) - np.nanmin(grid) + 1e-9)
                color = 'white' if norm_val > 0.65 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=color)

        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f'_Ku{ku_slice}' if ku_slice is not None else ''
    out = _FIGURES_DIR / f'ablation_heatmap_{model_name}{suffix}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ---------------------------------------------------------------------------
# Pareto front: loss vs communication reduction
# ---------------------------------------------------------------------------
def plot_pareto_front(all_results: Dict[str, List[Dict]]) -> None:
    """Scatter plot of final_loss vs comm_reduction_ratio across all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ['o', 's', '^', 'D']
    colors = ['#2176AE', '#D64045', '#57A773', '#E8871E']

    for idx, (model_name, results) in enumerate(sorted(all_results.items())):
        cr = [r['comm_reduction_ratio'] for r in results]
        loss = [r['final_loss'] for r in results]
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.scatter(cr, loss, c=c, marker=m, s=20, alpha=0.5, label=model_name)

        # Highlight Pareto-optimal points (non-dominated in loss × comm)
        pareto_idx = _pareto_front_indices(cr, loss)
        p_cr = [cr[i] for i in pareto_idx]
        p_loss = [loss[i] for i in pareto_idx]
        # Sort for line plotting
        order = sorted(range(len(p_cr)), key=lambda k: p_cr[k])
        p_cr = [p_cr[k] for k in order]
        p_loss = [p_loss[k] for k in order]
        ax.plot(p_cr, p_loss, color=c, linewidth=1.5, linestyle='--', alpha=0.7)
        ax.scatter(p_cr, p_loss, c=c, marker=m, s=60, edgecolors='k', linewidths=0.6, zorder=5)

    ax.set_xlabel('Communication Reduction Ratio')
    ax.set_ylabel('Final Loss')
    ax.set_title('DES-LOC Ablation: Loss vs Communication Reduction (Pareto Front)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = _FIGURES_DIR / 'ablation_pareto_front.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


def _pareto_front_indices(x_vals: List[float], y_vals: List[float]) -> List[int]:
    """Find indices of Pareto-optimal points (maximise x, minimise y)."""
    n = len(x_vals)
    is_dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has >= x and <= y, with at least one strict
            if x_vals[j] >= x_vals[i] and y_vals[j] <= y_vals[i]:
                if x_vals[j] > x_vals[i] or y_vals[j] < y_vals[i]:
                    is_dominated[i] = True
                    break
    return [i for i in range(n) if not is_dominated[i]]


# ---------------------------------------------------------------------------
# Ku sensitivity subplot
# ---------------------------------------------------------------------------
def plot_ku_sensitivity(all_results: Dict[str, List[Dict]]) -> None:
    """Line plot showing how Ku affects final loss at a few representative (Kx, Kv) pairs."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4), squeeze=False)

    representative_pairs = [(4, 16), (8, 32), (2, 8)]  # (Kx, Kv) points to highlight
    colors = ['#2176AE', '#D64045', '#57A773']

    for col, (model_name, results) in enumerate(sorted(all_results.items())):
        ax = axes[0, col]
        for (kx, kv), color in zip(representative_pairs, colors):
            subset = [r for r in results if r['Kx'] == kx and r['Kv'] == kv]
            if not subset:
                continue
            subset.sort(key=lambda r: r['Ku'])
            ku_vals = [r['Ku'] for r in subset]
            losses = [r['final_loss'] for r in subset]
            ax.plot(ku_vals, losses, 'o-', color=color, label=f'$K_x$={kx}, $K_v$={kv}')

        ax.set_xlabel('$K_u$ (momentum sync period)')
        ax.set_ylabel('Final Loss')
        ax.set_title(model_name)
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('DES-LOC: Sensitivity to Momentum Sync Period $K_u$', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = _FIGURES_DIR / 'ablation_ku_sensitivity.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot DES-LOC ablation heatmaps from run_ablation.py results')
    parser.add_argument(
        '--model', default=None,
        help='Plot for a single model (e.g. tiny_70m). Default: all.')
    parser.add_argument(
        '--ku-slice', type=int, default=None,
        help='Fix Ku to this value in heatmaps instead of averaging over Ku.')
    parser.add_argument(
        '--results-dir', default=str(_RESULTS_DIR),
        help='Directory containing ablation JSON results')
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)

    print('Loading ablation results...')
    all_results = load_ablation_results(model_name=args.model, results_dir=results_dir)

    if not all_results:
        print('[plot_ablation] No results found. Run run_ablation.py first.')
        return

    print(f'Found results for {len(all_results)} model(s): {list(all_results.keys())}')

    # Per-model heatmaps
    for model_name, results in sorted(all_results.items()):
        print(f'\nPlotting heatmaps for {model_name} ({len(results)} configs)...')
        plot_model_heatmaps(model_name, results, ku_slice=args.ku_slice)

    # Cross-model plots
    if len(all_results) >= 1:
        print('\nPlotting Pareto front...')
        plot_pareto_front(all_results)
        print('Plotting Ku sensitivity...')
        plot_ku_sensitivity(all_results)

    print(f'\nAll figures saved to {_FIGURES_DIR}/')


if __name__ == '__main__':
    main()
