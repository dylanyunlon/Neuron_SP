# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
plot_figures.py — Generate paper figures for NeurIPS 2026 submission
====================================================================
Produces five matplotlib figures for the DES-LOC + AutoSP paper:

  Fig 1: Scaling law curve (loss vs tokens) with 7B prediction overlay
  Fig 2: Training throughput comparison (DES-LOC vs baseline)
  Fig 3: Communication reduction ratio vs convergence (Kx, Ku, Kv sweep)
  Fig 4: Per-GPU utilization heatmap (A6000 vs H100 vs Blackwell)
  Fig 5: Ablation — LOC cache hit rate vs training step

Data sources:
  - experiments/scaling_law/scaling_7b_predictions.json   (real fit)
  - experiments/scaling_law/scaling_fit_results.json       (real fit)
  - Placeholder data for Fig 2–5 (replace with real experiment runs)

Usage:
    python experiments/plot_figures.py
"""

import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_FIGURES_DIR = _SCRIPT_DIR / 'figures'
_SCALING_PRED = _SCRIPT_DIR / 'scaling_law' / 'scaling_7b_predictions.json'
_SCALING_FIT = _SCRIPT_DIR / 'scaling_law' / 'scaling_fit_results.json'

# Ensure output directory exists
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style defaults — clean, publication-quality
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
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

# Color palette: accessible, distinct
_COLORS = {
    'des_loc': '#2176AE',
    'baseline': '#D64045',
    'prediction': '#57A773',
    'fit_curve': '#2176AE',
    'measured': '#D64045',
    'kx': '#2176AE',
    'ku': '#E8871E',
    'kv': '#57A773',
    'hit_rate': '#2176AE',
    'hit_rate_ema': '#D64045',
}


# ===========================================================================
# Fig 1: Scaling law — loss vs tokens with 7B prediction overlay
# ===========================================================================
def fig1_scaling_law():
    """Chinchilla L(N,D) curve with measured points and 7B extrapolation."""
    with open(_SCALING_FIT) as f:
        fit = json.load(f)
    with open(_SCALING_PRED) as f:
        pred = json.load(f)

    # Scaling law coefficients
    E, A, alpha = fit['E'], fit['A'], fit['alpha']
    B, beta = fit['B'], fit['beta']

    # --- Measured points ---
    measured = fit['measured_points']
    m_tokens_b = [p['D'] / 1e9 for p in measured]
    m_loss = [p['L_measured'] for p in measured]
    m_names = [p['model_name'] for p in measured]

    # --- Continuous curve for 410M model (representative) ---
    N_410m = 410_000_000
    d_range = np.linspace(1e9, 25e9, 500)
    l_curve = E + A / N_410m**alpha + B / d_range**beta

    # --- 7B predictions ---
    N_7b = 7_000_000_000
    p_tokens_b = [p['tokens_B'] for p in pred['predictions']]
    p_loss = [p['predicted_loss'] for p in pred['predictions']]
    d_range_7b = np.linspace(15e9, 350e9, 500)
    l_curve_7b = E + A / N_7b**alpha + B / d_range_7b**beta

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Continuous fit curve (410M)
    ax.plot(d_range / 1e9, l_curve, color=_COLORS['fit_curve'], label='Fit: 410M $L(N,D)$')

    # Measured scatter
    ax.scatter(m_tokens_b, m_loss, color=_COLORS['measured'], zorder=5, s=50, edgecolors='k',
               linewidths=0.5, label='Measured (70M–1B)')
    for name, x, y in zip(m_names, m_tokens_b, m_loss):
        # Only label a few to avoid clutter
        if 'half' not in name:
            ax.annotate(name.replace('_', ' '), (x, y), textcoords='offset points',
                        xytext=(6, 4), fontsize=7, color='#555')

    # 7B prediction curve + markers
    ax.plot(d_range_7b / 1e9, l_curve_7b, color=_COLORS['prediction'], linestyle='--',
            label='Predicted: 7B $L(N,D)$')
    ax.scatter(p_tokens_b, p_loss, color=_COLORS['prediction'], marker='D', zorder=5, s=40,
               edgecolors='k', linewidths=0.5, label='7B checkpoints')

    ax.set_xlabel('Training tokens (B)')
    ax.set_ylabel('Loss')
    ax.set_title('Fig 1: Chinchilla Scaling Law — Heterogeneous Cluster')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = _FIGURES_DIR / 'fig1_scaling_law.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ===========================================================================
# Fig 2: Training throughput — DES-LOC vs baseline
# ===========================================================================
def fig2_throughput():
    """Bar chart: throughput (samples/s) for heterogeneous vs homogeneous."""
    # Placeholder data — replace with real profiling runs
    configs = ['2×A6000\n+1×H100\n(hetero)', '3×A6000\n(homo)', '3×H100\n(homo)']
    baseline_tput = [48.2, 35.7, 72.1]  # samples/s, standard DP
    desloc_tput = [61.5, 36.9, 73.8]    # samples/s, DES-LOC

    x = np.arange(len(configs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, baseline_tput, width, color=_COLORS['baseline'],
                   edgecolor='k', linewidth=0.5, label='Baseline (sync DP)')
    bars2 = ax.bar(x + width / 2, desloc_tput, width, color=_COLORS['des_loc'],
                   edgecolor='k', linewidth=0.5, label='DES-LOC')

    # Speedup annotations
    for b1, b2 in zip(bars1, bars2):
        speedup = b2.get_height() / b1.get_height()
        ax.annotate(f'{speedup:.2f}×', xy=(b2.get_x() + b2.get_width() / 2, b2.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')

    ax.set_ylabel('Throughput (samples / s)')
    ax.set_title('Fig 2: Training Throughput — DES-LOC vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    out = _FIGURES_DIR / 'fig2_throughput.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ===========================================================================
# Fig 3: Communication reduction vs convergence (Kx, Ku, Kv sweep)
# ===========================================================================
def fig3_comm_reduction():
    """Line plot: comm reduction ratio and final loss for varying K periods."""
    # Placeholder data — replace with Kx/Ku/Kv sweep experiments
    k_values = [1, 2, 4, 8, 16, 32]

    # comm_reduction = 1 - 1/K  (theoretical, modulated by overlap)
    cr_kx = [0.0, 0.42, 0.68, 0.82, 0.90, 0.94]
    cr_ku = [0.0, 0.38, 0.61, 0.76, 0.85, 0.91]
    cr_kv = [0.0, 0.35, 0.55, 0.71, 0.81, 0.88]

    # Final validation loss (higher K → slightly worse convergence)
    loss_kx = [2.598, 2.601, 2.611, 2.637, 2.698, 2.814]
    loss_ku = [2.598, 2.600, 2.607, 2.628, 2.672, 2.765]
    loss_kv = [2.598, 2.599, 2.604, 2.619, 2.651, 2.731]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: communication reduction
    ax1.plot(k_values, cr_kx, 'o-', color=_COLORS['kx'], label='$K_x$ (params)')
    ax1.plot(k_values, cr_ku, 's-', color=_COLORS['ku'], label='$K_u$ (momentum)')
    ax1.plot(k_values, cr_kv, '^-', color=_COLORS['kv'], label='$K_v$ (variance)')
    ax1.set_xlabel('Sync period $K$')
    ax1.set_ylabel('Communication reduction ratio')
    ax1.set_xscale('log', base=2)
    ax1.set_title('(a) Comm. reduction')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Right: convergence (final loss)
    ax2.plot(k_values, loss_kx, 'o-', color=_COLORS['kx'], label='$K_x$')
    ax2.plot(k_values, loss_ku, 's-', color=_COLORS['ku'], label='$K_u$')
    ax2.plot(k_values, loss_kv, '^-', color=_COLORS['kv'], label='$K_v$')
    ax2.set_xlabel('Sync period $K$')
    ax2.set_ylabel('Final validation loss')
    ax2.set_xscale('log', base=2)
    ax2.set_title('(b) Convergence impact')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Fig 3: DES-LOC Sync Period Sweep ($K_x$, $K_u$, $K_v$)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = _FIGURES_DIR / 'fig3_comm_reduction.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ===========================================================================
# Fig 4: Per-GPU utilization heatmap
# ===========================================================================
def fig4_gpu_heatmap():
    """Heatmap: utilization breakdown across GPU types and workload phases."""
    # Placeholder data — replace with real nvprof / nsight measurements
    gpu_types = ['A6000-0', 'A6000-1', 'H100-0']
    phases = ['Forward', 'Backward', 'AllReduce', 'Optimizer', 'Idle/Bubble']

    # Utilization fractions (rows=GPUs, cols=phases), must sum to ~1.0 per row
    util = np.array([
        [0.28, 0.35, 0.18, 0.09, 0.10],  # A6000-0
        [0.27, 0.34, 0.19, 0.09, 0.11],  # A6000-1
        [0.32, 0.38, 0.12, 0.10, 0.08],  # H100-0
    ])

    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(util, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.45)

    ax.set_xticks(np.arange(len(phases)))
    ax.set_yticks(np.arange(len(gpu_types)))
    ax.set_xticklabels(phases)
    ax.set_yticklabels(gpu_types)

    # Annotate cells with percentage
    for i in range(len(gpu_types)):
        for j in range(len(phases)):
            val = util[i, j]
            color = 'white' if val > 0.28 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', color=color, fontsize=10)

    ax.set_title('Fig 4: Per-GPU Time Breakdown — Heterogeneous Cluster')
    fig.colorbar(im, ax=ax, label='Fraction of step time', shrink=0.8)
    fig.tight_layout()

    out = _FIGURES_DIR / 'fig4_gpu_heatmap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ===========================================================================
# Fig 5: Ablation — LOC cache hit rate vs training step
# ===========================================================================
def fig5_cache_hit_rate():
    """Line plot: LOC cache hit rate warming up over training."""
    # Placeholder data — replace with real LOC cache profiling logs
    steps = np.arange(0, 5001, 50)
    # Sigmoid-like warm-up: cache effectiveness grows as model stabilizes
    raw_hr = 1.0 / (1.0 + np.exp(-0.003 * (steps - 1200)))
    # Add realistic noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, size=steps.shape)
    hit_rate = np.clip(raw_hr + noise, 0, 1)

    # Exponential moving average for trend line
    ema = np.zeros_like(hit_rate)
    ema[0] = hit_rate[0]
    alpha_ema = 0.05
    for i in range(1, len(hit_rate)):
        ema[i] = alpha_ema * hit_rate[i] + (1 - alpha_ema) * ema[i - 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(steps, hit_rate, s=8, alpha=0.4, color=_COLORS['hit_rate'], label='Per-step hit rate')
    ax.plot(steps, ema, color=_COLORS['hit_rate_ema'], linewidth=2, label='EMA trend')

    # Mark warm-up threshold
    ax.axhline(y=0.90, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.annotate('90% threshold', xy=(100, 0.905), fontsize=8, color='gray')

    # Mark the step where EMA crosses 90%
    cross_idx = np.argmax(ema >= 0.90)
    if cross_idx > 0:
        cross_step = steps[cross_idx]
        ax.axvline(x=cross_step, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.annotate(f'step {cross_step}', xy=(cross_step + 50, 0.15), fontsize=8, color='gray')

    ax.set_xlabel('Training step')
    ax.set_ylabel('LOC cache hit rate')
    ax.set_title('Fig 5: LOC Cache Hit Rate Warm-up (1B model, $K_x$=4)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = _FIGURES_DIR / 'fig5_cache_hit_rate.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved {out}')


# ===========================================================================
# Main
# ===========================================================================
def main():
    print('Generating paper figures...')
    fig1_scaling_law()
    fig2_throughput()
    fig3_comm_reduction()
    fig4_gpu_heatmap()
    fig5_cache_hit_rate()
    print(f'All figures saved to {_FIGURES_DIR}/')


if __name__ == '__main__':
    main()
