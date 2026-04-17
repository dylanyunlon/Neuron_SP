#!/usr/bin/env python3
"""
===============================================================================
M029: Data Visualization Module (NKI-FA Standards)
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements publication-quality plotting following NKI-FA commit 
da964f3 standards. All data MUST come from parsed experiment logs, not 
hardcoded values.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Critical Standards:
1. Data labels: "0.9234" not "0.9" or "1"
2. All values from experiment logs, never fabricated
3. Professional color palettes
4. Precise annotations on all bars/lines
5. Proper axis labels and titles
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M029"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
import json
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: COLOR PALETTES AND STYLE CONSTANTS
# =============================================================================

class ColorPalette:
    """NKI-FA standard color palette."""
    
    # Primary method colors (consistent across all figures)
    DDP = '#264653'           # Dark teal
    DESLOC = '#2A9D8F'        # Teal
    LOCAL_ADAM = '#E9C46A'    # Yellow
    FAVG_OPT = '#F4A261'      # Orange
    RESET_STATES = '#E76F51'  # Coral
    MUON = '#7B2CBF'          # Purple
    
    # Secondary colors
    BASELINE = '#6C757D'      # Gray
    IMPROVEMENT = '#2E86AB'   # Blue
    DEGRADATION = '#C73E1D'   # Red
    
    # Ablation curve colors
    ABLATION_COLORS = [
        '#264653', '#2A9D8F', '#E9C46A', '#F4A261', 
        '#E76F51', '#7B2CBF', '#2E86AB', '#399E5A'
    ]
    
    @classmethod
    def get_method_color(cls, method: str) -> str:
        """Get color for a method name."""
        method_map = {
            'ddp': cls.DDP,
            'desloc': cls.DESLOC,
            'local_adam': cls.LOCAL_ADAM,
            'local adam': cls.LOCAL_ADAM,
            'favg_opt': cls.FAVG_OPT,
            'favg+opt': cls.FAVG_OPT,
            'reset_states': cls.RESET_STATES,
            'reset states': cls.RESET_STATES,
            'muon': cls.MUON,
            'desloc_muon': cls.MUON,
        }
        return method_map.get(method.lower(), cls.BASELINE)
    
    @classmethod
    def get_palette(cls, n: int) -> List[str]:
        """Get n distinct colors."""
        return cls.ABLATION_COLORS[:n]


class PlotStyle:
    """Plot style constants."""
    
    # Figure sizes
    SINGLE_COLUMN = (3.5, 2.8)
    DOUBLE_COLUMN = (7, 3)
    FULL_PAGE = (7, 7)
    
    # Font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 10
    ANNOTATION_SIZE = 9
    
    # Line styles
    SOLID = '-'
    DASHED = '--'
    DOTTED = ':'
    DASHDOT = '-.'
    
    # Marker styles
    CIRCLE = 'o'
    SQUARE = 's'
    TRIANGLE = '^'
    DIAMOND = 'D'
    STAR = '*'
    
    @classmethod
    def setup_matplotlib(cls):
        """Setup matplotlib with publication-quality defaults."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': cls.TICK_SIZE,
            'axes.titlesize': cls.TITLE_SIZE,
            'axes.labelsize': cls.LABEL_SIZE,
            'xtick.labelsize': cls.TICK_SIZE,
            'ytick.labelsize': cls.TICK_SIZE,
            'legend.fontsize': cls.LEGEND_SIZE,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        })


# =============================================================================
# PART 2: DATA CLASSES FOR PLOTTING
# =============================================================================

@dataclass
class PlotData:
    """Data for a single plot series."""
    name: str
    x: List[float]
    y: List[float]
    yerr: Optional[List[float]] = None
    color: Optional[str] = None
    linestyle: str = '-'
    marker: str = 'o'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data."""
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have same length: {len(self.x)} vs {len(self.y)}")
        if self.color is None:
            self.color = ColorPalette.get_method_color(self.name)


@dataclass
class BarData:
    """Data for bar chart."""
    categories: List[str]
    values: List[float]
    errors: Optional[List[float]] = None
    colors: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    
    def __post_init__(self):
        """Generate default labels with full precision."""
        if self.labels is None:
            # Following NKI-FA standard: show precise values
            self.labels = [f"{v:.3f}" if v < 10 else f"{v:.1f}" for v in self.values]


@dataclass
class HeatmapData:
    """Data for heatmap."""
    matrix: np.ndarray
    x_labels: List[str]
    y_labels: List[str]
    title: str = ""
    cmap: str = "viridis"
    annotate: bool = True


# =============================================================================
# PART 3: PLOTTING FUNCTIONS
# =============================================================================

class DESLOCPlotter:
    """Main plotting class following NKI-FA standards."""
    
    def __init__(self, output_dir: str = "./figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        PlotStyle.setup_matplotlib()
    
    def plot_convergence_curves(
        self,
        data: List[PlotData],
        title: str,
        xlabel: str = "Steps",
        ylabel: str = "Loss",
        log_scale: bool = False,
        filename: str = None,
        figsize: Tuple[float, float] = None,
    ) -> plt.Figure:
        """
        Plot convergence curves for multiple methods.
        
        Following NKI-FA standard: each curve comes from parsed log data.
        """
        fig, ax = plt.subplots(figsize=figsize or PlotStyle.DOUBLE_COLUMN)
        
        for series in data:
            ax.plot(
                series.x, series.y,
                label=series.name,
                color=series.color,
                linestyle=series.linestyle,
                marker=series.marker,
                markevery=max(1, len(series.x) // 10),
                markersize=6,
                linewidth=2,
            )
            
            if series.yerr is not None:
                ax.fill_between(
                    series.x,
                    np.array(series.y) - np.array(series.yerr),
                    np.array(series.y) + np.array(series.yerr),
                    alpha=0.2,
                    color=series.color,
                )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_bar_comparison(
        self,
        data: List[BarData],
        title: str,
        xlabel: str = "",
        ylabel: str = "",
        filename: str = None,
        grouped: bool = True,
        figsize: Tuple[float, float] = None,
    ) -> plt.Figure:
        """
        Plot bar comparison chart.
        
        Following NKI-FA standard: precise value annotations on each bar.
        """
        fig, ax = plt.subplots(figsize=figsize or PlotStyle.DOUBLE_COLUMN)
        
        n_groups = len(data[0].categories)
        n_bars = len(data) if grouped else 1
        
        if grouped:
            bar_width = 0.8 / n_bars
            indices = np.arange(n_groups)
            
            for i, bar_data in enumerate(data):
                offset = (i - n_bars/2 + 0.5) * bar_width
                bars = ax.bar(
                    indices + offset,
                    bar_data.values,
                    bar_width,
                    label=bar_data.categories[0] if hasattr(bar_data, 'series_name') else f"Series {i+1}",
                    color=bar_data.colors or [ColorPalette.ABLATION_COLORS[i]] * n_groups,
                    yerr=bar_data.errors,
                    capsize=3,
                )
                
                # Add value annotations (NKI-FA standard: precise values)
                for bar, label in zip(bars, bar_data.labels):
                    height = bar.get_height()
                    ax.annotate(
                        label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center',
                        va='bottom',
                        fontsize=PlotStyle.ANNOTATION_SIZE,
                    )
            
            ax.set_xticks(indices)
            ax.set_xticklabels(data[0].categories, rotation=45, ha='right')
        
        else:
            # Single bar chart
            bar_data = data[0]
            bars = ax.bar(
                range(n_groups),
                bar_data.values,
                color=bar_data.colors or ColorPalette.get_palette(n_groups),
                yerr=bar_data.errors,
                capsize=3,
            )
            
            for bar, label in zip(bars, bar_data.labels):
                height = bar.get_height()
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=PlotStyle.ANNOTATION_SIZE,
                )
            
            ax.set_xticks(range(n_groups))
            ax.set_xticklabels(bar_data.categories, rotation=45, ha='right')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_ablation_heatmap(
        self,
        data: HeatmapData,
        filename: str = None,
        figsize: Tuple[float, float] = None,
    ) -> plt.Figure:
        """
        Plot ablation study heatmap.
        
        Following NKI-FA standard: annotate each cell with precise value.
        """
        fig, ax = plt.subplots(figsize=figsize or PlotStyle.SINGLE_COLUMN)
        
        # Create heatmap
        im = ax.imshow(data.matrix, cmap=data.cmap, aspect='auto')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Set ticks
        ax.set_xticks(np.arange(len(data.x_labels)))
        ax.set_yticks(np.arange(len(data.y_labels)))
        ax.set_xticklabels(data.x_labels)
        ax.set_yticklabels(data.y_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add annotations
        if data.annotate:
            for i in range(len(data.y_labels)):
                for j in range(len(data.x_labels)):
                    value = data.matrix[i, j]
                    # Format with appropriate precision
                    text = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
                    text_color = "white" if value > data.matrix.mean() else "black"
                    ax.text(j, i, text, ha="center", va="center", 
                           color=text_color, fontsize=PlotStyle.ANNOTATION_SIZE)
        
        ax.set_title(data.title)
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_tflops_comparison(
        self,
        before_data: List[Dict[str, Any]],
        after_data: List[Dict[str, Any]],
        title: str = "TFLOPS Comparison",
        filename: str = None,
    ) -> plt.Figure:
        """
        Plot TFLOPS comparison following exact NKI-FA format.
        
        This matches draw_plot.py from NKI-FA commit da964f3.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        categories = [d['test_condition'] for d in before_data]
        before_tflops = [d['tflops'] for d in before_data]
        after_tflops = [d['tflops'] for d in after_data]
        before_time = [d['time_ms'] for d in before_data]
        after_time = [d['time_ms'] for d in after_data]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Plot 1: TFLOPS
        bars1 = ax1.bar(x - width/2, before_tflops, width, label='Before', 
                        color=ColorPalette.BASELINE)
        bars2 = ax1.bar(x + width/2, after_tflops, width, label='After',
                        color=ColorPalette.IMPROVEMENT)
        
        # Annotate with precise values
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        ax1.set_ylabel('TFLOPS (Higher is Better)')
        ax1.set_title(f'{title} - TFLOPS')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Time (ms)
        bars3 = ax2.bar(x - width/2, before_time, width, label='Before',
                        color=ColorPalette.BASELINE)
        bars4 = ax2.bar(x + width/2, after_time, width, label='After',
                        color=ColorPalette.IMPROVEMENT)
        
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        for bar in bars4:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        ax2.set_ylabel('Time (ms) (Lower is Better)')
        ax2.set_title(f'{title} - Execution Time')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_communication_reduction(
        self,
        methods: List[str],
        comm_costs: List[float],
        speedups: List[float],
        title: str = "Communication Reduction",
        filename: str = None,
    ) -> plt.Figure:
        """Plot communication reduction comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PlotStyle.DOUBLE_COLUMN)
        
        colors = [ColorPalette.get_method_color(m) for m in methods]
        x = range(len(methods))
        
        # Communication cost
        bars1 = ax1.bar(x, comm_costs, color=colors)
        ax1.set_ylabel('Communication Cost (normalized)')
        ax1.set_title('Communication Cost per Step')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, val in zip(bars1, comm_costs):
            ax1.annotate(f'{val:.2f}x',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        # Speedup
        bars2 = ax2.bar(x, speedups, color=colors)
        ax2.set_ylabel('Training Speedup')
        ax2.set_title('End-to-End Speedup')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars2, speedups):
            ax2.annotate(f'{val:.2f}x',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, filename: str):
        """Save figure to output directory."""
        # Save as both PNG and PDF
        for ext in ['.png', '.pdf']:
            path = self.output_dir / (filename + ext)
            fig.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {path}")


# =============================================================================
# PART 4: FIGURE GENERATORS FOR EACH BENCHMARK
# =============================================================================

class BenchmarkFigureGenerator:
    """Generate all benchmark figures."""
    
    def __init__(self, plotter: DESLOCPlotter):
        self.plotter = plotter
    
    def generate_figure1_rosenbrock(self, log_data: Dict[str, Any]) -> plt.Figure:
        """Generate Figure 1: Rosenbrock convergence."""
        # Extract data from logs
        methods = ['DDP', 'Local Adam', 'DES-LOC']
        plot_data = []
        
        for method in methods:
            if method.lower().replace(' ', '_') in log_data:
                d = log_data[method.lower().replace(' ', '_')]
                plot_data.append(PlotData(
                    name=method,
                    x=d['steps'],
                    y=d['loss'],
                    yerr=d.get('std'),
                ))
        
        return self.plotter.plot_convergence_curves(
            plot_data,
            title='Figure 1: Rosenbrock Optimization (M=256, σ=1.5)',
            xlabel='Steps',
            ylabel='Loss',
            log_scale=True,
            filename='figure1_rosenbrock'
        )
    
    def generate_figure2_momentum(self, log_data: Dict[str, Any]) -> plt.Figure:
        """Generate Figure 2: Momentum ablation."""
        beta1_values = [0.8, 0.9, 0.95, 0.99]
        plot_data = []
        
        for beta1 in beta1_values:
            key = f'beta1_{beta1}'
            if key in log_data:
                d = log_data[key]
                plot_data.append(PlotData(
                    name=f'β₁={beta1}',
                    x=d['steps'],
                    y=d['loss'],
                ))
        
        return self.plotter.plot_convergence_curves(
            plot_data,
            title='Figure 2: Momentum (β₁) Ablation',
            xlabel='Steps',
            ylabel='Loss',
            filename='figure2_momentum_ablation'
        )
    
    def generate_figure3_sync_ablation(self, log_data: Dict[str, Any]) -> plt.Figure:
        """Generate Figure 3: Sync period ablation (3 subplots)."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Kx ablation
        kx_values = [16, 32, 64, 128]
        for i, ax in enumerate(axes):
            ax.set_xlabel('Steps')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        axes[0].set_title('(a) Kx Ablation')
        axes[1].set_title('(b) Ku Ablation')
        axes[2].set_title('(c) Kv Ablation')
        
        plt.tight_layout()
        
        path = self.plotter.output_dir / 'figure3_sync_ablation.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        
        return fig


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Self-test
    plotter = DESLOCPlotter("./test_figures")
    
    # Test convergence plot with synthetic data
    test_data = [
        PlotData(
            name='DDP',
            x=list(range(0, 1000, 10)),
            y=[1.0 * np.exp(-0.003 * x) + 0.1 for x in range(0, 1000, 10)],
        ),
        PlotData(
            name='DES-LOC',
            x=list(range(0, 1000, 10)),
            y=[1.0 * np.exp(-0.005 * x) + 0.05 for x in range(0, 1000, 10)],
        ),
    ]
    
    fig = plotter.plot_convergence_curves(
        test_data,
        title='Test Convergence Plot',
        xlabel='Steps',
        ylabel='Loss',
        log_scale=True,
        filename='test_convergence'
    )
    
    # Test bar chart
    bar_data = [BarData(
        categories=['DDP', 'Local Adam', 'DES-LOC'],
        values=[1.0, 0.67, 0.33],
    )]
    
    fig2 = plotter.plot_bar_comparison(
        bar_data,
        title='Test Bar Chart',
        ylabel='Communication Cost (normalized)',
        filename='test_bar'
    )
    
    print("\n[M029] Visualization Module - Self-test PASSED")
    print(f"Figures saved to: {plotter.output_dir}")
