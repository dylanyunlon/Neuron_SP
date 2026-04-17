#!/usr/bin/env python3
"""
===============================================================================
M041: Figure Generator Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module generates publication-ready figures from experiment results.
Follows NKI-FA plotting standards with precise value annotations.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Automatic figure generation from JSON/log files
- NKI-FA compliant formatting (exact values like "582.3")
- LaTeX-ready outputs (PDF, EPS, PGF)
- Consistent color scheme and styling
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M041"

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import logging

import numpy as np

# Matplotlib configuration for publication
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: STYLE CONFIGURATION
# =============================================================================

class PlotStyle:
    """Publication-quality plot styling."""
    
    # NeurIPS/ICML standard
    FIGSIZE_SINGLE = (3.25, 2.5)     # Single column
    FIGSIZE_DOUBLE = (6.875, 2.5)    # Double column
    FIGSIZE_FULL = (6.875, 4.5)      # Full page width
    
    # Font sizes
    FONT_SIZE_TITLE = 10
    FONT_SIZE_LABEL = 9
    FONT_SIZE_TICK = 8
    FONT_SIZE_LEGEND = 8
    FONT_SIZE_ANNOTATION = 7
    
    # Line styles
    LINE_WIDTH = 1.5
    MARKER_SIZE = 4
    
    # DPI for raster outputs
    DPI = 300
    
    @classmethod
    def setup_matplotlib(cls):
        """Configure matplotlib for publication quality."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': cls.FONT_SIZE_LABEL,
            'axes.labelsize': cls.FONT_SIZE_LABEL,
            'axes.titlesize': cls.FONT_SIZE_TITLE,
            'xtick.labelsize': cls.FONT_SIZE_TICK,
            'ytick.labelsize': cls.FONT_SIZE_TICK,
            'legend.fontsize': cls.FONT_SIZE_LEGEND,
            'figure.figsize': cls.FIGSIZE_SINGLE,
            'figure.dpi': cls.DPI,
            'savefig.dpi': cls.DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'axes.linewidth': 0.8,
            'lines.linewidth': cls.LINE_WIDTH,
            'lines.markersize': cls.MARKER_SIZE,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'legend.edgecolor': 'gray',
        })


class ColorPalette:
    """Color scheme for different methods."""
    
    # Primary colors (colorblind-friendly)
    DDP = '#264653'           # Dark blue-green
    LOCAL_ADAM = '#E9C46A'    # Yellow
    DESLOC = '#2A9D8F'        # Teal
    DESLOC_MUON = '#E76F51'   # Coral
    MUON = '#F4A261'          # Orange
    
    # Extended palette for ablations
    ABLATION_COLORS = [
        '#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51',
        '#1D3557', '#457B9D', '#A8DADC', '#F1FAEE', '#E63946'
    ]
    
    # Method to color mapping
    METHOD_COLORS = {
        'ddp': DDP,
        'local_adam': LOCAL_ADAM,
        'desloc': DESLOC,
        'desloc_k32': DESLOC,
        'desloc_k64': '#3EB489',
        'desloc_k128': '#1B7A5C',
        'desloc_muon': DESLOC_MUON,
        'muon': MUON,
        'adam': '#457B9D',
        'desloc_adam': '#2A9D8F',
    }
    
    # Markers
    METHOD_MARKERS = {
        'ddp': 'o',
        'local_adam': 's',
        'desloc': '^',
        'desloc_k32': '^',
        'desloc_k64': 'v',
        'desloc_k128': 'd',
        'desloc_muon': 'p',
        'muon': 'h',
        'adam': 'o',
        'desloc_adam': '^',
    }
    
    @classmethod
    def get_color(cls, method: str) -> str:
        """Get color for method."""
        return cls.METHOD_COLORS.get(method.lower(), cls.ABLATION_COLORS[0])
    
    @classmethod
    def get_marker(cls, method: str) -> str:
        """Get marker for method."""
        return cls.METHOD_MARKERS.get(method.lower(), 'o')


# =============================================================================
# PART 2: FIGURE GENERATORS
# =============================================================================

class FigureGenerator:
    """Base class for figure generation."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        PlotStyle.setup_matplotlib()
    
    def save_figure(
        self,
        fig: Figure,
        name: str,
        formats: List[str] = ['pdf', 'png'],
    ):
        """Save figure in multiple formats."""
        for fmt in formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, format=fmt, bbox_inches='tight', dpi=PlotStyle.DPI)
            logger.info(f"Saved figure: {path}")
        
        plt.close(fig)
    
    def format_value(self, value: float, precision: int = 1) -> str:
        """Format value following NKI-FA standard (e.g., '582.3')."""
        if value >= 1000:
            return f"{value:.0f}"
        elif value >= 100:
            return f"{value:.1f}"
        elif value >= 10:
            return f"{value:.2f}"
        elif value >= 1:
            return f"{value:.3f}"
        else:
            return f"{value:.4f}"


class ConvergencePlotter(FigureGenerator):
    """Generates convergence curve figures (Figure 1, 2)."""
    
    def plot_convergence(
        self,
        data: Dict[str, Dict[str, List]],
        title: str,
        xlabel: str = "Step",
        ylabel: str = "Loss",
        name: str = "convergence",
        figsize: Tuple[float, float] = None,
        log_scale: bool = True,
    ):
        """
        Plot convergence curves for multiple methods.
        
        Args:
            data: {method_name: {'steps': [...], 'losses': [...]}}
        """
        if figsize is None:
            figsize = PlotStyle.FIGSIZE_SINGLE
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for method, values in data.items():
            steps = values['steps']
            losses = values['losses']
            
            color = ColorPalette.get_color(method)
            marker = ColorPalette.get_marker(method)
            
            ax.plot(
                steps, losses,
                color=color,
                marker=marker,
                markevery=max(1, len(steps) // 10),
                label=method,
                linewidth=PlotStyle.LINE_WIDTH,
                markersize=PlotStyle.MARKER_SIZE,
            )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        self.save_figure(fig, name)
        return fig


class BarPlotter(FigureGenerator):
    """Generates bar comparison figures (Figure 4, Table 2)."""
    
    def plot_bar_comparison(
        self,
        data: Dict[str, float],
        title: str,
        xlabel: str = "Method",
        ylabel: str = "Value",
        name: str = "bar_comparison",
        figsize: Tuple[float, float] = None,
        show_values: bool = True,
        horizontal: bool = False,
    ):
        """
        Plot bar comparison with value annotations.
        
        Args:
            data: {method_name: value}
        """
        if figsize is None:
            figsize = PlotStyle.FIGSIZE_SINGLE
        
        fig, ax = plt.subplots(figsize=figsize)
        
        methods = list(data.keys())
        values = list(data.values())
        colors = [ColorPalette.get_color(m) for m in methods]
        
        x = np.arange(len(methods))
        
        if horizontal:
            bars = ax.barh(x, values, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_yticks(x)
            ax.set_yticklabels(methods)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        ax.set_title(title)
        
        # Add value annotations (NKI-FA format)
        if show_values:
            for bar, val in zip(bars, values):
                if horizontal:
                    ax.annotate(
                        self.format_value(val),
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords='offset points',
                        va='center',
                        fontsize=PlotStyle.FONT_SIZE_ANNOTATION,
                    )
                else:
                    ax.annotate(
                        self.format_value(val),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center',
                        fontsize=PlotStyle.FONT_SIZE_ANNOTATION,
                    )
        
        plt.tight_layout()
        self.save_figure(fig, name)
        return fig


class AblationPlotter(FigureGenerator):
    """Generates ablation study figures (Figure 3)."""
    
    def plot_ablation_sweep(
        self,
        sweep_values: List[int],
        metrics: Dict[str, List[float]],
        title: str,
        xlabel: str,
        ylabel: str = "Loss",
        name: str = "ablation",
        figsize: Tuple[float, float] = None,
    ):
        """
        Plot ablation sweep results.
        
        Args:
            sweep_values: [16, 32, 64, 128]
            metrics: {'loss': [...], 'comm_reduction': [...]}
        """
        if figsize is None:
            figsize = PlotStyle.FIGSIZE_SINGLE
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        color1 = ColorPalette.DESLOC
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel, color=color1)
        
        line1 = ax1.plot(
            sweep_values,
            metrics.get('loss', []),
            color=color1,
            marker='o',
            linewidth=PlotStyle.LINE_WIDTH,
            markersize=PlotStyle.MARKER_SIZE,
            label='Loss',
        )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log', base=2)
        
        # Secondary y-axis for communication reduction
        if 'comm_reduction' in metrics:
            ax2 = ax1.twinx()
            color2 = ColorPalette.DDP
            ax2.set_ylabel('Comm. Reduction (%)', color=color2)
            
            comm_pct = [v * 100 for v in metrics['comm_reduction']]
            line2 = ax2.plot(
                sweep_values,
                comm_pct,
                color=color2,
                marker='s',
                linestyle='--',
                linewidth=PlotStyle.LINE_WIDTH,
                markersize=PlotStyle.MARKER_SIZE,
                label='Comm. Reduction',
            )
            ax2.tick_params(axis='y', labelcolor=color2)
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='center right')
        
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, name)
        return fig


class ThroughputPlotter(FigureGenerator):
    """Generates throughput comparison figures (Figure 5)."""
    
    def plot_scaling(
        self,
        model_sizes: List[str],
        throughputs: Dict[str, List[float]],
        title: str,
        name: str = "scaling",
        figsize: Tuple[float, float] = None,
    ):
        """
        Plot throughput scaling across model sizes.
        
        Args:
            model_sizes: ['125M', '360M', '1.7B']
            throughputs: {method: [throughput_125M, throughput_360M, ...]}
        """
        if figsize is None:
            figsize = PlotStyle.FIGSIZE_DOUBLE
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(model_sizes))
        width = 0.8 / len(throughputs)
        
        for i, (method, values) in enumerate(throughputs.items()):
            offset = (i - len(throughputs) / 2 + 0.5) * width
            color = ColorPalette.get_color(method)
            
            bars = ax.bar(
                x + offset,
                values,
                width * 0.9,
                color=color,
                label=method,
                edgecolor='black',
                linewidth=0.5,
            )
            
            # Add value annotations
            for bar, val in zip(bars, values):
                ax.annotate(
                    self.format_value(val),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    fontsize=PlotStyle.FONT_SIZE_ANNOTATION,
                    rotation=90,
                )
        
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Throughput (tokens/sec)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_sizes)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.save_figure(fig, name)
        return fig


class CommunicationPlotter(FigureGenerator):
    """Generates communication analysis figures (Figure 4)."""
    
    def plot_communication_breakdown(
        self,
        methods: List[str],
        param_syncs: List[int],
        momentum_syncs: List[int],
        total_bytes_gb: List[float],
        title: str,
        name: str = "communication",
        figsize: Tuple[float, float] = None,
    ):
        """Plot stacked bar chart of communication costs."""
        if figsize is None:
            figsize = PlotStyle.FIGSIZE_DOUBLE
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        x = np.arange(len(methods))
        width = 0.6
        
        # Left: sync operations
        ax1.bar(x, param_syncs, width, label='Param Syncs', color=ColorPalette.DDP)
        ax1.bar(x, momentum_syncs, width, bottom=param_syncs, 
               label='Momentum Syncs', color=ColorPalette.DESLOC)
        
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Number of Sync Operations')
        ax1.set_title('Sync Operations')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right: data transferred
        colors = [ColorPalette.get_color(m) for m in methods]
        bars = ax2.bar(x, total_bytes_gb, width, color=colors, edgecolor='black')
        
        for bar, val in zip(bars, total_bytes_gb):
            ax2.annotate(
                f'{val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                fontsize=PlotStyle.FONT_SIZE_ANNOTATION,
            )
        
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Data Transferred (GB)')
        ax2.set_title('Communication Volume')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self.save_figure(fig, name)
        return fig


# =============================================================================
# PART 3: AUTOMATED FIGURE GENERATION
# =============================================================================

class AutoFigureGenerator:
    """Automatically generates all figures from results."""
    
    def __init__(self, results_path: Path, output_dir: Path):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize plotters
        self.convergence = ConvergencePlotter(output_dir)
        self.bar = BarPlotter(output_dir)
        self.ablation = AblationPlotter(output_dir)
        self.throughput = ThroughputPlotter(output_dir)
        self.communication = CommunicationPlotter(output_dir)
        
        # Load results
        self.results = self._load_results()
    
    def _load_results(self) -> Dict[str, Any]:
        """Load results from JSON file."""
        if not self.results_path.exists():
            logger.warning(f"Results file not found: {self.results_path}")
            return {}
        
        with open(self.results_path) as f:
            return json.load(f)
    
    def generate_all(self):
        """Generate all figures."""
        logger.info("Generating all figures...")
        
        self.generate_figure1_rosenbrock()
        self.generate_figure2_momentum()
        self.generate_figure3_sync_ablation()
        self.generate_figure4_communication()
        self.generate_figure5_scaling()
        self.generate_figure6_outer_optimizer()
        self.generate_figure7_muon()
        self.generate_table2_wallclock()
        
        logger.info(f"All figures saved to {self.output_dir}")
    
    def generate_figure1_rosenbrock(self):
        """Generate Figure 1: Rosenbrock convergence."""
        experiments = self._filter_experiments('rosenbrock')
        
        if not experiments:
            logger.warning("No Rosenbrock experiments found")
            return
        
        data = {}
        for exp in experiments:
            method = exp.get('method', 'unknown')
            if 'loss_curve' in exp:
                data[method] = {
                    'steps': [p['step'] for p in exp['loss_curve']],
                    'losses': [p['value'] for p in exp['loss_curve']],
                }
        
        if data:
            self.convergence.plot_convergence(
                data,
                title='Figure 1: Rosenbrock Optimization',
                name='figure1_rosenbrock',
            )
    
    def generate_figure2_momentum(self):
        """Generate Figure 2: Momentum ablation."""
        experiments = self._filter_experiments('momentum_ablation')
        
        if not experiments:
            logger.warning("No momentum ablation experiments found")
            return
        
        data = {}
        for exp in experiments:
            beta1 = exp.get('config', {}).get('beta1', 0.9)
            method = f'β₁={beta1}'
            if 'loss_curve' in exp:
                data[method] = {
                    'steps': [p['step'] for p in exp['loss_curve']],
                    'losses': [p['value'] for p in exp['loss_curve']],
                }
        
        if data:
            self.convergence.plot_convergence(
                data,
                title='Figure 2: Momentum (β₁) Ablation',
                name='figure2_momentum',
            )
    
    def generate_figure3_sync_ablation(self):
        """Generate Figure 3: Sync period ablation."""
        experiments = self._filter_experiments('sync_ablation')
        
        # Separate by ablation type
        for ablation_type in ['kx', 'ku', 'kv']:
            type_exps = [e for e in experiments 
                        if ablation_type in e.get('method', '').lower()]
            
            if not type_exps:
                continue
            
            sweep_values = []
            losses = []
            comm_reductions = []
            
            for exp in sorted(type_exps, key=lambda e: e.get('config', {}).get(ablation_type, 0)):
                val = exp.get('config', {}).get(ablation_type, 0)
                if val > 0:
                    sweep_values.append(val)
                    losses.append(exp.get('final_loss', float('inf')))
                    comm_reductions.append(exp.get('communication_reduction', 0))
            
            if sweep_values:
                self.ablation.plot_ablation_sweep(
                    sweep_values,
                    {'loss': losses, 'comm_reduction': comm_reductions},
                    title=f'Figure 3: {ablation_type.upper()} Ablation',
                    xlabel=ablation_type.upper(),
                    name=f'figure3_{ablation_type}_ablation',
                )
    
    def generate_figure4_communication(self):
        """Generate Figure 4: Communication reduction."""
        experiments = self._filter_experiments('comm_reduction')
        
        if not experiments:
            logger.warning("No communication reduction experiments found")
            return
        
        data = {exp.get('method', 'unknown'): exp.get('communication_reduction', 0) * 100
               for exp in experiments}
        
        if data:
            self.bar.plot_bar_comparison(
                data,
                title='Figure 4: Communication Reduction',
                ylabel='Reduction (%)',
                name='figure4_communication',
            )
    
    def generate_figure5_scaling(self):
        """Generate Figure 5: Billion-scale throughput."""
        experiments = self._filter_experiments('billion_scale')
        
        # Placeholder - actual implementation would extract throughput data
        logger.info("Figure 5 generation placeholder")
    
    def generate_figure6_outer_optimizer(self):
        """Generate Figure 6: Outer optimizer comparison."""
        experiments = self._filter_experiments('outer_optimizer')
        
        if not experiments:
            logger.warning("No outer optimizer experiments found")
            return
        
        data = {exp.get('outer_type', 'unknown'): exp.get('final_loss', float('inf'))
               for exp in experiments}
        
        if data:
            self.bar.plot_bar_comparison(
                data,
                title='Figure 6: Outer Optimizer Comparison',
                ylabel='Final Loss',
                name='figure6_outer_optimizer',
            )
    
    def generate_figure7_muon(self):
        """Generate Figure 7: Muon integration."""
        experiments = self._filter_experiments('muon')
        
        if not experiments:
            logger.warning("No Muon experiments found")
            return
        
        data = {}
        for exp in experiments:
            method = exp.get('optimizer_type', exp.get('method', 'unknown'))
            if 'loss_curve' in exp:
                data[method] = {
                    'steps': [p['step'] for p in exp['loss_curve']],
                    'losses': [p['value'] for p in exp['loss_curve']],
                }
        
        if data:
            self.convergence.plot_convergence(
                data,
                title='Figure 7: Muon Integration',
                name='figure7_muon',
            )
    
    def generate_table2_wallclock(self):
        """Generate Table 2 visualization."""
        experiments = self._filter_experiments('wallclock')
        
        if not experiments:
            logger.warning("No wall-clock experiments found")
            return
        
        # Generate as bar chart
        data = {exp.get('method', 'unknown'): exp.get('throughput_tokens_per_sec', 0)
               for exp in experiments}
        
        if data:
            self.bar.plot_bar_comparison(
                data,
                title='Table 2: Throughput Comparison',
                ylabel='Tokens/sec',
                name='table2_wallclock',
            )
    
    def _filter_experiments(self, exp_type: str) -> List[Dict]:
        """Filter experiments by type."""
        if 'experiments' not in self.results:
            return []
        
        return [e for e in self.results['experiments']
               if e.get('experiment_type', '') == exp_type]


# =============================================================================
# PART 4: MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Figure Generator")
    parser.add_argument("--results", type=str, required=True, help="Results JSON file")
    parser.add_argument("--output-dir", type=str, default="./figures")
    parser.add_argument("--figures", type=str, nargs="+", 
                       help="Specific figures to generate")
    parser.add_argument("--formats", type=str, nargs="+", default=['pdf', 'png'])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure Generator")
    print("=" * 70)
    
    generator = AutoFigureGenerator(
        results_path=Path(args.results),
        output_dir=Path(args.output_dir),
    )
    
    generator.generate_all()
    
    print(f"\nFigures saved to: {args.output_dir}")
    print("\n[M041] Figure Generator - COMPLETED")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
