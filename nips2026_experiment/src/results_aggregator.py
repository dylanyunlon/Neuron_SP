#!/usr/bin/env python3
"""
===============================================================================
M040: Results Aggregator Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module aggregates results from all experiment runs, parses logs,
and generates a unified JSON database for analysis and visualization.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Parse all experiment log files
- Extract metrics following NKI-FA format
- Generate unified results database
- Compute summary statistics
- Export to multiple formats (JSON, CSV, LaTeX)
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M040"

import os
import sys
import json
import re
import csv
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import logging
from collections import defaultdict
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

@dataclass
class MetricPoint:
    """Single metric data point."""
    step: int
    value: float
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {'step': self.step, 'value': self.value, 'timestamp': self.timestamp}


@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment."""
    experiment_id: str
    experiment_type: str
    method: str
    model_size: str
    
    # Training metrics
    loss_curve: List[MetricPoint] = field(default_factory=list)
    final_loss: float = float('inf')
    best_loss: float = float('inf')
    convergence_step: int = -1
    
    # Communication metrics
    total_syncs: int = 0
    communication_reduction: float = 0.0
    
    # Timing metrics
    total_time_seconds: float = 0.0
    avg_step_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    log_file: str = ""
    timestamp: str = ""
    hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['loss_curve'] = [p.to_dict() for p in self.loss_curve]
        return d


@dataclass
class AggregatedResults:
    """Aggregated results from all experiments."""
    run_id: str
    timestamp: str
    total_experiments: int
    successful_experiments: int
    failed_experiments: int
    
    experiments: List[ExperimentMetrics] = field(default_factory=list)
    
    # Summary by figure
    figure_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Summary statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['experiments'] = [e.to_dict() for e in self.experiments]
        return d


# =============================================================================
# PART 2: LOG PARSERS
# =============================================================================

class LogPatterns:
    """Regular expression patterns for parsing logs."""
    
    # Step and loss patterns
    STEP_LOSS = re.compile(r'\[Step\s+(\d+)\]\s*[Ll]oss[:\s]+([0-9.]+)')
    STEP_LOSS_LR = re.compile(r'\[Step\s+(\d+)\]\s*loss=([0-9.]+),?\s*LR[:\s=]+([0-9.eE+-]+)')
    
    # Final metrics
    FINAL_LOSS = re.compile(r'Final\s+[Ll]oss[:\s]+([0-9.]+)')
    TOTAL_SYNCS = re.compile(r'Total\s+Syncs?[:\s]+(\d+)')
    COMM_REDUCTION = re.compile(r'[Cc]ommunication\s+[Rr]eduction[:\s]+([0-9.]+)%?')
    
    # Timing
    RUNTIME = re.compile(r'Runtime[:\s]+([0-9.]+)s')
    AVG_STEP_MS = re.compile(r'(?:avg|average)\s*(?:step|time)[:\s]+([0-9.]+)\s*ms', re.IGNORECASE)
    THROUGHPUT = re.compile(r'[Tt]hroughput[:\s]+([0-9.]+)\s*(?:tok|tokens?)/s')
    
    # Configuration
    CONFIG_LINE = re.compile(r'(Kx|Ku|Kv|β₁|beta1|lr|model_size)[:\s=]+([0-9.]+)')
    
    # Experiment headers
    EXPERIMENT_HEADER = re.compile(r'###\s*(.+?)\s*###')
    METHOD_HEADER = re.compile(r'###\s*Method[:\s]+(.+?)\s*###')
    FIGURE_HEADER = re.compile(r'Figure\s+(\d+[abc]?)[:\s]+(.+)')
    
    # ψ-factor
    PSI_FACTOR = re.compile(r'[ψΨ][-_]?factor[:\s]+([0-9.]+)')


class LogParser:
    """Parses experiment log files."""
    
    def __init__(self):
        self.patterns = LogPatterns()
    
    def parse_log_file(self, filepath: Path) -> List[ExperimentMetrics]:
        """Parse a single log file."""
        experiments = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return experiments
        
        # Determine experiment type from filename
        exp_type = self._determine_experiment_type(filepath)
        
        # Parse based on type
        if exp_type == 'rosenbrock':
            experiments = self._parse_rosenbrock_log(content, filepath)
        elif exp_type == 'momentum_ablation':
            experiments = self._parse_momentum_log(content, filepath)
        elif exp_type == 'sync_ablation':
            experiments = self._parse_sync_ablation_log(content, filepath)
        elif exp_type == 'comm_reduction':
            experiments = self._parse_comm_reduction_log(content, filepath)
        elif exp_type == 'billion_scale':
            experiments = self._parse_billion_scale_log(content, filepath)
        elif exp_type == 'outer_optimizer':
            experiments = self._parse_outer_optimizer_log(content, filepath)
        elif exp_type == 'muon':
            experiments = self._parse_muon_log(content, filepath)
        elif exp_type == 'wallclock':
            experiments = self._parse_wallclock_log(content, filepath)
        else:
            experiments = self._parse_generic_log(content, filepath)
        
        return experiments
    
    def _determine_experiment_type(self, filepath: Path) -> str:
        """Determine experiment type from filename."""
        name = filepath.stem.lower()
        
        if 'rosenbrock' in name:
            return 'rosenbrock'
        elif 'momentum' in name:
            return 'momentum_ablation'
        elif 'sync' in name:
            return 'sync_ablation'
        elif 'comm' in name:
            return 'comm_reduction'
        elif 'billion' in name:
            return 'billion_scale'
        elif 'outer' in name:
            return 'outer_optimizer'
        elif 'muon' in name:
            return 'muon'
        elif 'wallclock' in name or 'table2' in name:
            return 'wallclock'
        else:
            return 'unknown'
    
    def _parse_loss_curve(self, content: str) -> List[MetricPoint]:
        """Extract loss curve from log content."""
        points = []
        
        for match in self.patterns.STEP_LOSS.finditer(content):
            step = int(match.group(1))
            loss = float(match.group(2))
            points.append(MetricPoint(step=step, value=loss))
        
        return points
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute MD5 hash of file for integrity."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:16]
    
    def _parse_rosenbrock_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse Rosenbrock experiment log."""
        experiments = []
        
        # Split by method sections
        sections = re.split(r'###\s*Method:\s*(.+?)\s*###', content)
        
        for i in range(1, len(sections), 2):
            method = sections[i].strip()
            section_content = sections[i + 1] if i + 1 < len(sections) else ""
            
            loss_curve = self._parse_loss_curve(section_content)
            
            exp = ExperimentMetrics(
                experiment_id=f"rosenbrock_{method}_{self._compute_file_hash(filepath)}",
                experiment_type='rosenbrock',
                method=method,
                model_size='N/A',
                loss_curve=loss_curve,
                final_loss=loss_curve[-1].value if loss_curve else float('inf'),
                best_loss=min(p.value for p in loss_curve) if loss_curve else float('inf'),
                log_file=str(filepath),
                timestamp=datetime.now().isoformat(),
                hash=self._compute_file_hash(filepath),
            )
            
            # Extract sync count
            sync_match = self.patterns.TOTAL_SYNCS.search(section_content)
            if sync_match:
                exp.total_syncs = int(sync_match.group(1))
            
            experiments.append(exp)
        
        return experiments
    
    def _parse_momentum_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse momentum ablation log."""
        experiments = []
        
        sections = re.split(r'###\s*β₁\s*=\s*([0-9.]+)\s*###', content)
        
        for i in range(1, len(sections), 2):
            beta1 = float(sections[i])
            section_content = sections[i + 1] if i + 1 < len(sections) else ""
            
            loss_curve = self._parse_loss_curve(section_content)
            
            psi_match = self.patterns.PSI_FACTOR.search(section_content)
            psi = float(psi_match.group(1)) if psi_match else 0.0
            
            exp = ExperimentMetrics(
                experiment_id=f"momentum_beta1_{beta1}_{self._compute_file_hash(filepath)}",
                experiment_type='momentum_ablation',
                method=f'DES-LOC(β₁={beta1})',
                model_size='125M',
                loss_curve=loss_curve,
                final_loss=loss_curve[-1].value if loss_curve else float('inf'),
                config={'beta1': beta1, 'psi_factor': psi},
                log_file=str(filepath),
                timestamp=datetime.now().isoformat(),
                hash=self._compute_file_hash(filepath),
            )
            
            experiments.append(exp)
        
        return experiments
    
    def _parse_sync_ablation_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse sync period ablation log."""
        experiments = []
        
        # Find ablation sections
        for ablation_type in ['kx', 'ku', 'kv']:
            pattern = re.compile(rf'###\s*{ablation_type}=(\d+)\s*.*?###', re.IGNORECASE)
            
            for match in pattern.finditer(content):
                value = int(match.group(1))
                start = match.end()
                
                # Find next section or end
                next_match = pattern.search(content, start)
                end = next_match.start() if next_match else len(content)
                
                section_content = content[start:end]
                loss_curve = self._parse_loss_curve(section_content)
                
                comm_match = self.patterns.COMM_REDUCTION.search(section_content)
                comm_reduction = float(comm_match.group(1)) / 100 if comm_match else 0.0
                
                exp = ExperimentMetrics(
                    experiment_id=f"sync_{ablation_type}_{value}_{self._compute_file_hash(filepath)}",
                    experiment_type='sync_ablation',
                    method=f'{ablation_type.upper()}={value}',
                    model_size='125M',
                    loss_curve=loss_curve,
                    final_loss=loss_curve[-1].value if loss_curve else float('inf'),
                    communication_reduction=comm_reduction,
                    config={ablation_type: value},
                    log_file=str(filepath),
                    timestamp=datetime.now().isoformat(),
                    hash=self._compute_file_hash(filepath),
                )
                
                experiments.append(exp)
        
        return experiments
    
    def _parse_comm_reduction_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse communication reduction log."""
        experiments = []
        
        # Parse table-like format
        lines = content.split('\n')
        for line in lines:
            # Match: method model_size comm_ops gb time reduction
            match = re.match(r'(\w+)\s+(\w+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%?', line)
            if match:
                method, model_size, ops, gb, time_ms, reduction = match.groups()
                
                exp = ExperimentMetrics(
                    experiment_id=f"comm_{method}_{model_size}_{self._compute_file_hash(filepath)}",
                    experiment_type='comm_reduction',
                    method=method,
                    model_size=model_size,
                    total_syncs=int(ops),
                    communication_reduction=float(reduction) / 100,
                    total_time_seconds=float(time_ms) / 1000,
                    log_file=str(filepath),
                    timestamp=datetime.now().isoformat(),
                    hash=self._compute_file_hash(filepath),
                )
                
                experiments.append(exp)
        
        return experiments
    
    def _parse_billion_scale_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse billion-scale experiment log."""
        return self._parse_generic_log(content, filepath)
    
    def _parse_outer_optimizer_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse outer optimizer experiment log."""
        return self._parse_generic_log(content, filepath)
    
    def _parse_muon_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse Muon experiment log."""
        return self._parse_generic_log(content, filepath)
    
    def _parse_wallclock_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse wall-clock timing log."""
        experiments = []
        
        # Parse table format
        lines = content.split('\n')
        for line in lines:
            match = re.match(
                r'(\w+)\s+(\w+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',
                line
            )
            if match:
                method, model_size, steps, time_s, ms_step, samples_s, tokens_s = match.groups()
                
                exp = ExperimentMetrics(
                    experiment_id=f"wallclock_{method}_{model_size}_{self._compute_file_hash(filepath)}",
                    experiment_type='wallclock',
                    method=method,
                    model_size=model_size,
                    total_time_seconds=float(time_s),
                    avg_step_ms=float(ms_step),
                    throughput_tokens_per_sec=float(tokens_s),
                    config={'total_steps': int(steps)},
                    log_file=str(filepath),
                    timestamp=datetime.now().isoformat(),
                    hash=self._compute_file_hash(filepath),
                )
                
                experiments.append(exp)
        
        return experiments
    
    def _parse_generic_log(self, content: str, filepath: Path) -> List[ExperimentMetrics]:
        """Parse generic log format."""
        loss_curve = self._parse_loss_curve(content)
        
        exp = ExperimentMetrics(
            experiment_id=f"generic_{self._compute_file_hash(filepath)}",
            experiment_type='unknown',
            method='unknown',
            model_size='unknown',
            loss_curve=loss_curve,
            final_loss=loss_curve[-1].value if loss_curve else float('inf'),
            log_file=str(filepath),
            timestamp=datetime.now().isoformat(),
            hash=self._compute_file_hash(filepath),
        )
        
        return [exp]


# =============================================================================
# PART 3: RESULTS AGGREGATOR
# =============================================================================

class ResultsAggregator:
    """Aggregates results from all experiments."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.parser = LogParser()
        self.experiments: List[ExperimentMetrics] = []
    
    def collect_results(self, log_dirs: List[Path] = None, json_dirs: List[Path] = None):
        """Collect results from log and JSON files."""
        # Parse log files
        if log_dirs:
            for log_dir in log_dirs:
                self._collect_from_logs(Path(log_dir))
        
        # Parse JSON files
        if json_dirs:
            for json_dir in json_dirs:
                self._collect_from_json(Path(json_dir))
        
        logger.info(f"Collected {len(self.experiments)} experiments")
    
    def _collect_from_logs(self, log_dir: Path):
        """Collect from log files."""
        if not log_dir.exists():
            logger.warning(f"Log directory not found: {log_dir}")
            return
        
        for log_file in log_dir.glob("*.log"):
            experiments = self.parser.parse_log_file(log_file)
            self.experiments.extend(experiments)
            logger.debug(f"Parsed {len(experiments)} experiments from {log_file}")
    
    def _collect_from_json(self, json_dir: Path):
        """Collect from JSON result files."""
        if not json_dir.exists():
            logger.warning(f"JSON directory not found: {json_dir}")
            return
        
        for json_file in json_dir.glob("*_results.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Convert JSON results to ExperimentMetrics
                if 'results' in data:
                    for result in data['results']:
                        exp = self._json_to_experiment(result, json_file)
                        if exp:
                            self.experiments.append(exp)
            except Exception as e:
                logger.error(f"Failed to parse {json_file}: {e}")
    
    def _json_to_experiment(self, result: Dict, filepath: Path) -> Optional[ExperimentMetrics]:
        """Convert JSON result to ExperimentMetrics."""
        try:
            loss_curve = []
            if 'steps' in result and 'losses' in result:
                for step, loss in zip(result['steps'], result['losses']):
                    loss_curve.append(MetricPoint(step=step, value=loss))
            
            return ExperimentMetrics(
                experiment_id=result.get('experiment_id', f"json_{filepath.stem}"),
                experiment_type=result.get('experiment_type', filepath.stem.split('_')[0]),
                method=result.get('method', result.get('optimizer_type', 'unknown')),
                model_size=result.get('model_size', result.get('model_scale', 'unknown')),
                loss_curve=loss_curve,
                final_loss=result.get('final_loss', float('inf')),
                communication_reduction=result.get('communication_reduction', 0.0),
                total_time_seconds=result.get('runtime_seconds', result.get('total_time_seconds', 0.0)),
                config=result.get('config', {}),
                log_file=str(filepath),
                timestamp=datetime.now().isoformat(),
            )
        except Exception as e:
            logger.error(f"Failed to convert JSON result: {e}")
            return None
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        stats = {
            'total_experiments': len(self.experiments),
            'by_type': defaultdict(int),
            'by_method': defaultdict(int),
            'by_model_size': defaultdict(int),
        }
        
        final_losses = []
        comm_reductions = []
        
        for exp in self.experiments:
            stats['by_type'][exp.experiment_type] += 1
            stats['by_method'][exp.method] += 1
            stats['by_model_size'][exp.model_size] += 1
            
            if exp.final_loss < float('inf'):
                final_losses.append(exp.final_loss)
            
            if exp.communication_reduction > 0:
                comm_reductions.append(exp.communication_reduction)
        
        if final_losses:
            stats['loss_stats'] = {
                'mean': statistics.mean(final_losses),
                'std': statistics.stdev(final_losses) if len(final_losses) > 1 else 0,
                'min': min(final_losses),
                'max': max(final_losses),
            }
        
        if comm_reductions:
            stats['comm_reduction_stats'] = {
                'mean': statistics.mean(comm_reductions),
                'std': statistics.stdev(comm_reductions) if len(comm_reductions) > 1 else 0,
                'min': min(comm_reductions),
                'max': max(comm_reductions),
            }
        
        # Convert defaultdicts to regular dicts
        stats['by_type'] = dict(stats['by_type'])
        stats['by_method'] = dict(stats['by_method'])
        stats['by_model_size'] = dict(stats['by_model_size'])
        
        return stats
    
    def generate_figure_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Generate summaries organized by figure."""
        summaries = {}
        
        # Group experiments by type
        by_type = defaultdict(list)
        for exp in self.experiments:
            by_type[exp.experiment_type].append(exp)
        
        # Figure 1: Rosenbrock
        if 'rosenbrock' in by_type:
            summaries['figure1'] = self._summarize_figure(by_type['rosenbrock'], 'Rosenbrock')
        
        # Figure 2: Momentum
        if 'momentum_ablation' in by_type:
            summaries['figure2'] = self._summarize_figure(by_type['momentum_ablation'], 'Momentum Ablation')
        
        # Figure 3: Sync Period
        if 'sync_ablation' in by_type:
            summaries['figure3'] = self._summarize_figure(by_type['sync_ablation'], 'Sync Period Ablation')
        
        # Figure 4: Communication
        if 'comm_reduction' in by_type:
            summaries['figure4'] = self._summarize_figure(by_type['comm_reduction'], 'Communication Reduction')
        
        # Figure 5: Billion Scale
        if 'billion_scale' in by_type:
            summaries['figure5'] = self._summarize_figure(by_type['billion_scale'], 'Billion Scale')
        
        # Figure 6: Outer Optimizer
        if 'outer_optimizer' in by_type:
            summaries['figure6'] = self._summarize_figure(by_type['outer_optimizer'], 'Outer Optimizer')
        
        # Figure 7: Muon
        if 'muon' in by_type:
            summaries['figure7'] = self._summarize_figure(by_type['muon'], 'Muon Integration')
        
        # Table 2: Wall-Clock
        if 'wallclock' in by_type:
            summaries['table2'] = self._summarize_figure(by_type['wallclock'], 'Wall-Clock Time')
        
        return summaries
    
    def _summarize_figure(self, experiments: List[ExperimentMetrics], title: str) -> Dict[str, Any]:
        """Summarize experiments for a figure."""
        return {
            'title': title,
            'num_experiments': len(experiments),
            'methods': list(set(exp.method for exp in experiments)),
            'best_result': min(experiments, key=lambda e: e.final_loss).to_dict() if experiments else None,
        }
    
    def save(self, run_id: str = None):
        """Save aggregated results."""
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Compute statistics
        stats = self.compute_statistics()
        figure_summaries = self.generate_figure_summaries()
        
        # Create aggregated results
        results = AggregatedResults(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            total_experiments=len(self.experiments),
            successful_experiments=len([e for e in self.experiments if e.final_loss < float('inf')]),
            failed_experiments=len([e for e in self.experiments if e.final_loss == float('inf')]),
            experiments=self.experiments,
            figure_summaries=figure_summaries,
            statistics=stats,
        )
        
        # Save JSON
        output_path = self.output_dir / f"aggregated_results_{run_id}.json"
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"Saved aggregated results to {output_path}")
        
        # Save CSV
        self._save_csv(run_id)
        
        return output_path
    
    def _save_csv(self, run_id: str):
        """Save results as CSV."""
        csv_path = self.output_dir / f"aggregated_results_{run_id}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'experiment_id', 'type', 'method', 'model_size',
                'final_loss', 'best_loss', 'comm_reduction',
                'total_time_s', 'throughput_tok_s'
            ])
            
            # Data
            for exp in self.experiments:
                writer.writerow([
                    exp.experiment_id,
                    exp.experiment_type,
                    exp.method,
                    exp.model_size,
                    exp.final_loss,
                    exp.best_loss,
                    exp.communication_reduction,
                    exp.total_time_seconds,
                    exp.throughput_tokens_per_sec,
                ])
        
        logger.info(f"Saved CSV to {csv_path}")


# =============================================================================
# PART 4: MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Results Aggregator")
    parser.add_argument("--log-dirs", type=str, nargs="+", help="Log directories")
    parser.add_argument("--json-dirs", type=str, nargs="+", help="JSON result directories")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--run-id", type=str, help="Run ID for output files")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Results Aggregator")
    print("=" * 70)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    aggregator = ResultsAggregator(output_dir)
    
    # Collect results
    log_dirs = [Path(d) for d in args.log_dirs] if args.log_dirs else [output_dir]
    json_dirs = [Path(d) for d in args.json_dirs] if args.json_dirs else [output_dir]
    
    aggregator.collect_results(log_dirs=log_dirs, json_dirs=json_dirs)
    
    # Save
    output_path = aggregator.save(args.run_id)
    
    # Print summary
    stats = aggregator.compute_statistics()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {stats['total_experiments']}")
    print(f"By type: {stats['by_type']}")
    print(f"By method: {stats['by_method']}")
    
    print(f"\nResults saved to: {output_path}")
    print("\n[M040] Results Aggregator - COMPLETED")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
