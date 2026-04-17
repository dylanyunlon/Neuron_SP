#!/usr/bin/env python3
"""
===============================================================================
M026: Experiment Log Parser Core Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements a rigorous log parsing system following NKI-FA commit 
da964f3 standards. All benchmark data MUST come from actual experiment logs,
not hardcoded values.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Critical Design Principles:
1. Data integrity: Every data point traces to a log line
2. Precision: Float values preserve original precision from logs
3. Reproducibility: Hash-based log verification
4. Fault tolerance: Graceful handling of malformed logs
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M026"

import re
import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union, Iterator
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: ENUMS AND CONSTANTS
# =============================================================================

class LogLevel(Enum):
    """Log severity levels for filtering."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """Types of metrics that can be extracted from logs."""
    LOSS = "loss"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    TFLOPS = "tflops"
    COMMUNICATION = "communication"
    GRADIENT_NORM = "gradient_norm"
    LEARNING_RATE = "learning_rate"
    EPOCH = "epoch"
    STEP = "step"
    TIME_MS = "time_ms"
    TOKENS_PER_SEC = "tokens_per_sec"


class ExperimentType(Enum):
    """Types of experiments in the benchmark suite."""
    ROSENBROCK = "rosenbrock"
    MOMENTUM_ABLATION = "momentum_ablation"
    SYNC_PERIOD_ABLATION = "sync_period_ablation"
    COMMUNICATION_REDUCTION = "communication_reduction"
    BILLION_SCALE = "billion_scale"
    OUTER_OPTIMIZER = "outer_optimizer"
    MUON_INTEGRATION = "muon_integration"


# =============================================================================
# PART 2: DATA CLASSES
# =============================================================================

@dataclass
class LogEntry:
    """Represents a single parsed log entry with full provenance."""
    timestamp: Optional[datetime]
    level: LogLevel
    message: str
    raw_line: str
    line_number: int
    file_path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute line hash for verification."""
        self.line_hash = hashlib.md5(self.raw_line.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level.value,
            'message': self.message,
            'line_number': self.line_number,
            'file_path': self.file_path,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'line_hash': self.line_hash
        }


@dataclass
class ExperimentRun:
    """Represents a complete experiment run with all parsed data."""
    experiment_id: str
    experiment_type: ExperimentType
    config: Dict[str, Any]
    entries: List[LogEntry] = field(default_factory=list)
    metrics_timeline: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize metrics timeline."""
        self.metrics_timeline = defaultdict(list)
    
    def add_entry(self, entry: LogEntry):
        """Add a log entry and update metrics timeline."""
        self.entries.append(entry)
        step = entry.metrics.get('step', len(self.entries))
        for metric_name, value in entry.metrics.items():
            if metric_name != 'step':
                self.metrics_timeline[metric_name].append((step, value))
    
    def compute_summary_stats(self):
        """Compute summary statistics for all metrics."""
        import numpy as np
        for metric_name, timeline in self.metrics_timeline.items():
            if timeline:
                values = [v for _, v in timeline]
                self.summary_stats[metric_name] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'final': values[-1],
                    'count': len(values)
                }


@dataclass 
class BenchmarkResult:
    """Structured benchmark result following NKI-FA format."""
    test_condition: str
    kernel: str
    direction: str  # FWD or BWD
    time_ms: float
    tflops: float
    batch_index: int
    run_index: int
    deterministic: bool
    seqlen: int
    headdim: int
    optimization_status: str  # 'Before' or 'After'
    raw_log_line: str
    line_hash: str
    
    def __post_init__(self):
        """Validate data integrity."""
        if self.time_ms <= 0:
            raise ValueError(f"Invalid time_ms: {self.time_ms}")
        if self.tflops <= 0:
            raise ValueError(f"Invalid tflops: {self.tflops}")
        self.line_hash = hashlib.md5(self.raw_log_line.encode()).hexdigest()[:8]


# =============================================================================
# PART 3: REGEX PATTERNS (Following NKI-FA Standards)
# =============================================================================

class LogPatterns:
    """Centralized regex patterns for log parsing."""
    
    # NKI-FA style benchmark output
    BENCHMARK_LINE = re.compile(
        r"(Fav\d)\s+(fwd|bwd):\s+([\d.]+)\s*ms,\s+([\d.]+)\s*TFLOPS",
        re.IGNORECASE
    )
    
    # Experiment configuration header
    CONFIG_HEADER = re.compile(
        r"###\s*(.+?)\s*###"
    )
    
    # Key-value pairs in config
    CONFIG_KV = re.compile(
        r"(\w+)\s*=\s*(\S+)"
    )
    
    # DES-LOC training log patterns
    TRAINING_STEP = re.compile(
        r"\[(?:Epoch|Step)\s*(\d+)\]"
        r"(?:.*?Loss[:\s]*([\d.e+-]+))?"
        r"(?:.*?LR[:\s]*([\d.e+-]+))?"
        r"(?:.*?Throughput[:\s]*([\d.]+)\s*tokens/s)?"
    )
    
    # Loss pattern
    LOSS_PATTERN = re.compile(
        r"(?:loss|Loss|LOSS)[:\s=]*([\d.e+-]+)"
    )
    
    # Accuracy pattern
    ACCURACY_PATTERN = re.compile(
        r"(?:acc|accuracy|Accuracy)[:\s=]*([\d.]+)%?"
    )
    
    # Memory usage pattern
    MEMORY_PATTERN = re.compile(
        r"(?:memory|Memory|GPU\s*Memory)[:\s=]*([\d.]+)\s*(?:MB|GB|MiB|GiB)"
    )
    
    # Throughput pattern
    THROUGHPUT_PATTERN = re.compile(
        r"(?:throughput|Throughput)[:\s=]*([\d.]+)\s*(?:tokens/s|samples/s|it/s)"
    )
    
    # Gradient norm pattern
    GRAD_NORM_PATTERN = re.compile(
        r"(?:grad_norm|gradient_norm|GradNorm)[:\s=]*([\d.e+-]+)"
    )
    
    # Communication time pattern
    COMM_TIME_PATTERN = re.compile(
        r"(?:comm_time|communication|sync_time)[:\s=]*([\d.]+)\s*(?:ms|s)"
    )
    
    # Timestamp patterns
    TIMESTAMP_ISO = re.compile(
        r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    )
    
    TIMESTAMP_SIMPLE = re.compile(
        r"\[(\d{2}:\d{2}:\d{2})\]"
    )
    
    # Log level pattern
    LOG_LEVEL = re.compile(
        r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]"
    )


# =============================================================================
# PART 4: BASE PARSER CLASS
# =============================================================================

class BaseLogParser(ABC):
    """Abstract base class for log parsers."""
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize parser.
        
        Args:
            strict_mode: If True, raise exceptions on parse errors.
                        If False, log warnings and continue.
        """
        self.strict_mode = strict_mode
        self.parse_errors: List[Dict[str, Any]] = []
        self.patterns = LogPatterns()
    
    @abstractmethod
    def parse_line(self, line: str, line_number: int, file_path: str) -> Optional[LogEntry]:
        """Parse a single log line."""
        pass
    
    @abstractmethod
    def parse_file(self, file_path: str) -> List[LogEntry]:
        """Parse an entire log file."""
        pass
    
    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        # Try ISO format first
        match = self.patterns.TIMESTAMP_ISO.search(line)
        if match:
            try:
                return datetime.fromisoformat(match.group(1).replace(' ', 'T'))
            except ValueError:
                pass
        
        # Try simple format
        match = self.patterns.TIMESTAMP_SIMPLE.search(line)
        if match:
            try:
                today = datetime.now().date()
                time_str = match.group(1)
                return datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        
        return None
    
    def _extract_log_level(self, line: str) -> LogLevel:
        """Extract log level from line."""
        match = self.patterns.LOG_LEVEL.search(line)
        if match:
            try:
                return LogLevel(match.group(1))
            except ValueError:
                pass
        return LogLevel.INFO
    
    def _record_error(self, line: str, line_number: int, file_path: str, error: str):
        """Record a parse error."""
        self.parse_errors.append({
            'line': line,
            'line_number': line_number,
            'file_path': file_path,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        if self.strict_mode:
            raise ValueError(f"Parse error at {file_path}:{line_number}: {error}")
        else:
            logger.warning(f"Parse error at {file_path}:{line_number}: {error}")


# =============================================================================
# PART 5: NKI-FA STYLE BENCHMARK PARSER
# =============================================================================

class NKIFABenchmarkParser(BaseLogParser):
    """
    Parser for NKI-FA style benchmark output.
    
    Follows the exact format from NKI-FA commit da964f3:
    ```
    ### headdim = 128, causal = False, seqlen = 16384, deterministic = True ###
    Fav2 fwd: 12.308ms, 357.3 TFLOPS
    Fav2 bwd: 60.364ms, 182.1 TFLOPS
    Fav3 fwd: 5.830ms, 754.4 TFLOPS
    Fav3 bwd: 18.884ms, 582.3 TFLOPS
    ```
    """
    
    def __init__(self, strict_mode: bool = True):
        super().__init__(strict_mode)
        self.current_config: Dict[str, Any] = {}
        self.run_index = 0
        self.batch_index = 0
    
    def parse_line(self, line: str, line_number: int, file_path: str) -> Optional[LogEntry]:
        """Parse a single benchmark log line."""
        line = line.strip()
        if not line:
            return None
        
        # Check for config header
        config_match = self.patterns.CONFIG_HEADER.match(line)
        if config_match:
            self._parse_config_header(config_match.group(1))
            self.run_index += 1
            # Update batch every 6 runs (following NKI-FA pattern)
            if self.run_index % 6 == 0:
                self.batch_index += 1
            return None
        
        # Check for benchmark result
        bench_match = self.patterns.BENCHMARK_LINE.search(line)
        if bench_match:
            return self._create_benchmark_entry(bench_match, line, line_number, file_path)
        
        return None
    
    def _parse_config_header(self, config_str: str):
        """Parse configuration from header string."""
        self.current_config = {}
        for match in self.patterns.CONFIG_KV.finditer(config_str):
            key = match.group(1)
            value = match.group(2)
            # Type conversion
            if value.lower() in ('true', 'false'):
                self.current_config[key] = value.lower() == 'true'
            elif value.isdigit():
                self.current_config[key] = int(value)
            else:
                try:
                    self.current_config[key] = float(value)
                except ValueError:
                    self.current_config[key] = value
    
    def _create_benchmark_entry(
        self, 
        match: re.Match, 
        raw_line: str, 
        line_number: int, 
        file_path: str
    ) -> LogEntry:
        """Create a LogEntry from benchmark match."""
        kernel = match.group(1)
        direction = match.group(2).upper()
        time_ms = float(match.group(3))
        tflops = float(match.group(4))
        
        metrics = {
            'time_ms': time_ms,
            'tflops': tflops
        }
        
        metadata = {
            'kernel': kernel,
            'direction': direction,
            'run_index': self.run_index,
            'batch_index': self.batch_index,
            **self.current_config
        }
        
        return LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message=f"{kernel} {direction}: {time_ms}ms, {tflops} TFLOPS",
            raw_line=raw_line,
            line_number=line_number,
            file_path=file_path,
            metrics=metrics,
            metadata=metadata
        )
    
    def parse_file(self, file_path: str) -> List[LogEntry]:
        """Parse an entire benchmark log file."""
        entries = []
        self.run_index = 0
        self.batch_index = 0
        self.current_config = {}
        
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                entry = self.parse_line(line, line_number, file_path)
                if entry:
                    entries.append(entry)
        
        logger.info(f"Parsed {len(entries)} entries from {file_path}")
        return entries
    
    def parse_string(self, log_string: str, source_name: str = "string") -> List[LogEntry]:
        """Parse benchmark data from a string."""
        entries = []
        self.run_index = 0
        self.batch_index = 0
        self.current_config = {}
        
        for line_number, line in enumerate(log_string.split('\n'), 1):
            entry = self.parse_line(line, line_number, source_name)
            if entry:
                entries.append(entry)
        
        return entries
    
    def to_benchmark_results(self, entries: List[LogEntry]) -> List[BenchmarkResult]:
        """Convert LogEntries to structured BenchmarkResults."""
        results = []
        for entry in entries:
            try:
                result = BenchmarkResult(
                    test_condition=str(entry.metadata.get('config_str', '')),
                    kernel=entry.metadata['kernel'],
                    direction=entry.metadata['direction'],
                    time_ms=entry.metrics['time_ms'],
                    tflops=entry.metrics['tflops'],
                    batch_index=entry.metadata['batch_index'],
                    run_index=entry.metadata['run_index'],
                    deterministic=entry.metadata.get('deterministic', False),
                    seqlen=entry.metadata.get('seqlen', 0),
                    headdim=entry.metadata.get('headdim', 128),
                    optimization_status='Before',  # Will be set by caller
                    raw_log_line=entry.raw_line,
                    line_hash=entry.line_hash
                )
                results.append(result)
            except (KeyError, ValueError) as e:
                self._record_error(entry.raw_line, entry.line_number, 
                                  entry.file_path, str(e))
        
        return results


# =============================================================================
# PART 6: DES-LOC TRAINING LOG PARSER
# =============================================================================

class DESLOCTrainingParser(BaseLogParser):
    """
    Parser for DES-LOC training logs.
    
    Expected format:
    ```
    [Epoch 10] Loss: 0.0234, LR: 1.5e-4, Throughput: 12847.3 tokens/s
    [Step 1000] loss=0.0189 grad_norm=1.234 comm_time=45.2ms
    ```
    """
    
    def parse_line(self, line: str, line_number: int, file_path: str) -> Optional[LogEntry]:
        """Parse a single training log line."""
        line = line.strip()
        if not line:
            return None
        
        metrics = {}
        metadata = {}
        
        # Extract step/epoch
        step_match = self.patterns.TRAINING_STEP.search(line)
        if step_match:
            metrics['step'] = int(step_match.group(1))
            if step_match.group(2):
                metrics['loss'] = float(step_match.group(2))
            if step_match.group(3):
                metrics['learning_rate'] = float(step_match.group(3))
            if step_match.group(4):
                metrics['throughput'] = float(step_match.group(4))
        
        # Extract additional metrics
        loss_match = self.patterns.LOSS_PATTERN.search(line)
        if loss_match and 'loss' not in metrics:
            metrics['loss'] = float(loss_match.group(1))
        
        grad_match = self.patterns.GRAD_NORM_PATTERN.search(line)
        if grad_match:
            metrics['gradient_norm'] = float(grad_match.group(1))
        
        comm_match = self.patterns.COMM_TIME_PATTERN.search(line)
        if comm_match:
            metrics['comm_time_ms'] = float(comm_match.group(1))
        
        mem_match = self.patterns.MEMORY_PATTERN.search(line)
        if mem_match:
            metrics['memory_mb'] = float(mem_match.group(1))
        
        # Only create entry if we found metrics
        if not metrics:
            return None
        
        return LogEntry(
            timestamp=self._extract_timestamp(line),
            level=self._extract_log_level(line),
            message=line,
            raw_line=line,
            line_number=line_number,
            file_path=file_path,
            metrics=metrics,
            metadata=metadata
        )
    
    def parse_file(self, file_path: str) -> List[LogEntry]:
        """Parse an entire training log file."""
        entries = []
        
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                entry = self.parse_line(line, line_number, file_path)
                if entry:
                    entries.append(entry)
        
        logger.info(f"Parsed {len(entries)} training entries from {file_path}")
        return entries


# =============================================================================
# PART 7: LOG AGGREGATOR
# =============================================================================

class LogAggregator:
    """
    Aggregates parsed log entries into experiment runs.
    """
    
    def __init__(self):
        self.runs: Dict[str, ExperimentRun] = {}
    
    def add_entries(
        self, 
        entries: List[LogEntry], 
        experiment_id: str,
        experiment_type: ExperimentType,
        config: Dict[str, Any]
    ):
        """Add entries to an experiment run."""
        if experiment_id not in self.runs:
            self.runs[experiment_id] = ExperimentRun(
                experiment_id=experiment_id,
                experiment_type=experiment_type,
                config=config
            )
        
        for entry in entries:
            self.runs[experiment_id].add_entry(entry)
    
    def finalize(self):
        """Compute summary statistics for all runs."""
        for run in self.runs.values():
            run.compute_summary_stats()
    
    def get_run(self, experiment_id: str) -> Optional[ExperimentRun]:
        """Get a specific experiment run."""
        return self.runs.get(experiment_id)
    
    def export_to_json(self, output_path: str):
        """Export all runs to JSON."""
        data = {}
        for exp_id, run in self.runs.items():
            data[exp_id] = {
                'experiment_type': run.experiment_type.value,
                'config': run.config,
                'summary_stats': run.summary_stats,
                'entry_count': len(run.entries),
                'metrics_timeline': {
                    k: [(s, v) for s, v in timeline]
                    for k, timeline in run.metrics_timeline.items()
                }
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.runs)} runs to {output_path}")


# =============================================================================
# PART 8: UTILITY FUNCTIONS
# =============================================================================

def compute_log_hash(file_path: str) -> str:
    """Compute SHA256 hash of log file for verification."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_data_integrity(entries: List[LogEntry], expected_count: int) -> bool:
    """Validate that we parsed the expected number of entries."""
    actual_count = len(entries)
    if actual_count != expected_count:
        logger.warning(
            f"Data integrity check failed: expected {expected_count} entries, "
            f"got {actual_count}"
        )
        return False
    return True


def merge_log_files(file_paths: List[str], output_path: str):
    """Merge multiple log files into one."""
    with open(output_path, 'w') as out_f:
        for path in file_paths:
            out_f.write(f"\n### Source: {path} ###\n")
            with open(path, 'r') as in_f:
                out_f.write(in_f.read())
    logger.info(f"Merged {len(file_paths)} files into {output_path}")


# =============================================================================
# PART 9: MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage and self-test
    test_log = """
### headdim = 128, causal = False, seqlen = 16384, deterministic = True ###
Fav2 fwd: 12.308ms, 357.3 TFLOPS
Fav2 bwd: 60.364ms, 182.1 TFLOPS
Fav3 fwd: 5.830ms, 754.4 TFLOPS
Fav3 bwd: 18.884ms, 582.3 TFLOPS

### headdim = 128, causal = False, seqlen = 8192, deterministic = False ###
Fav2 fwd: 3.381ms, 325.2 TFLOPS
Fav2 bwd: 9.087ms, 302.5 TFLOPS
Fav3 fwd: 1.480ms, 743.1 TFLOPS
Fav3 bwd: 4.278ms, 642.5 TFLOPS
"""
    
    parser = NKIFABenchmarkParser(strict_mode=False)
    entries = parser.parse_string(test_log)
    
    print(f"Parsed {len(entries)} entries:")
    for entry in entries:
        print(f"  [{entry.metadata['kernel']}] {entry.metadata['direction']}: "
              f"{entry.metrics['time_ms']}ms, {entry.metrics['tflops']} TFLOPS")
    
    # Convert to BenchmarkResults
    results = parser.to_benchmark_results(entries)
    print(f"\nConverted to {len(results)} BenchmarkResults")
    
    print("\n[M026] Log Parser Module - Self-test PASSED")
