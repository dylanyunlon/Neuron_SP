#!/usr/bin/env python3
"""
===============================================================================
M028: Experiment Runner Entry Point
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides the main entry point for running experiments.
It orchestrates batch execution of 100+ experiments with proper logging,
checkpoint management, and result collection.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Design Principles (from llm4walking.sh reference):
1. Environment setup before any experiment
2. Centralized entry point for all experiments
3. Parallel execution where possible
4. Comprehensive logging to enable post-hoc analysis
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M028"

import os
import sys
import json
import time
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import signal
import traceback
import hashlib

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import (
    ConfigManager, ExperimentConfig, FigureType, 
    HardwareTarget, ModelScale, OptimizerType
)
from log_parser import LogAggregator, DESLOCTrainingParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: CONSTANTS AND ENUMS
# =============================================================================

# GPU Device mapping based on user's hardware
GPU_DEVICES = {
    0: {"name": "NVIDIA RTX A6000", "memory_gb": 48, "cuda_id": 0},
    1: {"name": "NVIDIA RTX A6000", "memory_gb": 48, "cuda_id": 1},
    2: {"name": "NVIDIA H100 NVL", "memory_gb": 96, "cuda_id": 2},
}

# Maximum concurrent experiments per GPU type
MAX_CONCURRENT = {
    "nvidia_a6000": 2,
    "nvidia_h100": 4,
    "aws_trainium2": 8,
}

# Experiment timeout in seconds
DEFAULT_TIMEOUT = 3600 * 4  # 4 hours


# =============================================================================
# PART 2: DATA CLASSES
# =============================================================================

@dataclass
class ExperimentStatus:
    """Status of an experiment run."""
    experiment_id: str
    status: str  # "pending", "running", "completed", "failed", "timeout"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None
    gpu_id: Optional[int] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Compute duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_id': self.experiment_id,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'exit_code': self.exit_code,
            'error_message': self.error_message,
            'log_file': self.log_file,
            'gpu_id': self.gpu_id,
        }


@dataclass
class ExperimentBatch:
    """A batch of experiments to run together."""
    batch_id: str
    experiments: List[ExperimentConfig]
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.experiments)


@dataclass
class RunnerConfig:
    """Configuration for the experiment runner."""
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    max_concurrent: int = 4
    timeout_seconds: int = DEFAULT_TIMEOUT
    retry_failed: bool = True
    max_retries: int = 3
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2])
    dry_run: bool = False
    verbose: bool = True


# =============================================================================
# PART 3: ENVIRONMENT SETUP (Following llm4walking.sh pattern)
# =============================================================================

class EnvironmentManager:
    """Manages experiment environment setup."""
    
    REQUIRED_ENV_VARS = [
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
    ]
    
    OPTIONAL_ENV_VARS = [
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "TORCH_DISTRIBUTED_DEBUG",
    ]
    
    def __init__(self):
        self.original_env = dict(os.environ)
        self.modified_vars: Dict[str, str] = {}
    
    def setup_cuda_environment(self, gpu_ids: List[int]):
        """Setup CUDA environment variables."""
        cuda_devices = ",".join(str(g) for g in gpu_ids)
        self._set_env("CUDA_VISIBLE_DEVICES", cuda_devices)
        
        # Set CUDA home if not set
        if "CUDA_HOME" not in os.environ:
            cuda_paths = [
                "/usr/local/cuda",
                "/usr/local/cuda-12.4",
                "/opt/cuda",
            ]
            for path in cuda_paths:
                if os.path.exists(path):
                    self._set_env("CUDA_HOME", path)
                    break
    
    def setup_distributed_environment(self, world_size: int, rank: int = 0):
        """Setup distributed training environment."""
        self._set_env("WORLD_SIZE", str(world_size))
        self._set_env("RANK", str(rank))
        self._set_env("LOCAL_RANK", str(rank))
        self._set_env("MASTER_ADDR", "localhost")
        self._set_env("MASTER_PORT", "29500")
    
    def setup_nccl_environment(self, debug: bool = False):
        """Setup NCCL environment for collective communications."""
        if debug:
            self._set_env("NCCL_DEBUG", "INFO")
        self._set_env("NCCL_IB_DISABLE", "1")  # Disable InfiniBand if not available
        self._set_env("NCCL_P2P_DISABLE", "0")
    
    def setup_python_environment(self, extra_paths: List[str] = None):
        """Setup Python environment."""
        pythonpath = os.environ.get("PYTHONPATH", "")
        paths = [p for p in pythonpath.split(":") if p]
        
        if extra_paths:
            paths.extend(extra_paths)
        
        # Add current directory
        paths.insert(0, str(Path(__file__).parent.parent))
        
        self._set_env("PYTHONPATH", ":".join(paths))
    
    def _set_env(self, key: str, value: str):
        """Set environment variable and track modification."""
        os.environ[key] = value
        self.modified_vars[key] = value
        logger.debug(f"Set {key}={value}")
    
    def get_env_summary(self) -> Dict[str, str]:
        """Get summary of modified environment variables."""
        return dict(self.modified_vars)
    
    def restore(self):
        """Restore original environment."""
        for key in self.modified_vars:
            if key in self.original_env:
                os.environ[key] = self.original_env[key]
            else:
                del os.environ[key]
        self.modified_vars.clear()


# =============================================================================
# PART 4: EXPERIMENT EXECUTOR
# =============================================================================

class ExperimentExecutor:
    """Executes individual experiments."""
    
    def __init__(self, runner_config: RunnerConfig):
        self.config = runner_config
        self.env_manager = EnvironmentManager()
        
        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def execute(
        self, 
        experiment: ExperimentConfig, 
        gpu_id: int
    ) -> ExperimentStatus:
        """Execute a single experiment."""
        status = ExperimentStatus(
            experiment_id=experiment.experiment_id,
            status="running",
            start_time=datetime.now(),
            gpu_id=gpu_id
        )
        
        # Setup log file
        log_file = Path(self.config.log_dir) / f"{experiment.experiment_id}.log"
        status.log_file = str(log_file)
        
        try:
            # Setup environment
            self.env_manager.setup_cuda_environment([gpu_id])
            self.env_manager.setup_distributed_environment(
                experiment.training.num_workers
            )
            self.env_manager.setup_nccl_environment(debug=self.config.verbose)
            
            # Build command
            cmd = self._build_command(experiment)
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
                status.status = "completed"
                status.exit_code = 0
            else:
                # Execute with timeout
                with open(log_file, 'w') as log_f:
                    # Write header
                    log_f.write(f"### Experiment: {experiment.experiment_id} ###\n")
                    log_f.write(f"### Config Hash: {experiment.config_hash} ###\n")
                    log_f.write(f"### Start Time: {status.start_time.isoformat()} ###\n")
                    log_f.write(f"### GPU: {gpu_id} ###\n")
                    log_f.write(f"### Command: {' '.join(cmd)} ###\n")
                    log_f.write("=" * 80 + "\n\n")
                    log_f.flush()
                    
                    process = subprocess.Popen(
                        cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                    )
                    
                    try:
                        exit_code = process.wait(timeout=self.config.timeout_seconds)
                        status.exit_code = exit_code
                        status.status = "completed" if exit_code == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        process.kill()
                        status.status = "timeout"
                        status.error_message = f"Timeout after {self.config.timeout_seconds}s"
                    
                    # Write footer
                    log_f.write("\n" + "=" * 80 + "\n")
                    log_f.write(f"### End Time: {datetime.now().isoformat()} ###\n")
                    log_f.write(f"### Status: {status.status} ###\n")
        
        except Exception as e:
            status.status = "failed"
            status.error_message = str(e)
            logger.error(f"Experiment {experiment.experiment_id} failed: {e}")
            traceback.print_exc()
        
        finally:
            status.end_time = datetime.now()
            self.env_manager.restore()
        
        return status
    
    def _build_command(self, experiment: ExperimentConfig) -> List[str]:
        """Build command to execute experiment."""
        # Determine which script to run based on figure type
        script_map = {
            FigureType.FIGURE_1: "run_rosenbrock.py",
            FigureType.FIGURE_2: "run_momentum_ablation.py",
            FigureType.FIGURE_3A: "run_sync_ablation.py",
            FigureType.FIGURE_3B: "run_sync_ablation.py",
            FigureType.FIGURE_3C: "run_sync_ablation.py",
            FigureType.FIGURE_4: "run_comm_reduction.py",
            FigureType.FIGURE_5A: "run_billion_scale.py",
            FigureType.FIGURE_5B: "run_billion_scale.py",
            FigureType.FIGURE_5C: "run_billion_scale.py",
            FigureType.FIGURE_6: "run_outer_optimizer.py",
            FigureType.FIGURE_7: "run_muon.py",
            FigureType.TABLE_2: "run_wallclock.py",
        }
        
        script_name = script_map.get(experiment.figure_type, "run_generic.py")
        script_path = Path(__file__).parent / "experiments" / script_name
        
        # Build config file path
        config_path = Path(self.config.output_dir) / f"{experiment.experiment_id}_config.json"
        experiment.save(str(config_path))
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", str(config_path),
            "--output-dir", self.config.output_dir,
            "--checkpoint-dir", self.config.checkpoint_dir,
        ]
        
        if self.config.verbose:
            cmd.append("--verbose")
        
        return cmd


# =============================================================================
# PART 5: BATCH RUNNER
# =============================================================================

class BatchRunner:
    """Runs batches of experiments with scheduling."""
    
    def __init__(self, runner_config: RunnerConfig):
        self.config = runner_config
        self.executor = ExperimentExecutor(runner_config)
        self.statuses: Dict[str, ExperimentStatus] = {}
        self.gpu_availability: Dict[int, bool] = {
            gpu_id: True for gpu_id in runner_config.gpu_ids
        }
    
    def run_all(
        self, 
        experiments: List[ExperimentConfig],
        parallel: bool = True
    ) -> Dict[str, ExperimentStatus]:
        """Run all experiments."""
        logger.info(f"Starting batch run of {len(experiments)} experiments")
        
        if parallel:
            return self._run_parallel(experiments)
        else:
            return self._run_sequential(experiments)
    
    def _run_sequential(
        self, 
        experiments: List[ExperimentConfig]
    ) -> Dict[str, ExperimentStatus]:
        """Run experiments sequentially."""
        for i, exp in enumerate(experiments):
            logger.info(f"Running experiment {i+1}/{len(experiments)}: {exp.experiment_id}")
            gpu_id = self._get_available_gpu(exp)
            status = self.executor.execute(exp, gpu_id)
            self.statuses[exp.experiment_id] = status
            
            # Handle retries
            if status.status == "failed" and self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    logger.info(f"Retrying {exp.experiment_id} (attempt {retry + 2})")
                    status = self.executor.execute(exp, gpu_id)
                    self.statuses[exp.experiment_id] = status
                    if status.status == "completed":
                        break
        
        return self.statuses
    
    def _run_parallel(
        self, 
        experiments: List[ExperimentConfig]
    ) -> Dict[str, ExperimentStatus]:
        """Run experiments in parallel."""
        with ProcessPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            futures = {}
            exp_queue = list(experiments)
            
            while exp_queue or futures:
                # Submit new experiments if GPUs available
                while exp_queue and len(futures) < self.config.max_concurrent:
                    exp = exp_queue.pop(0)
                    gpu_id = self._get_available_gpu(exp)
                    if gpu_id is not None:
                        self.gpu_availability[gpu_id] = False
                        future = executor.submit(
                            self._execute_with_gpu,
                            exp,
                            gpu_id
                        )
                        futures[future] = (exp, gpu_id)
                
                # Wait for at least one to complete
                if futures:
                    done_futures = []
                    for future in as_completed(futures, timeout=10):
                        exp, gpu_id = futures[future]
                        try:
                            status = future.result()
                            self.statuses[exp.experiment_id] = status
                        except Exception as e:
                            status = ExperimentStatus(
                                experiment_id=exp.experiment_id,
                                status="failed",
                                error_message=str(e)
                            )
                            self.statuses[exp.experiment_id] = status
                        
                        self.gpu_availability[gpu_id] = True
                        done_futures.append(future)
                    
                    for future in done_futures:
                        del futures[future]
        
        return self.statuses
    
    def _execute_with_gpu(
        self, 
        experiment: ExperimentConfig, 
        gpu_id: int
    ) -> ExperimentStatus:
        """Execute experiment with specific GPU (for process pool)."""
        return self.executor.execute(experiment, gpu_id)
    
    def _get_available_gpu(self, experiment: ExperimentConfig) -> int:
        """Get an available GPU for the experiment."""
        # Prefer H100 for large models
        if experiment.model.scale in [ModelScale.XXL, ModelScale.XL]:
            if 2 in self.gpu_availability and self.gpu_availability[2]:
                return 2
        
        # Use any available GPU
        for gpu_id, available in self.gpu_availability.items():
            if available:
                return gpu_id
        
        # Default to first GPU
        return self.config.gpu_ids[0]
    
    def save_results(self, output_path: str):
        """Save all experiment results to file."""
        results = {
            exp_id: status.to_dict()
            for exp_id, status in self.statuses.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def print_summary(self):
        """Print summary of all experiments."""
        total = len(self.statuses)
        completed = sum(1 for s in self.statuses.values() if s.status == "completed")
        failed = sum(1 for s in self.statuses.values() if s.status == "failed")
        timeout = sum(1 for s in self.statuses.values() if s.status == "timeout")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT BATCH SUMMARY")
        print("=" * 60)
        print(f"Total:     {total}")
        print(f"Completed: {completed} ({100*completed/total:.1f}%)")
        print(f"Failed:    {failed} ({100*failed/total:.1f}%)")
        print(f"Timeout:   {timeout} ({100*timeout/total:.1f}%)")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed experiments:")
            for exp_id, status in self.statuses.items():
                if status.status == "failed":
                    print(f"  - {exp_id}: {status.error_message}")


# =============================================================================
# PART 6: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DES-LOC Benchmark Experiment Runner"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Log directory"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=4,
        help="Maximum concurrent experiments"
    )
    parser.add_argument(
        "--gpu-ids", type=int, nargs="+", default=[0, 1, 2],
        help="GPU IDs to use"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run experiments sequentially"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--figures", type=str, nargs="+",
        help="Specific figures to run (e.g., figure1 figure2)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create runner config
    runner_config = RunnerConfig(
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_concurrent=args.max_concurrent,
        gpu_ids=args.gpu_ids,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    
    # Generate experiment configs
    config_manager = ConfigManager()
    all_configs = config_manager.generate_all_benchmark_configs()
    
    # Filter by figure if specified
    if args.figures:
        figure_filter = [f.lower() for f in args.figures]
        all_configs = [
            c for c in all_configs
            if any(f in c.figure_type.value.lower() for f in figure_filter)
        ]
    
    logger.info(f"Running {len(all_configs)} experiments")
    
    # Run experiments
    runner = BatchRunner(runner_config)
    statuses = runner.run_all(all_configs, parallel=not args.sequential)
    
    # Save results and print summary
    results_path = Path(args.output_dir) / "experiment_results.json"
    runner.save_results(str(results_path))
    runner.print_summary()
    
    print(f"\n[M028] Experiment Runner - Completed")
    return 0 if all(s.status == "completed" for s in statuses.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
