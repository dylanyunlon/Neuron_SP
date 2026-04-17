#!/usr/bin/env python3
"""
===============================================================================
M036: Communication Reduction Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the communication cost analysis for Figure 4.
Compares DDP, Local Adam, and DES-LOC communication patterns and costs.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- Measures: AllReduce calls, bytes transferred, latency
- Model sizes: 125M, 360M, 1.7B
- Methods: DDP (every step), Local Adam (none), DES-LOC (periodic)
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M036"

import os
import sys
import json
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import logging
import random
import numpy as np
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: COMMUNICATION PROFILER
# =============================================================================

class CommunicationProfiler:
    """Profiles communication operations in distributed training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.all_reduce_count = 0
        self.broadcast_count = 0
        self.all_gather_count = 0
        self.reduce_scatter_count = 0
        
        self.bytes_sent = 0
        self.bytes_received = 0
        
        self.total_comm_time_ms = 0.0
        self.comm_times: List[float] = []
        
        self.step_comm_bytes: List[int] = []
        self.step_comm_time: List[float] = []
    
    def record_all_reduce(self, tensor: torch.Tensor, time_ms: float = 0.0):
        """Record an AllReduce operation."""
        self.all_reduce_count += 1
        
        # Bytes = tensor size * 2 (send + receive) * dtype size
        tensor_bytes = tensor.numel() * tensor.element_size()
        self.bytes_sent += tensor_bytes
        self.bytes_received += tensor_bytes
        
        self.total_comm_time_ms += time_ms
        self.comm_times.append(time_ms)
    
    def record_broadcast(self, tensor: torch.Tensor, time_ms: float = 0.0):
        """Record a Broadcast operation."""
        self.broadcast_count += 1
        
        tensor_bytes = tensor.numel() * tensor.element_size()
        self.bytes_received += tensor_bytes
        
        self.total_comm_time_ms += time_ms
        self.comm_times.append(time_ms)
    
    def record_step(self, step: int):
        """Record end of a training step."""
        self.step_comm_bytes.append(self.bytes_sent + self.bytes_received)
        self.step_comm_time.append(self.total_comm_time_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        total_bytes = self.bytes_sent + self.bytes_received
        
        return {
            'all_reduce_count': self.all_reduce_count,
            'broadcast_count': self.broadcast_count,
            'total_comm_ops': self.all_reduce_count + self.broadcast_count,
            'bytes_sent_gb': self.bytes_sent / (1024 ** 3),
            'bytes_received_gb': self.bytes_received / (1024 ** 3),
            'total_bytes_gb': total_bytes / (1024 ** 3),
            'total_comm_time_ms': self.total_comm_time_ms,
            'avg_comm_time_ms': np.mean(self.comm_times) if self.comm_times else 0.0,
            'comm_time_std_ms': np.std(self.comm_times) if self.comm_times else 0.0,
        }


# =============================================================================
# PART 2: COMMUNICATION SIMULATOR
# =============================================================================

@dataclass
class NetworkConfig:
    """Network configuration for simulation."""
    bandwidth_gbps: float = 100.0  # Network bandwidth in Gbps
    latency_us: float = 5.0        # Base latency in microseconds
    num_gpus: int = 8              # Number of GPUs
    
    def compute_allreduce_time_ms(self, tensor_bytes: int) -> float:
        """
        Compute AllReduce time using ring algorithm model.
        
        Time = 2(n-1) * (α + β * m/n)
        where:
            n = number of GPUs
            α = latency
            β = 1/bandwidth
            m = message size in bytes
        """
        n = self.num_gpus
        alpha = self.latency_us / 1000  # Convert to ms
        beta = 8.0 / (self.bandwidth_gbps * 1000)  # ms per byte
        m = tensor_bytes
        
        # Ring AllReduce: 2(n-1) rounds
        rounds = 2 * (n - 1)
        time_per_round = alpha + beta * m / n
        
        return rounds * time_per_round


class CommunicationSimulator:
    """Simulates communication costs for different training methods."""
    
    def __init__(self, network_config: NetworkConfig):
        self.network = network_config
        self.profiler = CommunicationProfiler()
    
    def simulate_ddp_step(self, param_bytes: int, momentum_bytes: int = 0) -> Dict[str, float]:
        """Simulate DDP communication for one step."""
        # DDP AllReduces gradients after every backward
        allreduce_time = self.network.compute_allreduce_time_ms(param_bytes)
        
        self.profiler.record_all_reduce(
            torch.zeros(param_bytes // 4),  # Assume float32
            allreduce_time
        )
        
        return {
            'comm_time_ms': allreduce_time,
            'bytes_transferred': param_bytes * 2,
        }
    
    def simulate_local_adam_step(self) -> Dict[str, float]:
        """Simulate Local Adam communication (none)."""
        return {
            'comm_time_ms': 0.0,
            'bytes_transferred': 0,
        }
    
    def simulate_desloc_step(
        self,
        step: int,
        param_bytes: int,
        momentum1_bytes: int,
        momentum2_bytes: int,
        kx: int = 32,
        ku: int = 64,
        kv: int = 128,
    ) -> Dict[str, float]:
        """Simulate DES-LOC communication for one step."""
        total_time = 0.0
        total_bytes = 0
        
        # Parameter sync
        if step % kx == 0:
            time_x = self.network.compute_allreduce_time_ms(param_bytes)
            self.profiler.record_all_reduce(torch.zeros(param_bytes // 4), time_x)
            total_time += time_x
            total_bytes += param_bytes * 2
        
        # First momentum sync
        if step % ku == 0:
            time_u = self.network.compute_allreduce_time_ms(momentum1_bytes)
            self.profiler.record_all_reduce(torch.zeros(momentum1_bytes // 4), time_u)
            total_time += time_u
            total_bytes += momentum1_bytes * 2
        
        # Second momentum sync
        if kv > 0 and step % kv == 0:
            time_v = self.network.compute_allreduce_time_ms(momentum2_bytes)
            self.profiler.record_all_reduce(torch.zeros(momentum2_bytes // 4), time_v)
            total_time += time_v
            total_bytes += momentum2_bytes * 2
        
        return {
            'comm_time_ms': total_time,
            'bytes_transferred': total_bytes,
        }
    
    def reset(self):
        """Reset profiler."""
        self.profiler.reset()


# =============================================================================
# PART 3: MODEL SIZE CONFIGURATIONS
# =============================================================================

@dataclass
class ModelSizeConfig:
    """Configuration for model size in communication analysis."""
    name: str
    param_count: int
    dtype_bytes: int = 4  # float32
    
    @property
    def param_bytes(self) -> int:
        return self.param_count * self.dtype_bytes
    
    @property
    def momentum_bytes(self) -> int:
        return self.param_bytes  # Same size as params


MODEL_SIZES = {
    '125M': ModelSizeConfig('125M', 125_000_000),
    '360M': ModelSizeConfig('360M', 360_000_000),
    '1.7B': ModelSizeConfig('1.7B', 1_700_000_000),
    '7B': ModelSizeConfig('7B', 7_000_000_000),
}


# =============================================================================
# PART 4: EXPERIMENT RUNNER
# =============================================================================

class CommunicationMethod(Enum):
    """Communication methods to compare."""
    DDP = "ddp"
    LOCAL_ADAM = "local_adam"
    DESLOC_K32 = "desloc_k32"    # Kx=32, Ku=64, Kv=128
    DESLOC_K64 = "desloc_k64"    # Kx=64, Ku=128, Kv=256
    DESLOC_K128 = "desloc_k128"  # Kx=128, Ku=256, Kv=512


@dataclass
class CommExperimentResult:
    """Result from communication experiment."""
    method: str
    model_size: str
    total_steps: int
    total_comm_ops: int
    total_bytes_gb: float
    total_time_ms: float
    avg_time_per_step_ms: float
    communication_reduction: float
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CommunicationExperiment:
    """Runs communication cost experiments."""
    
    def __init__(
        self,
        network_config: NetworkConfig = None,
        output_dir: str = "./outputs",
    ):
        if network_config is None:
            network_config = NetworkConfig(
                bandwidth_gbps=100.0,  # 100 Gbps InfiniBand
                latency_us=5.0,
                num_gpus=8,
            )
        
        self.simulator = CommunicationSimulator(network_config)
        self.network = network_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CommExperiment: {network_config.num_gpus} GPUs, "
                   f"{network_config.bandwidth_gbps} Gbps")
    
    def run_single(
        self,
        method: CommunicationMethod,
        model_size: str,
        total_steps: int = 10000,
    ) -> CommExperimentResult:
        """Run single communication experiment."""
        self.simulator.reset()
        
        model_config = MODEL_SIZES[model_size]
        param_bytes = model_config.param_bytes
        momentum_bytes = model_config.momentum_bytes
        
        # Get K values for DES-LOC
        if method == CommunicationMethod.DESLOC_K32:
            kx, ku, kv = 32, 64, 128
        elif method == CommunicationMethod.DESLOC_K64:
            kx, ku, kv = 64, 128, 256
        elif method == CommunicationMethod.DESLOC_K128:
            kx, ku, kv = 128, 256, 512
        else:
            kx, ku, kv = 0, 0, 0
        
        total_time = 0.0
        total_bytes = 0
        
        print(f"\n### {method.value} on {model_size} model ###")
        
        for step in range(1, total_steps + 1):
            if method == CommunicationMethod.DDP:
                result = self.simulator.simulate_ddp_step(param_bytes)
            elif method == CommunicationMethod.LOCAL_ADAM:
                result = self.simulator.simulate_local_adam_step()
            else:
                result = self.simulator.simulate_desloc_step(
                    step, param_bytes, momentum_bytes, momentum_bytes,
                    kx, ku, kv
                )
            
            total_time += result['comm_time_ms']
            total_bytes += result['bytes_transferred']
            
            # Log progress
            if step % 1000 == 0:
                print(f"[Step {step:5d}] Cumulative comm: {total_bytes/(1024**3):.2f} GB, "
                      f"{total_time:.1f} ms")
        
        # Get stats
        stats = self.simulator.profiler.get_stats()
        
        # Compute reduction vs DDP baseline
        ddp_ops = total_steps
        actual_ops = stats['total_comm_ops']
        reduction = 1.0 - (actual_ops / max(1, ddp_ops))
        
        return CommExperimentResult(
            method=method.value,
            model_size=model_size,
            total_steps=total_steps,
            total_comm_ops=stats['total_comm_ops'],
            total_bytes_gb=stats['total_bytes_gb'],
            total_time_ms=stats['total_comm_time_ms'],
            avg_time_per_step_ms=stats['total_comm_time_ms'] / total_steps,
            communication_reduction=reduction,
            config={
                'bandwidth_gbps': self.network.bandwidth_gbps,
                'latency_us': self.network.latency_us,
                'num_gpus': self.network.num_gpus,
                'kx': kx,
                'ku': ku,
                'kv': kv,
            }
        )
    
    def run_all(
        self,
        model_sizes: List[str] = ['125M', '360M'],
        total_steps: int = 5000,
    ) -> List[CommExperimentResult]:
        """Run all method-size combinations."""
        results = []
        
        for model_size in model_sizes:
            for method in CommunicationMethod:
                result = self.run_single(method, model_size, total_steps)
                results.append(result)
                
                print(f"  {method.value}: {result.total_bytes_gb:.2f} GB, "
                      f"reduction={result.communication_reduction:.1%}")
        
        return results
    
    def run_scaling_analysis(
        self,
        model_size: str = '125M',
        gpu_counts: List[int] = [4, 8, 16, 32, 64],
        total_steps: int = 1000,
    ) -> List[CommExperimentResult]:
        """Analyze scaling behavior with different GPU counts."""
        results = []
        
        for num_gpus in gpu_counts:
            # Update network config
            self.network.num_gpus = num_gpus
            
            print(f"\n### Scaling: {num_gpus} GPUs ###")
            
            for method in [CommunicationMethod.DDP, CommunicationMethod.DESLOC_K32]:
                result = self.run_single(method, model_size, total_steps)
                results.append(result)
        
        return results
    
    def save_results(self, results: List[CommExperimentResult]):
        """Save results to file."""
        output_path = self.output_dir / "communication_reduction_results.json"
        
        data = {
            'experiment': 'communication_reduction',
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Save log
        log_path = self.output_dir / "communication_reduction.log"
        with open(log_path, 'w') as f:
            f.write("### Figure 4: Communication Reduction Analysis ###\n")
            f.write(f"### Timestamp: {datetime.now().isoformat()} ###\n\n")
            
            for result in results:
                f.write(f"### {result.method} on {result.model_size} ###\n")
                f.write(f"Total Steps: {result.total_steps}\n")
                f.write(f"Total Comm Ops: {result.total_comm_ops}\n")
                f.write(f"Total Bytes: {result.total_bytes_gb:.4f} GB\n")
                f.write(f"Total Time: {result.total_time_ms:.2f} ms\n")
                f.write(f"Avg Time/Step: {result.avg_time_per_step_ms:.4f} ms\n")
                f.write(f"Communication Reduction: {result.communication_reduction:.1%}\n\n")
        
        logger.info(f"Saved log to {log_path}")
    
    def generate_comparison_table(self, results: List[CommExperimentResult]) -> str:
        """Generate comparison table."""
        lines = []
        lines.append("=" * 80)
        lines.append("Communication Cost Comparison")
        lines.append("=" * 80)
        lines.append(f"{'Method':<15} {'Model':<8} {'Comm Ops':<10} {'GB':<10} {'Time(ms)':<12} {'Reduction':<10}")
        lines.append("-" * 80)
        
        for r in results:
            lines.append(f"{r.method:<15} {r.model_size:<8} {r.total_comm_ops:<10} "
                        f"{r.total_bytes_gb:<10.2f} {r.total_time_ms:<12.1f} "
                        f"{r.communication_reduction:<10.1%}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Communication Reduction Experiment")
    parser.add_argument("--model-sizes", type=str, nargs="+", default=["125M", "360M"])
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--bandwidth-gbps", type=float, default=100.0)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--scaling-analysis", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure 4: Communication Reduction Analysis")
    print("=" * 70)
    
    network_config = NetworkConfig(
        bandwidth_gbps=args.bandwidth_gbps,
        latency_us=5.0,
        num_gpus=args.num_gpus,
    )
    
    experiment = CommunicationExperiment(
        network_config=network_config,
        output_dir=args.output_dir,
    )
    
    if args.scaling_analysis:
        results = experiment.run_scaling_analysis(
            model_size=args.model_sizes[0],
            total_steps=args.total_steps,
        )
    else:
        results = experiment.run_all(
            model_sizes=args.model_sizes,
            total_steps=args.total_steps,
        )
    
    experiment.save_results(results)
    
    # Print comparison table
    print("\n" + experiment.generate_comparison_table(results))
    
    print("\n[M036] Communication Reduction - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
