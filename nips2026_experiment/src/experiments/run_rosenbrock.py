#!/usr/bin/env python3
"""
===============================================================================
M031: Rosenbrock Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the Rosenbrock optimization experiment for Figure 1.
The Rosenbrock function serves as a canonical test for distributed optimization
convergence analysis.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- M = 256 workers (simulated)
- σ = 1.5 noise level
- Optimizers: DDP, Local Adam, DES-LOC (multiple K values)
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M031"

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import hashlib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.desloc_optimizer import DESLOCOptimizer, DESLOCConfig, BaseOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: ROSENBROCK FUNCTION
# =============================================================================

class RosenbrockFunction:
    """
    Rosenbrock function for optimization benchmarking.
    
    f(x, y) = (a - x)² + b(y - x²)²
    
    With a=1, b=100 (standard), the global minimum is at (1, 1) with f(1, 1) = 0.
    """
    
    def __init__(
        self,
        dim: int = 2,
        a: float = 1.0,
        b: float = 100.0,
        noise_sigma: float = 0.0,
    ):
        self.dim = dim
        self.a = a
        self.b = b
        self.noise_sigma = noise_sigma
        self.optimal_value = 0.0
        self.optimal_point = torch.ones(dim)
    
    def __call__(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Compute Rosenbrock function value.
        
        Args:
            x: Input tensor of shape (batch_size, dim) or (dim,)
            add_noise: Whether to add stochastic gradient noise
        
        Returns:
            Function value(s)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Sum over pairs (x_i, x_{i+1})
        # f = sum_i [(a - x_i)² + b(x_{i+1} - x_i²)²]
        x_even = x[:, :-1]
        x_odd = x[:, 1:]
        
        term1 = (self.a - x_even) ** 2
        term2 = self.b * (x_odd - x_even ** 2) ** 2
        
        value = (term1 + term2).sum(dim=1)
        
        if add_noise and self.noise_sigma > 0:
            noise = torch.randn_like(value) * self.noise_sigma
            value = value + noise
        
        return value.squeeze()
    
    def gradient(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Compute gradient of Rosenbrock function.
        
        Args:
            x: Input tensor of shape (dim,) or (batch_size, dim)
            add_noise: Whether to add stochastic gradient noise
        
        Returns:
            Gradient tensor
        """
        x = x.clone().requires_grad_(True)
        value = self(x, add_noise=False)
        value.backward(torch.ones_like(value))
        grad = x.grad
        
        if add_noise and self.noise_sigma > 0:
            noise = torch.randn_like(grad) * self.noise_sigma
            grad = grad + noise
        
        return grad


# =============================================================================
# PART 2: WORKER SIMULATION
# =============================================================================

@dataclass
class WorkerState:
    """State for a simulated distributed worker."""
    worker_id: int
    position: torch.Tensor
    momentum: torch.Tensor = None
    velocity: torch.Tensor = None
    step: int = 0
    
    def __post_init__(self):
        if self.momentum is None:
            self.momentum = torch.zeros_like(self.position)
        if self.velocity is None:
            self.velocity = torch.zeros_like(self.position)


class DistributedWorkerSimulator:
    """
    Simulates distributed workers for Rosenbrock optimization.
    
    Each worker maintains its own position and momentum states,
    with synchronization at specified intervals.
    """
    
    def __init__(
        self,
        num_workers: int,
        dim: int = 2,
        init_range: Tuple[float, float] = (-2.0, 2.0),
    ):
        self.num_workers = num_workers
        self.dim = dim
        self.workers: List[WorkerState] = []
        
        # Initialize workers with random positions
        for i in range(num_workers):
            position = torch.empty(dim).uniform_(*init_range)
            self.workers.append(WorkerState(worker_id=i, position=position))
        
        logger.info(f"Initialized {num_workers} workers with dim={dim}")
    
    def get_average_position(self) -> torch.Tensor:
        """Get average position across all workers."""
        positions = torch.stack([w.position for w in self.workers])
        return positions.mean(dim=0)
    
    def get_average_momentum(self) -> torch.Tensor:
        """Get average momentum across all workers."""
        momentums = torch.stack([w.momentum for w in self.workers])
        return momentums.mean(dim=0)
    
    def sync_positions(self):
        """Synchronize positions across all workers (AllReduce simulation)."""
        avg_position = self.get_average_position()
        for worker in self.workers:
            worker.position = avg_position.clone()
    
    def sync_momentum(self):
        """Synchronize momentum across all workers."""
        avg_momentum = self.get_average_momentum()
        for worker in self.workers:
            worker.momentum = avg_momentum.clone()
    
    def sync_velocity(self):
        """Synchronize velocity (second moment) across all workers."""
        velocities = torch.stack([w.velocity for w in self.workers])
        avg_velocity = velocities.mean(dim=0)
        for worker in self.workers:
            worker.velocity = avg_velocity.clone()


# =============================================================================
# PART 3: OPTIMIZERS FOR ROSENBROCK
# =============================================================================

class RosenbrockDDP:
    """DDP baseline: sync every step."""
    
    def __init__(self, simulator: DistributedWorkerSimulator, lr: float = 0.01):
        self.simulator = simulator
        self.lr = lr
        self.step_count = 0
    
    def step(self, function: RosenbrockFunction):
        """Perform one optimization step."""
        # Each worker computes gradient at current position
        for worker in self.simulator.workers:
            grad = function.gradient(worker.position)
            worker.position = worker.position - self.lr * grad
        
        # AllReduce after every step
        self.simulator.sync_positions()
        self.step_count += 1
    
    def get_loss(self, function: RosenbrockFunction) -> float:
        """Get current loss at average position."""
        avg_pos = self.simulator.get_average_position()
        return function(avg_pos, add_noise=False).item()


class RosenbrockLocalAdam:
    """Local Adam: no synchronization."""
    
    def __init__(
        self,
        simulator: DistributedWorkerSimulator,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.simulator = simulator
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0
    
    def step(self, function: RosenbrockFunction):
        """Perform one optimization step."""
        self.step_count += 1
        
        for worker in self.simulator.workers:
            grad = function.gradient(worker.position)
            
            # Update biased first moment
            worker.momentum = self.beta1 * worker.momentum + (1 - self.beta1) * grad
            
            # Update biased second moment
            worker.velocity = self.beta2 * worker.velocity + (1 - self.beta2) * grad ** 2
            
            # Bias correction
            m_hat = worker.momentum / (1 - self.beta1 ** self.step_count)
            v_hat = worker.velocity / (1 - self.beta2 ** self.step_count)
            
            # Update position
            worker.position = worker.position - self.lr * m_hat / (v_hat.sqrt() + self.eps)
    
    def get_loss(self, function: RosenbrockFunction) -> float:
        """Get current loss at average position."""
        avg_pos = self.simulator.get_average_position()
        return function(avg_pos, add_noise=False).item()


class RosenbrockDESLOC:
    """DES-LOC: sync at specified periods."""
    
    def __init__(
        self,
        simulator: DistributedWorkerSimulator,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        kx: int = 32,
        ku: int = 64,
        kv: int = 128,
    ):
        self.simulator = simulator
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.kx = kx
        self.ku = ku
        self.kv = kv
        self.step_count = 0
        
        # Compute ψ-factor
        px = 1.0 / kx
        pu = 1.0 / ku
        self.psi_factor = 4 * (1 - px) / (px ** 2) * (1 - beta1) * (1 - pu) / (6 * (1 - (1 - pu) * beta1))
    
    def step(self, function: RosenbrockFunction):
        """Perform one optimization step."""
        self.step_count += 1
        
        # Local updates
        for worker in self.simulator.workers:
            grad = function.gradient(worker.position)
            
            # Update momentum
            worker.momentum = self.beta1 * worker.momentum + (1 - self.beta1) * grad
            worker.velocity = self.beta2 * worker.velocity + (1 - self.beta2) * grad ** 2
            
            # Bias correction with ψ-factor
            m_hat = worker.momentum / (1 - self.beta1 ** self.step_count) * self.psi_factor
            v_hat = worker.velocity / (1 - self.beta2 ** self.step_count)
            
            # Update position
            worker.position = worker.position - self.lr * m_hat / (v_hat.sqrt() + self.eps)
        
        # Periodic synchronization
        if self.step_count % self.kx == 0:
            self.simulator.sync_positions()
        
        if self.step_count % self.ku == 0:
            self.simulator.sync_momentum()
        
        if self.kv > 0 and self.step_count % self.kv == 0:
            self.simulator.sync_velocity()
    
    def get_loss(self, function: RosenbrockFunction) -> float:
        """Get current loss at average position."""
        avg_pos = self.simulator.get_average_position()
        return function(avg_pos, add_noise=False).item()
    
    def get_communication_count(self) -> int:
        """Get total number of sync operations."""
        return (self.step_count // self.kx + 
                self.step_count // self.ku + 
                (self.step_count // self.kv if self.kv > 0 else 0))


# =============================================================================
# PART 4: EXPERIMENT RUNNER
# =============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    method: str
    steps: List[int]
    losses: List[float]
    final_loss: float
    total_syncs: int
    config: Dict[str, Any]
    runtime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RosenbrockExperiment:
    """Runs the complete Rosenbrock experiment suite."""
    
    def __init__(
        self,
        num_workers: int = 256,
        dim: int = 2,
        noise_sigma: float = 1.5,
        total_steps: int = 1000,
        log_interval: int = 10,
        output_dir: str = "./outputs",
    ):
        self.num_workers = num_workers
        self.dim = dim
        self.noise_sigma = noise_sigma
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create function
        self.function = RosenbrockFunction(dim=dim, noise_sigma=noise_sigma)
        
        logger.info(f"Rosenbrock Experiment: M={num_workers}, σ={noise_sigma}")
    
    def run_single(
        self,
        method: str,
        lr: float = 0.01,
        **kwargs
    ) -> ExperimentResult:
        """Run a single experiment."""
        # Create fresh simulator
        simulator = DistributedWorkerSimulator(self.num_workers, self.dim)
        
        # Create optimizer
        if method == "ddp":
            optimizer = RosenbrockDDP(simulator, lr=lr)
        elif method == "local_adam":
            optimizer = RosenbrockLocalAdam(simulator, lr=lr, **kwargs)
        elif method == "desloc":
            optimizer = RosenbrockDESLOC(simulator, lr=lr, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Run optimization
        steps = []
        losses = []
        start_time = time.time()
        
        for step in range(self.total_steps):
            optimizer.step(self.function)
            
            if step % self.log_interval == 0:
                loss = optimizer.get_loss(self.function)
                steps.append(step)
                losses.append(loss)
                
                # Log output in NKI-FA format
                print(f"[Step {step:5d}] Loss: {loss:.6f}")
        
        runtime = time.time() - start_time
        final_loss = optimizer.get_loss(self.function)
        
        # Get sync count
        if hasattr(optimizer, 'get_communication_count'):
            total_syncs = optimizer.get_communication_count()
        elif method == "ddp":
            total_syncs = self.total_steps
        else:
            total_syncs = 0
        
        config = {
            'method': method,
            'lr': lr,
            'num_workers': self.num_workers,
            'noise_sigma': self.noise_sigma,
            **kwargs
        }
        
        return ExperimentResult(
            method=method,
            steps=steps,
            losses=losses,
            final_loss=final_loss,
            total_syncs=total_syncs,
            config=config,
            runtime_seconds=runtime,
        )
    
    def run_all(self) -> List[ExperimentResult]:
        """Run all experiment configurations."""
        results = []
        
        # DDP baseline
        logger.info("Running DDP baseline...")
        print("\n### Method: DDP (AllReduce every step) ###")
        results.append(self.run_single("ddp", lr=0.01))
        
        # Local Adam
        logger.info("Running Local Adam...")
        print("\n### Method: Local Adam (No sync) ###")
        results.append(self.run_single("local_adam", lr=0.01))
        
        # DES-LOC with different K values
        k_configs = [
            (16, 32, 64),
            (32, 64, 128),
            (64, 128, 256),
        ]
        
        for kx, ku, kv in k_configs:
            logger.info(f"Running DES-LOC Kx={kx}, Ku={ku}, Kv={kv}...")
            print(f"\n### Method: DES-LOC (Kx={kx}, Ku={ku}, Kv={kv}) ###")
            results.append(self.run_single(
                "desloc", lr=0.01, kx=kx, ku=ku, kv=kv
            ))
        
        return results
    
    def save_results(self, results: List[ExperimentResult]):
        """Save results to file."""
        output_path = self.output_dir / "rosenbrock_results.json"
        
        data = {
            'experiment': 'rosenbrock',
            'config': {
                'num_workers': self.num_workers,
                'dim': self.dim,
                'noise_sigma': self.noise_sigma,
                'total_steps': self.total_steps,
            },
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Also save as log file for parsing
        log_path = self.output_dir / "rosenbrock_experiment.log"
        with open(log_path, 'w') as f:
            f.write(f"### Rosenbrock Experiment M={self.num_workers}, σ={self.noise_sigma} ###\n")
            f.write(f"### Timestamp: {datetime.now().isoformat()} ###\n\n")
            
            for result in results:
                f.write(f"### Method: {result.method} ###\n")
                f.write(f"Config: {json.dumps(result.config)}\n")
                for step, loss in zip(result.steps, result.losses):
                    f.write(f"[Step {step:5d}] Loss: {loss:.6f}\n")
                f.write(f"Final Loss: {result.final_loss:.6f}\n")
                f.write(f"Total Syncs: {result.total_syncs}\n")
                f.write(f"Runtime: {result.runtime_seconds:.2f}s\n\n")
        
        logger.info(f"Saved log to {log_path}")


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Rosenbrock Experiment")
    parser.add_argument("--num-workers", type=int, default=256)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure 1: Rosenbrock Optimization")
    print("=" * 70)
    
    experiment = RosenbrockExperiment(
        num_workers=args.num_workers,
        dim=args.dim,
        noise_sigma=args.noise_sigma,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
    )
    
    results = experiment.run_all()
    experiment.save_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        comm_reduction = 1.0 - (r.total_syncs / args.total_steps) if r.total_syncs > 0 else 1.0
        print(f"{r.method:20s}: final_loss={r.final_loss:.6f}, "
              f"syncs={r.total_syncs}, comm_reduction={comm_reduction:.1%}")
    
    print("\n[M031] Rosenbrock Experiment - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
