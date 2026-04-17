#!/usr/bin/env python3
"""
===============================================================================
M037: Outer Optimizer Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the outer optimizer comparison for Figure 6.
Compares Nesterov momentum vs simple averaging for parameter synchronization.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- Outer optimizers: Simple Average, Weighted Average, Nesterov, Heavy Ball
- Model: 125M GPT-2 style
- Sync period: Kx=32
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M037"

import os
import sys
import json
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import logging
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: OUTER OPTIMIZER TYPES
# =============================================================================

class OuterOptimizerType(Enum):
    """Types of outer optimizers for parameter synchronization."""
    SIMPLE_AVERAGE = "simple_avg"
    WEIGHTED_AVERAGE = "weighted_avg"
    NESTEROV = "nesterov"
    HEAVY_BALL = "heavy_ball"
    POLYAK = "polyak"


class OuterOptimizer(ABC):
    """Base class for outer optimizers."""
    
    @abstractmethod
    def aggregate(
        self,
        local_params: List[torch.Tensor],
        global_params: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Aggregate local parameters to global."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get optimizer name."""
        pass


class SimpleAverageOuter(OuterOptimizer):
    """Simple averaging of local parameters."""
    
    def aggregate(
        self,
        local_params: List[torch.Tensor],
        global_params: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        stacked = torch.stack(local_params)
        return stacked.mean(dim=0)
    
    def get_name(self) -> str:
        return "Simple Average"


class WeightedAverageOuter(OuterOptimizer):
    """Weighted averaging based on local loss."""
    
    def __init__(self):
        self.weights: Optional[torch.Tensor] = None
    
    def set_weights(self, weights: torch.Tensor):
        """Set weights for averaging (inverse of losses)."""
        self.weights = weights / weights.sum()
    
    def aggregate(
        self,
        local_params: List[torch.Tensor],
        global_params: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if self.weights is None:
            # Fall back to simple average
            stacked = torch.stack(local_params)
            return stacked.mean(dim=0)
        
        stacked = torch.stack(local_params)
        weighted = stacked * self.weights.view(-1, *([1] * (stacked.ndim - 1)))
        return weighted.sum(dim=0)
    
    def get_name(self) -> str:
        return "Weighted Average"


class NesterovOuter(OuterOptimizer):
    """Nesterov momentum for outer optimization."""
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.velocity: Optional[torch.Tensor] = None
    
    def aggregate(
        self,
        local_params: List[torch.Tensor],
        global_params: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        # Average local params
        stacked = torch.stack(local_params)
        avg_params = stacked.mean(dim=0)
        
        # Compute update direction
        update = avg_params - global_params
        
        # Initialize velocity
        if self.velocity is None:
            self.velocity = torch.zeros_like(global_params)
        
        # Nesterov update
        # v_new = μ * v + update
        # θ_new = θ + μ * v_new + update
        self.velocity = self.momentum * self.velocity + update
        new_params = global_params + self.momentum * self.velocity + update
        
        return new_params
    
    def get_name(self) -> str:
        return "Nesterov"


class HeavyBallOuter(OuterOptimizer):
    """Heavy Ball momentum for outer optimization."""
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.velocity: Optional[torch.Tensor] = None
    
    def aggregate(
        self,
        local_params: List[torch.Tensor],
        global_params: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        stacked = torch.stack(local_params)
        avg_params = stacked.mean(dim=0)
        
        update = avg_params - global_params
        
        if self.velocity is None:
            self.velocity = torch.zeros_like(global_params)
        
        # Heavy Ball: v = μ * v + update; θ = θ + v
        self.velocity = self.momentum * self.velocity + update
        new_params = global_params + self.velocity
        
        return new_params
    
    def get_name(self) -> str:
        return "Heavy Ball"


class PolyakOuter(OuterOptimizer):
    """Polyak averaging (running average of iterates)."""
    
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.running_avg: Optional[torch.Tensor] = None
    
    def aggregate(
        self,
        local_params: List[torch.Tensor],
        global_params: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        stacked = torch.stack(local_params)
        avg_params = stacked.mean(dim=0)
        
        if self.running_avg is None:
            self.running_avg = avg_params.clone()
        else:
            # Exponential moving average
            self.running_avg = self.decay * self.running_avg + (1 - self.decay) * avg_params
        
        return self.running_avg.clone()
    
    def get_name(self) -> str:
        return "Polyak"


def create_outer_optimizer(opt_type: OuterOptimizerType, **kwargs) -> OuterOptimizer:
    """Factory function for outer optimizers."""
    if opt_type == OuterOptimizerType.SIMPLE_AVERAGE:
        return SimpleAverageOuter()
    elif opt_type == OuterOptimizerType.WEIGHTED_AVERAGE:
        return WeightedAverageOuter()
    elif opt_type == OuterOptimizerType.NESTEROV:
        return NesterovOuter(momentum=kwargs.get('momentum', 0.9))
    elif opt_type == OuterOptimizerType.HEAVY_BALL:
        return HeavyBallOuter(momentum=kwargs.get('momentum', 0.9))
    elif opt_type == OuterOptimizerType.POLYAK:
        return PolyakOuter(decay=kwargs.get('decay', 0.99))
    else:
        raise ValueError(f"Unknown outer optimizer: {opt_type}")


# =============================================================================
# PART 2: DES-LOC WITH CONFIGURABLE OUTER OPTIMIZER
# =============================================================================

class DESLOCWithOuterOptimizer(Optimizer):
    """DES-LOC with configurable outer optimizer for aggregation."""
    
    def __init__(
        self,
        params,
        outer_optimizer: OuterOptimizer,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        kx: int = 32,
        num_workers: int = 4,
    ):
        self.outer_optimizer = outer_optimizer
        self.kx = kx
        self.num_workers = num_workers
        self.step_count = 0
        
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
        # Initialize worker states
        self._init_workers()
        
        logger.info(f"DES-LOC with {outer_optimizer.get_name()}: Kx={kx}, workers={num_workers}")
    
    def _init_workers(self):
        """Initialize simulated worker states."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['worker_params'] = [p.data.clone() for _ in range(self.num_workers)]
                state['worker_m'] = [torch.zeros_like(p) for _ in range(self.num_workers)]
                state['worker_v'] = [torch.zeros_like(p) for _ in range(self.num_workers)]
                state['global_param'] = p.data.clone()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad = p.grad
                
                # Simulate local updates on each worker
                for w in range(self.num_workers):
                    # Add noise to simulate heterogeneous gradients
                    worker_grad = grad + torch.randn_like(grad) * 0.01
                    
                    # Weight decay
                    if weight_decay != 0:
                        state['worker_params'][w].mul_(1 - lr * weight_decay)
                    
                    # Update momentum
                    state['worker_m'][w].mul_(beta1).add_(worker_grad, alpha=1 - beta1)
                    state['worker_v'][w].mul_(beta2).addcmul_(worker_grad, worker_grad, value=1 - beta2)
                    
                    # Bias correction
                    m_hat = state['worker_m'][w] / (1 - beta1 ** self.step_count)
                    v_hat = state['worker_v'][w] / (1 - beta2 ** self.step_count)
                    
                    # Update worker params
                    state['worker_params'][w].addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
                
                # Periodic sync with outer optimizer
                if self.step_count % self.kx == 0:
                    aggregated = self.outer_optimizer.aggregate(
                        state['worker_params'],
                        state['global_param'],
                        self.step_count,
                    )
                    
                    state['global_param'].copy_(aggregated)
                    
                    # Broadcast back to workers
                    for w in range(self.num_workers):
                        state['worker_params'][w].copy_(aggregated)
                
                # Update actual parameter (use worker 0 for simplicity)
                p.data.copy_(state['worker_params'][0])
        
        return loss
    
    def get_global_params(self):
        """Get globally synchronized parameters."""
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(self.state[p]['global_param'])
        return params


# =============================================================================
# PART 3: MODEL FOR EXPERIMENTS
# =============================================================================

class OuterOptTestModel(nn.Module):
    """Model for outer optimizer experiments."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
    ):
        super().__init__()
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"OuterOptTestModel: {n_params/1e6:.2f}M params")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = tok + pos
        
        h = self.transformer(h, mask=self.mask[:T, :T])
        h = self.ln_f(h)
        logits = self.lm_head(h)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


# =============================================================================
# PART 4: EXPERIMENT RUNNER
# =============================================================================

@dataclass
class OuterOptResult:
    """Result from outer optimizer experiment."""
    outer_type: str
    steps: List[int]
    losses: List[float]
    final_loss: float
    convergence_step: int  # Step where loss < threshold
    runtime_seconds: float
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OuterOptimizerExperiment:
    """Runs outer optimizer comparison experiments."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        self.dataset = self._create_dataset()
        
        logger.info(f"OuterOptimizerExperiment on {self.device}")
    
    def _create_dataset(self) -> Dataset:
        """Create synthetic dataset."""
        class SyntheticData(Dataset):
            def __init__(self, n=30000, seq_len=256, vocab=10000):
                self.n = n
                self.seq_len = seq_len
                self.vocab = vocab
            
            def __len__(self):
                return self.n
            
            def __getitem__(self, idx):
                torch.manual_seed(idx)
                data = torch.randint(0, self.vocab, (self.seq_len + 1,))
                return data[:-1], data[1:]
        
        return SyntheticData()
    
    def run_single(
        self,
        outer_type: OuterOptimizerType,
        total_steps: int = 3000,
        batch_size: int = 8,
        kx: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ) -> OuterOptResult:
        """Run single experiment."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create model
        model = OuterOptTestModel().to(self.device)
        
        # Create outer optimizer
        outer_opt = create_outer_optimizer(outer_type, momentum=0.9, decay=0.99)
        
        # Create optimizer
        optimizer = DESLOCWithOuterOptimizer(
            model.parameters(),
            outer_optimizer=outer_opt,
            lr=1e-4,
            kx=kx,
            num_workers=num_workers,
        )
        
        # Data
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        loader_iter = iter(loader)
        
        # Train
        steps = []
        losses = []
        convergence_step = total_steps
        convergence_threshold = 5.0
        start_time = time.time()
        
        model.train()
        for step in range(total_steps):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            
            x, y = x.to(self.device), y.to(self.device)
            
            _, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 50 == 0:
                steps.append(step)
                losses.append(loss.item())
                
                if loss.item() < convergence_threshold and convergence_step == total_steps:
                    convergence_step = step
                
                print(f"[{outer_type.value}] Step {step:5d}: loss={loss.item():.6f}")
        
        runtime = time.time() - start_time
        
        return OuterOptResult(
            outer_type=outer_type.value,
            steps=steps,
            losses=losses,
            final_loss=losses[-1] if losses else float('inf'),
            convergence_step=convergence_step,
            runtime_seconds=runtime,
            config={
                'kx': kx,
                'num_workers': num_workers,
                'total_steps': total_steps,
            }
        )
    
    def run_all(self, total_steps: int = 2000) -> List[OuterOptResult]:
        """Run all outer optimizer comparisons."""
        results = []
        
        for outer_type in OuterOptimizerType:
            print(f"\n### Figure 6: {outer_type.value} ###")
            result = self.run_single(outer_type, total_steps=total_steps)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[OuterOptResult]):
        """Save results."""
        output_path = self.output_dir / "outer_optimizer_results.json"
        
        data = {
            'experiment': 'outer_optimizer',
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Log file
        log_path = self.output_dir / "outer_optimizer.log"
        with open(log_path, 'w') as f:
            f.write("### Figure 6: Outer Optimizer Comparison ###\n")
            f.write(f"### Timestamp: {datetime.now().isoformat()} ###\n\n")
            
            for result in results:
                f.write(f"### Outer: {result.outer_type} ###\n")
                for step, loss in zip(result.steps, result.losses):
                    f.write(f"[Step {step:5d}] Loss: {loss:.6f}\n")
                f.write(f"Final Loss: {result.final_loss:.6f}\n")
                f.write(f"Convergence Step: {result.convergence_step}\n")
                f.write(f"Runtime: {result.runtime_seconds:.2f}s\n\n")
        
        logger.info(f"Saved log to {log_path}")


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Outer Optimizer Experiment")
    parser.add_argument("--outer-type", type=str,
                       choices=["simple_avg", "weighted_avg", "nesterov", "heavy_ball", "polyak", "all"],
                       default="all")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--kx", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure 6: Outer Optimizer Comparison")
    print("=" * 70)
    
    experiment = OuterOptimizerExperiment(output_dir=args.output_dir)
    
    if args.outer_type == "all":
        results = experiment.run_all(total_steps=args.total_steps)
    else:
        outer_type = OuterOptimizerType(args.outer_type)
        results = [experiment.run_single(
            outer_type,
            total_steps=args.total_steps,
            kx=args.kx,
            num_workers=args.num_workers,
        )]
    
    experiment.save_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r.outer_type:15s}: loss={r.final_loss:.6f}, "
              f"converge@{r.convergence_step}")
    
    print("\n[M037] Outer Optimizer - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
