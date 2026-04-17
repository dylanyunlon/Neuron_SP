#!/usr/bin/env python3
"""
===============================================================================
M035: Muon Integration Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the DES-LOC + Muon optimizer integration for Figure 7.
Muon uses orthogonalized momentum updates, requiring specialized handling
in the desynchronized setting.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- Muon with Newton-Schulz orthogonalization
- DES-LOC sync periods: Kx=32, Ku=64 (no Kv for Muon)
- Comparison: Adam baseline, Muon baseline, DES-LOC+Adam, DES-LOC+Muon
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M035"

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
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import logging
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.desloc_optimizer import DESLOCConfig, BaseOptimizer, SyncScheduler, DistributedOps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: MUON OPTIMIZER IMPLEMENTATION
# =============================================================================

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute G * (G^T G)^{-1/2}.
    
    This orthogonalizes the columns of G, which is the key operation in Muon.
    Uses 5 iterations for convergence.
    
    Args:
        G: Input matrix of shape (m, n) where m >= n
        steps: Number of Newton-Schulz iterations
        eps: Small constant for numerical stability
    
    Returns:
        Orthogonalized matrix of same shape as G
    """
    assert G.ndim >= 2
    
    # Transpose if needed (want tall matrix)
    a, b = G.shape[-2:]
    transpose = a < b
    if transpose:
        G = G.transpose(-2, -1)
    
    # Normalize
    G_norm = G.norm()
    if G_norm < eps:
        return G
    G = G / G_norm
    
    # Newton-Schulz iteration
    # X_{k+1} = X_k (3I - X_k^T X_k) / 2
    # converges to orthogonal matrix
    
    for _ in range(steps):
        A = G @ G.transpose(-2, -1)
        # (3I - A) / 2
        B = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype) * 1.5 - A * 0.5
        G = B @ G
    
    # Denormalize
    G = G * G_norm
    
    if transpose:
        G = G.transpose(-2, -1)
    
    return G


class MuonOptimizer(torch.optim.Optimizer):
    """
    Muon (MomentUm Orthogonalized by Newton-schulz) optimizer.
    
    Muon applies Newton-Schulz orthogonalization to the momentum,
    which helps with training stability and convergence.
    
    Paper: "Muon: An Optimizer for Hidden Layers" 
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        backend: str = "newtonschulz5",
    ):
        """
        Initialize Muon optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum coefficient (β)
            nesterov: Use Nesterov momentum
            ns_steps: Number of Newton-Schulz iterations
            weight_decay: Weight decay (L2 regularization)
            backend: Orthogonalization backend
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.backend = backend
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                
                # Update momentum
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # Orthogonalize momentum for 2D+ params (matrices)
                if p.ndim >= 2:
                    # Reshape to 2D for orthogonalization
                    original_shape = buf.shape
                    buf_2d = buf.view(buf.shape[0], -1)
                    
                    # Apply Newton-Schulz
                    buf_orth = zeropower_via_newtonschulz5(buf_2d, steps=ns_steps)
                    
                    # Reshape back
                    buf_orth = buf_orth.view(original_shape)
                    
                    # Nesterov momentum
                    if nesterov:
                        update = buf_orth.mul(momentum).add_(grad, alpha=1 - momentum)
                    else:
                        update = buf_orth
                else:
                    # For 1D params (biases), use regular momentum
                    if nesterov:
                        update = buf.mul(momentum).add_(grad, alpha=1 - momentum)
                    else:
                        update = buf
                
                # Update parameters
                p.add_(update, alpha=-lr)
        
        return loss


# =============================================================================
# PART 2: DES-LOC MUON OPTIMIZER
# =============================================================================

class DESLOCMuonOptimizer(torch.optim.Optimizer):
    """
    DES-LOC with Muon base optimizer.
    
    Combines desynchronized updates with Muon's orthogonalized momentum.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        kx: int = 32,
        ku: int = 64,
    ):
        """Initialize DES-LOC Muon optimizer."""
        self.kx = kx
        self.ku = ku
        self.step_count = 0
        
        # Compute ψ-factor for Muon (simplified, no Kv)
        px = 1.0 / kx
        pu = 1.0 / ku
        beta = momentum
        self.psi_factor = 4 * (1 - px) / (px ** 2) * (1 - beta) * (1 - pu) / (6 * (1 - (1 - pu) * beta))
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p)
                state['local_momentum'] = torch.zeros_like(p)
        
        logger.info(f"DES-LOC Muon: Kx={kx}, Ku={ku}, ψ={self.psi_factor:.4f}")
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step with DES-LOC sync."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_count += 1
        
        # Determine sync operations
        sync_params = (self.step_count % self.kx == 0)
        sync_momentum = (self.step_count % self.ku == 0)
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                local_buf = state['local_momentum']
                
                # Update local momentum
                local_buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # Orthogonalize for 2D+ params
                if p.ndim >= 2:
                    original_shape = local_buf.shape
                    buf_2d = local_buf.view(local_buf.shape[0], -1)
                    buf_orth = zeropower_via_newtonschulz5(buf_2d, steps=ns_steps)
                    buf_orth = buf_orth.view(original_shape)
                    
                    # Apply ψ-factor correction
                    buf_orth = buf_orth * self.psi_factor
                    
                    if nesterov:
                        update = buf_orth.mul(momentum).add_(grad, alpha=1 - momentum)
                    else:
                        update = buf_orth
                else:
                    if nesterov:
                        update = local_buf.mul(momentum).add_(grad, alpha=1 - momentum)
                    else:
                        update = local_buf
                
                # Update parameters
                p.add_(update, alpha=-lr)
        
        # Synchronization
        if sync_params:
            self._sync_params()
        
        if sync_momentum:
            self._sync_momentum()
        
        return loss
    
    def _sync_params(self):
        """Synchronize parameters across workers."""
        for group in self.param_groups:
            for p in group['params']:
                if DistributedOps.get_world_size() > 1:
                    p.data = DistributedOps.all_reduce_mean(p.data)
    
    def _sync_momentum(self):
        """Synchronize momentum across workers."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if DistributedOps.get_world_size() > 1:
                    state['momentum_buffer'] = DistributedOps.all_reduce_mean(state['local_momentum'])
                else:
                    state['momentum_buffer'].copy_(state['local_momentum'])
                state['local_momentum'].copy_(state['momentum_buffer'])
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        total_syncs = (self.step_count // self.kx) + (self.step_count // self.ku)
        ddp_syncs = self.step_count * 2  # params + momentum every step
        
        return {
            'total_steps': self.step_count,
            'param_syncs': self.step_count // self.kx,
            'momentum_syncs': self.step_count // self.ku,
            'total_syncs': total_syncs,
            'ddp_equivalent': ddp_syncs,
            'communication_reduction': 1.0 - (total_syncs / max(1, ddp_syncs)),
            'psi_factor': self.psi_factor,
        }


# =============================================================================
# PART 3: MODEL FOR MUON EXPERIMENTS
# =============================================================================

class MuonTestModel(nn.Module):
    """Model designed for Muon optimizer experiments."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MuonTestModel: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        x = self.encoder(x, mask=self.causal_mask[:T, :T])
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss


# =============================================================================
# PART 4: EXPERIMENT RUNNER
# =============================================================================

class OptimizerType(Enum):
    """Optimizer types for comparison."""
    ADAM = "adam"
    MUON = "muon"
    DESLOC_ADAM = "desloc_adam"
    DESLOC_MUON = "desloc_muon"


@dataclass
class MuonExperimentResult:
    """Result from Muon experiment."""
    optimizer_type: str
    steps: List[int]
    losses: List[float]
    final_loss: float
    communication_reduction: float
    runtime_seconds: float
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MuonExperiment:
    """Runs Muon integration experiments."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset
        self.dataset = self._create_dataset()
        
        logger.info(f"MuonExperiment on {self.device}")
    
    def _create_dataset(self) -> Dataset:
        """Create synthetic dataset."""
        class SyntheticData(Dataset):
            def __init__(self, n_samples=50000, seq_len=512, vocab_size=10000):
                self.n_samples = n_samples
                self.seq_len = seq_len
                self.vocab_size = vocab_size
            
            def __len__(self):
                return self.n_samples
            
            def __getitem__(self, idx):
                torch.manual_seed(idx)
                data = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
                return data[:-1], data[1:]
        
        return SyntheticData()
    
    def _create_optimizer(
        self,
        opt_type: OptimizerType,
        params,
        lr: float,
        kx: int = 32,
        ku: int = 64,
    ):
        """Create optimizer based on type."""
        if opt_type == OptimizerType.ADAM:
            return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        elif opt_type == OptimizerType.MUON:
            return MuonOptimizer(params, lr=lr, momentum=0.95, weight_decay=0.0)
        elif opt_type == OptimizerType.DESLOC_ADAM:
            from optimizers.desloc_optimizer import DESLOCOptimizer, DESLOCConfig
            config = DESLOCConfig(lr=lr, kx=kx, ku=ku, kv=128, weight_decay=0.01)
            return DESLOCOptimizer(params, config)
        elif opt_type == OptimizerType.DESLOC_MUON:
            return DESLOCMuonOptimizer(params, lr=lr, kx=kx, ku=ku)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    def run_single(
        self,
        opt_type: OptimizerType,
        total_steps: int = 3000,
        batch_size: int = 8,
        lr: float = 0.001,
        seed: int = 42,
    ) -> MuonExperimentResult:
        """Run single experiment with specified optimizer."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create model
        model = MuonTestModel().to(self.device)
        
        # Adjust LR for Muon (typically needs higher LR)
        actual_lr = lr * 10 if "muon" in opt_type.value else lr
        
        # Create optimizer
        optimizer = self._create_optimizer(opt_type, model.parameters(), actual_lr)
        
        # Data loader
        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        train_iter = iter(train_loader)
        
        # Training
        steps = []
        losses = []
        start_time = time.time()
        
        model.train()
        for step in range(total_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            _, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 50 == 0:
                steps.append(step)
                losses.append(loss.item())
                print(f"[{opt_type.value}] Step {step:5d}: loss={loss.item():.6f}")
        
        runtime = time.time() - start_time
        
        # Get communication stats
        if hasattr(optimizer, 'get_communication_stats'):
            comm_stats = optimizer.get_communication_stats()
            comm_reduction = comm_stats['communication_reduction']
        else:
            comm_reduction = 0.0
        
        return MuonExperimentResult(
            optimizer_type=opt_type.value,
            steps=steps,
            losses=losses,
            final_loss=losses[-1] if losses else float('inf'),
            communication_reduction=comm_reduction,
            runtime_seconds=runtime,
            config={
                'lr': actual_lr,
                'total_steps': total_steps,
                'batch_size': batch_size,
            }
        )
    
    def run_all(self, total_steps: int = 2000) -> List[MuonExperimentResult]:
        """Run all optimizer comparisons."""
        results = []
        
        for opt_type in OptimizerType:
            print(f"\n### Figure 7: {opt_type.value.upper()} ###")
            result = self.run_single(opt_type, total_steps=total_steps)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[MuonExperimentResult]):
        """Save results."""
        output_path = self.output_dir / "muon_experiment_results.json"
        
        data = {
            'experiment': 'muon_integration',
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Save log
        log_path = self.output_dir / "muon_experiment.log"
        with open(log_path, 'w') as f:
            f.write("### Figure 7: Muon Integration Experiment ###\n")
            f.write(f"### Timestamp: {datetime.now().isoformat()} ###\n\n")
            
            for result in results:
                f.write(f"### Optimizer: {result.optimizer_type} ###\n")
                for step, loss in zip(result.steps, result.losses):
                    f.write(f"[Step {step:5d}] Loss: {loss:.6f}\n")
                f.write(f"Final Loss: {result.final_loss:.6f}\n")
                f.write(f"Communication Reduction: {result.communication_reduction:.1%}\n")
                f.write(f"Runtime: {result.runtime_seconds:.2f}s\n\n")
        
        logger.info(f"Saved log to {log_path}")


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Muon Integration Experiment")
    parser.add_argument("--optimizer", type=str, 
                       choices=["adam", "muon", "desloc_adam", "desloc_muon", "all"],
                       default="all")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure 7: Muon Integration")
    print("=" * 70)
    
    experiment = MuonExperiment(output_dir=args.output_dir)
    
    if args.optimizer == "all":
        results = experiment.run_all(total_steps=args.total_steps)
    else:
        opt_type = OptimizerType(args.optimizer)
        results = [experiment.run_single(opt_type, total_steps=args.total_steps)]
    
    experiment.save_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r.optimizer_type:15s}: loss={r.final_loss:.6f}, "
              f"comm_reduction={r.communication_reduction:.1%}")
    
    print("\n[M035] Muon Integration - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
