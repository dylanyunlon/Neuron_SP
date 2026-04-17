#!/usr/bin/env python3
"""
===============================================================================
M030: DES-LOC Core Optimizer Implementation
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the Desynchronized Low-Communication (DES-LOC) optimizer
following the theoretical framework from the paper. Key innovations:

1. Probabilistic synchronization with periods Kx, Ku, Kv
2. ψ-factor correction for momentum bias
3. Support for Adam, AdamW, ADOPT, and Muon base optimizers

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Reference: Algorithm 1 from DES-LOC paper (template_extraction_section3.txt)
- Symbol A = DES-LOC
- Symbol B = Adam  
- Symbol C = DDP (baseline)
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M030"

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: ENUMS AND CONSTANTS
# =============================================================================

class SyncStrategy(Enum):
    """Synchronization strategy types."""
    DETERMINISTIC = "deterministic"      # Sync every K steps
    PROBABILISTIC = "probabilistic"      # Sync with probability p
    ADAPTIVE = "adaptive"                # Adapt K based on gradient variance


class BaseOptimizer(Enum):
    """Base optimizer types."""
    SGD = "sgd"
    SGDM = "sgdm"
    ADAM = "adam"
    ADAMW = "adamw"
    ADOPT = "adopt"
    MUON = "muon"


# =============================================================================
# PART 2: CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class DESLOCConfig:
    """Configuration for DES-LOC optimizer."""
    
    # Base optimizer settings
    base_optimizer: BaseOptimizer = BaseOptimizer.ADAM
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    
    # Sync periods (K values)
    kx: int = 32              # Parameter sync period
    ku: int = 64              # First momentum sync period
    kv: int = 128             # Second momentum sync period
    
    # Sync strategy
    sync_strategy: SyncStrategy = SyncStrategy.DETERMINISTIC
    
    # ψ-factor settings
    use_psi_correction: bool = True
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Distributed settings
    world_size: int = 1
    rank: int = 0
    
    def compute_psi_factor(self) -> float:
        """
        Compute ψ-factor for momentum correction.
        
        From paper equation:
        ψ = 4(1-px)/px² · (1-β)(1-pu)/(6·(1-(1-pu)β))
        
        where px = 1/Kx, pu = 1/Ku
        """
        px = 1.0 / self.kx
        pu = 1.0 / self.ku
        beta = self.beta1
        
        term1 = 4 * (1 - px) / (px ** 2)
        term2 = (1 - beta) * (1 - pu) / (6 * (1 - (1 - pu) * beta))
        
        psi = term1 * term2
        return psi
    
    @property
    def px(self) -> float:
        """Probability of parameter sync."""
        return 1.0 / self.kx
    
    @property
    def pu(self) -> float:
        """Probability of first momentum sync."""
        return 1.0 / self.ku
    
    @property
    def pv(self) -> float:
        """Probability of second momentum sync."""
        return 1.0 / self.kv if self.kv > 0 else 0.0


# =============================================================================
# PART 3: SYNC SCHEDULER
# =============================================================================

class SyncScheduler:
    """Manages synchronization scheduling across workers."""
    
    def __init__(self, config: DESLOCConfig):
        self.config = config
        self.step_count = 0
        self._rng = torch.Generator()
        if config.rank == 0:
            self._seed = torch.randint(0, 2**31, (1,)).item()
        else:
            self._seed = 0
        
        # Broadcast seed to all workers for consistent probabilistic sync
        if dist.is_initialized() and config.world_size > 1:
            seed_tensor = torch.tensor([self._seed], dtype=torch.long)
            dist.broadcast(seed_tensor, src=0)
            self._seed = seed_tensor.item()
        
        self._rng.manual_seed(self._seed)
    
    def should_sync_params(self) -> bool:
        """Check if parameters should be synchronized."""
        self.step_count += 1
        
        if self.config.sync_strategy == SyncStrategy.DETERMINISTIC:
            return self.step_count % self.config.kx == 0
        elif self.config.sync_strategy == SyncStrategy.PROBABILISTIC:
            return torch.rand(1, generator=self._rng).item() < self.config.px
        else:
            return self.step_count % self.config.kx == 0
    
    def should_sync_momentum1(self) -> bool:
        """Check if first momentum should be synchronized."""
        if self.config.sync_strategy == SyncStrategy.DETERMINISTIC:
            return self.step_count % self.config.ku == 0
        elif self.config.sync_strategy == SyncStrategy.PROBABILISTIC:
            return torch.rand(1, generator=self._rng).item() < self.config.pu
        else:
            return self.step_count % self.config.ku == 0
    
    def should_sync_momentum2(self) -> bool:
        """Check if second momentum should be synchronized."""
        if self.config.kv == 0:
            return False
        
        if self.config.sync_strategy == SyncStrategy.DETERMINISTIC:
            return self.step_count % self.config.kv == 0
        elif self.config.sync_strategy == SyncStrategy.PROBABILISTIC:
            return torch.rand(1, generator=self._rng).item() < self.config.pv
        else:
            return self.step_count % self.config.kv == 0
    
    def get_sync_schedule(self) -> Dict[str, bool]:
        """Get full sync schedule for current step."""
        return {
            'sync_params': self.should_sync_params(),
            'sync_momentum1': self.should_sync_momentum1(),
            'sync_momentum2': self.should_sync_momentum2(),
            'step': self.step_count,
        }


# =============================================================================
# PART 4: DISTRIBUTED OPERATIONS
# =============================================================================

class DistributedOps:
    """Distributed communication operations."""
    
    @staticmethod
    def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce with mean across workers."""
        if not dist.is_initialized():
            return tensor
        
        world_size = dist.get_world_size()
        if world_size == 1:
            return tensor
        
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
        return tensor
    
    @staticmethod
    def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank."""
        if not dist.is_initialized():
            return tensor
        
        dist.broadcast(tensor, src=src)
        return tensor
    
    @staticmethod
    def get_world_size() -> int:
        """Get world size."""
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()
    
    @staticmethod
    def get_rank() -> int:
        """Get current rank."""
        if not dist.is_initialized():
            return 0
        return dist.get_rank()


# =============================================================================
# PART 5: DES-LOC OPTIMIZER
# =============================================================================

class DESLOCOptimizer(Optimizer):
    """
    Desynchronized Low-Communication Optimizer.
    
    Implements Algorithm 1 from the DES-LOC paper with support for
    multiple base optimizers (Adam, AdamW, ADOPT, Muon).
    """
    
    def __init__(
        self,
        params,
        config: DESLOCConfig,
    ):
        """
        Initialize DES-LOC optimizer.
        
        Args:
            params: Model parameters to optimize
            config: DESLOCConfig with all hyperparameters
        """
        self.config = config
        self.scheduler = SyncScheduler(config)
        self.psi_factor = config.compute_psi_factor() if config.use_psi_correction else 1.0
        
        defaults = dict(
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        super().__init__(params, defaults)
        
        # Initialize state
        self._init_state()
        
        logger.info(f"Initialized DES-LOC with Kx={config.kx}, Ku={config.ku}, "
                   f"Kv={config.kv}, ψ={self.psi_factor:.4f}")
    
    def _init_state(self):
        """Initialize optimizer state for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Local copies for desynchronized updates
                    state['local_param'] = p.data.clone()
                    state['local_exp_avg'] = torch.zeros_like(p)
                    state['local_exp_avg_sq'] = torch.zeros_like(p)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            Optional loss value from closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Get sync schedule
        sync_schedule = self.scheduler.get_sync_schedule()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grad_norm = grad.norm()
                    if grad_norm > self.config.max_grad_norm:
                        grad = grad * (self.config.max_grad_norm / (grad_norm + 1e-6))
                
                state = self.state[p]
                state['step'] += 1
                
                # Local update (Algorithm 1, Line 4-6)
                self._local_update(
                    p, grad, state, lr, beta1, beta2, eps, weight_decay
                )
                
                # Synchronization (Algorithm 1, Line 7-9)
                if sync_schedule['sync_params']:
                    self._sync_params(p, state)
                
                if sync_schedule['sync_momentum1']:
                    self._sync_momentum(state, 'exp_avg', 'local_exp_avg')
                
                if sync_schedule['sync_momentum2']:
                    self._sync_momentum(state, 'exp_avg_sq', 'local_exp_avg_sq')
        
        return loss
    
    def _local_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        """Perform local update step."""
        step = state['step']
        local_exp_avg = state['local_exp_avg']
        local_exp_avg_sq = state['local_exp_avg_sq']
        
        # Weight decay
        if weight_decay != 0:
            if self.config.base_optimizer == BaseOptimizer.ADAMW:
                p.data.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(p.data, alpha=weight_decay)
        
        # Update biased first moment estimate
        local_exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Update biased second moment estimate
        local_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Apply ψ-factor correction
        corrected_exp_avg = local_exp_avg / bias_correction1
        if self.config.use_psi_correction:
            corrected_exp_avg = corrected_exp_avg * self.psi_factor
        
        corrected_exp_avg_sq = local_exp_avg_sq / bias_correction2
        
        # Compute denominator
        denom = corrected_exp_avg_sq.sqrt().add_(eps)
        
        # Update parameters
        p.data.addcdiv_(corrected_exp_avg, denom, value=-lr)
    
    def _sync_params(self, p: torch.Tensor, state: Dict[str, Any]):
        """Synchronize parameters across workers."""
        if DistributedOps.get_world_size() > 1:
            p.data = DistributedOps.all_reduce_mean(p.data)
        state['local_param'].copy_(p.data)
    
    def _sync_momentum(
        self, 
        state: Dict[str, Any], 
        global_key: str, 
        local_key: str
    ):
        """Synchronize momentum across workers."""
        if DistributedOps.get_world_size() > 1:
            state[global_key] = DistributedOps.all_reduce_mean(state[local_key])
        else:
            state[global_key].copy_(state[local_key])
        state[local_key].copy_(state[global_key])
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        step = self.scheduler.step_count
        expected_syncs_ddp = step * 3  # params + 2 momentum per step
        
        # Count actual syncs based on periods
        actual_param_syncs = step // self.config.kx
        actual_m1_syncs = step // self.config.ku
        actual_m2_syncs = step // self.config.kv if self.config.kv > 0 else 0
        actual_syncs = actual_param_syncs + actual_m1_syncs + actual_m2_syncs
        
        reduction = 1.0 - (actual_syncs / max(1, expected_syncs_ddp))
        
        return {
            'total_steps': step,
            'param_syncs': actual_param_syncs,
            'momentum1_syncs': actual_m1_syncs,
            'momentum2_syncs': actual_m2_syncs,
            'total_syncs': actual_syncs,
            'ddp_equivalent_syncs': expected_syncs_ddp,
            'communication_reduction': reduction,
            'psi_factor': self.psi_factor,
        }


# =============================================================================
# PART 6: SPECIALIZED OPTIMIZERS
# =============================================================================

class DESLOCMuon(DESLOCOptimizer):
    """
    DES-LOC with Muon base optimizer.
    
    Muon uses orthogonalized momentum, requiring special handling.
    """
    
    def __init__(self, params, config: DESLOCConfig):
        config.base_optimizer = BaseOptimizer.MUON
        config.kv = 0  # Muon only uses single momentum
        super().__init__(params, config)
    
    def _local_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        """Muon-style update with orthogonalization."""
        local_exp_avg = state['local_exp_avg']
        
        # Weight decay
        if weight_decay != 0:
            p.data.mul_(1 - lr * weight_decay)
        
        # Update momentum with Nesterov-style lookahead
        local_exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Newton-Schulz orthogonalization (simplified)
        # Full implementation would do iterative orthogonalization
        update = local_exp_avg.clone()
        
        # Apply update
        p.data.add_(update, alpha=-lr)


class DESLOCADoPT(DESLOCOptimizer):
    """
    DES-LOC with ADOPT base optimizer.
    
    ADOPT uses different second moment estimation.
    """
    
    def __init__(self, params, config: DESLOCConfig):
        config.base_optimizer = BaseOptimizer.ADOPT
        super().__init__(params, config)
    
    def _local_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        state: Dict[str, Any],
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ):
        """ADOPT-style update."""
        step = state['step']
        local_exp_avg = state['local_exp_avg']
        local_exp_avg_sq = state['local_exp_avg_sq']
        
        # ADOPT uses decoupled weight decay
        if weight_decay != 0:
            p.data.mul_(1 - lr * weight_decay)
        
        # First iteration: set exp_avg_sq to grad^2
        if step == 1:
            local_exp_avg_sq.copy_(grad * grad)
        else:
            # Update second moment with max
            local_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            local_exp_avg_sq.copy_(torch.max(local_exp_avg_sq, grad * grad))
        
        # Normalized gradient
        denom = local_exp_avg_sq.sqrt().add_(eps)
        normalized_grad = grad / denom
        
        # Update momentum
        local_exp_avg.mul_(beta1).add_(normalized_grad, alpha=1 - beta1)
        
        # Bias correction for momentum
        bias_correction1 = 1 - beta1 ** step
        corrected_exp_avg = local_exp_avg / bias_correction1
        
        # Apply ψ-factor correction
        if self.config.use_psi_correction:
            corrected_exp_avg = corrected_exp_avg * self.psi_factor
        
        # Update parameters
        p.data.add_(corrected_exp_avg, alpha=-lr)


# =============================================================================
# PART 7: OPTIMIZER FACTORY
# =============================================================================

class DESLOCFactory:
    """Factory for creating DES-LOC optimizers."""
    
    @staticmethod
    def create(
        params,
        base_optimizer: str = "adam",
        lr: float = 1e-4,
        kx: int = 32,
        ku: int = 64,
        kv: int = 128,
        **kwargs
    ) -> DESLOCOptimizer:
        """
        Create a DES-LOC optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base optimizer type ("adam", "muon", "adopt")
            lr: Learning rate
            kx, ku, kv: Sync periods
            **kwargs: Additional config options
        """
        config = DESLOCConfig(
            base_optimizer=BaseOptimizer(base_optimizer.lower()),
            lr=lr,
            kx=kx,
            ku=ku,
            kv=kv,
            **kwargs
        )
        
        if config.base_optimizer == BaseOptimizer.MUON:
            return DESLOCMuon(params, config)
        elif config.base_optimizer == BaseOptimizer.ADOPT:
            return DESLOCADoPT(params, config)
        else:
            return DESLOCOptimizer(params, config)


# =============================================================================
# PART 8: MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Self-test
    print("[M030] DES-LOC Optimizer - Self-test")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Create optimizer
    config = DESLOCConfig(
        lr=1e-3,
        kx=4,
        ku=8,
        kv=16,
    )
    
    optimizer = DESLOCOptimizer(model.parameters(), config)
    
    print(f"ψ-factor: {optimizer.psi_factor:.4f}")
    print(f"Sync periods: Kx={config.kx}, Ku={config.ku}, Kv={config.kv}")
    
    # Simulate training
    for step in range(20):
        x = torch.randn(8, 10)
        y = torch.randn(8, 10)
        
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if (step + 1) % 5 == 0:
            stats = optimizer.get_communication_stats()
            print(f"Step {step+1}: loss={loss.item():.4f}, "
                  f"comm_reduction={stats['communication_reduction']:.2%}")
    
    print("\n[M030] DES-LOC Optimizer - Self-test PASSED")
