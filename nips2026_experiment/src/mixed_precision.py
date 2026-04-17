#!/usr/bin/env python3
"""
===============================================================================
M047: Mixed Precision Training Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides comprehensive mixed precision training support including
FP16, BF16, and FP8 precision modes with integration into DES-LOC optimizer.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Automatic mixed precision (AMP) with GradScaler
- BF16 training for newer GPUs (A100, H100)
- FP8 support for H100 (experimental)
- Loss scaling strategies
- Precision-aware gradient synchronization
- Integration with DES-LOC optimizer
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M047"

import os
import sys
import json
import math
import functools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Iterator
)
from datetime import datetime
from enum import Enum, auto
from contextlib import contextmanager, nullcontext
import logging
import warnings

# Optional imports
try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    autocast = None
    GradScaler = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# PART 1: PRECISION MODES
# =============================================================================

class PrecisionMode(Enum):
    """Supported precision modes."""
    FP32 = "fp32"           # Full precision
    FP16 = "fp16"           # Half precision with loss scaling
    BF16 = "bf16"           # Brain floating point (no loss scaling needed)
    FP8_E4M3 = "fp8_e4m3"   # FP8 with 4-bit exponent (H100)
    FP8_E5M2 = "fp8_e5m2"   # FP8 with 5-bit exponent (H100)
    MIXED = "mixed"         # Automatic mixed precision


class LossScaleStrategy(Enum):
    """Loss scaling strategies for FP16 training."""
    STATIC = auto()         # Fixed loss scale
    DYNAMIC = auto()        # Dynamic loss scaling (default)
    BACKOFF = auto()        # Aggressive backoff on overflow
    GRADUAL = auto()        # Gradual scale increase


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    mode: PrecisionMode = PrecisionMode.BF16
    
    # Loss scaling (for FP16)
    loss_scale_strategy: LossScaleStrategy = LossScaleStrategy.DYNAMIC
    initial_loss_scale: float = 65536.0
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_growth_interval: int = 2000
    min_loss_scale: float = 1.0
    max_loss_scale: float = 2**24
    
    # Gradient clipping in mixed precision
    max_grad_norm: Optional[float] = 1.0
    
    # FP8 settings (H100)
    fp8_format: str = "e4m3"  # or "e5m2"
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1024
    fp8_amax_compute_algo: str = "max"
    
    # DES-LOC integration
    sync_in_fp32: bool = True  # Sync gradients in FP32 for numerical stability
    master_weights: bool = True  # Keep FP32 master weights
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'mode': self.mode.value,
            'loss_scale_strategy': self.loss_scale_strategy.name,
        }


# =============================================================================
# PART 2: LOSS SCALER
# =============================================================================

class DESLOCGradScaler:
    """Enhanced gradient scaler for DES-LOC with multiple scaling strategies."""
    
    def __init__(self, config: MixedPrecisionConfig = None):
        self.config = config or MixedPrecisionConfig()
        
        self._scale = self.config.initial_loss_scale
        self._growth_tracker = 0
        self._overflow_count = 0
        self._total_steps = 0
        self._scale_history: List[float] = []
        
        # PyTorch GradScaler for FP16
        if TORCH_AVAILABLE and self.config.mode == PrecisionMode.FP16:
            self._scaler = GradScaler(
                init_scale=self.config.initial_loss_scale,
                growth_factor=self.config.loss_scale_growth_factor,
                backoff_factor=self.config.loss_scale_backoff_factor,
                growth_interval=self.config.loss_scale_growth_interval,
                enabled=True,
            )
        else:
            self._scaler = None
    
    @property
    def scale(self) -> float:
        """Get current loss scale."""
        if self._scaler is not None:
            return self._scaler.get_scale()
        return self._scale
    
    def scale_loss(self, loss: 'torch.Tensor') -> 'torch.Tensor':
        """Scale the loss for backward pass."""
        if self._scaler is not None:
            return self._scaler.scale(loss)
        elif self.config.mode == PrecisionMode.FP16:
            return loss * self._scale
        return loss
    
    def unscale_gradients(self, optimizer: 'torch.optim.Optimizer'):
        """Unscale gradients before optimizer step."""
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
        elif self.config.mode == PrecisionMode.FP16:
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.div_(self._scale)
    
    def step(self, optimizer: 'torch.optim.Optimizer') -> bool:
        """Perform optimizer step with overflow checking."""
        self._total_steps += 1
        
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
            self._scale_history.append(self.scale)
            return True
        
        # Manual overflow check for custom implementation
        if self.config.mode == PrecisionMode.FP16:
            overflow = self._check_overflow(optimizer)
            
            if overflow:
                self._handle_overflow()
                return False
            else:
                optimizer.step()
                self._handle_success()
                return True
        
        optimizer.step()
        return True
    
    def _check_overflow(self, optimizer: 'torch.optim.Optimizer') -> bool:
        """Check for gradient overflow."""
        if not TORCH_AVAILABLE:
            return False
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        return True
        return False
    
    def _handle_overflow(self):
        """Handle gradient overflow."""
        self._overflow_count += 1
        self._growth_tracker = 0
        
        if self.config.loss_scale_strategy == LossScaleStrategy.DYNAMIC:
            self._scale *= self.config.loss_scale_backoff_factor
        elif self.config.loss_scale_strategy == LossScaleStrategy.BACKOFF:
            # More aggressive backoff
            self._scale *= self.config.loss_scale_backoff_factor ** 2
        
        self._scale = max(self._scale, self.config.min_loss_scale)
        self._scale_history.append(self._scale)
        
        logger.warning(f"Gradient overflow detected. Scale reduced to {self._scale}")
    
    def _handle_success(self):
        """Handle successful step (no overflow)."""
        self._growth_tracker += 1
        
        if self.config.loss_scale_strategy in (LossScaleStrategy.DYNAMIC, LossScaleStrategy.GRADUAL):
            if self._growth_tracker >= self.config.loss_scale_growth_interval:
                self._scale *= self.config.loss_scale_growth_factor
                self._scale = min(self._scale, self.config.max_loss_scale)
                self._growth_tracker = 0
        
        self._scale_history.append(self._scale)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scaler statistics."""
        return {
            'current_scale': self.scale,
            'total_steps': self._total_steps,
            'overflow_count': self._overflow_count,
            'overflow_rate': self._overflow_count / max(1, self._total_steps),
            'scale_history_len': len(self._scale_history),
            'min_scale_seen': min(self._scale_history) if self._scale_history else self._scale,
            'max_scale_seen': max(self._scale_history) if self._scale_history else self._scale,
        }


# =============================================================================
# PART 3: MIXED PRECISION CONTEXT
# =============================================================================

class MixedPrecisionContext:
    """Context manager for mixed precision training."""
    
    def __init__(self, config: MixedPrecisionConfig = None):
        self.config = config or MixedPrecisionConfig()
        self.scaler = DESLOCGradScaler(self.config)
        self._enabled = True
        
        # Determine dtype
        self._dtype = self._get_dtype()
        
        # Check hardware support
        self._check_hardware_support()
    
    def _get_dtype(self):
        """Get the appropriate dtype for the precision mode."""
        if not TORCH_AVAILABLE:
            return None
        
        dtype_map = {
            PrecisionMode.FP32: torch.float32,
            PrecisionMode.FP16: torch.float16,
            PrecisionMode.BF16: torch.bfloat16,
            PrecisionMode.MIXED: torch.float16,
        }
        
        return dtype_map.get(self.config.mode, torch.float32)
    
    def _check_hardware_support(self):
        """Check if hardware supports the precision mode."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return
        
        device_cap = torch.cuda.get_device_capability()
        
        if self.config.mode == PrecisionMode.BF16:
            if device_cap < (8, 0):
                logger.warning(
                    f"BF16 requires compute capability >= 8.0, "
                    f"current: {device_cap}. Falling back to FP16."
                )
                self.config.mode = PrecisionMode.FP16
                self._dtype = torch.float16
        
        if self.config.mode in (PrecisionMode.FP8_E4M3, PrecisionMode.FP8_E5M2):
            if device_cap < (9, 0):
                logger.warning(
                    f"FP8 requires compute capability >= 9.0 (H100), "
                    f"current: {device_cap}. Falling back to BF16."
                )
                self.config.mode = PrecisionMode.BF16
                self._dtype = torch.bfloat16
    
    @contextmanager
    def forward_context(self):
        """Context manager for forward pass."""
        if not TORCH_AVAILABLE or not self._enabled:
            yield
            return
        
        if self.config.mode == PrecisionMode.FP32:
            yield
        elif self.config.mode in (PrecisionMode.FP16, PrecisionMode.MIXED):
            with autocast(dtype=torch.float16):
                yield
        elif self.config.mode == PrecisionMode.BF16:
            with autocast(dtype=torch.bfloat16):
                yield
        else:
            # FP8 or unsupported
            with autocast(dtype=torch.bfloat16):
                yield
    
    def scale_loss(self, loss: 'torch.Tensor') -> 'torch.Tensor':
        """Scale loss for backward pass."""
        return self.scaler.scale_loss(loss)
    
    def backward(self, loss: 'torch.Tensor'):
        """Perform backward pass with loss scaling."""
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward()
    
    def unscale_and_clip(
        self,
        optimizer: 'torch.optim.Optimizer',
        max_norm: Optional[float] = None,
    ) -> float:
        """Unscale gradients and optionally clip."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        self.scaler.unscale_gradients(optimizer)
        
        max_norm = max_norm or self.config.max_grad_norm
        
        if max_norm is not None:
            # Collect all gradients
            params = []
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        params.append(param)
            
            if params:
                total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
                return total_norm.item()
        
        return 0.0
    
    def step(self, optimizer: 'torch.optim.Optimizer') -> bool:
        """Perform optimizer step."""
        return self.scaler.step(optimizer)
    
    def enable(self):
        """Enable mixed precision."""
        self._enabled = True
    
    def disable(self):
        """Disable mixed precision."""
        self._enabled = False


# =============================================================================
# PART 4: MASTER WEIGHT HANDLER
# =============================================================================

class MasterWeightHandler:
    """Handles FP32 master weights for mixed precision training."""
    
    def __init__(self, model: 'nn.Module', config: MixedPrecisionConfig = None):
        self.config = config or MixedPrecisionConfig()
        self.model = model
        self._master_params: Dict[str, 'torch.Tensor'] = {}
        self._fp16_params: Dict[str, 'torch.nn.Parameter'] = {}
        
        if self.config.master_weights:
            self._setup_master_weights()
    
    def _setup_master_weights(self):
        """Setup FP32 master weights."""
        if not TORCH_AVAILABLE:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store FP32 copy
                self._master_params[name] = param.data.float().clone()
                self._fp16_params[name] = param
                
                # Convert model param to low precision
                if self.config.mode == PrecisionMode.FP16:
                    param.data = param.data.half()
                elif self.config.mode == PrecisionMode.BF16:
                    param.data = param.data.bfloat16()
    
    def copy_grads_to_master(self):
        """Copy gradients from FP16 params to master weights."""
        if not TORCH_AVAILABLE:
            return
        
        for name, master_param in self._master_params.items():
            fp16_param = self._fp16_params[name]
            if fp16_param.grad is not None:
                if master_param.grad is None:
                    master_param.grad = torch.zeros_like(master_param)
                master_param.grad.copy_(fp16_param.grad.float())
    
    def copy_master_to_model(self):
        """Copy master weights back to model."""
        if not TORCH_AVAILABLE:
            return
        
        for name, master_param in self._master_params.items():
            fp16_param = self._fp16_params[name]
            fp16_param.data.copy_(master_param.data)
    
    def get_master_params(self) -> Iterator[Tuple[str, 'torch.Tensor']]:
        """Get iterator over master parameters."""
        return iter(self._master_params.items())
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict with master weights."""
        return {
            'master_params': {k: v.clone() for k, v in self._master_params.items()},
            'config': self.config.to_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict with master weights."""
        if 'master_params' in state_dict:
            for name, param in state_dict['master_params'].items():
                if name in self._master_params:
                    self._master_params[name].copy_(param)
            self.copy_master_to_model()


# =============================================================================
# PART 5: DES-LOC MIXED PRECISION INTEGRATION
# =============================================================================

class DESLOCMixedPrecision:
    """Integrates mixed precision with DES-LOC optimizer."""
    
    def __init__(
        self,
        model: 'nn.Module',
        optimizer: 'torch.optim.Optimizer',
        config: MixedPrecisionConfig = None,
        Kx: int = 32,
        Ku: int = 64,
        Kv: int = 128,
    ):
        self.config = config or MixedPrecisionConfig()
        self.model = model
        self.optimizer = optimizer
        
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        
        # Components
        self.precision_ctx = MixedPrecisionContext(self.config)
        self.master_weights = MasterWeightHandler(model, self.config) if self.config.master_weights else None
        
        self.current_step = 0
        self._sync_dtype = torch.float32 if self.config.sync_in_fp32 else self.precision_ctx._dtype
    
    @contextmanager
    def forward_backward(self):
        """Context for forward and backward pass."""
        with self.precision_ctx.forward_context():
            yield
    
    def backward(self, loss: 'torch.Tensor'):
        """Perform backward pass."""
        self.precision_ctx.backward(loss)
    
    def step(self) -> Dict[str, Any]:
        """Perform optimizer step with DES-LOC sync considerations."""
        self.current_step += 1
        
        # Determine if this is a sync step
        is_Kx_sync = self.current_step % self.Kx == 0
        is_Ku_sync = self.current_step % self.Ku == 0
        is_Kv_sync = self.current_step % self.Kv == 0
        
        # Unscale and clip gradients
        grad_norm = self.precision_ctx.unscale_and_clip(self.optimizer)
        
        # Handle master weights
        if self.master_weights:
            self.master_weights.copy_grads_to_master()
        
        # Perform optimizer step
        success = self.precision_ctx.step(self.optimizer)
        
        if success:
            # Update model from master weights
            if self.master_weights:
                self.master_weights.copy_master_to_model()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        return {
            'step': self.current_step,
            'success': success,
            'grad_norm': grad_norm,
            'loss_scale': self.precision_ctx.scaler.scale,
            'is_sync_step': is_Kx_sync or is_Ku_sync or is_Kv_sync,
            'sync_type': 'Kv' if is_Kv_sync else ('Ku' if is_Ku_sync else ('Kx' if is_Kx_sync else None)),
        }
    
    def prepare_for_sync(self, gradients: Dict[str, 'torch.Tensor']) -> Dict[str, 'torch.Tensor']:
        """Prepare gradients for synchronization (convert to sync dtype)."""
        if not TORCH_AVAILABLE:
            return gradients
        
        if self.config.sync_in_fp32:
            return {
                name: grad.float() for name, grad in gradients.items()
            }
        return gradients
    
    def after_sync(self, gradients: Dict[str, 'torch.Tensor']) -> Dict[str, 'torch.Tensor']:
        """Convert synced gradients back to training dtype."""
        if not TORCH_AVAILABLE:
            return gradients
        
        target_dtype = self.precision_ctx._dtype
        return {
            name: grad.to(target_dtype) for name, grad in gradients.items()
        }
    
    def get_training_dtype(self):
        """Get the dtype used for training."""
        return self.precision_ctx._dtype
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        state = {
            'config': self.config.to_dict(),
            'current_step': self.current_step,
            'scaler_stats': self.precision_ctx.scaler.get_statistics(),
        }
        
        if self.master_weights:
            state['master_weights'] = self.master_weights.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        self.current_step = state_dict.get('current_step', 0)
        
        if self.master_weights and 'master_weights' in state_dict:
            self.master_weights.load_state_dict(state_dict['master_weights'])


# =============================================================================
# PART 6: PRECISION CONVERSION UTILITIES
# =============================================================================

def convert_model_precision(
    model: 'nn.Module',
    mode: PrecisionMode,
    keep_batch_norm_fp32: bool = True,
) -> 'nn.Module':
    """Convert model to specified precision."""
    if not TORCH_AVAILABLE:
        return model
    
    dtype_map = {
        PrecisionMode.FP32: torch.float32,
        PrecisionMode.FP16: torch.float16,
        PrecisionMode.BF16: torch.bfloat16,
    }
    
    target_dtype = dtype_map.get(mode, torch.float32)
    
    for name, module in model.named_modules():
        # Keep certain layers in FP32
        if keep_batch_norm_fp32 and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        
        # Convert parameters
        for param_name, param in module.named_parameters(recurse=False):
            param.data = param.data.to(target_dtype)
    
    return model


def get_recommended_precision(
    model_size_params: int,
    gpu_memory_gb: float,
    gpu_compute_cap: Tuple[int, int] = (8, 0),
) -> PrecisionMode:
    """Recommend precision mode based on model and hardware."""
    # H100 or newer: Use BF16 or FP8
    if gpu_compute_cap >= (9, 0):
        if model_size_params > 10e9:  # > 10B params
            return PrecisionMode.FP8_E4M3
        return PrecisionMode.BF16
    
    # A100 or similar: Use BF16
    if gpu_compute_cap >= (8, 0):
        return PrecisionMode.BF16
    
    # Older GPUs: Use FP16 with loss scaling
    return PrecisionMode.FP16


# =============================================================================
# PART 7: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate mixed precision training capabilities."""
    print("=" * 70)
    print("DES-LOC Mixed Precision Training Demo")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping demo")
        return
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Setup mixed precision with DES-LOC integration
    config = MixedPrecisionConfig(
        mode=PrecisionMode.BF16,
        max_grad_norm=1.0,
        sync_in_fp32=True,
        master_weights=True,
    )
    
    mp = DESLOCMixedPrecision(
        model=model,
        optimizer=optimizer,
        config=config,
        Kx=32, Ku=64, Kv=128,
    )
    
    print(f"\nPrecision mode: {config.mode.value}")
    print(f"Training dtype: {mp.get_training_dtype()}")
    
    # Simulate training steps
    for step in range(5):
        # Forward pass
        with mp.forward_backward():
            x = torch.randn(32, 512)
            y = model(x)
            loss = y.sum()
        
        # Backward pass
        mp.backward(loss)
        
        # Optimizer step
        result = mp.step()
        print(f"Step {result['step']}: loss_scale={result['loss_scale']:.1f}, "
              f"grad_norm={result['grad_norm']:.4f}, sync={result['sync_type']}")
    
    # Get scaler statistics
    print("\nScaler Statistics:")
    print(json.dumps(mp.precision_ctx.scaler.get_statistics(), indent=2))
    
    print("\n[M047] Mixed Precision Demo - COMPLETED")


if __name__ == "__main__":
    demo()
