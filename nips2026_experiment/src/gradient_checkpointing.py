#!/usr/bin/env python3
"""
===============================================================================
M046: Gradient Checkpointing Integration Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides gradient checkpointing utilities integrated with DES-LOC
optimizer, enabling memory-efficient training of large models.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Selective layer checkpointing
- Automatic checkpointing policy
- Memory-compute tradeoff optimization
- Integration with DES-LOC sync schedule
- Activation recomputation strategies
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M046"

import os
import sys
import json
import math
import functools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Set, Sequence
)
from datetime import datetime
from enum import Enum, auto
from contextlib import contextmanager
import logging
import weakref

# Optional imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# PART 1: CHECKPOINTING POLICIES
# =============================================================================

class CheckpointPolicy(Enum):
    """Policies for selecting which layers to checkpoint."""
    NONE = auto()           # No checkpointing
    ALL = auto()            # Checkpoint all eligible layers
    EVERY_N = auto()        # Checkpoint every N layers
    SQRT = auto()           # Checkpoint sqrt(N) layers
    MEMORY_BUDGET = auto()  # Fit within memory budget
    SELECTIVE = auto()      # Manual selection
    ADAPTIVE = auto()       # Adapt based on memory pressure


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing."""
    policy: CheckpointPolicy = CheckpointPolicy.SQRT
    
    # For EVERY_N policy
    checkpoint_every_n: int = 2
    
    # For MEMORY_BUDGET policy
    memory_budget_gb: float = 40.0
    
    # For SELECTIVE policy
    checkpoint_layer_names: List[str] = field(default_factory=list)
    
    # Recomputation strategy
    preserve_rng_state: bool = True
    use_reentrant: bool = False  # Non-reentrant is more memory efficient
    
    # DES-LOC integration
    sync_checkpoint_boundary: bool = True  # Align checkpoints with sync boundaries
    Kx: int = 32  # Parameter sync period
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'policy': self.policy.name,
        }


# =============================================================================
# PART 2: CHECKPOINTING WRAPPER
# =============================================================================

class CheckpointedModule(nn.Module if TORCH_AVAILABLE else object):
    """Wrapper that applies gradient checkpointing to a module."""
    
    def __init__(
        self,
        module: 'nn.Module',
        config: CheckpointConfig = None,
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        self.module = module
        self.config = config or CheckpointConfig()
        self._checkpoint_enabled = True
    
    def forward(self, *args, **kwargs):
        if not TORCH_AVAILABLE:
            return self.module(*args, **kwargs)
        
        if self._checkpoint_enabled and self.training:
            # Use checkpoint
            def custom_forward(*inputs):
                return self.module(*inputs, **kwargs)
            
            return checkpoint(
                custom_forward,
                *args,
                preserve_rng_state=self.config.preserve_rng_state,
                use_reentrant=self.config.use_reentrant,
            )
        else:
            return self.module(*args, **kwargs)
    
    def enable_checkpointing(self):
        """Enable gradient checkpointing."""
        self._checkpoint_enabled = True
    
    def disable_checkpointing(self):
        """Disable gradient checkpointing."""
        self._checkpoint_enabled = False


# =============================================================================
# PART 3: CHECKPOINTING MANAGER
# =============================================================================

class CheckpointingManager:
    """Manages gradient checkpointing for a model."""
    
    def __init__(self, config: CheckpointConfig = None):
        self.config = config or CheckpointConfig()
        self.checkpointed_modules: List[weakref.ref] = []
        self.layer_memory_estimates: Dict[str, float] = {}
        self.total_memory_saved_mb: float = 0.0
    
    def apply_checkpointing(
        self,
        model: 'nn.Module',
        layer_types: Tuple[type, ...] = None,
    ) -> 'nn.Module':
        """Apply checkpointing to model according to policy."""
        if not TORCH_AVAILABLE:
            return model
        
        # Default layer types to checkpoint
        if layer_types is None:
            layer_types = self._get_default_layer_types()
        
        # Get layers to checkpoint based on policy
        layers_to_checkpoint = self._select_layers(model, layer_types)
        
        # Apply checkpointing
        for name, indices in layers_to_checkpoint.items():
            self._apply_to_layer(model, name, indices)
        
        logger.info(f"Applied checkpointing to {len(layers_to_checkpoint)} layer groups")
        return model
    
    def _get_default_layer_types(self) -> Tuple:
        """Get default layer types to checkpoint."""
        if not TORCH_AVAILABLE:
            return ()
        
        # Common transformer layer types
        types = []
        
        try:
            from transformers.models.bert.modeling_bert import BertLayer
            types.append(BertLayer)
        except ImportError:
            pass
        
        try:
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            types.append(GPT2Block)
        except ImportError:
            pass
        
        # Fallback to generic Sequential
        types.append(nn.Sequential)
        
        return tuple(types)
    
    def _select_layers(
        self,
        model: 'nn.Module',
        layer_types: Tuple[type, ...],
    ) -> Dict[str, List[int]]:
        """Select which layers to checkpoint based on policy."""
        # Find all matching layers
        matching_layers = []
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                matching_layers.append((name, module))
        
        if not matching_layers:
            return {}
        
        total_layers = len(matching_layers)
        
        # Apply policy
        if self.config.policy == CheckpointPolicy.NONE:
            return {}
        
        elif self.config.policy == CheckpointPolicy.ALL:
            selected_indices = list(range(total_layers))
        
        elif self.config.policy == CheckpointPolicy.EVERY_N:
            n = self.config.checkpoint_every_n
            selected_indices = list(range(0, total_layers, n))
        
        elif self.config.policy == CheckpointPolicy.SQRT:
            # Checkpoint every sqrt(N) layers
            step = max(1, int(math.sqrt(total_layers)))
            selected_indices = list(range(0, total_layers, step))
        
        elif self.config.policy == CheckpointPolicy.SELECTIVE:
            selected_indices = [
                i for i, (name, _) in enumerate(matching_layers)
                if any(n in name for n in self.config.checkpoint_layer_names)
            ]
        
        elif self.config.policy == CheckpointPolicy.MEMORY_BUDGET:
            selected_indices = self._select_for_memory_budget(
                matching_layers,
                self.config.memory_budget_gb,
            )
        
        elif self.config.policy == CheckpointPolicy.ADAPTIVE:
            selected_indices = self._adaptive_selection(matching_layers)
        
        else:
            selected_indices = []
        
        # DES-LOC sync boundary alignment
        if self.config.sync_checkpoint_boundary:
            selected_indices = self._align_with_sync(selected_indices, total_layers)
        
        # Group by parent module for efficient application
        result = {}
        for idx in selected_indices:
            name, _ = matching_layers[idx]
            parent_name = '.'.join(name.split('.')[:-1]) or 'root'
            if parent_name not in result:
                result[parent_name] = []
            result[parent_name].append(idx)
        
        return result
    
    def _select_for_memory_budget(
        self,
        layers: List[Tuple[str, 'nn.Module']],
        budget_gb: float,
    ) -> List[int]:
        """Select layers to fit within memory budget."""
        # Estimate memory per layer
        layer_memories = []
        for name, module in layers:
            mem_mb = self._estimate_layer_memory(module)
            layer_memories.append(mem_mb)
            self.layer_memory_estimates[name] = mem_mb
        
        total_memory_mb = sum(layer_memories)
        budget_mb = budget_gb * 1024
        
        if total_memory_mb <= budget_mb:
            # No checkpointing needed
            return []
        
        # Checkpoint layers to reduce memory
        memory_to_save = total_memory_mb - budget_mb
        
        # Sort by memory (largest first)
        sorted_indices = sorted(
            range(len(layers)),
            key=lambda i: layer_memories[i],
            reverse=True
        )
        
        selected = []
        saved = 0
        for idx in sorted_indices:
            if saved >= memory_to_save:
                break
            selected.append(idx)
            # Checkpointing saves ~2/3 of activation memory
            saved += layer_memories[idx] * 0.67
        
        self.total_memory_saved_mb = saved
        return sorted(selected)
    
    def _estimate_layer_memory(self, module: 'nn.Module') -> float:
        """Estimate activation memory for a layer in MB."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        # Count parameters as proxy for activation size
        total_params = sum(p.numel() for p in module.parameters())
        
        # Rough estimate: activations ~ 2x parameters for transformers
        activation_elements = total_params * 2
        
        # Assume fp16/bf16 (2 bytes per element)
        memory_bytes = activation_elements * 2
        
        return memory_bytes / (1024**2)
    
    def _adaptive_selection(
        self,
        layers: List[Tuple[str, 'nn.Module']],
    ) -> List[int]:
        """Adaptively select layers based on current memory pressure."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return []
        
        # Get current memory state
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        
        utilization = reserved / total
        
        if utilization < 0.5:
            # Low memory pressure, minimal checkpointing
            return list(range(0, len(layers), 4))
        elif utilization < 0.75:
            # Moderate pressure
            return list(range(0, len(layers), 2))
        else:
            # High pressure, checkpoint all
            return list(range(len(layers)))
    
    def _align_with_sync(
        self,
        indices: List[int],
        total_layers: int,
    ) -> List[int]:
        """Align checkpoint boundaries with DES-LOC sync periods."""
        if not indices:
            return indices
        
        Kx = self.config.Kx
        
        # Assume layers map roughly to steps
        # Align to Kx boundaries
        aligned = []
        for idx in indices:
            # Round to nearest Kx boundary
            aligned_idx = (idx // Kx) * Kx
            aligned_idx = min(aligned_idx, total_layers - 1)
            if aligned_idx not in aligned:
                aligned.append(aligned_idx)
        
        return sorted(aligned)
    
    def _apply_to_layer(
        self,
        model: 'nn.Module',
        parent_name: str,
        indices: List[int],
    ):
        """Apply checkpointing to specific layer indices."""
        if not TORCH_AVAILABLE:
            return
        
        if parent_name == 'root':
            parent = model
        else:
            parent = dict(model.named_modules())[parent_name]
        
        # Enable gradient checkpointing if supported
        if hasattr(parent, 'gradient_checkpointing_enable'):
            parent.gradient_checkpointing_enable()
            return
        
        # Manual wrapping for Sequential
        if isinstance(parent, nn.Sequential):
            for idx in indices:
                if idx < len(parent):
                    original = parent[idx]
                    parent[idx] = CheckpointedModule(original, self.config)
                    self.checkpointed_modules.append(weakref.ref(parent[idx]))


# =============================================================================
# PART 4: SELECTIVE RECOMPUTATION
# =============================================================================

class SelectiveRecomputation:
    """Enables selective recomputation of specific operations."""
    
    def __init__(self):
        self.recompute_ops: Set[str] = set()
        self.preserve_ops: Set[str] = set()
    
    def mark_for_recomputation(self, op_name: str):
        """Mark an operation for recomputation."""
        self.recompute_ops.add(op_name)
        self.preserve_ops.discard(op_name)
    
    def mark_for_preservation(self, op_name: str):
        """Mark an operation to preserve (not recompute)."""
        self.preserve_ops.add(op_name)
        self.recompute_ops.discard(op_name)
    
    def should_recompute(self, op_name: str) -> bool:
        """Check if operation should be recomputed."""
        return op_name in self.recompute_ops


class RecomputeFunction(torch.autograd.Function if TORCH_AVAILABLE else object):
    """Custom autograd function for selective recomputation."""
    
    @staticmethod
    def forward(ctx, func, preserve_rng_state, *args):
        if not TORCH_AVAILABLE:
            return func(*args)
        
        ctx.func = func
        ctx.preserve_rng_state = preserve_rng_state
        ctx.save_for_backward(*args)
        
        if preserve_rng_state:
            ctx.rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.cuda_rng_state = torch.cuda.get_rng_state()
        
        with torch.no_grad():
            outputs = func(*args)
        
        return outputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        if not TORCH_AVAILABLE:
            return (None, None) + grad_outputs
        
        args = ctx.saved_tensors
        
        # Restore RNG state if needed
        if ctx.preserve_rng_state:
            torch.set_rng_state(ctx.rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(ctx.cuda_rng_state)
        
        # Recompute forward
        with torch.enable_grad():
            detached_args = [a.detach().requires_grad_(a.requires_grad) for a in args]
            outputs = ctx.func(*detached_args)
        
        # Compute gradients
        if isinstance(outputs, tuple):
            torch.autograd.backward(outputs, grad_outputs)
        else:
            torch.autograd.backward(outputs, grad_outputs[0])
        
        grads = [a.grad if a.requires_grad else None for a in detached_args]
        
        return (None, None) + tuple(grads)


def recompute(func: Callable, *args, preserve_rng_state: bool = True):
    """Apply selective recomputation to a function."""
    if not TORCH_AVAILABLE:
        return func(*args)
    
    return RecomputeFunction.apply(func, preserve_rng_state, *args)


# =============================================================================
# PART 5: DES-LOC CHECKPOINTING INTEGRATION
# =============================================================================

class DESLOCCheckpointingIntegration:
    """Integrates gradient checkpointing with DES-LOC optimizer."""
    
    def __init__(
        self,
        config: CheckpointConfig = None,
        Kx: int = 32,
        Ku: int = 64,
        Kv: int = 128,
    ):
        self.config = config or CheckpointConfig()
        self.config.Kx = Kx
        
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        
        self.manager = CheckpointingManager(self.config)
        self.current_step = 0
        self.checkpoint_boundaries: List[int] = []
    
    def setup(self, model: 'nn.Module') -> 'nn.Module':
        """Setup checkpointing for model."""
        # Calculate optimal checkpoint boundaries
        self._calculate_boundaries(model)
        
        # Apply checkpointing
        return self.manager.apply_checkpointing(model)
    
    def _calculate_boundaries(self, model: 'nn.Module'):
        """Calculate checkpoint boundaries aligned with sync schedule."""
        if not TORCH_AVAILABLE:
            return
        
        # Count layers
        num_layers = sum(1 for _ in model.modules())
        
        # Create boundaries at sync points
        boundaries = []
        
        # Kx boundaries (most frequent)
        for i in range(0, num_layers, self.Kx):
            boundaries.append(('Kx', i))
        
        # Ku boundaries
        for i in range(0, num_layers, self.Ku):
            boundaries.append(('Ku', i))
        
        # Kv boundaries
        for i in range(0, num_layers, self.Kv):
            boundaries.append(('Kv', i))
        
        self.checkpoint_boundaries = sorted(set(b[1] for b in boundaries))
    
    def step(self, step: int):
        """Update step and potentially adjust checkpointing."""
        self.current_step = step
        
        # Adaptive adjustment based on sync schedule
        is_sync_step = (
            step % self.Kx == 0 or
            step % self.Ku == 0 or
            step % self.Kv == 0
        )
        
        if is_sync_step:
            # At sync boundaries, we might adjust checkpointing
            self._adjust_for_sync()
    
    def _adjust_for_sync(self):
        """Adjust checkpointing behavior at sync boundaries."""
        # During sync, we might want different memory behavior
        # This is a hook for future optimization
        pass
    
    def get_memory_savings_report(self) -> Dict[str, Any]:
        """Get report on memory savings from checkpointing."""
        return {
            'policy': self.config.policy.name,
            'total_memory_saved_mb': self.manager.total_memory_saved_mb,
            'layer_estimates': self.manager.layer_memory_estimates,
            'checkpoint_boundaries': self.checkpoint_boundaries,
            'sync_periods': {
                'Kx': self.Kx,
                'Ku': self.Ku,
                'Kv': self.Kv,
            },
        }


# =============================================================================
# PART 6: UTILITIES
# =============================================================================

def enable_checkpointing_for_transformers(model, policy: str = 'sqrt'):
    """Convenience function to enable checkpointing for transformer models."""
    config = CheckpointConfig(
        policy=CheckpointPolicy[policy.upper()],
    )
    
    manager = CheckpointingManager(config)
    return manager.apply_checkpointing(model)


def estimate_checkpointing_savings(
    model,
    batch_size: int = 32,
    seq_length: int = 512,
) -> Dict[str, float]:
    """Estimate memory savings from checkpointing."""
    if not TORCH_AVAILABLE:
        return {'error': 'PyTorch not available'}
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate activation memory
    # For transformers: activations ~ batch_size * seq_length * hidden_dim * num_layers
    # Rough estimate: 2-4x parameter memory per batch
    activation_factor = 3.0
    activation_memory_mb = (total_params * 4 * activation_factor) / (1024**2)
    
    # Checkpointing typically saves ~67% of activation memory
    # at cost of ~33% more compute
    savings_mb = activation_memory_mb * 0.67
    
    return {
        'total_params': total_params,
        'estimated_activation_memory_mb': activation_memory_mb,
        'estimated_savings_mb': savings_mb,
        'compute_overhead_pct': 33.0,
    }


# =============================================================================
# PART 7: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate gradient checkpointing capabilities."""
    print("=" * 70)
    print("DES-LOC Gradient Checkpointing Demo")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping demo")
        return
    
    # Create a simple model
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_size, 8)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
        
        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.mlp(x))
            return x
    
    class SimpleModel(nn.Module):
        def __init__(self, num_layers=12, hidden_size=768):
            super().__init__()
            self.layers = nn.Sequential(*[
                SimpleTransformerBlock(hidden_size) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel(num_layers=12)
    
    # Setup DES-LOC checkpointing integration
    integration = DESLOCCheckpointingIntegration(
        config=CheckpointConfig(policy=CheckpointPolicy.SQRT),
        Kx=32, Ku=64, Kv=128,
    )
    
    model = integration.setup(model)
    
    # Get report
    report = integration.get_memory_savings_report()
    print("\nMemory Savings Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Estimate savings
    savings = estimate_checkpointing_savings(model, batch_size=32)
    print("\nEstimated Savings:")
    print(json.dumps(savings, indent=2))
    
    print("\n[M046] Gradient Checkpointing Demo - COMPLETED")


if __name__ == "__main__":
    demo()
