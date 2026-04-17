#!/usr/bin/env python3
"""
===============================================================================
M045: Memory Profiling Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides comprehensive memory profiling for GPU and CPU memory
usage during distributed training, with optimization recommendations.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- GPU memory tracking and visualization
- CPU memory profiling
- Peak memory detection
- Memory leak detection
- Optimization recommendations
- Integration with gradient checkpointing
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M045"

import os
import sys
import json
import time
import gc
import threading
import functools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Set
)
from datetime import datetime
from enum import Enum, auto
from contextlib import contextmanager
from collections import defaultdict, deque
import logging
import traceback

# Optional imports
try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    cuda = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================

@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: str
    step: int
    
    # GPU memory (bytes)
    gpu_allocated: int = 0
    gpu_reserved: int = 0
    gpu_max_allocated: int = 0
    gpu_max_reserved: int = 0
    
    # CPU memory (bytes)
    cpu_used: int = 0
    cpu_available: int = 0
    cpu_total: int = 0
    cpu_percent: float = 0.0
    
    # Additional info
    device: str = "cuda:0"
    operation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert to MB for readability
        for key in ['gpu_allocated', 'gpu_reserved', 'gpu_max_allocated', 
                    'gpu_max_reserved', 'cpu_used', 'cpu_available', 'cpu_total']:
            d[f'{key}_mb'] = d[key] / (1024**2)
        return d


@dataclass
class TensorMemoryInfo:
    """Memory information for a tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    device: str
    requires_grad: bool
    is_leaf: bool
    grad_fn: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'size_mb': self.size_bytes / (1024**2),
        }


@dataclass
class MemoryAllocation:
    """Record of a memory allocation event."""
    timestamp: float
    size_bytes: int
    operation: str
    stack_trace: str
    device: str
    is_allocation: bool  # True = allocation, False = deallocation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'size_mb': self.size_bytes / (1024**2),
        }


# =============================================================================
# PART 2: GPU MEMORY PROFILER
# =============================================================================

class GPUMemoryProfiler:
    """Profiles GPU memory usage."""
    
    def __init__(
        self,
        device: int = 0,
        history_size: int = 10000,
        track_allocations: bool = False,
    ):
        self.device = device
        self.history_size = history_size
        self.track_allocations = track_allocations
        
        self.snapshots: deque = deque(maxlen=history_size)
        self.allocations: deque = deque(maxlen=history_size)
        self.peak_allocated = 0
        self.peak_reserved = 0
        self._lock = threading.Lock()
        
        # Initialize CUDA if available
        if TORCH_AVAILABLE and cuda.is_available():
            cuda.set_device(device)
    
    def get_current_memory(self) -> Dict[str, int]:
        """Get current GPU memory usage."""
        if not TORCH_AVAILABLE or not cuda.is_available():
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'max_reserved': 0,
            }
        
        return {
            'allocated': cuda.memory_allocated(self.device),
            'reserved': cuda.memory_reserved(self.device),
            'max_allocated': cuda.max_memory_allocated(self.device),
            'max_reserved': cuda.max_memory_reserved(self.device),
        }
    
    def snapshot(self, step: int = 0, operation: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        gpu_mem = self.get_current_memory()
        cpu_mem = self._get_cpu_memory()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            step=step,
            gpu_allocated=gpu_mem['allocated'],
            gpu_reserved=gpu_mem['reserved'],
            gpu_max_allocated=gpu_mem['max_allocated'],
            gpu_max_reserved=gpu_mem['max_reserved'],
            cpu_used=cpu_mem['used'],
            cpu_available=cpu_mem['available'],
            cpu_total=cpu_mem['total'],
            cpu_percent=cpu_mem['percent'],
            device=f"cuda:{self.device}",
            operation=operation,
        )
        
        with self._lock:
            self.snapshots.append(snapshot)
            self.peak_allocated = max(self.peak_allocated, gpu_mem['allocated'])
            self.peak_reserved = max(self.peak_reserved, gpu_mem['reserved'])
        
        return snapshot
    
    def _get_cpu_memory(self) -> Dict[str, Any]:
        """Get CPU memory usage."""
        if not PSUTIL_AVAILABLE:
            return {'used': 0, 'available': 0, 'total': 0, 'percent': 0.0}
        
        mem = psutil.virtual_memory()
        return {
            'used': mem.used,
            'available': mem.available,
            'total': mem.total,
            'percent': mem.percent,
        }
    
    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if TORCH_AVAILABLE and cuda.is_available():
            cuda.reset_peak_memory_stats(self.device)
        self.peak_allocated = 0
        self.peak_reserved = 0
    
    def get_tensor_memory_breakdown(self) -> List[TensorMemoryInfo]:
        """Get memory breakdown by tensor."""
        if not TORCH_AVAILABLE:
            return []
        
        tensors = []
        
        # Iterate through all objects in memory
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensors.append(TensorMemoryInfo(
                        name=f"tensor_{id(obj)}",
                        shape=tuple(obj.shape),
                        dtype=str(obj.dtype),
                        size_bytes=obj.element_size() * obj.nelement(),
                        device=str(obj.device),
                        requires_grad=obj.requires_grad,
                        is_leaf=obj.is_leaf,
                        grad_fn=str(obj.grad_fn) if obj.grad_fn else None,
                    ))
            except Exception:
                pass
        
        # Sort by size
        tensors.sort(key=lambda x: x.size_bytes, reverse=True)
        return tensors
    
    def get_largest_tensors(self, n: int = 10) -> List[TensorMemoryInfo]:
        """Get the n largest tensors in memory."""
        return self.get_tensor_memory_breakdown()[:n]
    
    @contextmanager
    def track(self, operation: str, step: int = 0):
        """Context manager to track memory during an operation."""
        # Snapshot before
        before = self.snapshot(step, f"{operation}_start")
        
        try:
            yield
        finally:
            # Snapshot after
            after = self.snapshot(step, f"{operation}_end")
            
            # Log memory change
            delta = after.gpu_allocated - before.gpu_allocated
            if abs(delta) > 1024 * 1024:  # > 1MB change
                logger.debug(
                    f"[{operation}] Memory delta: {delta / (1024**2):.2f} MB "
                    f"(now: {after.gpu_allocated / (1024**2):.2f} MB)"
                )


# =============================================================================
# PART 3: MEMORY LEAK DETECTOR
# =============================================================================

class MemoryLeakDetector:
    """Detects potential memory leaks."""
    
    def __init__(
        self,
        threshold_mb: float = 100.0,
        window_steps: int = 100,
    ):
        self.threshold_mb = threshold_mb
        self.window_steps = window_steps
        
        self.memory_history: deque = deque(maxlen=window_steps * 2)
        self.baseline: Optional[int] = None
        self.warnings: List[Dict[str, Any]] = []
    
    def record(self, step: int, allocated_bytes: int):
        """Record memory usage at a step."""
        self.memory_history.append({
            'step': step,
            'allocated': allocated_bytes,
            'timestamp': time.time(),
        })
        
        if self.baseline is None and len(self.memory_history) >= 10:
            # Establish baseline after warmup
            self.baseline = sum(m['allocated'] for m in list(self.memory_history)[:10]) // 10
    
    def check_leak(self) -> Optional[Dict[str, Any]]:
        """Check for potential memory leak."""
        if len(self.memory_history) < self.window_steps:
            return None
        
        if self.baseline is None:
            return None
        
        recent = list(self.memory_history)[-self.window_steps:]
        
        # Check for monotonic increase
        increasing_count = 0
        for i in range(1, len(recent)):
            if recent[i]['allocated'] > recent[i-1]['allocated']:
                increasing_count += 1
        
        increase_ratio = increasing_count / (len(recent) - 1)
        
        # Check total increase
        start_mem = recent[0]['allocated']
        end_mem = recent[-1]['allocated']
        total_increase_mb = (end_mem - start_mem) / (1024**2)
        
        # Check against baseline
        baseline_increase_mb = (end_mem - self.baseline) / (1024**2)
        
        if total_increase_mb > self.threshold_mb and increase_ratio > 0.7:
            warning = {
                'detected_at_step': recent[-1]['step'],
                'timestamp': datetime.now().isoformat(),
                'increase_mb': total_increase_mb,
                'baseline_increase_mb': baseline_increase_mb,
                'increase_ratio': increase_ratio,
                'severity': 'high' if total_increase_mb > self.threshold_mb * 2 else 'medium',
                'recommendation': self._get_recommendation(total_increase_mb, increase_ratio),
            }
            
            self.warnings.append(warning)
            return warning
        
        return None
    
    def _get_recommendation(self, increase_mb: float, ratio: float) -> str:
        """Generate recommendation for potential leak."""
        if ratio > 0.9:
            return "Strong monotonic increase. Check for tensor accumulation in lists/dicts."
        elif increase_mb > self.threshold_mb * 5:
            return "Large memory growth. Consider gradient checkpointing or smaller batch."
        else:
            return "Monitor for continued growth. May indicate gradual leak."


# =============================================================================
# PART 4: MODEL MEMORY ESTIMATOR
# =============================================================================

class ModelMemoryEstimator:
    """Estimates memory requirements for a model."""
    
    # Memory overhead factors
    OPTIMIZER_MEMORY_FACTOR = {
        'sgd': 1.0,
        'adam': 2.0,     # First and second moments
        'adamw': 2.0,
        'adafactor': 0.5,  # Factored second moment
        'muon': 1.0,     # Newton-Schulz is transient
    }
    
    GRADIENT_CHECKPOINTING_FACTOR = 0.3  # ~70% reduction in activations
    MIXED_PRECISION_FACTOR = 0.5  # Half precision weights
    
    def __init__(self, model=None):
        self.model = model
    
    def estimate_model_memory(self, model=None) -> Dict[str, Any]:
        """Estimate memory requirements for a model."""
        model = model or self.model
        if model is None:
            return {'error': 'No model provided'}
        
        if not TORCH_AVAILABLE:
            return {'error': 'PyTorch not available'}
        
        total_params = 0
        trainable_params = 0
        param_memory = 0
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            
            param_memory += param.element_size() * num_params
        
        # Gradient memory (same as trainable params)
        gradient_memory = param_memory  # Assuming same dtype
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'param_memory_mb': param_memory / (1024**2),
            'gradient_memory_mb': gradient_memory / (1024**2),
            'total_static_mb': (param_memory + gradient_memory) / (1024**2),
        }
    
    def estimate_training_memory(
        self,
        model=None,
        batch_size: int = 32,
        seq_length: int = 512,
        optimizer: str = 'adam',
        gradient_checkpointing: bool = False,
        mixed_precision: bool = True,
    ) -> Dict[str, Any]:
        """Estimate total training memory."""
        base = self.estimate_model_memory(model or self.model)
        
        if 'error' in base:
            return base
        
        param_mb = base['param_memory_mb']
        grad_mb = base['gradient_memory_mb']
        
        # Optimizer states
        opt_factor = self.OPTIMIZER_MEMORY_FACTOR.get(optimizer.lower(), 2.0)
        optimizer_mb = param_mb * opt_factor
        
        # Activation memory (rough estimate)
        # Assuming transformer: activations ~ batch_size * seq_len * hidden_dim * num_layers
        # Rough heuristic: 2-4x parameter memory
        activation_factor = 3.0
        if gradient_checkpointing:
            activation_factor *= self.GRADIENT_CHECKPOINTING_FACTOR
        activation_mb = param_mb * activation_factor * (batch_size / 32)
        
        # Mixed precision savings
        if mixed_precision:
            param_mb *= self.MIXED_PRECISION_FACTOR
            activation_mb *= self.MIXED_PRECISION_FACTOR
        
        total_mb = param_mb + grad_mb + optimizer_mb + activation_mb
        
        return {
            **base,
            'optimizer': optimizer,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'gradient_checkpointing': gradient_checkpointing,
            'mixed_precision': mixed_precision,
            'optimizer_states_mb': optimizer_mb,
            'activation_memory_mb': activation_mb,
            'estimated_total_mb': total_mb,
            'estimated_total_gb': total_mb / 1024,
            'recommended_gpu': self._recommend_gpu(total_mb),
        }
    
    def _recommend_gpu(self, required_mb: float) -> str:
        """Recommend GPU based on memory requirements."""
        gpu_options = [
            ('RTX 3090', 24 * 1024),
            ('RTX A6000', 48 * 1024),
            ('A100 40GB', 40 * 1024),
            ('A100 80GB', 80 * 1024),
            ('H100 80GB', 80 * 1024),
            ('H100 NVL 96GB', 96 * 1024),
        ]
        
        # Add 20% headroom
        required_with_headroom = required_mb * 1.2
        
        for name, memory_mb in gpu_options:
            if memory_mb >= required_with_headroom:
                return f"{name} ({memory_mb // 1024}GB)"
        
        return "Multiple GPUs required (model parallelism)"


# =============================================================================
# PART 5: MEMORY OPTIMIZER
# =============================================================================

class MemoryOptimizer:
    """Provides memory optimization recommendations."""
    
    def __init__(self, profiler: GPUMemoryProfiler):
        self.profiler = profiler
    
    def analyze_and_recommend(self) -> Dict[str, Any]:
        """Analyze memory usage and provide recommendations."""
        recommendations = []
        
        # Get current state
        current = self.profiler.get_current_memory()
        largest_tensors = self.profiler.get_largest_tensors(10)
        
        allocated_mb = current['allocated'] / (1024**2)
        reserved_mb = current['reserved'] / (1024**2)
        fragmentation = 1 - (allocated_mb / reserved_mb) if reserved_mb > 0 else 0
        
        # Check fragmentation
        if fragmentation > 0.3:
            recommendations.append({
                'type': 'fragmentation',
                'severity': 'medium',
                'message': f"High memory fragmentation ({fragmentation*100:.1f}%). "
                          "Consider torch.cuda.empty_cache() periodically.",
            })
        
        # Check for large tensors
        if largest_tensors:
            largest = largest_tensors[0]
            if largest.size_bytes > 1024**3:  # > 1GB
                recommendations.append({
                    'type': 'large_tensor',
                    'severity': 'high',
                    'message': f"Large tensor detected: {largest.size_bytes/(1024**3):.2f}GB. "
                              f"Shape: {largest.shape}, dtype: {largest.dtype}",
                })
        
        # Check for unused gradients
        grad_tensors = [t for t in largest_tensors if t.requires_grad and not t.is_leaf]
        if len(grad_tensors) > 50:
            recommendations.append({
                'type': 'gradient_accumulation',
                'severity': 'medium',
                'message': "Many intermediate gradients in memory. "
                          "Consider gradient checkpointing.",
            })
        
        # Memory utilization
        if allocated_mb > 0.9 * reserved_mb:
            recommendations.append({
                'type': 'near_capacity',
                'severity': 'warning',
                'message': "Near GPU memory capacity. Risk of OOM.",
            })
        
        return {
            'current_allocated_mb': allocated_mb,
            'current_reserved_mb': reserved_mb,
            'fragmentation_pct': fragmentation * 100,
            'largest_tensor_mb': largest_tensors[0].size_bytes / (1024**2) if largest_tensors else 0,
            'recommendations': recommendations,
            'optimization_tips': self._get_optimization_tips(recommendations),
        }
    
    def _get_optimization_tips(self, recommendations: List[Dict]) -> List[str]:
        """Generate optimization tips based on analysis."""
        tips = []
        
        severities = [r['severity'] for r in recommendations]
        types = [r['type'] for r in recommendations]
        
        if 'high' in severities or 'near_capacity' in types:
            tips.extend([
                "Enable gradient checkpointing: model.gradient_checkpointing_enable()",
                "Use mixed precision: torch.cuda.amp.autocast()",
                "Reduce batch size or sequence length",
            ])
        
        if 'fragmentation' in types:
            tips.extend([
                "Call torch.cuda.empty_cache() after large operations",
                "Use memory-efficient attention (FlashAttention)",
            ])
        
        if 'gradient_accumulation' in types:
            tips.extend([
                "Use gradient accumulation instead of large batches",
                "Enable activation checkpointing for transformer layers",
            ])
        
        return tips


# =============================================================================
# PART 6: MEMORY PROFILING CONTEXT
# =============================================================================

class MemoryProfilingContext:
    """Unified context for memory profiling."""
    
    def __init__(
        self,
        device: int = 0,
        output_dir: Optional[Path] = None,
        track_leaks: bool = True,
    ):
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("./memory_profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiler = GPUMemoryProfiler(device)
        self.leak_detector = MemoryLeakDetector()
        self.estimator = ModelMemoryEstimator()
        self.optimizer = MemoryOptimizer(self.profiler)
        self.track_leaks = track_leaks
    
    def profile_step(self, step: int, operation: str = "training_step"):
        """Profile a training step."""
        snapshot = self.profiler.snapshot(step, operation)
        
        if self.track_leaks:
            self.leak_detector.record(step, snapshot.gpu_allocated)
            leak_warning = self.leak_detector.check_leak()
            if leak_warning:
                logger.warning(f"Potential memory leak detected: {leak_warning}")
        
        return snapshot
    
    @contextmanager
    def track_operation(self, operation: str, step: int = 0):
        """Track memory during an operation."""
        with self.profiler.track(operation, step):
            yield
    
    def estimate_requirements(
        self,
        model,
        batch_size: int = 32,
        seq_length: int = 512,
        optimizer: str = 'adam',
    ) -> Dict[str, Any]:
        """Estimate memory requirements for training."""
        return self.estimator.estimate_training_memory(
            model, batch_size, seq_length, optimizer
        )
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get memory optimization recommendations."""
        return self.optimizer.analyze_and_recommend()
    
    def save_profile(self, prefix: str = "memory_profile"):
        """Save memory profile to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        profile = {
            'timestamp': timestamp,
            'device': self.device,
            'snapshots': [s.to_dict() for s in self.profiler.snapshots],
            'peak_allocated_mb': self.profiler.peak_allocated / (1024**2),
            'peak_reserved_mb': self.profiler.peak_reserved / (1024**2),
            'leak_warnings': self.leak_detector.warnings,
            'recommendations': self.get_recommendations(),
        }
        
        path = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Memory profile saved to {path}")
        return path


# =============================================================================
# PART 7: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate memory profiling capabilities."""
    print("=" * 70)
    print("DES-LOC Memory Profiling Demo")
    print("=" * 70)
    
    ctx = MemoryProfilingContext(device=0)
    
    # Simulate training steps
    for step in range(10):
        snapshot = ctx.profile_step(step, f"step_{step}")
        print(f"Step {step}: GPU allocated = {snapshot.gpu_allocated / (1024**2):.2f} MB")
    
    # Get recommendations
    print("\nRecommendations:")
    recs = ctx.get_recommendations()
    print(json.dumps(recs, indent=2))
    
    # Save profile
    ctx.save_profile("demo")
    
    print("\n[M045] Memory Profiling Demo - COMPLETED")


if __name__ == "__main__":
    demo()
