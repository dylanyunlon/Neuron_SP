#!/usr/bin/env python3
"""
===============================================================================
M044: Distributed Debugging Tools Module
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module provides tools for debugging distributed training issues including
gradient divergence detection, communication tracing, and state inspection.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Features:
- Gradient divergence detection across workers
- Communication operation tracing
- Distributed state inspection and comparison
- Deadlock detection
- Worker synchronization barriers with diagnostics
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M044"

import os
import sys
import json
import time
import threading
import functools
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    List, Dict, Optional, Any, Callable, TypeVar, Union,
    Tuple, Set
)
from datetime import datetime, timedelta
from enum import Enum, auto
from contextlib import contextmanager
from collections import defaultdict, deque
import logging
import hashlib
import pickle
import socket

# Optional imports with fallbacks
try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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


# =============================================================================
# PART 1: GRADIENT DIVERGENCE DETECTOR
# =============================================================================

@dataclass
class GradientStats:
    """Statistics for a gradient tensor."""
    name: str
    shape: Tuple[int, ...]
    norm: float
    mean: float
    std: float
    min_val: float
    max_val: float
    num_zeros: int
    num_nans: int
    num_infs: int
    hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DivergenceReport:
    """Report of gradient divergence between workers."""
    step: int
    timestamp: str
    rank: int
    divergent_params: List[Dict[str, Any]]
    max_divergence: float
    mean_divergence: float
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GradientDivergenceDetector:
    """Detects gradient divergence across distributed workers."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        history_size: int = 100,
        check_frequency: int = 10,
    ):
        self.threshold = threshold
        self.history_size = history_size
        self.check_frequency = check_frequency
        
        self.history: deque = deque(maxlen=history_size)
        self.step_count = 0
        self.divergence_count = 0
    
    def compute_stats(self, name: str, tensor) -> GradientStats:
        """Compute statistics for a gradient tensor."""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().float().numpy()
        elif NUMPY_AVAILABLE:
            tensor_np = np.asarray(tensor)
        else:
            # Fallback for testing
            return GradientStats(
                name=name, shape=(0,), norm=0.0, mean=0.0, std=0.0,
                min_val=0.0, max_val=0.0, num_zeros=0, num_nans=0,
                num_infs=0, hash=""
            )
        
        return GradientStats(
            name=name,
            shape=tensor_np.shape,
            norm=float(np.linalg.norm(tensor_np)),
            mean=float(np.mean(tensor_np)),
            std=float(np.std(tensor_np)),
            min_val=float(np.min(tensor_np)),
            max_val=float(np.max(tensor_np)),
            num_zeros=int(np.sum(tensor_np == 0)),
            num_nans=int(np.sum(np.isnan(tensor_np))),
            num_infs=int(np.sum(np.isinf(tensor_np))),
            hash=hashlib.md5(tensor_np.tobytes()).hexdigest()[:16],
        )
    
    def check_divergence(
        self,
        model,
        step: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> Optional[DivergenceReport]:
        """Check for gradient divergence across workers."""
        self.step_count = step
        
        if step % self.check_frequency != 0:
            return None
        
        # Collect local gradient stats
        local_stats = []
        if TORCH_AVAILABLE and hasattr(model, 'named_parameters'):
            for name, param in model.named_parameters():
                if param.grad is not None:
                    stats = self.compute_stats(name, param.grad)
                    local_stats.append(stats)
        
        if not local_stats:
            return None
        
        # In distributed setting, gather stats from all workers
        divergent_params = []
        max_divergence = 0.0
        
        if TORCH_AVAILABLE and dist.is_initialized() and world_size > 1:
            # Serialize and gather stats
            local_data = pickle.dumps(local_stats)
            gathered_data = [None] * world_size
            
            try:
                dist.all_gather_object(gathered_data, local_data)
                
                # Deserialize
                all_stats = [pickle.loads(d) for d in gathered_data]
                
                # Compare stats across workers
                for param_idx, local_stat in enumerate(local_stats):
                    norms = [all_stats[r][param_idx].norm for r in range(world_size)]
                    norm_std = np.std(norms)
                    norm_mean = np.mean(norms)
                    
                    if norm_mean > 0:
                        relative_divergence = norm_std / norm_mean
                        
                        if relative_divergence > self.threshold:
                            divergent_params.append({
                                'name': local_stat.name,
                                'norms': norms,
                                'divergence': relative_divergence,
                            })
                            max_divergence = max(max_divergence, relative_divergence)
            except Exception as e:
                logger.warning(f"Failed to gather gradient stats: {e}")
        
        if divergent_params:
            self.divergence_count += 1
            
            report = DivergenceReport(
                step=step,
                timestamp=datetime.now().isoformat(),
                rank=rank,
                divergent_params=divergent_params,
                max_divergence=max_divergence,
                mean_divergence=np.mean([p['divergence'] for p in divergent_params]),
                recommendation=self._get_recommendation(divergent_params),
            )
            
            self.history.append(report)
            return report
        
        return None
    
    def _get_recommendation(self, divergent_params: List[Dict]) -> str:
        """Generate recommendation based on divergence pattern."""
        if len(divergent_params) > 10:
            return "Widespread divergence detected. Check learning rate and batch size."
        elif any('norm' in p['name'].lower() for p in divergent_params):
            return "Normalization layer divergence. Consider syncing running stats."
        elif self.divergence_count > 5:
            return "Persistent divergence. Consider reducing sync period (Kx, Ku, Kv)."
        else:
            return "Minor divergence. Monitor for persistence."


# =============================================================================
# PART 2: COMMUNICATION TRACER
# =============================================================================

class CommunicationType(Enum):
    """Types of communication operations."""
    ALL_REDUCE = auto()
    ALL_GATHER = auto()
    BROADCAST = auto()
    REDUCE_SCATTER = auto()
    POINT_TO_POINT = auto()
    BARRIER = auto()


@dataclass
class CommOperation:
    """Record of a communication operation."""
    op_type: CommunicationType
    tensor_name: str
    tensor_size_bytes: int
    start_time: float
    end_time: float
    rank: int
    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'op_type': self.op_type.name,
            'duration_ms': self.duration_ms,
        }


class CommunicationTracer:
    """Traces distributed communication operations."""
    
    def __init__(self, max_history: int = 10000):
        self.operations: deque = deque(maxlen=max_history)
        self.active_ops: Dict[str, float] = {}  # op_id -> start_time
        self._lock = threading.Lock()
        
        # Statistics
        self.total_bytes = 0
        self.total_time_ms = 0.0
        self.op_counts: Dict[CommunicationType, int] = defaultdict(int)
    
    @contextmanager
    def trace(
        self,
        op_type: CommunicationType,
        tensor_name: str,
        tensor_size_bytes: int,
        rank: int = 0,
    ):
        """Context manager to trace a communication operation."""
        op_id = f"{op_type.name}_{tensor_name}_{time.time()}"
        start_time = time.perf_counter()
        error_msg = None
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            
            op = CommOperation(
                op_type=op_type,
                tensor_name=tensor_name,
                tensor_size_bytes=tensor_size_bytes,
                start_time=start_time,
                end_time=end_time,
                rank=rank,
                success=success,
                error_message=error_msg,
            )
            
            with self._lock:
                self.operations.append(op)
                self.total_bytes += tensor_size_bytes
                self.total_time_ms += op.duration_ms
                self.op_counts[op_type] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        with self._lock:
            if not self.operations:
                return {'total_operations': 0}
            
            durations = [op.duration_ms for op in self.operations]
            
            return {
                'total_operations': len(self.operations),
                'total_bytes_gb': self.total_bytes / (1024**3),
                'total_time_ms': self.total_time_ms,
                'avg_duration_ms': np.mean(durations) if durations else 0,
                'max_duration_ms': max(durations) if durations else 0,
                'min_duration_ms': min(durations) if durations else 0,
                'by_type': {k.name: v for k, v in self.op_counts.items()},
                'failed_ops': sum(1 for op in self.operations if not op.success),
            }
    
    def get_slowest_operations(self, n: int = 10) -> List[Dict]:
        """Get the n slowest operations."""
        with self._lock:
            sorted_ops = sorted(
                self.operations,
                key=lambda x: x.duration_ms,
                reverse=True
            )
            return [op.to_dict() for op in sorted_ops[:n]]
    
    def save_trace(self, path: Path):
        """Save communication trace to file."""
        with self._lock:
            trace_data = {
                'statistics': self.get_statistics(),
                'operations': [op.to_dict() for op in self.operations],
                'generated_at': datetime.now().isoformat(),
            }
            
            with open(path, 'w') as f:
                json.dump(trace_data, f, indent=2)


# =============================================================================
# PART 3: STATE INSPECTOR
# =============================================================================

@dataclass
class TensorSnapshot:
    """Snapshot of a tensor's state."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    norm: float
    mean: float
    hash: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DistributedStateInspector:
    """Inspects and compares state across distributed workers."""
    
    def __init__(self):
        self.snapshots: Dict[int, Dict[str, TensorSnapshot]] = {}  # step -> {name: snapshot}
    
    def snapshot_model(
        self,
        model,
        step: int,
        include_grads: bool = True,
    ) -> Dict[str, TensorSnapshot]:
        """Take a snapshot of model state."""
        snapshots = {}
        
        if not TORCH_AVAILABLE:
            return snapshots
        
        for name, param in model.named_parameters():
            # Parameter snapshot
            snapshots[f"param.{name}"] = self._tensor_snapshot(
                f"param.{name}", param.data
            )
            
            # Gradient snapshot
            if include_grads and param.grad is not None:
                snapshots[f"grad.{name}"] = self._tensor_snapshot(
                    f"grad.{name}", param.grad
                )
        
        self.snapshots[step] = snapshots
        return snapshots
    
    def _tensor_snapshot(self, name: str, tensor) -> TensorSnapshot:
        """Create snapshot of a single tensor."""
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().float().numpy()
            return TensorSnapshot(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                device=str(tensor.device),
                norm=float(torch.norm(tensor).item()),
                mean=float(torch.mean(tensor.float()).item()),
                hash=hashlib.md5(tensor_np.tobytes()).hexdigest()[:16],
            )
        else:
            return TensorSnapshot(
                name=name, shape=(0,), dtype="unknown", device="cpu",
                norm=0.0, mean=0.0, hash=""
            )
    
    def compare_snapshots(
        self,
        step1: int,
        step2: int,
    ) -> Dict[str, Any]:
        """Compare snapshots between two steps."""
        if step1 not in self.snapshots or step2 not in self.snapshots:
            return {'error': 'Snapshot not found'}
        
        snap1 = self.snapshots[step1]
        snap2 = self.snapshots[step2]
        
        changes = []
        for name in snap1:
            if name in snap2:
                s1, s2 = snap1[name], snap2[name]
                
                if s1.hash != s2.hash:
                    norm_change = abs(s2.norm - s1.norm) / (s1.norm + 1e-8)
                    mean_change = abs(s2.mean - s1.mean)
                    
                    changes.append({
                        'name': name,
                        'norm_change_pct': norm_change * 100,
                        'mean_change': mean_change,
                        'norm_before': s1.norm,
                        'norm_after': s2.norm,
                    })
        
        return {
            'step1': step1,
            'step2': step2,
            'total_params': len(snap1),
            'changed_params': len(changes),
            'changes': sorted(changes, key=lambda x: x['norm_change_pct'], reverse=True),
        }
    
    def compare_across_ranks(
        self,
        local_snapshot: Dict[str, TensorSnapshot],
        rank: int,
        world_size: int,
    ) -> Dict[str, Any]:
        """Compare local snapshot with other ranks."""
        if not TORCH_AVAILABLE or not dist.is_initialized():
            return {'error': 'Distributed not initialized'}
        
        # Serialize local snapshot
        local_data = {
            name: {
                'norm': snap.norm,
                'mean': snap.mean,
                'hash': snap.hash,
            }
            for name, snap in local_snapshot.items()
        }
        
        # Gather from all ranks
        gathered = [None] * world_size
        try:
            dist.all_gather_object(gathered, local_data)
        except Exception as e:
            return {'error': str(e)}
        
        # Compare
        mismatches = []
        for name in local_snapshot:
            hashes = [gathered[r].get(name, {}).get('hash', '') for r in range(world_size)]
            unique_hashes = set(hashes)
            
            if len(unique_hashes) > 1:
                norms = [gathered[r].get(name, {}).get('norm', 0) for r in range(world_size)]
                mismatches.append({
                    'name': name,
                    'unique_hashes': len(unique_hashes),
                    'norms_by_rank': norms,
                    'norm_std': np.std(norms),
                })
        
        return {
            'rank': rank,
            'world_size': world_size,
            'total_tensors': len(local_snapshot),
            'mismatched_tensors': len(mismatches),
            'mismatches': mismatches,
        }


# =============================================================================
# PART 4: DEADLOCK DETECTOR
# =============================================================================

class DeadlockDetector:
    """Detects potential deadlocks in distributed training."""
    
    def __init__(
        self,
        timeout_seconds: float = 300.0,
        check_interval: float = 10.0,
    ):
        self.timeout_seconds = timeout_seconds
        self.check_interval = check_interval
        
        self.last_progress: Dict[int, float] = {}  # rank -> timestamp
        self.operation_stack: Dict[int, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
    
    def record_progress(self, rank: int, operation: str = "step"):
        """Record progress for a rank."""
        with self._lock:
            self.last_progress[rank] = time.time()
            if operation:
                # Keep recent operations for debugging
                ops = self.operation_stack[rank]
                ops.append(f"{datetime.now().isoformat()}: {operation}")
                if len(ops) > 100:
                    ops.pop(0)
    
    def enter_operation(self, rank: int, operation: str):
        """Record entering a blocking operation."""
        with self._lock:
            self.operation_stack[rank].append(f"ENTER: {operation} at {time.time()}")
    
    def exit_operation(self, rank: int, operation: str):
        """Record exiting a blocking operation."""
        with self._lock:
            self.operation_stack[rank].append(f"EXIT: {operation} at {time.time()}")
    
    def check_deadlock(self) -> Optional[Dict[str, Any]]:
        """Check for potential deadlock."""
        with self._lock:
            current_time = time.time()
            stalled_ranks = []
            
            for rank, last_time in self.last_progress.items():
                if current_time - last_time > self.timeout_seconds:
                    stalled_ranks.append({
                        'rank': rank,
                        'stalled_seconds': current_time - last_time,
                        'recent_operations': self.operation_stack[rank][-10:],
                    })
            
            if stalled_ranks:
                return {
                    'detected_at': datetime.now().isoformat(),
                    'timeout_threshold': self.timeout_seconds,
                    'stalled_ranks': stalled_ranks,
                    'total_ranks': len(self.last_progress),
                    'diagnosis': self._diagnose(stalled_ranks),
                }
            
            return None
    
    def _diagnose(self, stalled_ranks: List[Dict]) -> str:
        """Diagnose potential cause of deadlock."""
        if len(stalled_ranks) == len(self.last_progress):
            return "All ranks stalled. Likely collective operation deadlock."
        elif len(stalled_ranks) == 1:
            return "Single rank stalled. Possible data loading or hardware issue."
        else:
            return f"{len(stalled_ranks)} ranks stalled. Possible synchronization mismatch."
    
    def start_monitoring(self, callback: Optional[Callable] = None):
        """Start background deadlock monitoring."""
        def monitor():
            while not self._stop_flag.is_set():
                result = self.check_deadlock()
                if result:
                    logger.critical(f"Potential deadlock detected: {result}")
                    if callback:
                        callback(result)
                time.sleep(self.check_interval)
        
        self._stop_flag.clear()
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_flag.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)


# =============================================================================
# PART 5: DIAGNOSTIC BARRIER
# =============================================================================

class DiagnosticBarrier:
    """Synchronization barrier with diagnostics."""
    
    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout_seconds = timeout_seconds
        self.barrier_times: List[Dict] = []
    
    def barrier(
        self,
        name: str = "barrier",
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, Any]:
        """Execute barrier with timing diagnostics."""
        start_time = time.time()
        local_ready_time = start_time
        
        result = {
            'name': name,
            'rank': rank,
            'world_size': world_size,
            'local_ready_time': local_ready_time,
            'success': True,
            'error': None,
        }
        
        if TORCH_AVAILABLE and dist.is_initialized() and world_size > 1:
            try:
                # Record when this rank is ready
                ready_times = [torch.tensor([local_ready_time])]
                gathered = [torch.zeros(1) for _ in range(world_size)]
                
                dist.all_gather(gathered, ready_times[0])
                
                # Actual barrier
                dist.barrier()
                
                end_time = time.time()
                
                result.update({
                    'ready_times': [t.item() for t in gathered],
                    'barrier_start': start_time,
                    'barrier_end': end_time,
                    'total_wait_ms': (end_time - start_time) * 1000,
                    'straggler_rank': int(np.argmax([t.item() for t in gathered])),
                })
                
            except Exception as e:
                result['success'] = False
                result['error'] = str(e)
        else:
            result['total_wait_ms'] = 0
        
        self.barrier_times.append(result)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get barrier statistics."""
        if not self.barrier_times:
            return {'total_barriers': 0}
        
        wait_times = [b['total_wait_ms'] for b in self.barrier_times if b['success']]
        
        return {
            'total_barriers': len(self.barrier_times),
            'failed_barriers': sum(1 for b in self.barrier_times if not b['success']),
            'avg_wait_ms': np.mean(wait_times) if wait_times else 0,
            'max_wait_ms': max(wait_times) if wait_times else 0,
            'total_wait_ms': sum(wait_times),
        }


# =============================================================================
# PART 6: DEBUG TOOLKIT
# =============================================================================

class DistributedDebugToolkit:
    """Unified toolkit for distributed debugging."""
    
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        output_dir: Optional[Path] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.output_dir = Path(output_dir) if output_dir else Path("./debug_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.gradient_detector = GradientDivergenceDetector()
        self.comm_tracer = CommunicationTracer()
        self.state_inspector = DistributedStateInspector()
        self.deadlock_detector = DeadlockDetector()
        self.barrier = DiagnosticBarrier()
    
    def record_step(self, step: int):
        """Record progress for deadlock detection."""
        self.deadlock_detector.record_progress(self.rank, f"step_{step}")
    
    def check_gradients(self, model, step: int) -> Optional[DivergenceReport]:
        """Check for gradient divergence."""
        return self.gradient_detector.check_divergence(
            model, step, self.rank, self.world_size
        )
    
    @contextmanager
    def trace_comm(self, op_type: CommunicationType, tensor_name: str, size_bytes: int):
        """Trace a communication operation."""
        with self.comm_tracer.trace(op_type, tensor_name, size_bytes, self.rank):
            yield
    
    def snapshot(self, model, step: int):
        """Take a model snapshot."""
        return self.state_inspector.snapshot_model(model, step)
    
    def sync_barrier(self, name: str = "barrier"):
        """Execute diagnostic barrier."""
        return self.barrier.barrier(name, self.rank, self.world_size)
    
    def save_diagnostics(self, prefix: str = "debug"):
        """Save all diagnostic data."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save communication trace
        self.comm_tracer.save_trace(
            self.output_dir / f"{prefix}_comm_trace_rank{self.rank}_{timestamp}.json"
        )
        
        # Save comprehensive report
        report = {
            'rank': self.rank,
            'world_size': self.world_size,
            'timestamp': timestamp,
            'communication': self.comm_tracer.get_statistics(),
            'barriers': self.barrier.get_statistics(),
            'gradient_divergences': len(self.gradient_detector.history),
        }
        
        with open(self.output_dir / f"{prefix}_report_rank{self.rank}_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Diagnostics saved to {self.output_dir}")


# =============================================================================
# PART 7: MAIN / DEMO
# =============================================================================

def demo():
    """Demonstrate debugging tools."""
    print("=" * 70)
    print("DES-LOC Distributed Debugging Tools Demo")
    print("=" * 70)
    
    toolkit = DistributedDebugToolkit(rank=0, world_size=1)
    
    # Simulate some operations
    for step in range(10):
        toolkit.record_step(step)
        
        with toolkit.trace_comm(
            CommunicationType.ALL_REDUCE,
            f"gradient_{step}",
            1024 * 1024,  # 1MB
        ):
            time.sleep(0.01)  # Simulate communication
    
    # Get statistics
    print("\nCommunication Statistics:")
    print(json.dumps(toolkit.comm_tracer.get_statistics(), indent=2))
    
    # Save diagnostics
    toolkit.save_diagnostics("demo")
    
    print("\n[M044] Distributed Debugging Demo - COMPLETED")


if __name__ == "__main__":
    demo()
