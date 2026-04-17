# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed Communication Backend.

In the future, directly use NCCL/MPI/Gloo/etc without requiring torch.distributed. Simply wrap torch.distributed for now.

# Custom DS Backends -- Direct C/Ops
 - NCCL -- [EXPERIMENTAL]
 - MPI -- [EXPERIMENTAL]
 - RCCL -- [EXPERIMENTAL]
 - GLOO -- [EXPERIMENTAL]
# DS backend wrapper for torch.distributed [DEFAULT]
 - T-NCCL -- [DEFAULT]
 - T-GLOO
 - T-MPI

# Backend is the base class
     -- NcclBackend, MpiBackend, and TorchBackend are the main subclasses. TorchBackend is the only officially supported backend for now.
"""


class Backend(object):

    def __init__(self, name='backend', rank=0, size=1):
        self.name = name
        # The world size and rank of the world process group
        self.world_group = None
        self.world_size = size
        self.world_rank = rank
        # Single process group (pg) implementation for now but keep a list for future
        self.process_groups = []
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def new_group(self):
        # create a new pg and add it to pg list
        pass

    def init_process_group(self):
        # subclasses will initialize them fully
        # - initialize a default world process group and add it to pg list
        self.initialized = True


# =====================================================================
# M060: DES-LOC Communication Backend Abstraction (400 lines)
# =====================================================================
# Provides a backend-agnostic interface for DES-LOC communication.
# Supports NCCL (GPU), Gloo (CPU), and future Trainium/NeuronCore
# backends. Each backend implements the same sync schedule but
# with hardware-specific collective operations.
#
# Reference: des_loc_reconstructed.tex Algorithm 1
# Architecture: follows CCCL thrust/system/ backend pattern
# =====================================================================

import time as _time
from collections import deque as _deque


class DESLOCBackend:
    """
    Abstract DES-LOC communication backend.

    Provides the interface for desynchronized allreduce operations
    across different hardware backends (NCCL, Gloo, custom).

    Subclasses implement:
    - allreduce_avg(): average tensors across workers
    - barrier(): synchronize all workers
    - get_bandwidth(): measure communication bandwidth
    """

    def __init__(self, world_size, rank, Kx=32, Ku=96, Kv=192,
                 clip_radius=1.0):
        self.world_size = world_size
        self.rank = rank
        self._Kx = Kx
        self._Ku = Ku
        self._Kv = Kv
        self._clip_radius = clip_radius
        self._step = 0

        # Statistics
        self._allreduce_count = 0
        self._allreduce_bytes = 0
        self._allreduce_skip_count = 0
        self._allreduce_times_ms = _deque(maxlen=1000)
        self._sync_x_count = 0
        self._sync_u_count = 0
        self._sync_v_count = 0

    def advance_step(self):
        """Advance the global step counter."""
        self._step += 1

    @property
    def step(self):
        return self._step

    def should_sync(self, state_type):
        """Determine if state_type should be synced at current step."""
        if state_type == 'x':
            return self._step % self._Kx == 0
        elif state_type == 'u':
            return self._step % self._Ku == 0
        elif state_type == 'v':
            return self._step % self._Kv == 0
        return True

    def allreduce_avg(self, tensor, group=None, async_op=False):
        """Perform allreduce with averaging. Must be overridden."""
        raise NotImplementedError

    def barrier(self, group=None):
        """Synchronize all workers. Must be overridden."""
        raise NotImplementedError

    def conditional_allreduce(self, tensor, state_type='x',
                               group=None, async_op=False):
        """Allreduce only if DES-LOC schedule permits."""
        if self.world_size <= 1:
            return False

        if not self.should_sync(state_type):
            self._allreduce_skip_count += 1
            return False

        t0 = _time.time()
        self.allreduce_avg(tensor, group=group, async_op=async_op)
        elapsed_ms = (_time.time() - t0) * 1000

        tensor_bytes = tensor.numel() * tensor.element_size()
        self._allreduce_count += 1
        self._allreduce_bytes += tensor_bytes
        self._allreduce_times_ms.append(elapsed_ms)

        if state_type == 'x':
            self._sync_x_count += 1
        elif state_type == 'u':
            self._sync_u_count += 1
        elif state_type == 'v':
            self._sync_v_count += 1

        return True

    def clip_and_sync(self, tensor, state_type='x', group=None):
        """Apply per-coordinate clipping then conditional allreduce."""
        if self._clip_radius > 0:
            tensor.clamp_(-self._clip_radius, self._clip_radius)
        return self.conditional_allreduce(tensor, state_type, group)

    def get_stats(self):
        """Return communication statistics."""
        n = max(self._step, 1)
        total_syncs = self._sync_x_count + self._sync_u_count + self._sync_v_count
        ddp_equivalent = n * 3  # DDP syncs all 3 states every step
        times = list(self._allreduce_times_ms)
        return {
            'backend': self.__class__.__name__,
            'world_size': self.world_size,
            'rank': self.rank,
            'step': self._step,
            'Kx': self._Kx,
            'Ku': self._Ku,
            'Kv': self._Kv,
            'clip_radius': self._clip_radius,
            'allreduce_count': self._allreduce_count,
            'allreduce_bytes': self._allreduce_bytes,
            'allreduce_gb': round(self._allreduce_bytes / 1e9, 4),
            'allreduce_skip_count': self._allreduce_skip_count,
            'sync_x_count': self._sync_x_count,
            'sync_u_count': self._sync_u_count,
            'sync_v_count': self._sync_v_count,
            'comm_reduction': round(ddp_equivalent / max(total_syncs, 1), 2),
            'avg_latency_ms': round(sum(times) / len(times), 3) if times else 0,
            'p99_latency_ms': round(
                sorted(times)[int(len(times) * 0.99)], 3) if times else 0,
        }


class DESLOCNCCLBackend(DESLOCBackend):
    """DES-LOC backend using NCCL via torch.distributed."""

    def __init__(self, world_size, rank, **kwargs):
        super().__init__(world_size, rank, **kwargs)
        self._name = 'desloc_nccl'

    def allreduce_avg(self, tensor, group=None, async_op=False):
        """NCCL allreduce with averaging."""
        import torch.distributed as td
        return td.all_reduce(tensor, op=td.ReduceOp.AVG,
                              group=group, async_op=async_op)

    def barrier(self, group=None):
        """NCCL barrier."""
        import torch.distributed as td
        td.barrier(group=group)


class DESLOCGlooBackend(DESLOCBackend):
    """DES-LOC backend using Gloo (CPU-based communication)."""

    def __init__(self, world_size, rank, **kwargs):
        super().__init__(world_size, rank, **kwargs)
        self._name = 'desloc_gloo'

    def allreduce_avg(self, tensor, group=None, async_op=False):
        """Gloo allreduce with manual averaging."""
        import torch.distributed as td
        td.all_reduce(tensor, op=td.ReduceOp.SUM,
                       group=group, async_op=async_op)
        tensor.div_(self.world_size)

    def barrier(self, group=None):
        import torch.distributed as td
        td.barrier(group=group)


class DESLOCSingleGPUBackend(DESLOCBackend):
    """DES-LOC backend for single GPU (no-op communication)."""

    def __init__(self, **kwargs):
        super().__init__(world_size=1, rank=0, **kwargs)
        self._name = 'desloc_single'

    def allreduce_avg(self, tensor, group=None, async_op=False):
        """No-op: single GPU, no communication needed."""
        return None

    def barrier(self, group=None):
        """No-op barrier."""
        pass

    def conditional_allreduce(self, tensor, state_type='x',
                               group=None, async_op=False):
        """No-op: always returns False (no communication)."""
        return False


def create_desloc_backend(world_size, rank, backend_type='nccl',
                           **kwargs):
    """Factory function to create the appropriate DES-LOC backend.

    Args:
        world_size: number of workers
        rank: current worker rank
        backend_type: 'nccl', 'gloo', or 'single'
        **kwargs: Kx, Ku, Kv, clip_radius

    Returns:
        DESLOCBackend instance
    """
    if world_size <= 1 or backend_type == 'single':
        return DESLOCSingleGPUBackend(**kwargs)
    elif backend_type == 'nccl':
        return DESLOCNCCLBackend(world_size, rank, **kwargs)
    elif backend_type == 'gloo':
        return DESLOCGlooBackend(world_size, rank, **kwargs)
    else:
        raise ValueError(f"Unknown DES-LOC backend: {backend_type}. "
                         f"Supported: nccl, gloo, single")


# =================================================================
# M066: Backend-Agnostic DES-LOC Collective Ops (400 lines)
# =================================================================
# Extends the backend abstraction with DES-LOC-specific collective
# operations that work across NCCL, Gloo, and single-GPU backends.
#
# Implements the full DES-LOC Algorithm 1 communication pattern:
# - Selective state synchronization (x at Kx, u at Ku, v at Kv)
# - ServerOpt integration point
# - Per-coordinate clipping before allreduce
#
# Reference: Algorithm 1 lines 9-21
# Reference: template_extraction_from_latex.txt Ⅱ=Ring-AllReduce
# =================================================================

import time as _time
import math as _math


class DESLOCCollectiveOps:
    """Backend-agnostic implementation of DES-LOC collective ops.

    Wraps any backend (NCCL/Gloo/single) and applies DES-LOC
    sync schedule on top. This is the main integration point
    between DeepSpeed's communication layer and DES-LOC.

    Algorithm 1 pseudocode mapping:
      line 14: if t mod Kj = 0 then sync s^j
      line 15: s_t^{j,m} <- UPDATE^j(E_m[s_{t-1}^j], g_t^m)
      line 18: if t mod Kx = 0 then sync x
      line 19: x <- OPT(SERVEROPT(E_m[x_t^m])...)
    """

    def __init__(self, backend, Kx=32, Ku=96, Kv=192,
                 clip_radius=1.0, server_opt=None):
        self.backend = backend
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        self.clip_radius = clip_radius
        self.server_opt = server_opt
        self.step = 0
        self.stats = {
            'x_syncs': 0, 'u_syncs': 0, 'v_syncs': 0,
            'x_bytes': 0, 'u_bytes': 0, 'v_bytes': 0,
            'x_time_ms': 0.0, 'u_time_ms': 0.0, 'v_time_ms': 0.0,
            'clip_events': 0,
        }

    def _should_sync(self, state_type):
        """Check if state_type should sync at current step."""
        if state_type == 'x':
            return self.step % self.Kx == 0
        elif state_type == 'u':
            return self.step % self.Ku == 0
        elif state_type == 'v':
            return self.step % self.Kv == 0
        return False

    def clip_gradients(self, params):
        """Per-coordinate gradient clipping (Algorithm 1 line 12).

        clip(g, ρ)_i = sign(g_i) * min(|g_i|, ρ)
        """
        for p in params:
            if p.grad is not None:
                p.grad.data.clamp_(-self.clip_radius, self.clip_radius)
                self.stats['clip_events'] += 1

    def sync_parameters(self, params, group=None):
        """Sync parameters (x) if schedule allows.

        Algorithm 1 line 18: if t mod Kx = 0 then sync x
        Algorithm 1 line 19: x <- SERVEROPT(E_m[x])

        Returns True if sync was performed.
        """
        if not self._should_sync('x'):
            return False

        start = _time.monotonic()
        total_bytes = 0

        for p in params:
            if not p.requires_grad:
                continue
            self.backend.all_reduce(p.data, group=group)
            total_bytes += p.numel() * p.element_size()

        # Apply ServerOpt if configured
        if self.server_opt is not None:
            self.server_opt.step(params)

        elapsed_ms = (_time.monotonic() - start) * 1000.0
        self.stats['x_syncs'] += 1
        self.stats['x_bytes'] += total_bytes
        self.stats['x_time_ms'] += elapsed_ms
        return True

    def sync_first_moment(self, optimizer, group=None):
        """Sync first moment (u / exp_avg) if schedule allows.

        Algorithm 1 line 14: if t mod Ku = 0 then sync s^u
        """
        if not self._should_sync('u'):
            return False

        start = _time.monotonic()
        total_bytes = 0

        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]
                for key in ('exp_avg', 'momentum_buffer'):
                    if key in state:
                        self.backend.all_reduce(
                            state[key], group=group)
                        total_bytes += (state[key].numel() *
                                        state[key].element_size())

        elapsed_ms = (_time.monotonic() - start) * 1000.0
        self.stats['u_syncs'] += 1
        self.stats['u_bytes'] += total_bytes
        self.stats['u_time_ms'] += elapsed_ms
        return True

    def sync_second_moment(self, optimizer, group=None):
        """Sync second moment (v / exp_avg_sq) if schedule allows.

        Algorithm 1 line 14: if t mod Kv = 0 then sync s^v
        """
        if not self._should_sync('v'):
            return False

        start = _time.monotonic()
        total_bytes = 0

        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    self.backend.all_reduce(
                        state['exp_avg_sq'], group=group)
                    total_bytes += (state['exp_avg_sq'].numel() *
                                    state['exp_avg_sq'].element_size())

        elapsed_ms = (_time.monotonic() - start) * 1000.0
        self.stats['v_syncs'] += 1
        self.stats['v_bytes'] += total_bytes
        self.stats['v_time_ms'] += elapsed_ms
        return True

    def full_step(self, params, optimizer, group=None):
        """Execute full DES-LOC communication step.

        Called after optimizer.step(). Performs:
        1. Gradient clipping (if not already done)
        2. Parameter sync (if t mod Kx == 0)
        3. First moment sync (if t mod Ku == 0)
        4. Second moment sync (if t mod Kv == 0)

        Returns dict describing what was synced.
        """
        result = {
            'step': self.step,
            'x_synced': False,
            'u_synced': False,
            'v_synced': False,
            'total_bytes': 0,
        }

        result['x_synced'] = self.sync_parameters(params, group)
        result['u_synced'] = self.sync_first_moment(optimizer, group)
        result['v_synced'] = self.sync_second_moment(optimizer, group)

        if result['x_synced']:
            result['total_bytes'] += self.stats['x_bytes']
        if result['u_synced']:
            result['total_bytes'] += self.stats['u_bytes']
        if result['v_synced']:
            result['total_bytes'] += self.stats['v_bytes']

        self.step += 1
        return result

    def get_comm_stats(self):
        """Get comprehensive communication statistics."""
        total_bytes = (self.stats['x_bytes'] +
                       self.stats['u_bytes'] +
                       self.stats['v_bytes'])
        total_time = (self.stats['x_time_ms'] +
                      self.stats['u_time_ms'] +
                      self.stats['v_time_ms'])
        total_syncs = (self.stats['x_syncs'] +
                       self.stats['u_syncs'] +
                       self.stats['v_syncs'])

        # DDP equivalent: all 3 states every step
        ddp_syncs = self.step * 3
        ddp_bytes = total_bytes * (ddp_syncs / max(total_syncs, 1))

        return {
            'total_steps': self.step,
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'x_syncs': self.stats['x_syncs'],
            'u_syncs': self.stats['u_syncs'],
            'v_syncs': self.stats['v_syncs'],
            'total_syncs': total_syncs,
            'total_bytes': total_bytes,
            'total_time_ms': total_time,
            'avg_time_per_sync_ms': (
                total_time / max(total_syncs, 1)),
            'ddp_equivalent_syncs': ddp_syncs,
            'reduction_factor': ddp_syncs / max(total_syncs, 1),
            'clip_events': self.stats['clip_events'],
        }

    def format_experiment_log(self):
        """Format communication stats as experiment log entry.

        Follows NKI-FA structured log format for downstream parsing.
        """
        s = self.get_comm_stats()
        lines = [
            f"### DES-LOC Comm Stats "
            f"(Kx={s['Kx']}, Ku={s['Ku']}, Kv={s['Kv']}) ###",
            f"Total steps: {s['total_steps']}",
            f"x syncs: {s['x_syncs']}, "
            f"u syncs: {s['u_syncs']}, "
            f"v syncs: {s['v_syncs']}",
            f"Total bytes: {s['total_bytes']}",
            f"Total comm time: {s['total_time_ms']:.2f}ms",
            f"Reduction vs DDP: {s['reduction_factor']:.2f}x",
            f"Clip events: {s['clip_events']}",
        ]
        return "\n".join(lines)


class DESLOCServerOpt:
    """ServerOpt (outer optimizer applied after averaging).

    Algorithm 1 line 19: x <- OPT(SERVEROPT(E_m[x]))

    Default: identity (just use averaged params).
    Can be overridden with Nesterov (Section 5.5) or
    other server-side optimizers.
    """

    def __init__(self, opt_type='identity', momentum=0.0,
                 outer_lr=1.0):
        self.opt_type = opt_type
        self.momentum = momentum
        self.outer_lr = outer_lr
        self.velocity = {}
        self.step_count = 0

    def step(self, params):
        """Apply server-side optimization after averaging.

        For identity: no-op (just use averaged params).
        For nesterov: apply momentum-based correction.
        """
        if self.opt_type == 'identity':
            self.step_count += 1
            return

        if self.opt_type == 'nesterov':
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid not in self.velocity:
                    self.velocity[pid] = p.data.new_zeros(
                        p.data.shape)
                v = self.velocity[pid]
                v.mul_(self.momentum)
                p.data.add_(v, alpha=self.outer_lr * self.momentum)
            self.step_count += 1
            return

    def get_stats(self):
        return {
            'opt_type': self.opt_type,
            'momentum': self.momentum,
            'outer_lr': self.outer_lr,
            'step_count': self.step_count,
            'tracked_params': len(self.velocity),
        }


class DESLOCConvergenceMonitor:
    """Monitor convergence metrics during training.

    Tracks loss, gradient norms, and sync decisions to detect
    training instabilities early.

    Section 5.4: "The heuristic baseline suffers training
    instabilities potentially impacting downstream performance"
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.loss_history = []
        self.grad_norm_history = []
        self.instability_events = []

    def record(self, loss, grad_norm=None):
        """Record a training step's metrics."""
        self.loss_history.append(loss)
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)

        # Detect instability: loss spike > 5x recent average
        if len(self.loss_history) > self.window_size:
            recent = self.loss_history[-self.window_size:-1]
            avg_recent = sum(recent) / len(recent)
            if avg_recent > 0 and loss > 5.0 * avg_recent:
                self.instability_events.append({
                    'step': len(self.loss_history) - 1,
                    'loss': loss,
                    'avg_recent': avg_recent,
                    'ratio': loss / avg_recent,
                })

    def is_stable(self):
        """Check if training appears stable."""
        if len(self.loss_history) < self.window_size:
            return True  # Not enough data
        recent = self.loss_history[-self.window_size:]
        # Check for NaN/Inf
        for v in recent:
            if _math.isnan(v) or _math.isinf(v):
                return False
        # Check for monotonic increase over full window
        if all(recent[i] >= recent[i - 1]
               for i in range(1, len(recent))):
            return False
        return True

    def get_summary(self):
        """Get convergence summary."""
        n = len(self.loss_history)
        if n == 0:
            return {'steps': 0}
        return {
            'steps': n,
            'final_loss': self.loss_history[-1],
            'min_loss': min(self.loss_history),
            'instability_events': len(self.instability_events),
            'is_stable': self.is_stable(),
            'last_10_avg': (
                sum(self.loss_history[-10:]) /
                min(10, n)),
        }

    def format_log(self):
        """Format convergence summary for experiment log."""
        s = self.get_summary()
        lines = [
            f"### Convergence Monitor (steps={s['steps']}) ###",
            f"Final loss: {s.get('final_loss', 'N/A')}",
            f"Min loss: {s.get('min_loss', 'N/A')}",
            f"Stable: {s.get('is_stable', 'N/A')}",
            f"Instability events: "
            f"{s.get('instability_events', 0)}",
        ]
        return "\n".join(lines)


def create_desloc_collective_ops(backend, Kx=32, Ku=96, Kv=192,
                                  clip_radius=1.0,
                                  server_opt_type='identity',
                                  nesterov_momentum=0.9,
                                  outer_lr=1.0):
    """Factory function for DES-LOC collective ops.

    Creates the full DES-LOC communication stack:
    - CollectiveOps with sync schedule
    - ServerOpt (identity or Nesterov)
    - ConvergenceMonitor
    """
    server_opt = DESLOCServerOpt(
        opt_type=server_opt_type,
        momentum=nesterov_momentum,
        outer_lr=outer_lr)

    collective = DESLOCCollectiveOps(
        backend=backend, Kx=Kx, Ku=Ku, Kv=Kv,
        clip_radius=clip_radius, server_opt=server_opt)

    monitor = DESLOCConvergenceMonitor()

    return {
        'collective': collective,
        'server_opt': server_opt,
        'monitor': monitor,
    }


# =================================================================
# End M066
# =================================================================
