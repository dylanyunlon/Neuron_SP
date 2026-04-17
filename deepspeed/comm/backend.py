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
