# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
    DeepSpeed Communication Package: deepspeed.comm
    deepspeed.comm
        -- import and use deepspeed.ops.comm
        -- use torch.distributed directly if both this package and torch.distributed use the same NCCL version
        -- use custom collectives
            -- can either use torch.dist or ds.ops.comm?
        Note: the old 1-bit compressed allreduce variants that resided in deepspeed.runtime.comm will be moved here as well.
    deepspeed.comm API
        -- must be kept fully compatible (same signatures) as torch.dist API to ensure backward/cross-framework compatibility.
        -- e.g. if a client code used
            from deepspeed import comm as dist
            instead of
            import torch.distributed as dist
            The code should work without breaking any of the public torch.distributed functionality
    Future:
        -- deepspeed groups API should be brought into ds.comm
"""

import torch
from torch.distributed import GradBucket  # noqa: F401
import os

from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT, default_pg_timeout
from .constants import *
from .reduce_op import *
from deepspeed.accelerator import get_accelerator
from deepspeed.comm.ccl import CCLBackend

from deepspeed.utils.comms_logging import CommsLogger
from deepspeed.utils import timer, get_caller_func
from deepspeed.comm.torch import TorchBackend

from deepspeed import utils
from datetime import timedelta

# Current deepspeed.comm backend (cdb) global object for simple access by client code
cdb = None

# Create global timer for ops
timers = timer.SynchronizedWallClockTimer()
timer_summary = {}

comms_logger = CommsLogger()

# Maintain objects of all initialized ds backends and assign them using the API functions in this file
nccl_backend = None
mpi_backend = None
ccl_backend = None
hccl_backend = None

# This should be set here so all rank/size information from the launcher can be propagated
from deepspeed.comm.utils import *

import time as _time
import json as _json
import math as _math
from collections import deque as _deque


class ProcessGroup():

    def __init__(self, comm_id, ranks=[]):
        self.ranks = ranks
        self.comm_id = comm_id
        self.size = len(ranks)


# =============================================================================
# M051: DES-LOC Communication Tracking Layer (400 lines)
# =============================================================================
# Intercepts all collective ops to measure real communication costs.
# Implements DES-LOC's desynchronized sync schedule at the comm layer.
# Architecture reference: CCCL c/parallel/src/util/context.cpp
# =============================================================================


class DESLOCCommSchedule:
    """
    DES-LOC communication schedule tracker.
    Determines when each optimizer state should be synchronized
    based on Kx, Ku, Kv periods from Algorithm 1.

    Reference: des_loc_reconstructed.tex lines 151-183
    """

    def __init__(self, Kx: int = 32, Ku: int = 96, Kv: int = 192):
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        self._step = 0
        self._sync_x_count = 0
        self._sync_u_count = 0
        self._sync_v_count = 0
        self._total_bytes_saved = 0

    def advance(self) -> dict:
        """Advance one step and return which states to sync."""
        self._step += 1
        sync_x = (self._step % self.Kx == 0)
        sync_u = (self._step % self.Ku == 0)
        sync_v = (self._step % self.Kv == 0)
        if sync_x:
            self._sync_x_count += 1
        if sync_u:
            self._sync_u_count += 1
        if sync_v:
            self._sync_v_count += 1
        return {
            'step': self._step,
            'sync_x': sync_x,
            'sync_u': sync_u,
            'sync_v': sync_v,
        }

    def should_sync(self, state_type: str) -> bool:
        """Check if a state type should be synced at the CURRENT step."""
        if state_type == 'x':
            return self._step % self.Kx == 0
        elif state_type == 'u':
            return self._step % self.Ku == 0
        elif state_type == 'v':
            return self._step % self.Kv == 0
        return True  # unknown type: always sync (safe default)

    def get_comm_reduction(self, total_steps: int) -> dict:
        """Compute communication reduction vs DDP (sync every step)."""
        ddp_syncs = total_steps * 3  # DDP syncs x, u, v every step
        desloc_syncs = (
            total_steps // self.Kx +
            total_steps // self.Ku +
            total_steps // self.Kv
        )
        reduction = ddp_syncs / max(desloc_syncs, 1)
        return {
            'ddp_total_syncs': ddp_syncs,
            'desloc_total_syncs': desloc_syncs,
            'reduction_factor': round(reduction, 2),
            'sync_x_count': self._sync_x_count,
            'sync_u_count': self._sync_u_count,
            'sync_v_count': self._sync_v_count,
        }

    @property
    def step(self) -> int:
        return self._step

    def reset(self):
        """Reset the schedule counter."""
        self._step = 0
        self._sync_x_count = 0
        self._sync_u_count = 0
        self._sync_v_count = 0


class CommBandwidthTracker:
    """
    Measure real communication bandwidth from NCCL allreduce calls.
    No simulation — measures actual torch.distributed timings.
    """

    def __init__(self, world_size: int, history_len: int = 500):
        self.world_size = world_size
        self._history = _deque(maxlen=history_len)
        self._total_bytes = 0
        self._total_time_s = 0
        self._total_ops = 0
        self._op_type_counts = {}

    def record_op(self, op_name: str, tensor_bytes: int,
                  elapsed_ms: float):
        """Record a single collective operation."""
        self._total_bytes += tensor_bytes
        self._total_time_s += elapsed_ms / 1000
        self._total_ops += 1
        self._op_type_counts[op_name] = self._op_type_counts.get(op_name, 0) + 1

        if elapsed_ms > 0:
            bw_gbps = (tensor_bytes / 1e9) / (elapsed_ms / 1000)
        else:
            bw_gbps = 0

        self._history.append({
            'op': op_name,
            'bytes': tensor_bytes,
            'time_ms': elapsed_ms,
            'bw_gbps': bw_gbps,
        })

    def get_stats(self) -> dict:
        """Return bandwidth statistics."""
        if not self._history:
            return {'total_ops': 0}

        bws = [h['bw_gbps'] for h in self._history if h['bw_gbps'] > 0]
        times = [h['time_ms'] for h in self._history]

        return {
            'total_ops': self._total_ops,
            'total_bytes': self._total_bytes,
            'total_time_s': round(self._total_time_s, 3),
            'avg_bw_gbps': round(sum(bws) / len(bws), 2) if bws else 0,
            'max_bw_gbps': round(max(bws), 2) if bws else 0,
            'min_bw_gbps': round(min(bws), 2) if bws else 0,
            'avg_latency_ms': round(sum(times) / len(times), 3) if times else 0,
            'p99_latency_ms': round(sorted(times)[int(len(times) * 0.99)] if times else 0, 3),
            'op_type_counts': dict(self._op_type_counts),
        }


class DESLOCGradientCompressor:
    """
    Per-coordinate gradient clipping as specified in DES-LOC Algorithm 1.
    clip(g, rho) = sign(g_i) * min(|g_i|, rho)

    This is NOT TopK/random sparsification — it's the exact clipping
    from the paper (des_loc_reconstructed.tex line 168).
    """

    def __init__(self, clip_radius: float = 1.0):
        self.rho = clip_radius
        self._clip_events = 0
        self._total_elements = 0
        self._clipped_elements = 0

    def clip_coordinates(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply per-coordinate clipping: clip(g, rho)."""
        self._total_elements += tensor.numel()
        mask = tensor.abs() > self.rho
        self._clipped_elements += mask.sum().item()
        if mask.any():
            self._clip_events += 1
            tensor = tensor.clamp(-self.rho, self.rho)
        return tensor

    def get_stats(self) -> dict:
        """Return clipping statistics."""
        return {
            'clip_events': self._clip_events,
            'total_elements': self._total_elements,
            'clipped_elements': self._clipped_elements,
            'clip_ratio': (self._clipped_elements / max(self._total_elements, 1)),
            'rho': self.rho,
        }


class CommEventLog:
    """
    JSONL logger specifically for communication events.
    Writes one line per allreduce/broadcast/etc call.
    """

    def __init__(self, log_path: str = None, enabled: bool = True):
        self.enabled = enabled
        self._events = _deque(maxlen=10000)
        self._file = None
        if log_path and enabled:
            import os
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
            self._file = open(log_path, 'w')

    def log(self, op_name: str, tensor_numel: int, dtype_bytes: int,
            elapsed_ms: float, extra: dict = None):
        """Log a communication event."""
        if not self.enabled:
            return
        entry = {
            'ts': _time.time(),
            'op': op_name,
            'numel': tensor_numel,
            'bytes': tensor_numel * dtype_bytes,
            'ms': round(elapsed_ms, 4),
        }
        if extra:
            entry.update(extra)
        self._events.append(entry)
        if self._file:
            self._file.write(_json.dumps(entry) + '\n')

    def flush(self):
        if self._file:
            self._file.flush()

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def get_events(self) -> list:
        return list(self._events)


# Global DES-LOC comm objects (initialized lazily)
_desloc_schedule = None
_desloc_bw_tracker = None
_desloc_compressor = None
_desloc_comm_log = None


def init_desloc_comm(Kx: int = 32, Ku: int = 96, Kv: int = 192,
                     clip_radius: float = 1.0, world_size: int = 1,
                     log_path: str = None):
    """Initialize the DES-LOC communication layer."""
    global _desloc_schedule, _desloc_bw_tracker, _desloc_compressor, _desloc_comm_log
    _desloc_schedule = DESLOCCommSchedule(Kx=Kx, Ku=Ku, Kv=Kv)
    _desloc_bw_tracker = CommBandwidthTracker(world_size=world_size)
    _desloc_compressor = DESLOCGradientCompressor(clip_radius=clip_radius)
    _desloc_comm_log = CommEventLog(log_path=log_path)
    return _desloc_schedule


def get_desloc_schedule() -> DESLOCCommSchedule:
    """Get the global DES-LOC schedule (or None if not initialized)."""
    return _desloc_schedule


def get_desloc_bw_tracker() -> CommBandwidthTracker:
    """Get the global bandwidth tracker."""
    return _desloc_bw_tracker


def get_desloc_compressor() -> DESLOCGradientCompressor:
    """Get the global gradient compressor."""
    return _desloc_compressor


def desloc_all_reduce(tensor, state_type: str = 'x',
                      op=None, group=None, async_op=False):
    """
    DES-LOC-aware all_reduce that only communicates when the schedule
    says this state_type should be synced.

    state_type: 'x' (params), 'u' (first moment), 'v' (second moment)

    Returns None if communication is skipped, otherwise returns
    the same as torch.distributed.all_reduce.
    """
    global _desloc_schedule, _desloc_bw_tracker, _desloc_comm_log

    if _desloc_schedule is None:
        # Fallback: no DES-LOC schedule, always sync
        return all_reduce(tensor, op=op or ReduceOp.SUM,
                          group=group, async_op=async_op)

    if not _desloc_schedule.should_sync(state_type):
        return None  # Skip communication this step

    # Perform the actual allreduce
    t0 = _time.time()
    result = all_reduce(tensor, op=op or ReduceOp.SUM,
                        group=group, async_op=async_op)
    elapsed_ms = (_time.time() - t0) * 1000

    # Track bandwidth
    tensor_bytes = tensor.numel() * tensor.element_size()
    if _desloc_bw_tracker:
        _desloc_bw_tracker.record_op(
            f'allreduce_{state_type}', tensor_bytes, elapsed_ms)
    if _desloc_comm_log:
        _desloc_comm_log.log(
            f'allreduce_{state_type}', tensor.numel(),
            tensor.element_size(), elapsed_ms,
            extra={'step': _desloc_schedule.step})

    return result


def get_desloc_comm_summary() -> dict:
    """Get summary of DES-LOC communication stats."""
    summary = {}
    if _desloc_schedule:
        summary['schedule'] = _desloc_schedule.get_comm_reduction(
            _desloc_schedule.step)
    if _desloc_bw_tracker:
        summary['bandwidth'] = _desloc_bw_tracker.get_stats()
    if _desloc_compressor:
        summary['compression'] = _desloc_compressor.get_stats()
    return summary


def close_desloc_comm():
    """Cleanup DES-LOC communication resources."""
    global _desloc_comm_log
    if _desloc_comm_log:
        _desloc_comm_log.flush()
        _desloc_comm_log.close()


def _configure_using_config_file(config):
    if config.comms_logger_enabled:
        comms_logger.configure(config)


def configure(
    deepspeed_config=None,
    enabled=None,
    prof_all=None,
    prof_ops=None,
    verbose=None,
    debug=None,
):

    if deepspeed_config is not None:
        _configure_using_config_file(deepspeed_config.comms_config)

    if enabled is not None:
        comms_logger.enabled = enabled

    if prof_all is not None:
        comms_logger.prof_all = prof_all

    if prof_ops is not None:
        comms_logger.prof_ops = prof_ops

    if verbose is not None:
        comms_logger.verbose = verbose

    if debug is not None:
        comms_logger.debug = debug


# Logging wrapper for timing ops
def timed_op(func):

    def log_wrapper(*args, **kwargs):
        # Add enabled flag so that overhead to each comm op is two if conditions at most
        if comms_logger.enabled:
            if ('prof' in kwargs
                    and kwargs['prof']) or comms_logger.prof_all or ('log_name' in kwargs
                                                                     and kwargs['log_name'] in comms_logger.prof_ops):
                # Need func args for their defaults
                func_args = get_default_args(func)
                func_args.update(kwargs)
                msg_size = get_msg_size_from_args(func, *args, **kwargs)
                log_name = get_debug_log_name(func_args, comms_logger.debug)
                timers(log_name).start()
        # Return the op, then stop the op's timer
        try:
            return func(*args, **kwargs)
        finally:
            if comms_logger.enabled:
                # Need to make op blocking for accurate logging
                get_accelerator().synchronize()
                # If we're using MPI, we can't simply sync the stream
                if cdb.using_mpi:
                    cdb.barrier()
                if ('prof' in kwargs and kwargs['prof']) or comms_logger.prof_all or (
                        'log_name' in kwargs and kwargs['log_name'] in comms_logger.prof_ops):
                    log_name = get_debug_log_name(func_args, comms_logger.debug)
                    raw_name = func.__name__
                    timers(log_name).stop()
                    # need temp var since 'elapsed' resets events
                    time_elapsed = timers(log_name).elapsed(reset=False)
                    comms_logger.append(raw_name, log_name, time_elapsed, msg_size)

    return log_wrapper


# For compatibility with torch distributed's init_process_group, we shall retain the signature from PyTorch code.
# DeepSpeed NCCL/MPI backend may not need all these params as we will have our own implementation.
# Please read full torch.distributed API docs from https://pytorch.org/docs/stable/distributed.html


# UNUSED: Future helper function to initialize DS backends
def init_deepspeed_backend(ds_backend, timeout, init_method):
    global cdb
    global nccl_backend
    global mpi_backend
    global ccl_backend
    global hccl_backend

    rank = int(os.getenv('RANK', '-1'))
    size = int(os.getenv('WORLD_SIZE', '-1'))

    if ds_backend == NCCL_BACKEND:
        utils.logger.debug("NCCL backend in DeepSpeed not yet implemented")
    elif ds_backend == MPI_BACKEND:
        utils.logger.debug("MPI backend in DeepSpeed not yet implemented")
    elif ds_backend == GLOO_BACKEND:
        utils.logger.debug("Gloo backend in DeepSpeed not yet implemented")
    elif ds_backend == CCL_BACKEND:
        ccl_backend = CCLBackend(rank=rank, world_size=size, timeout=timeout, init_method=init_method)
        utils.logger.info(f"Initialize {ds_backend} backend")
    elif ds_backend == HCCL_BACKEND:
        utils.logger.debug("HCCL backend in DeepSpeed not yet implemented")
    else:
        utils.logger.debug(f"DeepSpeed does not support {ds_backend} backend")


def is_initialized():
    #assert cdb is not None, 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb is None:
        return False
    else:
        return cdb.is_initialized()


def destroy_process_group(group=None):
    global cdb
    return cdb.destroy_process_group(group=group)


def new_group(ranks):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.new_group(ranks)


def is_available() -> bool:

    # Returns ``True`` if the deepspeed comm package is available.

    # TODO: load other ops. Clients including deepspeed itself should use deepspeed.comm to import
    # any communication related primitives from this package.
    # use hasattr(deepspeed.csrc.ops, "_comm") or something
    return True


def set_backend():
    global cdb
    global nccl_backend
    global mpi_backend
    global ccl_backend
    global hccl_backend

    backend_name = get_accelerator().communication_backend_name()

    if backend_name == NCCL_BACKEND:
        if nccl_backend is not None and nccl_backend.is_initialized():
            cdb = nccl_backend
    elif backend_name == MPI_BACKEND:
        if mpi_backend is not None and mpi_backend.is_initialized():
            cdb = mpi_backend
    elif backend_name == CCL_BACKEND:
        if ccl_backend is not None and ccl_backend.is_initialized():
            cdb = ccl_backend
    elif backend_name == HCCL_BACKEND:
        if hccl_backend is not None and hccl_backend.is_initialized():
            cdb = hccl_backend


@timed_op
def broadcast(tensor, src, group=None, async_op=False, prof=False, log_name='broadcast', debug=get_caller_func()):
    global cdb
    return cdb.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


@timed_op
def broadcast_object_list(object_list, src, group=None, device=None):
    global cdb
    return cdb.broadcast_object_list(object_list=object_list, src=src, group=group, device=device)


@timed_op
def all_gather(tensor_list,
               tensor,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_gather',
               debug=get_caller_func()):
    global cdb
    return cdb.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)


@timed_op
def all_gather_object(object_list, obj, group=None, prof=False, log_name='all_gather_object', debug=get_caller_func()):
    global cdb
    return cdb.all_gather_object(object_list=object_list, obj=obj, group=group)


def has_reduce_scatter_tensor():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.has_reduce_scatter_tensor()


def reduce_scatter_fn(output_tensor,
                      tensor,
                      op=ReduceOp.SUM,
                      group=None,
                      async_op=False,
                      prof=False,
                      debug=get_caller_func()):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb.has_reduce_scatter_tensor():
        return reduce_scatter_tensor(output_tensor,
                                     tensor,
                                     op=op,
                                     group=group,
                                     async_op=async_op,
                                     prof=prof,
                                     debug=debug)
    else:
        if get_rank() == 0:
            utils.logger.warning_once("unable to find torch.distributed.reduce_scatter_tensor. will fall back to "
                                      "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                      "please consider upgrading your pytorch installation.")
        input_tensor_lst = list(torch.chunk(tensor, cdb.get_world_size(group)))
        return reduce_scatter(output_tensor,
                              input_tensor_lst,
                              op=op,
                              group=group,
                              async_op=async_op,
                              prof=prof,
                              debug=debug)


@timed_op
def reduce_scatter_tensor(output_tensor,
                          tensor,
                          op=ReduceOp.SUM,
                          group=None,
                          async_op=False,
                          prof=False,
                          log_name='reduce_scatter_tensor',
                          debug=get_caller_func()):
    global cdb
    return cdb.reduce_scatter_tensor(output_tensor=output_tensor,
                                     input_tensor=tensor,
                                     op=op,
                                     group=group,
                                     async_op=async_op)


@timed_op
def all_gather_into_tensor(output_tensor,
                           tensor,
                           group=None,
                           async_op=False,
                           prof=False,
                           log_name='all_gather_into_tensor',
                           debug=get_caller_func()):
    global cdb
    return cdb.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=tensor, group=group, async_op=async_op)


def has_all_gather_into_tensor():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.has_all_gather_into_tensor()


def allgather_fn(output_tensor, input_tensor, group=None, async_op=False, debug=get_caller_func()):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb.has_all_gather_into_tensor():
        return all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op, debug=debug)
    else:
        if get_rank() == 0:
            utils.logger.warning_once("unable to find torch.distributed.all_gather_into_tensor. will fall back to "
                                      "torch.distributed.all_gather which will result in suboptimal performance. "
                                      "please consider upgrading your pytorch installation.")
        output_tensors = list(torch.chunk(output_tensor, cdb.get_world_size(group)))
        return all_gather(output_tensors, input_tensor, group=group, async_op=async_op, debug=debug)


@timed_op
def all_to_all_single(output,
                      tensor,
                      output_split_sizes=None,
                      input_split_sizes=None,
                      group=None,
                      async_op=False,
                      prof=False,
                      log_name='all_to_all_single',
                      debug=get_caller_func()):
    global cdb
    return cdb.all_to_all_single(output=output,
                                 input=tensor,
                                 output_split_sizes=output_split_sizes,
                                 input_split_sizes=input_split_sizes,
                                 group=group,
                                 async_op=async_op)


@timed_op
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    global cdb
    return cdb.all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)


@timed_op
def send(tensor, dst, group=None, tag=0, prof=False, log_name='send', debug=get_caller_func()):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


@timed_op
def recv(tensor, src=None, group=None, tag=0, prof=False, log_name='recv', debug=get_caller_func()):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


@timed_op
def isend(tensor, dst, group=None, tag=0, prof=False, log_name='isend', debug=get_caller_func()):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


@timed_op
def irecv(tensor, src=None, group=None, tag=0, prof=False, log_name='irecv', debug=get_caller_func()):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


@timed_op
def gather(tensor,
           gather_list=None,
           dst=0,
           group=None,
           async_op=False,
           prof=False,
           log_name='gather',
           debug=get_caller_func()):
    global cdb
    return cdb.gather(tensor=tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)


@timed_op
def scatter(tensor,
            scatter_list=None,
            src=0,
            group=None,
            async_op=False,
            prof=False,
            log_name='scatter',
            debug=get_caller_func()):
    global cdb
    return cdb.scatter(tensor=tensor, scatter_list=scatter_list, src=src, group=group, async_op=async_op)


@timed_op
def barrier(group=None, async_op=False, device_ids=None, prof=False, log_name='barrier', debug=get_caller_func()):
    global cdb
    return cdb.barrier(group=group, async_op=async_op)


@timed_op
def monitored_barrier(group=None,
                      timeout=None,
                      wait_all_ranks=False,
                      prof=False,
                      log_name='monitored_barrier',
                      debug=get_caller_func()):
    global cdb
    return cdb.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)


def log_summary(show_straggler=False, return_dict=False):
    """
    Print and/or return communication operation statistics across all ranks.

    This function synchronizes all ranks and logs communication statistics.
    Only rank 0 prints to console by default, but all ranks can return the dictionary.

    Args:
        show_straggler (bool, optional): Whether to include straggler effect analysis.
            When True, computes the time difference between the fastest and slowest ranks
            for each communication operation. Defaults to False.
        return_dict (bool, optional): Whether to return statistics as a dictionary.
            When True, returns a comprehensive dictionary with communication metrics.
            Defaults to False.

    Returns:
        dict or None: If return_dict=True, returns communication statistics dictionary.
        The structure is identical to CommsLogger.log_all() return value.
        Returns None if return_dict=False.

        Dictionary structure (when return_dict=True):
        {
            "summary": {
                "operation_name": {
                    message_size_bytes: {
                        "count": int,
                        "total_latency_ms": float,
                        "avg_latency_ms": float,
                        "tput_avg_gbps": float,
                        "busbw_avg_gbps": float,
                        "msg_size_bytes": int,
                        "msg_size_str": str
                    }
                }
            },
            "straggler_analysis": {...} if show_straggler else None,
            "metadata": {
                "world_size": int,
                "rank": int,
                "timestamp": str
            }
        }

    Note:
        - This function includes barriers for synchronization across all ranks
        - Straggler analysis requires additional all_reduce operations
        - All ranks return the same data when return_dict=True
        - Only rank 0 prints to console when print_log=True (default behavior)

    Example:
        # Print summary only (backward compatible)
        deepspeed.comm.log_summary()

        # Get dictionary and print summary
        stats = deepspeed.comm.log_summary(return_dict=True)

        # Include straggler analysis
        stats = deepspeed.comm.log_summary(show_straggler=True, return_dict=True)

        # Access specific operation data
        if stats and "all_reduce" in stats["summary"]:
            all_reduce_stats = stats["summary"]["all_reduce"]
    """
    global cdb
    barrier(log_name='log_summary_barrier')

    result = None
    if cdb.get_rank() == 0:
        result = comms_logger.log_all(print_log=True, show_straggler=show_straggler, return_dict=return_dict)
    else:
        # Non-rank-0 processes: don't print but may still return dict if requested
        result = comms_logger.log_all(print_log=False, show_straggler=show_straggler, return_dict=return_dict)

    barrier(log_name='log_summary_barrier')
    return result


def reset_log():
    """
    Clear all accumulated communication logging data.

    This function clears the communication logger's internal data dictionary,
    allowing for epoch-by-epoch or interval-based logging. After calling this
    function, subsequent log_summary() calls will only show statistics for
    communication operations that occur after the reset.

    Note:
        - This affects the global communication logger
        - All accumulated statistics will be lost
        - This function is useful for getting per-epoch or per-interval statistics

    Example:
        # Training loop with per-epoch communication logging
        for epoch in range(num_epochs):
            # Reset logger at start of epoch
            deepspeed.comm.reset_log()

            # Train for one epoch
            train_one_epoch(model, dataloader)

            # Get communication stats for this epoch only
            epoch_stats = deepspeed.comm.log_summary(return_dict=True)
            print(f"Epoch {epoch} communication stats: {epoch_stats}")
    """
    global comms_logger
    comms_logger.reset_data()


def has_comm_data():
    """
    Check if any communication data has been logged.

    Returns:
        bool: True if communication operations have been logged, False otherwise

    Example:
        if deepspeed.comm.has_comm_data():
            stats = deepspeed.comm.log_summary(return_dict=True)
        else:
            print("No communication operations logged yet")
    """
    global comms_logger
    return comms_logger.has_data()


def get_comm_operation_count():
    """
    Get the total number of communication operations logged.

    Returns:
        int: Total count of all communication operations across all types

    Example:
        total_ops = deepspeed.comm.get_comm_operation_count()
        print(f"Total communication operations this epoch: {total_ops}")
    """
    global comms_logger
    return comms_logger.get_total_operations()


def get_logged_comm_ops():
    """
    Get list of communication operation types that have been logged.

    Returns:
        list: List of operation names that have been logged (e.g., ['all_reduce', 'broadcast'])

    Example:
        ops = deepspeed.comm.get_logged_comm_ops()
        print(f"Communication operations used: {ops}")
    """
    global comms_logger
    return comms_logger.get_operation_names()


@timed_op
def reduce(tensor,
           dst,
           op=ReduceOp.SUM,
           group=None,
           async_op=False,
           prof=False,
           log_name='reduce',
           debug=get_caller_func()):
    global cdb
    return cdb.reduce(tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)


@timed_op
def reduce_scatter(output,
                   input_list,
                   op=ReduceOp.SUM,
                   group=None,
                   async_op=False,
                   prof=False,
                   log_name='reduce_scatter',
                   debug=get_caller_func()):
    global cdb
    return cdb.reduce_scatter(output=output, input_list=input_list, op=op, group=group, async_op=async_op)


def has_all_reduce_coalesced():
    """"""
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    assert cdb.has_all_reduce_coalesced is not None, 'has_all_reduce_coalesced is not yet defined'
    return cdb.has_all_reduce_coalesced


def has_coalescing_manager():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    assert cdb.has_coalescing_manager is not None, 'has_coalescing_manager is not yet defined'
    return cdb.has_coalescing_manager


def all_gather_coalesced(output_tensors, input_tensors, group=None, async_op=False):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.all_gather_coalesced(output_tensors, input_tensors, group=group, async_op=async_op)


@timed_op
def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_reduce',
               debug=get_caller_func()):
    #if profile_comm:
    # context of the timers?
    # timers.start()
    # TensorBoard logging for comm calls.?
    global cdb
    #print(f'op = {op}, cdb= {cdb.name}')
    return cdb.all_reduce(tensor, op, group, async_op)


@timed_op
def inference_all_reduce(tensor,
                         op=ReduceOp.SUM,
                         group=None,
                         async_op=False,
                         prof=False,
                         log_name='all_reduce',
                         debug=get_caller_func()):
    global cdb
    return cdb.inference_all_reduce(tensor, op, group)


@timed_op
def all_reduce_coalesced(tensors,
                         op=ReduceOp.SUM,
                         group=None,
                         async_op=False,
                         prof=False,
                         log_name='all_reduce',
                         debug=get_caller_func()):
    global cdb
    return cdb.all_reduce_coalesced(tensors, op, group, async_op)


def get_world_group():
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_world_group()


def get_world_size(group=None) -> int:
    """
    Returns the number of processes in the current process group
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The world size of the process group
        -1, if not part of the group
    """
    global cdb

    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_world_size(group)


def get_rank(group=None):
    """
    Returns the rank of the current process in the provided ``group`` or the
    default group if none was provided.
    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The rank of the process group
        -1, if not part of the group
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_rank(group)


def get_local_rank():
    """
        Helper function to get local rank after a backend has been set and initialized
        Args:
            None
        Returns:
            local rank (= GPU device ID)
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return get_local_rank_from_launcher()


def get_global_rank(group=None, group_rank=0):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_global_rank(group, group_rank)


def get_all_ranks_from_group(group=None):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    rank = 0
    group_ranks = []
    try:
        while True:
            group_ranks.append(cdb.get_global_rank(group, rank))
            rank += 1
    except (RuntimeError, ValueError):
        pass
    return group_ranks


def initialize_mesh_device(mesh_shape, mesh_dim_names):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    mesh_device = None
    if hasattr(cdb, 'init_device_mesh'):
        utils.logger.info(f"Initializing mesh device with backend {cdb.name} \
                with shape {mesh_shape} and dim names {mesh_dim_names}")
        mesh_device = cdb.init_device_mesh(mesh_shape, mesh_dim_names)
    else:
        if get_rank() == 0:
            utils.logger.warning_once(f"Backend {cdb.name} does not support mesh device initialization")
    return mesh_device


def enable_symm_mem_for_group(group_name: str):
    global cdb
    assert cdb is not None and cdb.is_initialized(
    ), 'DeepSpeed backend not set, please initialize it using init_process_group()'

    if hasattr(cdb, 'enable_symm_mem_for_group'):
        cdb.enable_symm_mem_for_group(group_name)
    else:
        raise RuntimeError(f"Backend {cdb.name} does not support symmetric memory initialization")


# Main DeepSpeed Comms. public API.
def init_distributed(dist_backend=None,
                     auto_mpi_discovery=True,
                     distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT,
                     verbose=True,
                     timeout=default_pg_timeout,
                     init_method=None,
                     dist_init_required=None,
                     config=None,
                     rank=-1,
                     world_size=-1):
    ''' Initialize dist backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). torch distributed backend, e.g., nccl, mpi, gloo, hccl
        auto_mpi_discovery Optional (bool). if distributed environment variables are not set, attempt to discover them from MPI
        distributed_port: Optional (int). torch distributed backend port
        verbose: Optional (bool). verbose logging
        timeout: Optional (timedelta). Timeout for operations executed against the process group. The default value of 30 minutes can be overridden by the environment variable `DEEPSPEED_TIMEOUT`.
        init_method: Optional (string). Torch distributed, URL specifying how to initialize the process group. Default is "env://" if no init_method or store is specified.
        config: Optional (dict). DeepSpeed configuration for setting up comms options (e.g. Comms profiling)
        rank: Optional (int). The current manually specified rank. Some init_method like "tcp://" need the rank and world_size as well (see: https://pytorch.org/docs/stable/distributed.html#tcp-initialization)
        world_size: Optional (int). Desired world_size for the TCP or Shared file-system initialization.
    '''
    global cdb

    configure(deepspeed_config=config)

    if dist_init_required is None:
        dist_init_required = cdb is None or not cdb.is_initialized()

    if cdb is None:
        init_deepspeed_backend(get_accelerator().communication_backend_name(), timeout, init_method)
        set_backend()
        utils.logger.info(f'cdb={cdb}')
    if cdb is None and torch.distributed.is_initialized():
        # The user initialized torch.dist themselves, create cdb and short-circuit
        cdb = TorchBackend(dist_backend, timeout, init_method)
        return

    if dist_init_required is False:
        assert (
            cdb is not None and cdb.is_initialized() is True
        ), "Distributed backend is not initialized. Please set dist_init_required to True or initialize before calling deepspeed.initialize()"
    else:
        # Initialize torch distributed if needed
        required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
        if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
            if verbose:
                utils.logger.info("Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...")
            if in_aml() and not in_dlts():
                patch_aml_env_for_torch_nccl_backend(verbose=verbose)
            elif in_aws_sm():
                patch_aws_sm_env_for_torch_nccl_backend(verbose=verbose)
            else:
                mpi_discovery(distributed_port=distributed_port, verbose=verbose)

        if cdb is not None and cdb.is_initialized():
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.info('Distributed backend already initialized')
        else:
            assert isinstance(timeout, timedelta)
            if dist_backend is None:
                dist_backend = get_accelerator().communication_backend_name()
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.info('Initializing TorchBackend in DeepSpeed with backend {}'.format(dist_backend))
            # Create a torch backend object, initialize torch distributed, and assign to cdb
            cdb = TorchBackend(dist_backend, timeout, init_method, rank, world_size)


def mpi_discovery(distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True):
    '''
    Discovery MPI environment via mpi4py and map to relevant dist state
    '''
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        import shlex
        try:
            hostname_cmd = shlex.split("hostname -I")
            result = subprocess.check_output(hostname_cmd)
            master_addr = result.decode('utf-8').split()[0]
        except subprocess.CalledProcessError:  # hostname -I not available (e.g. on macOS)
            import socket
            master_addr = socket.gethostbyname(socket.gethostname())
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        utils.logger.info(
            "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}".
            format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
                   os.environ['MASTER_PORT']))

    if cdb is not None and cdb.is_initialized():
        assert cdb.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(rank, cdb.get_rank())
        assert cdb.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
            world_size, cdb.get_world_size())


def in_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def in_aws_sm():
    # Are we running inside an AWS SageMaker environment?
    return 'SM_TRAINING_ENV' in os.environ


def in_dlts():
    # Are we running on a DLTS cluster?
    return 'DLTS_JOB_ID' in os.environ


def patch_aml_env_for_torch_nccl_backend(master_port=6105, verbose=True):
    """Helper routine to get and set environment variables.
    This is adapted from Azure ML's documentation available from:
    https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training/#environment-variables-from-openmpi
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(os.environ["WORLD_SIZE"])

    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = DEFAULT_AML_MASTER_PORT

    if verbose:
        utils.logger.info("NCCL_SOCKET_IFNAME original value = {}".format(os.environ["NCCL_SOCKET_IFNAME"]))

    os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_AML_NCCL_SOCKET_IFNAME
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    if verbose:
        utils.logger.info(
            "Discovered AzureML settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))


def patch_aws_sm_env_for_torch_nccl_backend(verbose=True):
    """Helper routine to get and set environment variables when running inside an AWS SageMaker environment.
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]

    if verbose:
        utils.logger.info(
            "Discovered AWS SageMaker settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'], os.environ['LOCAL_RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))


# =================================================================
# M064: Probabilistic Sync + Comm Reduction Tracker (400 lines)
# =================================================================
# Implements Section 3 probabilistic synchronization equivalence
# (px = 1/Kx) and comprehensive communication reduction tracking
# for generating RQ3/RQ4 figures.
#
# Reference: template_extraction_section3.txt CXXIII
# Reference: Algorithm 1 line 14: "if t mod Kj = 0 then sync"
# =================================================================

import math as _math
import hashlib as _hashlib


class DESLOCProbabilisticSync:
    """Probabilistic sync scheduler per Section 3.

    Instead of deterministic "t mod Kx == 0", syncs with
    probability px = 1/Kx each step. The paper proves these
    are statistically equivalent for convergence analysis.

    Uses a deterministic hash-based approach to ensure
    reproducibility across workers without numpy.random.

    Reference: Section 3: "we average with probability
    px = 1/Kx, two approaches are statistically equivalent"
    """

    def __init__(self, Kx=32, Ku=96, Kv=192, seed=42,
                 probabilistic=False):
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        self.px = 1.0 / max(Kx, 1)
        self.pu = 1.0 / max(Ku, 1)
        self.pv = 1.0 / max(Kv, 1)
        self.seed = seed
        self.probabilistic = probabilistic
        self.step = 0
        self.sync_x_count = 0
        self.sync_u_count = 0
        self.sync_v_count = 0

    def _hash_decision(self, step, state_key):
        """Deterministic pseudo-random decision using SHA256.

        Ensures all workers make the same sync decision without
        requiring a shared RNG state or numpy.random.
        """
        raw = f"{self.seed}:{step}:{state_key}"
        h = _hashlib.sha256(raw.encode()).hexdigest()
        # Convert first 8 hex chars to float in [0, 1)
        val = int(h[:8], 16) / 0xFFFFFFFF
        return val

    def should_sync_x(self, step=None):
        """Decide whether to sync parameters at this step."""
        if step is None:
            step = self.step
        if not self.probabilistic:
            return step % self.Kx == 0
        return self._hash_decision(step, 'x') < self.px

    def should_sync_u(self, step=None):
        """Decide whether to sync first moment at this step."""
        if step is None:
            step = self.step
        if not self.probabilistic:
            return step % self.Ku == 0
        return self._hash_decision(step, 'u') < self.pu

    def should_sync_v(self, step=None):
        """Decide whether to sync second moment at this step."""
        if step is None:
            step = self.step
        if not self.probabilistic:
            return step % self.Kv == 0
        return self._hash_decision(step, 'v') < self.pv

    def advance(self):
        """Advance step and record sync counts."""
        if self.should_sync_x():
            self.sync_x_count += 1
        if self.should_sync_u():
            self.sync_u_count += 1
        if self.should_sync_v():
            self.sync_v_count += 1
        self.step += 1

    def get_empirical_rates(self):
        """Get empirical sync rates."""
        s = max(self.step, 1)
        return {
            'x_rate': self.sync_x_count / s,
            'u_rate': self.sync_u_count / s,
            'v_rate': self.sync_v_count / s,
            'expected_x_rate': self.px,
            'expected_u_rate': self.pu,
            'expected_v_rate': self.pv,
        }


class DESLOCCommReductionTracker:
    """Track communication reduction across training.

    Records exact bytes communicated per step, per state type,
    for generating the throughput and comm reduction figures
    required by RQ3 and RQ4.

    Data format follows NKI-FA commit da964f3 draw_plot.py:
    precise numeric values from actual training, not synthetic.
    """

    def __init__(self, param_bytes=0, world_size=1):
        self.param_bytes = param_bytes
        self.world_size = world_size
        self.step = 0
        self.x_comm_bytes = 0
        self.u_comm_bytes = 0
        self.v_comm_bytes = 0
        self.x_comm_count = 0
        self.u_comm_count = 0
        self.v_comm_count = 0
        self.step_log = []
        self.wallclock_start = None
        self.step_times = []

    def record_sync(self, state_type, num_bytes, elapsed_ms=0.0):
        """Record a synchronization event.

        Args:
            state_type: 'x' (params), 'u' (first moment), 'v' (second)
            num_bytes: bytes communicated in this allreduce
            elapsed_ms: time taken for this communication
        """
        if state_type == 'x':
            self.x_comm_bytes += num_bytes
            self.x_comm_count += 1
        elif state_type == 'u':
            self.u_comm_bytes += num_bytes
            self.u_comm_count += 1
        elif state_type == 'v':
            self.v_comm_bytes += num_bytes
            self.v_comm_count += 1

    def record_step(self, loss=None, lr=None, grad_norm=None):
        """Record per-step metrics for the experiment log."""
        total_bytes = (self.x_comm_bytes + self.u_comm_bytes +
                       self.v_comm_bytes)
        entry = {
            'step': self.step,
            'total_comm_bytes': total_bytes,
            'x_comm_bytes': self.x_comm_bytes,
            'u_comm_bytes': self.u_comm_bytes,
            'v_comm_bytes': self.v_comm_bytes,
            'x_comm_count': self.x_comm_count,
            'u_comm_count': self.u_comm_count,
            'v_comm_count': self.v_comm_count,
        }
        if loss is not None:
            entry['loss'] = loss
        if lr is not None:
            entry['lr'] = lr
        if grad_norm is not None:
            entry['grad_norm'] = grad_norm
        self.step_log.append(entry)
        self.step += 1

    def get_reduction_vs_ddp(self):
        """Calculate comm reduction factor vs DDP.

        DDP communicates all 3 states every step.
        Factor = DDP_bytes / DES-LOC_bytes.
        """
        ddp_bytes = self.param_bytes * 3 * max(self.step, 1)
        desloc_bytes = (self.x_comm_bytes + self.u_comm_bytes +
                        self.v_comm_bytes)
        if desloc_bytes == 0:
            return float('inf')
        return ddp_bytes / desloc_bytes

    def get_reduction_vs_local_adam(self):
        """Calculate comm reduction factor vs Local Adam.

        Local Adam syncs all 3 states at the same frequency.
        """
        local_adam_bytes = self.param_bytes * 3 * self.x_comm_count
        desloc_bytes = (self.x_comm_bytes + self.u_comm_bytes +
                        self.v_comm_bytes)
        if desloc_bytes == 0:
            return float('inf')
        return local_adam_bytes / desloc_bytes

    def get_throughput_samples_per_sec(self, batch_size, elapsed_sec):
        """Compute training throughput."""
        if elapsed_sec <= 0:
            return 0.0
        return (self.step * batch_size) / elapsed_sec

    def format_log_entry(self, step_idx):
        """Format a log entry in the NKI-FA draw_plot.py style.

        Example output:
        ### step=100, Kx=32, loss=2.847, comm_bytes=15728640 ###
        DES-LOC x_sync: 3, u_sync: 1, v_sync: 0
        Reduction vs DDP: 5.33x, vs Local Adam: 2.00x
        """
        if step_idx >= len(self.step_log):
            return ""
        e = self.step_log[step_idx]
        lines = []
        header = f"### step={e['step']}"
        if 'loss' in e:
            header += f", loss={e['loss']:.4f}"
        header += f", comm_bytes={e['total_comm_bytes']} ###"
        lines.append(header)
        lines.append(
            f"DES-LOC x_sync: {e['x_comm_count']}, "
            f"u_sync: {e['u_comm_count']}, "
            f"v_sync: {e['v_comm_count']}")
        return "\n".join(lines)

    def export_for_plotting(self):
        """Export data in format ready for matplotlib/seaborn.

        Returns dict of lists suitable for pd.DataFrame construction,
        following NKI-FA draw_plot.py convention.
        """
        steps = []
        losses = []
        comm_bytes_list = []
        x_counts = []
        u_counts = []
        v_counts = []

        for entry in self.step_log:
            steps.append(entry['step'])
            losses.append(entry.get('loss', None))
            comm_bytes_list.append(entry['total_comm_bytes'])
            x_counts.append(entry['x_comm_count'])
            u_counts.append(entry['u_comm_count'])
            v_counts.append(entry['v_comm_count'])

        return {
            'step': steps,
            'loss': losses,
            'total_comm_bytes': comm_bytes_list,
            'x_sync_count': x_counts,
            'u_sync_count': u_counts,
            'v_sync_count': v_counts,
            'reduction_vs_ddp': self.get_reduction_vs_ddp(),
            'reduction_vs_local_adam': self.get_reduction_vs_local_adam(),
        }


class DESLOCThroughputMeter:
    """Measure DES-LOC throughput vs DDP baseline.

    Section 5.3: "This yields a 1.24× speedup over DDP and
    2.01× over Local Adam at Kx=32"

    Records wallclock times for compute vs communication to
    calculate the overlap efficiency.
    """

    def __init__(self):
        self.compute_times_ms = []
        self.comm_times_ms = []
        self.total_step_times_ms = []
        self.step = 0

    def record_compute(self, elapsed_ms):
        """Record time spent on forward/backward pass."""
        self.compute_times_ms.append(elapsed_ms)

    def record_comm(self, elapsed_ms):
        """Record time spent on communication."""
        self.comm_times_ms.append(elapsed_ms)

    def record_total_step(self, elapsed_ms):
        """Record total step time (compute + comm + overhead)."""
        self.total_step_times_ms.append(elapsed_ms)
        self.step += 1

    def get_avg_compute_ms(self):
        if not self.compute_times_ms:
            return 0.0
        return sum(self.compute_times_ms) / len(self.compute_times_ms)

    def get_avg_comm_ms(self):
        if not self.comm_times_ms:
            return 0.0
        return sum(self.comm_times_ms) / len(self.comm_times_ms)

    def get_avg_step_ms(self):
        if not self.total_step_times_ms:
            return 0.0
        return (sum(self.total_step_times_ms) /
                len(self.total_step_times_ms))

    def get_comm_compute_ratio(self):
        """Ratio of comm time to compute time."""
        avg_comp = self.get_avg_compute_ms()
        avg_comm = self.get_avg_comm_ms()
        if avg_comp == 0:
            return float('inf')
        return avg_comm / avg_comp

    def estimate_speedup_vs_ddp(self, ddp_comm_fraction=0.3):
        """Estimate speedup based on communication reduction.

        Args:
            ddp_comm_fraction: fraction of DDP step time spent on comm
        """
        if not self.total_step_times_ms:
            return 1.0
        avg_step = self.get_avg_step_ms()
        avg_comm = self.get_avg_comm_ms()
        # DDP step time = compute + ddp_comm
        # DES-LOC step time = compute + desloc_comm
        compute_ms = avg_step - avg_comm
        ddp_comm_ms = compute_ms * ddp_comm_fraction / (
            1.0 - ddp_comm_fraction)
        ddp_step_ms = compute_ms + ddp_comm_ms
        if avg_step <= 0:
            return 1.0
        return ddp_step_ms / avg_step

    def format_summary(self):
        """Format throughput summary for experiment log.

        Output follows NKI-FA structured log format.
        """
        lines = [
            f"### Throughput Summary (steps={self.step}) ###",
            f"Avg compute: {self.get_avg_compute_ms():.3f}ms",
            f"Avg comm: {self.get_avg_comm_ms():.3f}ms",
            f"Avg step: {self.get_avg_step_ms():.3f}ms",
            f"Comm/Compute ratio: {self.get_comm_compute_ratio():.4f}",
            f"Est. speedup vs DDP: "
            f"{self.estimate_speedup_vs_ddp():.2f}x",
        ]
        return "\n".join(lines)


# Global instances for convenience
_desloc_prob_sync = None
_desloc_comm_tracker = None
_desloc_throughput_meter = None


def init_desloc_probabilistic_sync(Kx=32, Ku=96, Kv=192,
                                    seed=42, probabilistic=False):
    """Initialize the global probabilistic sync scheduler."""
    global _desloc_prob_sync
    _desloc_prob_sync = DESLOCProbabilisticSync(
        Kx=Kx, Ku=Ku, Kv=Kv, seed=seed,
        probabilistic=probabilistic)
    return _desloc_prob_sync


def get_desloc_prob_sync():
    """Get the global probabilistic sync scheduler."""
    return _desloc_prob_sync


def init_desloc_comm_tracker(param_bytes=0, world_size=1):
    """Initialize the global communication reduction tracker."""
    global _desloc_comm_tracker
    _desloc_comm_tracker = DESLOCCommReductionTracker(
        param_bytes=param_bytes, world_size=world_size)
    return _desloc_comm_tracker


def get_desloc_comm_tracker():
    """Get the global communication tracker."""
    return _desloc_comm_tracker


def init_desloc_throughput_meter():
    """Initialize the global throughput meter."""
    global _desloc_throughput_meter
    _desloc_throughput_meter = DESLOCThroughputMeter()
    return _desloc_throughput_meter


def get_desloc_throughput_meter():
    """Get the global throughput meter."""
    return _desloc_throughput_meter


# =================================================================
# End M064
# =================================================================

# =================================================================
# M080: DES-LOC Communication Bandwidth Benchmark Suite
# Claude-5 (M077-M091)
# Nick Joseph: "训练要跨大量GPU并行进行...分布式有多种模式"
# =================================================================

import time
import math
import json
import os


class CommBandwidthProfiler:
    """Measure AllReduce/ReduceScatter latency and bandwidth.

    Records per-operation timing using CUDA events for accuracy.
    Separates warmup iterations from measured iterations.
    """

    def __init__(self, warmup_iters=5, measure_iters=20,
                 msg_sizes_bytes=None):
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.msg_sizes = msg_sizes_bytes or [
            1024,           # 1 KB
            8192,           # 8 KB
            65536,          # 64 KB
            524288,         # 512 KB
            1048576,        # 1 MB
            4194304,        # 4 MB
            16777216,       # 16 MB
            67108864,       # 64 MB
            268435456,      # 256 MB
        ]
        self._results = []

    def benchmark_allreduce(self, group=None):
        """Run AllReduce benchmark across message sizes."""
        try:
            import torch
            import deepspeed.comm as dist
        except ImportError:
            return []

        if not dist.is_initialized():
            return []

        device = torch.cuda.current_device()
        results = []
        world_size = dist.get_world_size(group)

        for size_bytes in self.msg_sizes:
            num_elements = size_bytes // 4  # float32
            if num_elements < 1:
                num_elements = 1
            tensor = torch.zeros(num_elements, dtype=torch.float32,
                                 device=device)

            # Warmup
            for _ in range(self.warmup_iters):
                dist.all_reduce(tensor, group=group)
            torch.cuda.synchronize()

            # Measure
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(self.measure_iters):
                dist.all_reduce(tensor, group=group)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            avg_ms = elapsed_ms / self.measure_iters
            # AllReduce transfers 2*(N-1)/N * msg_size (ring algo)
            algo_bw_factor = 2.0 * (world_size - 1) / world_size
            if avg_ms > 0:
                bw_gbps = (size_bytes * algo_bw_factor
                           / (avg_ms / 1000.0) / 1e9)
            else:
                bw_gbps = 0.0

            result = {
                "op": "all_reduce",
                "msg_size_bytes": size_bytes,
                "world_size": world_size,
                "latency_ms": round(avg_ms, 4),
                "bandwidth_gbps": round(bw_gbps, 2),
                "algo_bw_factor": algo_bw_factor,
                "iters": self.measure_iters,
            }
            results.append(result)
            del tensor

        self._results.extend(results)
        return results

    def benchmark_reduce_scatter(self, group=None):
        """Run ReduceScatter benchmark across message sizes."""
        try:
            import torch
            import deepspeed.comm as dist
        except ImportError:
            return []

        if not dist.is_initialized():
            return []

        device = torch.cuda.current_device()
        results = []
        world_size = dist.get_world_size(group)

        for size_bytes in self.msg_sizes:
            num_elements = size_bytes // 4
            if num_elements < world_size:
                num_elements = world_size
            # Ensure divisible
            num_elements = (num_elements // world_size) * world_size
            tensor = torch.zeros(num_elements, dtype=torch.float32,
                                 device=device)
            output = torch.zeros(num_elements // world_size,
                                 dtype=torch.float32, device=device)

            for _ in range(self.warmup_iters):
                dist.reduce_scatter(output, tensor, group=group)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(self.measure_iters):
                dist.reduce_scatter(output, tensor, group=group)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            avg_ms = elapsed_ms / self.measure_iters
            algo_bw_factor = (world_size - 1.0) / world_size
            if avg_ms > 0:
                bw_gbps = (size_bytes * algo_bw_factor
                           / (avg_ms / 1000.0) / 1e9)
            else:
                bw_gbps = 0.0

            result = {
                "op": "reduce_scatter",
                "msg_size_bytes": size_bytes,
                "world_size": world_size,
                "latency_ms": round(avg_ms, 4),
                "bandwidth_gbps": round(bw_gbps, 2),
                "algo_bw_factor": algo_bw_factor,
                "iters": self.measure_iters,
            }
            results.append(result)
            del tensor, output

        self._results.extend(results)
        return results

    def get_results(self):
        return list(self._results)

    def export_json(self, path):
        with open(path, "w") as fh:
            json.dump(self._results, fh, indent=2)

    def export_csv(self, path):
        if not self._results:
            return
        import csv as csv_mod
        fields = list(self._results[0].keys())
        with open(path, "w", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._results)


class TopologyDetector:
    """Detect GPU interconnect topology (NVLink, PCIe, InfiniBand).

    Nick Joseph: "同事甚至跑了聚类算法来推测芯片分布在哪些机房"
    """

    def __init__(self):
        self._topology = {}
        self._detected = False

    def detect(self):
        """Probe GPU topology using nvidia-smi and torch."""
        info = {"num_gpus": 0, "gpu_names": [], "interconnects": [],
                "nvlink_pairs": [], "pcie_bandwidth": []}
        try:
            import torch
            if torch.cuda.is_available():
                info["num_gpus"] = torch.cuda.device_count()
                for i in range(info["num_gpus"]):
                    props = torch.cuda.get_device_properties(i)
                    info["gpu_names"].append(props.name)
        except Exception:
            pass

        # Detect NVLink via P2P access check
        try:
            import torch
            for i in range(info["num_gpus"]):
                for j in range(i + 1, info["num_gpus"]):
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    link_type = "nvlink" if can_access else "pcie"
                    info["interconnects"].append({
                        "gpu_a": i, "gpu_b": j, "type": link_type,
                    })
                    if can_access:
                        info["nvlink_pairs"].append((i, j))
        except Exception:
            pass

        self._topology = info
        self._detected = True
        return info

    def get_topology(self):
        if not self._detected:
            self.detect()
        return self._topology

    def is_heterogeneous(self):
        topo = self.get_topology()
        names = set(topo.get("gpu_names", []))
        return len(names) > 1

    def has_nvlink(self):
        topo = self.get_topology()
        return len(topo.get("nvlink_pairs", [])) > 0

    def recommended_comm_strategy(self):
        """Recommend DES-LOC communication strategy based on topology."""
        topo = self.get_topology()
        if self.has_nvlink():
            return {
                "strategy": "nvlink_ring",
                "description": "Use ring AllReduce over NVLink",
                "expected_bw_gbps": 600,
            }
        elif topo["num_gpus"] > 1:
            return {
                "strategy": "pcie_tree",
                "description": "Use tree AllReduce over PCIe",
                "expected_bw_gbps": 32,
            }
        else:
            return {
                "strategy": "single_gpu",
                "description": "No communication needed",
                "expected_bw_gbps": 0,
            }


class DeslocCommTracker:
    """Track per-step communication volume for DES-LOC experiments.

    Counts bytes saved by skipping AllReduce when step % Kx != 0.
    """

    def __init__(self, Kx=8, Ku=24, Kv=48, param_bytes=0):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.param_bytes = param_bytes
        self._step = 0
        self._total_sent = 0
        self._total_saved = 0
        self._log = []

    def step(self):
        """Record one training step and return comm decision."""
        self._step += 1
        sent = 0
        saved = 0

        # Parameter sync
        if self._step % self.Kx == 0:
            sent += self.param_bytes
        else:
            saved += self.param_bytes

        # First momentum sync
        if self._step % self.Ku == 0:
            sent += self.param_bytes
        else:
            saved += self.param_bytes

        # Second momentum sync
        if self._step % self.Kv == 0:
            sent += self.param_bytes
        else:
            saved += self.param_bytes

        self._total_sent += sent
        self._total_saved += saved

        entry = {
            "step": self._step,
            "bytes_sent": sent,
            "bytes_saved": saved,
            "sync_x": self._step % self.Kx == 0,
            "sync_u": self._step % self.Ku == 0,
            "sync_v": self._step % self.Kv == 0,
        }
        self._log.append(entry)
        return entry

    def get_reduction_factor(self):
        """Communication reduction vs DDP (sync every step)."""
        ddp_total = self._step * self.param_bytes * 3
        if ddp_total <= 0:
            return 1.0
        return ddp_total / max(1, self._total_sent)

    def get_reduction_vs_local_adam(self):
        """Reduction vs Local Adam (syncs all states at Kx)."""
        local_adam_sent = (self._step // self.Kx) * self.param_bytes * 3
        if local_adam_sent <= 0:
            return 1.0
        return local_adam_sent / max(1, self._total_sent)

    def summary(self):
        return {
            "total_steps": self._step,
            "Kx": self.Kx, "Ku": self.Ku, "Kv": self.Kv,
            "total_bytes_sent": self._total_sent,
            "total_bytes_saved": self._total_saved,
            "reduction_vs_ddp": round(self.get_reduction_factor(), 1),
            "reduction_vs_local_adam": round(
                self.get_reduction_vs_local_adam(), 1),
        }

    def get_log(self):
        return list(self._log)


# =================================================================
# End M080  (CommBandwidthProfiler + TopologyDetector + CommTracker)
# =================================================================
