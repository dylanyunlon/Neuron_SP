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


class ProcessGroup():

    def __init__(self, comm_id, ranks=[]):
        self.ranks = ranks
        self.comm_id = comm_id
        self.size = len(ranks)


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
               debug=get_caller_func(),
               desloc_tier=None):
    """Perform all_reduce collective.

    DES-LOC extension: desloc_tier classifies this op for per-tier
    communication accounting (tier 0=param, 1=momentum, 2=variance).
    The tier does NOT affect execution — gating happens in engine.py.

    Ref: Algorithm 1 — Ring-AllReduce for parameter averaging.
    Ref: Section 4.1 — 2*(N-1)/N * data_size bytes per allreduce.
    """
    global cdb
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


# =========================================================================
# DES-LOC Communication Primitives
# Ref: Algorithm 1 + Section 4.1 — independent sync period collectives
# Ref: NCCL src/collectives.cc — ncclAllReduce with grouped execution
# Ref: Megatron-LM param_and_grad_buffer.py — start_grad_sync pattern
# =========================================================================

DESLOC_COMM_TIER_PARAM = 0
DESLOC_COMM_TIER_MOMENTUM = 1
DESLOC_COMM_TIER_VARIANCE = 2




# M282: Bandwidth measurement
def desloc_measure_bandwidth(tensor_bytes=10000000, n_iters=5):
    import time
    try:
        import torch, torch.distributed as dist
        if not dist.is_initialized(): return {"bw_gbps":0}
        buf = torch.zeros(tensor_bytes//4, dtype=torch.float32, device=torch.cuda.current_device())
        dist.all_reduce(buf); torch.cuda.synchronize()
        times = []
        for _ in range(n_iters):
            torch.cuda.synchronize(); t0=time.monotonic()
            dist.all_reduce(buf); torch.cuda.synchronize()
            times.append(time.monotonic()-t0)
        avg = sum(times)/len(times)
        return {"bw_gbps": round(tensor_bytes/avg/1e9,4), "latency_us": round(avg*1e6,2)}
    except Exception: return {"bw_gbps":0}

# M290 — Claude-19: HierarchicalAR + RingOptimizer


# M306: Tier scheduling + comm tracking
def desloc_should_ar(step, Kx, warmup=512):
    if Kx <= 1 or step < warmup: return True
    return (step % Kx) == 0

def desloc_tier(tn, Kx=32, Ku=96, Kv=192):
    n = tn.lower() if tn else ''
    if 'exp_avg_sq' in n or 'moment2' in n: return (2, Kv)
    elif 'exp_avg' in n or 'moment1' in n: return (1, Ku)
    return (0, Kx)

def desloc_budget(mp, Kx, Ku, Kv, steps):
    pb = mp * 2; ddp = 3 * pb * steps; dl = pb * (steps / max(1, Kx) + steps / max(1, Ku) + steps / max(1, Kv))
    return {'ddp': ddp, 'dl': round(dl), 'red': round(ddp / max(1, dl), 2)}

class DeslocCommTracker:
    def __init__(self, Kx=1, Ku=3, Kv=6):
        self.Kx, self.Ku, self.Kv = Kx, Ku, Kv
        self._s = {0: 0, 1: 0, 2: 0}; self._k = {0: 0, 1: 0, 2: 0}; self._n = 0
    def sync(self, t, nb): self._s[t] = self._s.get(t, 0) + nb
    def skip(self, t, nb): self._k[t] = self._k.get(t, 0) + nb
    def step(self): self._n += 1
    def stats(self):
        ts = sum(self._s.values()); tk = sum(self._k.values())
        return {'sent': ts, 'skip': tk, 'red': round((ts + tk) / max(1, ts), 2), 'n': self._n}
    def state_dict(self): return {'s': dict(self._s), 'k': dict(self._k), 'n': self._n}
    def load_state_dict(self, d): self._s = {int(k): v for k, v in d.get('s', {}).items()}; self._k = {int(k): v for k, v in d.get('k', {}).items()}; self._n = d.get('n', 0)
# --- End M306 ---


# =========================================================================
# M332 (Claude-24): DES-LOC Scheduler + Tiered AllReduce + Profiler
# Required by deepspeed/runtime/engine.py:2494 — desloc_init_scheduler()
# Ref: Algorithm 1 — independent Kx/Ku/Kv sync periods
# =========================================================================

_desloc_scheduler_instance = None
_desloc_tiered_ar_instance = None
_desloc_profiler_instance = None


class DeslocScheduler:
    """Tracks per-step sync decisions for Kx (params), Ku (m1), Kv (m2).

    Kx=1 reproduces standard DDP (every-step sync).
    warmup_steps: force sync every step during warmup for stability.
    Ref: Section 3.2 — TWARM = 512 default.
    """

    def __init__(self, Kx=1, Ku=3, Kv=6, warmup_steps=512, group=None):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.warmup_steps = warmup_steps
        self.group = group
        self.local_step = 0
        self.total_syncs_x = 0
        self.total_syncs_u = 0
        self.total_syncs_v = 0
        self.total_skips = 0

    def advance(self):
        """Call once per optimizer step."""
        self.local_step += 1

    def should_sync_x(self):
        """Parameter sync: every Kx steps (or every step during warmup)."""
        if self.local_step < self.warmup_steps or self.Kx <= 1:
            return True
        return (self.local_step % self.Kx) == 0

    def should_sync_u(self):
        """First-moment sync: every Ku steps."""
        if self.local_step < self.warmup_steps or self.Ku <= 1:
            return True
        return (self.local_step % self.Ku) == 0

    def should_sync_v(self):
        """Second-moment sync: every Kv steps."""
        if self.local_step < self.warmup_steps or self.Kv <= 1:
            return True
        return (self.local_step % self.Kv) == 0

    def record_sync(self, tier):
        if tier == 'x':
            self.total_syncs_x += 1
        elif tier == 'u':
            self.total_syncs_u += 1
        elif tier == 'v':
            self.total_syncs_v += 1

    def record_skip(self):
        self.total_skips += 1

    def comm_reduction_ratio(self):
        total_ops = self.local_step * 3
        actual = self.total_syncs_x + self.total_syncs_u + self.total_syncs_v
        return round(total_ops / max(1, actual), 2)

    def state_dict(self):
        return {
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'local_step': self.local_step,
            'syncs': [self.total_syncs_x, self.total_syncs_u, self.total_syncs_v],
            'skips': self.total_skips,
        }

    def load_state_dict(self, sd):
        self.local_step = sd.get('local_step', 0)
        s = sd.get('syncs', [0, 0, 0])
        self.total_syncs_x, self.total_syncs_u, self.total_syncs_v = s[0], s[1], s[2]
        self.total_skips = sd.get('skips', 0)

    def __repr__(self):
        return (f"DeslocScheduler(Kx={self.Kx}, Ku={self.Ku}, Kv={self.Kv}, "
                f"step={self.local_step}, ratio={self.comm_reduction_ratio()}x)")


class DeslocGradBucket:
    """Contiguous gradient buffer for batched reduction.

    Ref: Megatron param_and_grad_buffer.py Bucket class — parameters
         registered into fixed-size buckets; reduced as single NCCL call.
    Ref: NCCL all_reduce.h:22 — chunkCount sizing for Ring protocol.
    """

    __slots__ = ('data', 'params', 'offsets', 'ready_count', 'tier', '_fill')

    def __init__(self, numel, dtype, device, tier='x'):
        self.data = torch.zeros(numel, dtype=dtype, device=device)
        self.params = []
        self.offsets = []
        self.ready_count = 0
        self.tier = tier
        self._fill = 0

    def register(self, param):
        off = self._fill
        n = param.data.numel()
        self.params.append(param)
        self.offsets.append((off, off + n))
        self._fill += n
        return off

    def copy_grad_in(self, idx):
        p = self.params[idx]
        if p.grad is None:
            return
        s, e = self.offsets[idx]
        self.data[s:e].copy_(p.grad.data.view(-1))
        self.ready_count += 1

    def all_ready(self):
        return self.ready_count >= len(self.params)

    def copy_grad_out(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                s, e = self.offsets[i]
                p.grad.data.copy_(self.data[s:e].view_as(p.grad.data))

    def reset(self):
        self.ready_count = 0


class DeslocTieredAllReduce:
    """Gated AllReduce with async bucket coalescing and tier routing.

    Key patterns from upstream codebases:

    1. Megatron start_grad_sync (param_and_grad_buffer.py:568):
       async_op dispatch with communication stream overlap.
    2. NCCL Ring AllReduce (all_reduce.h:22): chunkCount bucket sizing.
    3. veScale mesh_scatter_ragged (_collective_utils.py:83):
       async dispatch without blocking.
    4. neuronx-distributed _get_params_and_grad_norm
       (zero_redundancy_optimizer.py:71): per-shard grad norm.
    """

    # Megatron default: max(40M, 1M * world_size)
    DEFAULT_BUCKET_BYTES = 40_000_000

    def __init__(self, scheduler, group=None, bucket_bytes=None,
                 overlap_comm=True, fp32_reduce=False):
        self.scheduler = scheduler
        self.group = group
        self._bucket_cap = bucket_bytes or self.DEFAULT_BUCKET_BYTES
        self._overlap = overlap_comm
        self._fp32_reduce = fp32_reduce
        self._bytes_sent = 0
        self._bytes_skipped = 0
        self._op_count = 0
        self._pending = []  # async handles — Megatron pattern
        self._buckets = {'x': [], 'u': [], 'v': []}
        self._param_map = {}
        self._tier_norm_sq = {'x': 0.0, 'u': 0.0, 'v': 0.0}
        self._comm_stream = None  # Megatron DDP communication_stream

    def _get_stream(self):
        """Lazy-init comm stream. Ref: Megatron param_and_grad_buffer.py:577."""
        if self._comm_stream is None:
            try:
                self._comm_stream = torch.cuda.Stream()
            except Exception:
                pass
        return self._comm_stream

    def register_param(self, param, tier='x'):
        """Assign param to a tier bucket; create new bucket when full.
        Ref: Megatron _ParamAndGradBuffer.__init__ bucket allocation."""
        if param in self._param_map:
            return
        nbytes = param.data.numel() * param.data.element_size()
        bkts = self._buckets[tier]
        bkt = None
        if bkts:
            last = bkts[-1]
            if last._fill * last.data.element_size() + nbytes <= self._bucket_cap:
                bkt = last
        if bkt is None:
            cap = max(param.data.numel(),
                      self._bucket_cap // max(1, param.data.element_size()))
            bkt = DeslocGradBucket(cap, param.data.dtype,
                                   param.data.device, tier)
            bkts.append(bkt)
        idx = len(bkt.params)
        bkt.register(param)
        self._param_map[param] = (bkt, idx)

    def mark_grad_ready(self, param):
        entry = self._param_map.get(param)
        if entry is None:
            return
        bkt, idx = entry
        bkt.copy_grad_in(idx)

    def _should_sync(self, tier):
        if tier == 'x':
            return self.scheduler.should_sync_x()
        elif tier == 'u':
            return self.scheduler.should_sync_u()
        return self.scheduler.should_sync_v()

    def reduce_buckets(self):
        """Reduce ready buckets whose tier is due for sync.

        Ref: Megatron param_and_grad_buffer.py:594 —
             with stream_context, _coalescing_manager(..., async_ops=async_op)
        """
        import torch.distributed as tdist
        if not tdist.is_initialized():
            return {'x': 0, 'u': 0, 'v': 0}

        reduced = {'x': 0, 'u': 0, 'v': 0}
        stream = self._get_stream() if self._overlap else None

        for tier in ('x', 'u', 'v'):
            sync = self._should_sync(tier)
            for bkt in self._buckets[tier]:
                if not bkt.all_ready():
                    continue
                nb = bkt._fill * bkt.data.element_size()
                buf = bkt.data[:bkt._fill]

                # Per-tier grad norm — neuronx-distributed pattern
                self._tier_norm_sq[tier] += float(buf.float().norm().item()) ** 2

                if sync:
                    if self._fp32_reduce and buf.dtype != torch.float32:
                        fp32 = buf.float()
                        if stream is not None:
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                h = tdist.all_reduce(fp32, op=tdist.ReduceOp.AVG,
                                                     group=self.group, async_op=True)
                        else:
                            h = tdist.all_reduce(fp32, op=tdist.ReduceOp.AVG,
                                                 group=self.group, async_op=self._overlap)
                        self._pending.append((h, buf, fp32))
                    else:
                        if stream is not None:
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                h = tdist.all_reduce(buf, op=tdist.ReduceOp.AVG,
                                                     group=self.group, async_op=True)
                        else:
                            h = tdist.all_reduce(buf, op=tdist.ReduceOp.AVG,
                                                 group=self.group, async_op=self._overlap)
                        self._pending.append((h, None, None))
                    self._bytes_sent += nb
                    self._op_count += 1
                    self.scheduler.record_sync(tier)
                    reduced[tier] += 1
                else:
                    self._bytes_skipped += nb
                    self.scheduler.record_skip()
                bkt.reset()
        return reduced

    def finish_reduces(self):
        """Wait for async handles. Ref: Megatron finish_grad_sync pattern."""
        for h, dst, fp32 in self._pending:
            if h is not None:
                h.wait()
            if dst is not None and fp32 is not None:
                dst.copy_(fp32)
        self._pending.clear()

    def write_grads_back(self):
        for tier in ('x', 'u', 'v'):
            for bkt in self._buckets[tier]:
                bkt.copy_grad_out()

    def maybe_allreduce(self, tensor, tier='x'):
        """Single-tensor path for backward compat."""
        import torch.distributed as dist
        should = self._should_sync(tier)
        nbytes = tensor.numel() * tensor.element_size()
        if should:
            if dist.is_initialized():
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=self.group)
            self._bytes_sent += nbytes
            self._op_count += 1
            self.scheduler.record_sync(tier)
            return True
        self._bytes_skipped += nbytes
        self.scheduler.record_skip()
        return False

    def grad_norms_per_tier(self, norm_type=2.0):
        """Per-tier gradient norms.
        Ref: neuronx-distributed zero_redundancy_optimizer.py:71
        Ref: Megatron clip_grads.py:55 — get_grad_norm_fp32"""
        import math
        out = {}
        for t in ('x', 'u', 'v'):
            sq = self._tier_norm_sq[t]
            out[t] = math.sqrt(sq) if norm_type == 2.0 else sq ** (1.0 / norm_type)
        return out

    def reset_norms(self):
        self._tier_norm_sq = {'x': 0.0, 'u': 0.0, 'v': 0.0}

    def stats(self):
        total = self._bytes_sent + self._bytes_skipped
        return {
            'bytes_sent': self._bytes_sent,
            'bytes_skipped': self._bytes_skipped,
            'ops': self._op_count,
            'reduction': round(total / max(1, self._bytes_sent), 2),
            'buckets': {t: len(bs) for t, bs in self._buckets.items()},
        }


class DeslocProfiler:
    """Collects per-step timing and comm data for NKI-FA export.

    Ref: NKI-FA draw_plot.py — parse_data() expects structured blocks.
    Ref: flash-attention benchmark_attn.py — time_fwd()/flops() pattern.
    """

    def __init__(self):
        self._step_data = []
        self._comm_data = []
        self._current_step = {}

    def begin_step(self, step):
        import time
        self._current_step = {
            'step': step,
            't_start_ns': time.monotonic_ns(),
        }

    def end_step(self, loss=None, lr=None):
        import time
        if not self._current_step:
            return
        self._current_step['t_end_ns'] = time.monotonic_ns()
        dt_ms = (self._current_step['t_end_ns'] - self._current_step['t_start_ns']) / 1e6
        self._current_step['step_time_ms'] = round(dt_ms, 4)
        if loss is not None:
            self._current_step['loss'] = round(float(loss), 6)
        if lr is not None:
            self._current_step['lr'] = float(lr)
        self._step_data.append(self._current_step)
        self._current_step = {}

    def record_comm(self, tier, bytes_sent, synced):
        self._comm_data.append({
            'step': len(self._step_data),
            'tier': tier,
            'bytes': bytes_sent,
            'synced': synced,
        })

    def export_nkifa(self, path, config_str=""):
        """Write NKI-FA format log file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(f"### {config_str} ###\n")
            for sd in self._step_data:
                f.write(f"step {sd['step']}: "
                        f"loss={sd.get('loss', 0):.6f}, "
                        f"time_ms={sd.get('step_time_ms', 0):.4f}, "
                        f"lr={sd.get('lr', 0):.6e}\n")
            f.write("\n")
            synced = sum(1 for c in self._comm_data if c['synced'])
            skipped = sum(1 for c in self._comm_data if not c['synced'])
            f.write(f"total_comm_ops: {synced}\n")
            f.write(f"skipped_comm_ops: {skipped}\n")
            f.write(f"comm_reduction: {round((synced + skipped) / max(1, synced), 2)}\n")

    def get_loss_curve(self):
        return [sd.get('loss', 0) for sd in self._step_data]

    def get_step_times(self):
        return [sd.get('step_time_ms', 0) for sd in self._step_data]

    def state_dict(self):
        return {'steps': self._step_data, 'comms': self._comm_data}

    def load_state_dict(self, sd):
        self._step_data = sd.get('steps', [])
        self._comm_data = sd.get('comms', [])


def init_desloc_scheduler(Kx=1, Ku=3, Kv=6, warmup_steps=512, group=None):
    """Initialize global DES-LOC scheduler. Called from engine.desloc_init_scheduler()."""
    global _desloc_scheduler_instance, _desloc_tiered_ar_instance, _desloc_profiler_instance
    _desloc_scheduler_instance = DeslocScheduler(Kx=Kx, Ku=Ku, Kv=Kv,
                                                  warmup_steps=warmup_steps, group=group)
    _desloc_tiered_ar_instance = DeslocTieredAllReduce(_desloc_scheduler_instance, group=group)
    _desloc_profiler_instance = DeslocProfiler()


def get_desloc_scheduler():
    """Get global DES-LOC scheduler instance."""
    return _desloc_scheduler_instance


def get_desloc_tiered_ar():
    """Get global tiered AllReduce instance."""
    return _desloc_tiered_ar_instance


def get_desloc_profiler():
    """Get global DES-LOC profiler instance."""
    return _desloc_profiler_instance
# --- End M332 ---