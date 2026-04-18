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


# ═══════════════════════════════════════════════════════════════
# DES-LOC Communication Schedule & Topology (M191)
# ═══════════════════════════════════════════════════════════════
import time as _m191_time
import os as _m191_os


class DESLOCCommSchedule:
    """Communication schedule implementing Algorithm 1 sync logic.

    Manages the independent synchronization periods for:
    - Parameters (Kx): controls gradient allreduce frequency
    - First moment / momentum (Ku): controls exp_avg sync
    - Second moment (Kv): controls exp_avg_sq sync

    Section 3 (Theorem 1): convergence guaranteed when
    px = 1/Kx, pu = 1/Ku, pv = 1/Kv.

    The schedule supports both deterministic (mod Kx) and
    probabilistic (Bernoulli with px=1/Kx) sync modes.
    """

    def __init__(self, Kx=32, Ku=96, Kv=192,
                 probabilistic=False, warmup_steps=0):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.probabilistic = probabilistic
        self.warmup_steps = warmup_steps
        self.step = 0
        self._warmup_complete = False

    def should_sync_x(self, step=None):
        """Check if parameters should be synced.

        During warmup: always sync (Kx=1 behavior).
        After warmup: sync every Kx steps.
        """
        s = step if step is not None else self.step
        if s < self.warmup_steps:
            return True  # full sync during warmup
        if self.probabilistic:
            # Bernoulli with px = 1/Kx
            import torch
            return torch.rand(1).item() < (1.0 / self.Kx)
        return s % self.Kx == 0

    def should_sync_u(self, step=None):
        """Check if first moment should be synced."""
        s = step if step is not None else self.step
        if s < self.warmup_steps:
            return True
        if self.probabilistic:
            import torch
            return torch.rand(1).item() < (1.0 / self.Ku)
        return s % self.Ku == 0

    def should_sync_v(self, step=None):
        """Check if second moment should be synced."""
        s = step if step is not None else self.step
        if s < self.warmup_steps:
            return True
        if self.probabilistic:
            import torch
            return torch.rand(1).item() < (1.0 / self.Kv)
        return s % self.Kv == 0

    def advance(self):
        """Advance the schedule by one step."""
        self.step += 1
        if not self._warmup_complete and self.step >= self.warmup_steps:
            self._warmup_complete = True

    def is_warmup(self):
        """Check if still in warmup phase."""
        return self.step < self.warmup_steps

    def get_state(self):
        return {
            'step': self.step,
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'probabilistic': self.probabilistic,
            'warmup_steps': self.warmup_steps,
        }

    def load_state(self, state):
        self.step = state.get('step', 0)
        self._warmup_complete = self.step >= self.warmup_steps


class DESLOCTopologyDetector:
    """Detect GPU/network topology for DES-LOC optimization.

    Nick Joseph: "My colleague ran a clustering algorithm to
    infer which chips were in which data centers, because we
    suspected different cross-DC network latencies were causing
    training bottlenecks."

    This detector identifies:
    - Intra-node vs inter-node workers
    - NVLink vs PCIe vs network boundaries
    - Bandwidth tiers for adaptive Kx selection
    """

    def __init__(self):
        self.topology = {}
        self.bandwidth_tiers = []

    def detect(self):
        """Detect the current topology."""
        import socket
        hostname = socket.gethostname()
        self.topology['hostname'] = hostname

        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.topology['local_gpu_count'] = gpu_count
                self.topology['gpus'] = []
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.topology['gpus'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': round(props.total_mem / (1024**3), 1),
                        'sm_count': props.multi_processor_count,
                        'compute_capability': f"{props.major}.{props.minor}",
                    })
        except Exception:
            self.topology['local_gpu_count'] = 0
            self.topology['gpus'] = []

        # Check for NVLink (heuristic: multiple GPUs on same node)
        if self.topology.get('local_gpu_count', 0) > 1:
            self.topology['likely_nvlink'] = True
        else:
            self.topology['likely_nvlink'] = False

        return self.topology

    def recommend_Kx(self, bandwidth_gbps=None):
        """Recommend Kx based on detected topology.

        Higher bandwidth → can afford more frequent sync (lower Kx).
        Lower bandwidth → sync less often (higher Kx).

        Heuristic from Section 5.4:
        - NVLink (400+ GB/s): Kx=8 is fine
        - PCIe (32 GB/s): Kx=32 recommended
        - Network (10-25 Gbps): Kx=64-128 recommended
        """
        if bandwidth_gbps is None:
            if self.topology.get('likely_nvlink'):
                bandwidth_gbps = 400.0
            else:
                bandwidth_gbps = 25.0

        if bandwidth_gbps >= 200:
            return {'Kx': 8, 'Ku': 24, 'Kv': 48, 'tier': 'nvlink'}
        elif bandwidth_gbps >= 50:
            return {'Kx': 16, 'Ku': 48, 'Kv': 96, 'tier': 'pcie'}
        elif bandwidth_gbps >= 10:
            return {'Kx': 32, 'Ku': 96, 'Kv': 192, 'tier': 'network'}
        else:
            return {'Kx': 64, 'Ku': 192, 'Kv': 384, 'tier': 'slow_network'}


class DESLOCBandwidthProber:
    """Probe actual network bandwidth between workers.

    Sends test allreduce operations to measure real bandwidth,
    then uses the measurement to calibrate Kx.
    """

    def __init__(self, world_size=1, probe_sizes=None):
        self.world_size = world_size
        self.probe_sizes = probe_sizes or [1024, 65536, 1048576, 16777216]
        self.results = []

    def probe(self, group=None):
        """Run bandwidth probes and return measurements."""
        try:
            import torch
            import deepspeed.comm as dist
            if not dist.is_initialized() or self.world_size < 2:
                return {'bandwidth_gbps': 0, 'latency_us': 0}

            for size in self.probe_sizes:
                tensor = torch.zeros(size // 4, dtype=torch.float32,
                                     device=torch.cuda.current_device())
                # Warmup
                dist.all_reduce(tensor, group=group)
                torch.cuda.synchronize()

                # Timed run
                start = _m191_time.monotonic()
                for _ in range(5):
                    dist.all_reduce(tensor, group=group)
                torch.cuda.synchronize()
                elapsed = (_m191_time.monotonic() - start) / 5

                bw_gbps = (size * 2 * (self.world_size - 1) / self.world_size) / elapsed / 1e9 * 8
                self.results.append({
                    'size_bytes': size,
                    'time_sec': round(elapsed, 6),
                    'bandwidth_gbps': round(bw_gbps, 2),
                })

            # Return the measurement for largest probe
            best = max(self.results, key=lambda x: x['bandwidth_gbps'])
            return {
                'bandwidth_gbps': best['bandwidth_gbps'],
                'latency_us': round(self.results[0]['time_sec'] * 1e6, 1) if self.results else 0,
                'all_probes': self.results,
            }
        except Exception as e:
            return {'bandwidth_gbps': 0, 'error': str(e)}


# End M191
