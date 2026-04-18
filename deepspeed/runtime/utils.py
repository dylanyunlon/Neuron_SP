# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/Megatron

Helper functions and classes from multiple sources.
"""

from collections.abc import Iterable
import os
import psutil
import gc
from math import sqrt

from numpy import prod

import torch
from torch.nn import functional as F
try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
from typing import Union, List, Dict, Sequence
from deepspeed import comm as dist
from deepspeed.moe.utils import is_moe_param
from deepspeed.utils import groups, logger
from deepspeed.utils.bwc import (bwc_tensor_model_parallel_rank, bwc_pipeline_parallel_world_size,
                                 bwc_pipeline_parallel_group)
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.policy import transpose

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved


class DummyOptim():
    """
    Dummy optimizer presents model parameters as a param group, this is
    primarily used to allow ZeRO-3 without an optimizer
    """

    def __init__(self, params):
        self.param_groups = []
        self.param_groups.append({'params': params})


def filter_empty_parameters(params):
    """Filter out empty parameters (numel == 0) from optimizer params.

    This is useful for optimizers that perform operations like division by numel,
    which would produce NaNs for empty parameters.

    Args:
        params: Either a list/tuple of Parameters, or a list of parameter group dicts
                (each dict has 'params' key with list of Parameters)

    Returns:
        Filtered params in the same format as input (list of Parameters or list of dicts)
    """
    if not isinstance(params, (list, tuple)) or len(params) == 0:
        return params

    # Check if first element is a dict (parameter groups) or a Parameter
    if isinstance(params[0], dict):
        # params is a list of parameter group dicts
        filtered_params = []
        for param_group in params:
            filtered_group = {}
            trainable_params = []
            for key, value in param_group.items():
                if key == 'params':
                    # Filter out empty parameters
                    trainable_params = [p for p in value if p.numel() > 0]
                else:
                    filtered_group[key] = value
            # Only add group if it has non-empty parameters
            if len(trainable_params) > 0:
                filtered_group['params'] = trainable_params
                filtered_params.append(filtered_group)
        return filtered_params
    else:
        # params is a list of Parameters
        return [p for p in params if p.numel() > 0]


graph_cache = {}


def graph_process(replay_first_step, func, *args, **kwargs):
    # `func` should only contain operations on the GPU
    # Please ensure that the memory address of the data required by 'func' remains constant
    if func.__name__ not in graph_cache:
        cuda_stream = get_accelerator().Stream()
        cuda_stream.wait_stream(get_accelerator().current_stream())
        with get_accelerator().stream(cuda_stream):
            func(*args, **kwargs)
        get_accelerator().current_stream().wait_stream(cuda_stream)
        graph_cache[func.__name__] = get_accelerator().create_graph()
        with get_accelerator().capture_to_graph(graph_cache[func.__name__]):
            func(*args, **kwargs)
        if replay_first_step:
            get_accelerator().replay_graph(graph_cache[func.__name__])
    else:
        get_accelerator().replay_graph(graph_cache[func.__name__])


def noop_decorator(func):
    return func


class noop_context(object):

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def ensure_directory_exists(filename):
    """Create the directory path to ``filename`` if it does not already exist.

    Args:
        filename (str): A file path.
    """
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


def set_random_seed(seed):
    """Set the random seed for common PRNGs used during training.

    DES-LOC rewrite: removed numpy.random dependency.
    Uses only torch.manual_seed and Python random.
    Ref: Mandatory rule — zero numpy.random in DES-LOC code.

    Args:
        seed (int): the seed to use
    """
    import random
    random.seed(seed)
    # torch.manual_seed accepts only 64-bit integers
    torch.manual_seed(seed % (2**63))
    # DES-LOC: also seed the accelerator for reproducible GPU ops
    get_accelerator().manual_seed(seed % (2**63))


def is_model_parallel_parameter(p) -> bool:
    if hasattr(p, 'model_parallel') and p.model_parallel:
        return True

    if hasattr(p, 'tensor_model_parallel') and p.tensor_model_parallel:
        return True

    return False


def copy_to_device(item, device, criterion_func):
    """
    Return a copy of tensor on specified device.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to copy or (possibly nested) container of tensors to copy.
        device: target device
        criterion_func: Function to restrict copy operation to items meet criterion

    Returns:
        None
    """
    if criterion_func(item):
        return item.to(device)
    elif isinstance(item, list):
        return [copy_to_device(v, device, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([copy_to_device(v, device, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: copy_to_device(v, device, criterion_func) for k, v in item.items()}
    else:
        return item


def move_to_device(item, device, criterion_func=None):
    """
    Move tensor on to specified device by changing the storage.
    Works on individual tensors, and tensors contained/nested in lists, tuples, and dicts.
    Parameters:
        item: tensor to move or (possibly nested) container of tensors to move.
        device: target device
        criterion_func: Function to restrict move operation to items meet criterion, defaults to `None` which is an equivalent to always move

    Returns:
        None
    """
    if (criterion_func is not None and criterion_func(item)):
        device_copy = item.to(device)
        item.data = device_copy.data
        return item
    elif isinstance(item, list):
        return [move_to_device(v, device, criterion_func) for v in item]
    elif isinstance(item, tuple):
        return tuple([move_to_device(v, device, criterion_func) for v in item])
    elif isinstance(item, dict):
        return {k: move_to_device(v, device, criterion_func) for k, v in item.items()}
    else:
        return item.to(device)


def get_norm_with_moe_layers_fast(all_groups_norm, group):
    # This implementation standardizes the grad_norm across ranks. A more precise implementation can be found in 'get_norm_with_moe_layers'.
    # Need to allreduce (avg) the norms across different ranks because moe params will not be synced during allreduce
    scaled_norm = all_groups_norm * 1.0 / float(dist.get_world_size(group=group))
    scaled_norm_tensor = torch.tensor(scaled_norm, device=get_accelerator().current_device_name(), dtype=torch.float)
    dist.all_reduce(scaled_norm_tensor, group=group)
    all_groups_norm = scaled_norm_tensor.item()
    #print(f"old = {all_groups_norm_old} and new = {all_groups_norm} at rank: {deepspeed.comm.get_rank()}")
    return all_groups_norm


class CheckOverflow(object):
    '''Checks for overflow in gradient across parallel process'''

    def __init__(self, param_groups=None, mpu=None, zero_reduce_scatter=False, deepspeed=None):
        self.mpu = mpu
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        self.deepspeed = deepspeed
        self.has_moe_params = False
        # DES-LOC: overflow detection is ALWAYS synchronous, never gated by Kx.
        # This allreduce is safety-critical; skipping it could mask NaN/Inf
        # and corrupt all workers' parameters at the next sync boundary.
        # Ref: M092-M106 rule — MoE capacity allreduces also never gated.
        self.desloc_overflow_force_sync = True
        if param_groups:
            for group in param_groups:
                for param in group:
                    self.params.append(param)
                    if is_moe_param(param):
                        self.has_moe_params = True

    def check_using_norm(self, norm_group, reduce_overflow=True):
        # TODO: I don't think reduce_overflow is needed if mpu is None
        overflow = -1 in norm_group
        overflow_gpu = get_accelerator().FloatTensor([overflow])
        if self.has_moe_params:
            # In this case, we need to do an all_reduce across
            # the expert_parallel_group, so that if there was
            # an overflow due to expert weights, we detect it

            # Only need to check groups.get_largest_expert_parallel_group()
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        if self.mpu is not None:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        elif reduce_overflow:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX)
            dist.barrier()
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        has_moe_params = False
        if param_groups is None:
            params = self.params
            has_moe_params = self.has_moe_params
        else:
            assert param_groups is not None, \
                "self.params and param_groups both cannot be none"

            for group in param_groups:
                for param in group:
                    params.append(param)
                    if is_moe_param(param):
                        has_moe_params = True

        return self.has_overflow(params, has_moe_params=has_moe_params)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params, has_moe_params=None):
        if has_moe_params is None:
            has_moe_params = self.has_moe_params
        overflow = self.has_overflow_serial(params)
        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        overflow_gpu = get_accelerator().ByteTensor([overflow])
        # deepspeed.comm.all_reduce(overflow_gpu,
        #                             op=deepspeed.comm.ReduceOp.MAX,
        #                             group=mpu.get_model_parallel_group())
        if has_moe_params:
            # All reduce this across expert_parallel_group, so that if an expert
            # overflows, we detect it here
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=groups._get_max_expert_parallel_group())
        if self.zero_reduce_scatter:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=dist.get_world_group())
        elif self.mpu is not None:
            if self.deepspeed is not None:
                using_pipeline = hasattr(self.deepspeed, 'pipeline_enable_backward_allreduce')
                if (using_pipeline and self.deepspeed.pipeline_enable_backward_allreduce
                        is False) or (not using_pipeline and self.deepspeed.enable_backward_allreduce is False):
                    dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_data_parallel_group())
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=self.mpu.get_model_parallel_group())
        elif self.deepspeed is not None and self.deepspeed.enable_backward_allreduce is False:
            dist.all_reduce(overflow_gpu, op=dist.ReduceOp.MAX, group=dist.get_world_group())

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, i):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False


def _handle_overflow(cpu_sum, x, i):
    import math
    rank = dist.get_rank()
    if rank == 0:
        t_i = -1
        for v_i, v in enumerate(x.data.contiguous().view(-1)):
            if not math.isfinite(float(v)):
                t_i = v_i
                break
        logger.info(f"rank {rank} detected overflow {cpu_sum} in tensor {i}:{t_i} shape {x.shape}")


def get_global_norm(norm_list):
    """ Compute total from a list of norms
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm**2.0
    # logger.info(f'norm_list = {norm_list} global = {sqrt(total_norm)}')
    return sqrt(total_norm)


def clip_grad_norm_(parameters, max_norm, norm_type=2, mpu=None):
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    all_norms = []
    if norm_type == inf:
        for p in parameters:
            all_norms.append(p.grad.data.abs().max().float())
        total_norm = torch.stack(all_norms).max()
        total_norm = total_norm.to(get_accelerator().current_device_name())
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
    else:
        total_norm = 0
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank() == 0) or is_model_parallel_parameter(p):
                    param_norm = p.grad.data.detach().float().norm(norm_type)
                    all_norms.append(param_norm)
            else:
                param_norm = p.grad.data.detach().float().norm(norm_type)
                all_norms.append(param_norm)
        if len(all_norms) > 0:
            total_norm = torch.stack(all_norms).square().sum().float()
        else:
            total_norm = get_accelerator().FloatTensor([0.0])
        total_norm = total_norm.to(get_accelerator().current_device_name())
        # Sum across all model parallel GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm.pow(1. / norm_type)

    # Need to average total_norm across different GPUs due to the presence of moe params
    pg = groups._get_data_parallel_group()
    scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))
    scaled_norm_tensor = scaled_norm

    dist.all_reduce(scaled_norm_tensor, group=pg)
    total_norm = scaled_norm_tensor
    total_norm = total_norm.to(parameters[0].device)

    max_norm = torch.tensor([float(max_norm)], device=total_norm.device)
    clip_coef = max_norm / (total_norm + 1e-6)
    tmp_tensor = torch.tensor([1.0], device=clip_coef.device)
    clip_coef = torch.min(tmp_tensor, clip_coef)
    for p in parameters:
        p.grad.data.mul_(clip_coef)
    return total_norm


def get_flattened_grad_norm(parameters, norm_type=2, mpu=None, grad_norm_mask=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        grad_norm_mask (List[Tensor]): A list of Tensor, where
            each Tensor is a 2D Tensor containing ranges of [start_index, end_index].
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        for idx, p in enumerate(parameters):
            # Use grad_norm_mask to avoid redundant computation of flattened gradient norm
            if grad_norm_mask is not None and len(grad_norm_mask[idx]) > 0:

                # A loop-free implementation to create a mask tensor based on a range list
                # which is logically equivalent to the following implementation.
                # # mask_tensor_ = torch.zeros_like(p, device=p.device, dtype=bool)
                # # for mask_idx in grad_norm_mask[idx]:
                # #   mask_tensor_[mask_idx[0]:mask_idx[1]] = True
                cum_sum_pairs = torch.tensor([1, -1], device=get_accelerator().current_device_name(),
                                             dtype=p.dtype).repeat(grad_norm_mask[idx].shape[0], 1)
                mask_tensor = torch.zeros(p.shape[0] + 1,
                                          device=get_accelerator().current_device_name(),
                                          dtype=p.dtype)
                mask_tensor = mask_tensor.scatter_(0, grad_norm_mask[idx].view(-1),
                                                   cum_sum_pairs.view(-1)).cumsum(0).bool()[:-1]

                param_norm = torch.masked_fill(p.grad.data, mask_tensor, 0).float().norm(norm_type)

            else:
                param_norm = p.grad.data.float().norm(norm_type)
            total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def get_grad_zeros(parameters, mpu=None):
    """Compute the number of grads with zero values.

    This is adapted from get_grad_norm

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized

    Returns:
        Total number of params with zero values (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_zeros = 0.
    tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
    for p in parameters:
        # Pipeline parallelism may replicate parameters. Avoid multi-counting.
        if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
            continue

        # Filter to avoid over-counting replicated tensors from tensor
        # model parallelism
        if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
            continue

        count_zeros = p.grad.numel() - torch.count_nonzero(p.grad)
        total_zeros += count_zeros.item()

    # Sum across all model parallel GPUs.
    total_zeros_cuda = get_accelerator().FloatTensor([float(total_zeros)])
    if mpu is not None:
        dist.all_reduce(total_zeros_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
    total_zeros = total_zeros_cuda[0].item()

    return total_zeros


def get_weight_norm(parameters, norm_type=2, mpu=None):
    """Get norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
        -1 if the norm value is NaN or Inf.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for p in parameters:
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if (tensor_mp_rank > 0) and not is_model_parallel_parameter(p):
                continue

            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = get_accelerator().FloatTensor([float(total_norm)])
        if mpu is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float('inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm


def prefix_sum_inc(weights):
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_


def partition_uniform(num_items, num_parts):
    import numpy
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = num_items // num_parts
    residual = num_items - (chunksize * num_parts)

    parts = numpy.arange(0, (num_parts + 1) * chunksize, chunksize)

    for i in range(residual):
        parts[i + 1:] += 1
    parts = parts.tolist()

    return parts


def partition_balanced(weights, num_parts):
    """
    use dynamic programming solve `The Linear Partition Problem`.
    see https://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
    """
    import numpy as np
    n = len(weights)
    m = num_parts

    if n <= m:
        return partition_uniform(n, m)

    dp_max = np.full((n + 1, m + 1), np.inf)
    dp_min = np.full((n + 1, m + 1), np.inf)
    dp_cost = np.full((n + 1, m + 1), np.inf)
    position = np.zeros((n + 1, m + 1), dtype=int)
    prefix_sum = np.zeros((n + 1))
    prefix_sum[1:] = np.cumsum(weights)

    dp_max[0, 0] = 0
    dp_cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, min(i, m) + 1):
            for k in range(i):
                max_sum = max(dp_max[k, j - 1], prefix_sum[i] - prefix_sum[k])
                min_sum = min(dp_min[k, j - 1], prefix_sum[i] - prefix_sum[k])
                cost = max_sum - min_sum
                if dp_cost[i, j] >= cost:
                    dp_cost[i, j] = cost
                    dp_max[i, j] = max_sum
                    dp_min[i, j] = min_sum
                    position[i, j] = k

    parts = [n]
    for i in reversed(range(1, m + 1)):
        parts.append(position[parts[-1], i])
    parts.reverse()

    return parts


class PartitionedTensor:

    def __init__(self, tensor, group, partition_meta=None):
        super().__init__()

        self.group = group
        self.num_parts = dist.get_world_size(group=self.group)
        self.rank = dist.get_rank(group=self.group)
        self.orig_size = list(tensor.size())
        self.orig_device = tensor.device
        self.local_data, self.partition = self._partition_tensor(tensor)
        self.even_split = tensor.numel() % self.num_parts == 0

    @classmethod
    def from_meta(cls, meta, local_part, group, device=get_accelerator().device_name()):
        assert meta.dtype == torch.long
        dummy = torch.ones(dist.get_world_size(group=group))
        part_obj = cls(tensor=dummy, group=group)

        meta = meta.tolist()

        # [N, list0, ..., listN-1]
        part_obj.orig_size = meta[1:(1 + meta[0])]
        meta = meta[1 + meta[0]:]

        part_obj.orig_device = device
        part_obj.local_data = local_part.detach()

        part_obj.group = group

        # Partition is encoded like the rowptr of a CSR matrix:
        # [num_parts, rank, 0, part_1, ..., part_num_parts]
        # TODO: support shuffle between different partition granularities
        assert part_obj.num_parts == meta[0]
        assert part_obj.rank == meta[1]
        part_obj.partition = meta[2:]  # length num_parts+1

        return part_obj

    def _partition_tensor(self, tensor):
        partition = partition_uniform(num_items=tensor.numel(), num_parts=self.num_parts)
        start = partition[self.rank]
        length = partition[self.rank + 1] - start
        tensor_part = tensor.detach().contiguous().view(-1).narrow(0, start=start, length=length).clone()

        return tensor_part, partition

    def full(self, device=None):
        if device is None:
            device = self.orig_device

        # Allocate the full tensor as a flat buffer.
        full_numel = prod(self.full_size())
        flat_tensor = torch.zeros([full_numel], dtype=self.local_data.dtype, device=device)
        if self.even_split:
            # Collect the full tensor
            dist.all_gather_into_tensor(flat_tensor, self.local_data, group=self.group)
        else:
            for part_id in range(self.num_parts):
                part_size = self.partition[part_id + 1] - self.partition[part_id]
                buf = flat_tensor.narrow(0, start=self.partition[part_id], length=part_size)
                if part_id == self.rank:
                    buf.copy_(self.local_data)
                dist.broadcast(buf, part_id, self.group)
        return flat_tensor.view(self.full_size()).clone().detach()

    def to_meta(self):
        """Returns a torch.LongTensor that encodes partitioning information.

        Can be used along with ``data()`` to serialize a ``PartitionedTensor`` for
        communication.

        Returns:
            torch.LongTensor: a tensor encoding the meta-information for the partitioning
        """
        meta = []
        meta.append(len(self.orig_size))
        meta += list(self.orig_size)
        meta.append(self.num_parts)
        meta.append(self.rank)
        meta += self.partition
        return torch.LongTensor(data=meta).to(self.orig_device)

    def data(self):
        return self.local_data

    def local_size(self):
        return self.local_data.size()

    def full_size(self):
        return self.orig_size


mem_alloced = 0
mem_cached = 0


def memory_status(msg, print_rank=-1, reset_max=False):
    global mem_alloced, mem_cached

    rank = dist.get_rank()
    if print_rank != -1 and rank != print_rank:
        return

    get_accelerator().synchronize()

    if reset_max:
        get_accelerator().reset_max_memory_cached()
        get_accelerator().reset_max_memory_allocated()

    new_alloced = get_accelerator().memory_allocated()
    new_cached = get_accelerator().memory_cached()

    delta_alloced = new_alloced - mem_alloced
    delta_cached = new_cached - mem_cached

    mem_cached = new_cached
    mem_alloced = new_alloced

    max_alloced = get_accelerator().max_memory_allocated()
    max_cached = get_accelerator().max_memory_cached()

    # convert to GB for printing
    new_alloced /= 1024**3
    new_cached /= 1024**3
    delta_alloced /= 1024**3
    delta_cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f'RANK={rank} MEMSTATS', msg, f'device={get_accelerator().current_device_name()} '
        f'current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) '
        f'current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)')


def get_ma_status():
    if dist.is_initialized() and not dist.get_rank() == 0:
        return 0
    return get_accelerator().memory_allocated()


def empty_cache():
    get_accelerator().empty_cache()
    get_accelerator().reset_peak_memory_stats()


def see_memory_usage(message, force=False):
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(message)
    print(f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    print(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call
        args (tuple, optional): args to ``base``
        kwargs (dict, optional): kwargs supplied to ``base``

    Returns:
        str: A string representation of base(*args, **kwargs)
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name


def get_only_unique_item(items):
    item_set = set(items)
    if len(item_set) != 1:
        raise RuntimeError(f"expected there to be only one unique element in {items}")
    unique_item, = item_set

    return unique_item


def mask_nan_or_inf_with_val_inplace(input, device=None, val=-1.):
    norm_is_inf = input.isinf()
    norm_is_nan = input.isnan()
    inf_or_nan = norm_is_nan.logical_or(norm_is_inf)
    err = torch.tensor(-1.0, device=device, dtype=torch.float)
    input.masked_fill_(inf_or_nan, err)


def get_global_norm_of_tensors(input_tensors, norm_type=2, mpu=None, use_graph=False, moe_ep_group=None):
    """Get norm of an iterable of tensors.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Taken from Nvidia Megatron.

    Arguments:
        input_tensors (Iterable[Tensor]): an iterable of Tensors will have norm computed
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    assert isinstance(input_tensors, Iterable), f'expected Iterable type not {type(input_tensors)}'
    assert all([torch.is_tensor(t) for t in input_tensors]), 'expected list of only tensors'

    norm_type = float(norm_type)
    all_norms = []
    if norm_type == inf:
        for t in input_tensors:
            all_norms.append(t.data.abs().max().float())
        total_norm = torch.stack(all_norms).max()
        device_total_norm = total_norm.to(get_accelerator().current_device_name())
        # Max across model parallel
        if mpu is not None:
            # For MoE grads, max over model parallel only if MoE-TP is enabled
            if moe_ep_group is None or groups._get_expert_model_parallel_world_size() > 1:
                dist.all_reduce(device_total_norm, op=dist.ReduceOp.MAX, group=mpu.get_model_parallel_group())
            # If MoE grads and MoE-TP disabled, max over pipeline parallel
            elif bwc_pipeline_parallel_world_size(mpu) > 1:
                dist.all_reduce(device_total_norm, op=dist.ReduceOp.MAX, group=bwc_pipeline_parallel_group(mpu))

        # MoE grads: max across expert parallel group
        if moe_ep_group is not None:
            dist.all_reduce(device_total_norm, op=dist.ReduceOp.MAX, group=moe_ep_group)
        total_norm = device_total_norm.to(input_tensors[0].device)
    else:

        if 'norm_tensors_compute_buffer' not in graph_cache or len(
                graph_cache['norm_tensors_compute_buffer']) != len(input_tensors):
            graph_cache['norm_tensors_compute_buffer'] = [
                torch.empty([], dtype=torch.float, device=get_accelerator().current_device_name())
                for t in input_tensors
            ]
        compute_buffer = graph_cache['norm_tensors_compute_buffer']

        def _norm_tensors(tensor_list, _compute_buffer, _norm_type):
            for i, t in enumerate(tensor_list):
                _compute_buffer[i].data.copy_(t.data.float().norm(_norm_type)**_norm_type)
                if i != 0:
                    _compute_buffer[0].data.add_(_compute_buffer[i].data)

        if use_graph:
            graph_process(False, _norm_tensors, input_tensors, compute_buffer, norm_type)
        else:
            _norm_tensors(input_tensors, compute_buffer, norm_type)

        device_total_norm = compute_buffer[0].float().detach()

        # Sum across model parallel
        if mpu is not None:
            # For MoE grads, sum over model parallel only if MoE-TP is enabled
            if moe_ep_group is None or groups._get_expert_model_parallel_world_size() > 1:
                dist.all_reduce(device_total_norm, op=dist.ReduceOp.SUM, group=mpu.get_model_parallel_group())
            # If MoE grads and MoE-TP disabled, sum over pipeline parallel
            elif bwc_pipeline_parallel_world_size(mpu) > 1:
                dist.all_reduce(device_total_norm, op=dist.ReduceOp.SUM, group=bwc_pipeline_parallel_group(mpu))

        # MoE grads: sum across expert parallel group
        if moe_ep_group is not None:
            dist.all_reduce(device_total_norm, op=dist.ReduceOp.SUM, group=moe_ep_group)
        total_norm = device_total_norm.to(input_tensors[0].device).pow(1. / norm_type)

    mask_nan_or_inf_with_val_inplace(total_norm, device=total_norm.device)

    return total_norm


def clip_tensors_by_global_norm(input_tensors, max_norm=1.0, global_norm=None, mpu=None, eps=1e-6, use_graph=False):
    """Clip list of tensors by global norm.
    Args:
        input_tensors: List of tensors to be clipped
        global_norm (float, optional): Precomputed norm. Defaults to None.
        mpu (optional): model parallelism unit. Defaults to None.
        eps (float, optional): epsilon value added to grad norm. Defaults to 1e-6
    Returns:
        float: the global norm
    """
    if global_norm is None:
        global_norm = get_global_norm_of_tensors(input_tensors, mpu=mpu, use_graph=use_graph)
    clip_coef = max_norm / (global_norm + eps)
    if clip_coef < 1:
        if use_graph:

            def clip_tensors(_tensor_list, _clip_coef_tensor):
                for t in _tensor_list:
                    t.detach().mul_(_clip_coef_tensor)

            if 'clip_coef_tensor' not in graph_cache:
                # Alloc memory
                graph_cache['clip_coef_tensor'] = torch.tensor(clip_coef,
                                                               dtype=torch.float32).to(get_accelerator().device_name())
            clip_coef_tensor = graph_cache['clip_coef_tensor']
            clip_coef_tensor.copy_(torch.tensor(clip_coef, dtype=torch.float32))
            graph_process(False, clip_tensors, input_tensors, clip_coef_tensor)

        else:
            for t in input_tensors:
                t.detach().mul_(clip_coef)
    return global_norm


def align_dense_tensors(tensor_list, alignment):
    num_elements = sum(t.numel() for t in tensor_list)
    remaining = num_elements % alignment

    if remaining:
        elements_to_add = alignment - remaining
        pad_tensor = torch.zeros(elements_to_add, device=tensor_list[0].device, dtype=tensor_list[0].dtype)
        padded_tensor_list = tensor_list + [pad_tensor]
    else:
        padded_tensor_list = tensor_list

    return padded_tensor_list


def all_gather_into_tensor_dp_groups(groups_flat, partitioned_param_groups, dp_process_group):
    for group_id, (group_flat, partitioned_params) in enumerate(zip(groups_flat, partitioned_param_groups)):
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])
        if dp_world_size == 1:
            # no groups share optimizer states
            # pipeline parallel with bf16 will default call this even if dp size = 1.
            continue
        dist.all_gather_into_tensor(group_flat, partitioned_params[partition_id], dp_process_group[group_id])


def all_gather_dp_groups(groups_flat, partitioned_param_groups, dp_process_group, start_alignment_factor,
                         allgather_bucket_size):
    if dist.has_all_gather_into_tensor():
        return all_gather_into_tensor_dp_groups(groups_flat, partitioned_param_groups, dp_process_group)

    for group_id, partitioned_params in enumerate(partitioned_param_groups):
        # Sequential AllGather Best of both worlds
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])

        if dp_world_size == 1:
            # no groups share optimizer states
            # pipeline parallel with bf16 will default call this even if dp size = 1.
            continue
        num_shards = max(1, partitioned_params[partition_id].numel() * dp_world_size // allgather_bucket_size)

        shard_size = partitioned_params[partition_id].numel() // num_shards

        # Enforce nccl/rccl alignment of start location of each shard
        shard_size = shard_size - (shard_size % start_alignment_factor)

        num_elements = shard_size

        assert shard_size * num_shards <= partitioned_params[partition_id].numel()

        for shard_id in range(num_shards):

            if shard_id == (num_shards - 1):
                num_elements = partitioned_params[partition_id].numel() - shard_id * shard_size

            shard_list = []
            for dp_id in range(dp_world_size):
                curr_shard = partitioned_params[dp_id].narrow(0, shard_id * shard_size, num_elements).detach()
                shard_list.append(curr_shard)

            dist.all_gather(shard_list, shard_list[partition_id], dp_process_group[group_id])


def get_tensor_bytes(item):
    if torch.is_tensor(item):
        return item.numel() * item.element_size()
    elif isinstance(item, list):
        return sum([get_tensor_bytes(v) for v in item])
    elif isinstance(item, tuple):
        return sum([get_tensor_bytes(v) for v in item])
    elif isinstance(item, dict):
        return sum([get_tensor_bytes(v) for v in item.values()])
    else:
        return 0


def _get_folder_size(folder):
    size = 0
    for path, _, files in os.walk(folder):
        size += sum([os.path.getsize(os.path.join(path, f)) for f in files])
    return size


def get_checkpoint_folder_size(save_dir, tag, local_rank=None):
    if local_rank == 0:
        folder = os.path.join(save_dir, tag)
        size_tensor = torch.tensor(_get_folder_size(folder)).to(get_accelerator().device_name())
    else:
        size_tensor = torch.tensor(0).to(get_accelerator().device_name())

    dist.reduce(tensor=size_tensor, dst=0)
    return int(size_tensor)


class TLinear(torch.nn.Linear):

    def __init__(self, orig_layer, name=""):
        self.name = name
        super().__init__(orig_layer.weight.shape[1], orig_layer.weight.shape[0], bias=(orig_layer.bias is not None))
        self.weight.data = transpose(orig_layer.weight.data)
        self.bias = orig_layer.bias
        self._fwd_func = self._fwd_bias_add if self.bias is not None else self._fwd

    def _fwd(self, input):
        return F.linear(input, self.weight)

    def _fwd_bias_add(self, input):
        return F.linear(input, self.weight, bias=self.bias)

    def forward(self, input):
        return self._fwd_func(input)


def get_inactive_params(param_list):
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    return [param for param in param_list if (hasattr(param, 'ds_id') and \
                            param.ds_status == ZeroParamStatus.NOT_AVAILABLE)]


def get_norm_with_moe_layers(non_expert_norm, mpu, expert_tensors, norm_type=2):
    """ Compute the global norm with MoE experts

    Inputs:
    non_expert_norm (float) : the calculated norm of the non-expert params
    expert_tensors (Dict[ep_name, List[Tensor]): Dictionary of expert group name to list of grad tensors
    norm_type (int): the norm to use

    Returns:
        if norm is (-/+) inf, returns -1
        otherwise the global norm (float)
    """

    def to_tensor(v):
        return get_accelerator().FloatTensor(float(v)).detach()

    group_norms = [non_expert_norm]
    for exp_name, tensors in expert_tensors.items():
        group_norm = get_global_norm_of_tensors(input_tensors=tensors,
                                                mpu=mpu,
                                                norm_type=norm_type,
                                                use_graph=False,
                                                moe_ep_group=groups._get_expert_parallel_group(exp_name))
        group_norms.append(group_norm)

    # check if all norms are valid
    group_norms = torch.stack([to_tensor(norm) for norm in group_norms])
    if group_norms.eq(-1).any():
        return -1

    # combine norms
    if norm_type == inf:
        total_norm = group_norms.max().item()
    else:
        total_norm = group_norms.pow(norm_type).sum()
        total_norm = total_norm.item()**(1. / norm_type)
        if total_norm == float('inf') or total_norm == -float('inf'):
            total_norm = -1

    return total_norm


def _make_offload_state_key(key):
    return f"{key}_offload_buffer"


def offload_adam_states(optimizer, device, pin_memory: bool = False, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""

    def move_key(state, key):
        offload_buf_key = _make_offload_state_key(key)
        if offload_buf_key not in state:
            state[offload_buf_key] = torch.empty_like(state[key], device=device)
            if pin_memory:
                state[offload_buf_key] = get_accelerator().pin_memory(state[offload_buf_key])
        state[offload_buf_key].copy_(state[key], non_blocking=non_blocking)
        state[key].data = state[offload_buf_key]

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_key(state, "exp_avg_sq")


def reload_adam_states(optimizer, device, non_blocking: bool = False):
    """Move optimizer states to device. Note that this assumes the state structure of DeepSpeed Adam."""

    def move_back_key(state, key):
        state[key].data = state[_make_offload_state_key(key)].to(device, non_blocking=non_blocking)

    for _, state in optimizer.state.items():
        if "exp_avg" in state:
            move_back_key(state, "exp_avg")
        if "exp_avg_sq" in state:
            move_back_key(state, "exp_avg_sq")


def compare_tensors_in_structures(inputs1: Union[List, Dict], inputs2: Union[List, Dict]) -> bool:
    """
    Compare two lists or dictionaries for equality, including any tensors they may contain.

    Args:
        inputs1: First input, either a list or a dictionary.
        inputs2: Second input, either a list or a dictionary.

    Returns:
        True if inputs1 and inputs2 are equal; False otherwise.
    """
    if type(inputs1) != type(inputs2):  # Ensure types match
        return False

    if isinstance(inputs1, list) and isinstance(inputs2, list):
        if len(inputs1) != len(inputs2):
            return False
        for val1, val2 in zip(inputs1, inputs2):
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                val1 = val1.to(torch.device(get_accelerator().current_device_name()))
                val2 = val2.to(torch.device(get_accelerator().current_device_name()))
                if not torch.equal(val1, val2):
                    return False
            elif val1 != val2:
                return False
        return True

    elif isinstance(inputs1, dict) and isinstance(inputs2, dict):
        if inputs1.keys() != inputs2.keys():
            return False
        for key in inputs1:
            val1, val2 = inputs1[key], inputs2[key]
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                val1 = val1.to(torch.device(get_accelerator().current_device_name()))
                val2 = val2.to(torch.device(get_accelerator().current_device_name()))
                if not torch.equal(val1, val2):
                    return False
            elif val1 != val2:
                return False
        return True

    return False


def maybe_loss_for_backward(value) -> bool:
    """Check if the value is a loss tensor.
    Conditions:
    - The value must be a tensor.
    - The tensor must have exactly one element.
    - The tensor must have grad_fn defined.

    Args:
        value: The value to check.
    """
    return isinstance(value, torch.Tensor) and value.numel() == 1 and value.grad_fn is not None


class OutputBackwardHookManager:
    """
    Manages backward hooks on output tensors to trigger preprocessing only once.

    This is an alternative to register_full_backward_pre_hook that avoids warnings
    and provides more fine-grained control over when preprocessing occurs.

    The hook manager automatically manages its lifetime by attaching itself to the
    output tensors. When the outputs are freed, the hook manager is also freed.

    This manager handles two types of preprocessing:
    1. Global preprocessing (run once per backward pass): timers, flags, setup
    2. Per-tensor preprocessing (run for each output tensor): gradient scaling, loss logging

    Usage:
        # Only global preprocessing (run once)
        hook_manager = OutputBackwardHookManager(
            preprocess_once_fn=lambda: start_timers()
        )

        # Both global and per-tensor preprocessing
        hook_manager = OutputBackwardHookManager(
            preprocess_once_fn=lambda: start_timers(),
            preprocess_per_tensor_fn=lambda tensor: scale_gradient(tensor)
        )

        outputs = model(*inputs)
        hook_manager.register_hooks_on_outputs(outputs)
        # No need to manually clean up - it's freed when outputs are freed
    """

    def __init__(self, preprocess_once_fn, preprocess_per_tensor_fn=None):
        """
        Args:
            preprocess_once_fn: A callable that takes no arguments and performs
                               one-time preprocessing before backward (e.g., start timers).
                               Will only be called once per backward pass.
            preprocess_per_tensor_fn: Optional callable that takes a tensor and returns
                                     a potentially modified tensor. Called for each output
                                     tensor during backward (e.g., gradient scaling).
                                     If None, no per-tensor processing is done.
        """
        self.preprocess_once_fn = preprocess_once_fn
        self.preprocess_per_tensor_fn = preprocess_per_tensor_fn
        self.preprocess_done = False
        self.hook_handles = []

    def _make_backward_hook(self, tensor):
        """
        Creates a backward hook for a specific tensor.

        Args:
            tensor: The output tensor this hook is attached to
        """

        def backward_hook(grad):
            # First, ensure global preprocessing happens once
            if not self.preprocess_done:
                self.preprocess_done = True
                self.preprocess_once_fn()

            # Then apply per-tensor preprocessing if provided
            if self.preprocess_per_tensor_fn is not None:
                # Per-tensor preprocessing receives the tensor
                # It can perform operations like gradient scaling
                grad = self.preprocess_per_tensor_fn(grad)

            return grad

        return backward_hook

    def _traverse_and_register_hooks(self, outputs, first_tensor_holder):
        """
        Recursively traverse outputs to find tensors with grad_fn and register hooks.

        Args:
            outputs: Can be a tensor, tuple, list, dict, or nested structure of these.
            first_tensor_holder: List to hold the first tensor found (for attaching self)
        """
        if isinstance(outputs, torch.Tensor):
            if outputs.grad_fn is not None:
                # Store reference to first tensor to attach hook manager lifetime
                if not first_tensor_holder:
                    first_tensor_holder.append(outputs)
                # Pass the tensor to _make_backward_hook so per-tensor processing can access it
                hook_handle = outputs.register_hook(self._make_backward_hook(outputs))
                self.hook_handles.append(hook_handle)
        elif isinstance(outputs, (tuple, list)):
            for item in outputs:
                self._traverse_and_register_hooks(item, first_tensor_holder)
        elif isinstance(outputs, dict):
            for value in outputs.values():
                self._traverse_and_register_hooks(value, first_tensor_holder)

    def register_hooks_on_outputs(self, outputs):
        """
        Register backward hooks on all output tensors that have grad_fn.

        Args:
            outputs: The outputs from the forward pass. Can be a tensor or nested structure.
        """
        # Reset state for new forward pass
        self.preprocess_done = False
        self.remove_hooks()

        # Register hooks on all tensors with grad_fn
        first_tensor_holder = []
        self._traverse_and_register_hooks(outputs, first_tensor_holder)

        # Attach this hook manager instance to the first output tensor
        # This ensures the hook manager is kept alive as long as the outputs are alive
        # and automatically freed when outputs are freed
        if first_tensor_holder:
            first_tensor = first_tensor_holder[0]
            if not hasattr(first_tensor, '_backward_hook_managers'):
                first_tensor._backward_hook_managers = []
            first_tensor._backward_hook_managers.append(self)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def reset(self):
        """Reset the preprocessing flag without removing hooks."""
        self.preprocess_done = False


def register_output_backward_hooks(outputs, preprocess_once_fn, preprocess_per_tensor_fn=None):
    """
    Convenience function to register backward hooks on outputs.

    This function creates a hook manager that is automatically tied to the lifetime
    of the output tensors. When outputs are freed, the hook manager is also freed.

    Args:
        outputs: The outputs from forward pass (tensor, tuple, list, dict, or nested)
        preprocess_once_fn: A callable that takes no arguments and performs one-time
                           preprocessing before backward. Will only be called once per backward pass.
        preprocess_per_tensor_fn: Optional callable that takes a tensor and performs
                                 per-tensor preprocessing (e.g., gradient scaling).
                                 Called for each output tensor during backward.

    Returns:
        The hook manager instance (usually not needed, as lifetime is automatic)

    Example:
        # Only global preprocessing
        outputs = model(x)
        register_output_backward_hooks(outputs, lambda: print("Backward starting!"))

        # Both global and per-tensor preprocessing
        outputs = model(x)
        register_output_backward_hooks(
            outputs,
            preprocess_once_fn=lambda: start_timers(),
            preprocess_per_tensor_fn=lambda tensor: scale_tensor(tensor)
        )
        # Hook manager is automatically freed when outputs are freed
    """
    hook_manager = OutputBackwardHookManager(preprocess_once_fn, preprocess_per_tensor_fn)
    hook_manager.register_hooks_on_outputs(outputs)
    return hook_manager


def check_internal_apis_for_count_used_parameters() -> bool:
    """
    Ensure the Torch internal APIs needed by `count_used_parameters_in_backward` exist.
    """
    if not hasattr(torch.autograd.graph, '_get_grad_fn_or_grad_acc'):
        return False

    missing = [attr for attr in ("_current_graph_task_id", "_will_engine_execute_node") if not hasattr(torch._C, attr)]

    if missing:
        return False

    return True


def count_used_parameters_in_backward(parameters: Sequence[torch.nn.Parameter]) -> int:
    """
    Count the number of parameters that participate in the currently running backward graph.

    This helper is designed to be invoked from within a backward hook where a graph task
    is active. Parameters that do not require gradients, are detached, or are not touched
    by the current backward pass are ignored.

    torch.autograd.graph.register_multi_grad_hook is used for the purpose, but
    its verification on tensor shapes throws an error with ZeRO3 (it expects original tensor shape).
    So this function simplifies register_multi_grad_hook just to count used parameters.

    Args:
        parameters: Iterable of model parameters to inspect.

    Returns:
        The number of parameters whose gradient nodes will be executed by the autograd engine
        for the active backward call.
    """
    assert check_internal_apis_for_count_used_parameters(), (
        "count_used_parameters_in_backward requires internal PyTorch APIs that are not available "
        "in this PyTorch build.")

    from torch.autograd.graph import _get_grad_fn_or_grad_acc
    if torch._C._current_graph_task_id() == -1:
        raise RuntimeError("count_used_parameters_in_backward must be called during backward execution")

    seen_nodes = set()
    for param in parameters:
        if not isinstance(param, torch.Tensor) or not param.requires_grad:
            continue

        # Backward hooks run with grad mode disabled, but PyTorch <=2.4's
        # _get_grad_fn_or_grad_acc() requires grad mode for leaf params.
        with torch.enable_grad():
            grad_fn = _get_grad_fn_or_grad_acc(param)
        if grad_fn is None:
            continue

        if grad_fn in seen_nodes:
            continue

        seen_nodes.add(grad_fn)

    if not seen_nodes:
        return 0

    participating = sum(map(torch._C._will_engine_execute_node, seen_nodes))
    return int(participating)


# =========================================================================
# DES-LOC Training Utilities
# Ref: Algorithm 1, Section 2 (half-life), Section 3 (convergence)
# =========================================================================
import math as _math


def desloc_half_life(beta):
    """Half-life of EMA: tau = -1/ln(beta). Ref: Section 2, Eq.(1).
    beta=0.9 → 6.58, beta=0.95 → 13.51, beta=0.999 → 692.80."""
    if beta <= 0 or beta >= 1:
        return float('inf')
    return -1.0 / _math.log(beta)


def desloc_recommend_periods(beta1, beta2, Kx):
    """Recommend Ku, Kv from half-life ratios. Ref: Section 5.3.
    Default heuristic: Ku=3*Kx, Kv=6*Kx.
    Refined: Kv/Kx ~ tau(beta2)/tau(beta1) = ln(beta1)/ln(beta2)."""
    tau1 = desloc_half_life(beta1)
    tau2 = desloc_half_life(beta2)
    ratio = tau2 / tau1 if tau1 > 0 else 1.0
    Ku = max(1, int(round(Kx * 3)))
    Kv = max(Ku, int(round(Kx * min(ratio, 10))))
    return {'Ku': Ku, 'Kv': Kv, 'ratio': round(ratio, 4),
            'tau1': round(tau1, 4), 'tau2': round(tau2, 4)}


def desloc_psi_factor(Kx, Ku, beta1):
    """Compute psi from Theorem 1. Kx=1 → psi=0 (DDP bound).
    psi = 4(1-px)/px^2 * (1-beta1)(1-pu) / (6(1-(1-pu)*beta1))."""
    px = 1.0 / max(1, Kx)
    pu = 1.0 / max(1, Ku)
    num = 4.0 * (1 - px) * (1 - beta1) * (1 - pu)
    den = px * px * 6.0 * (1 - (1 - pu) * beta1)
    return num / den if abs(den) > 1e-15 else float('inf')


def desloc_comm_volume_per_step(param_bytes, num_workers, Kx, Ku, Kv):
    """Expected comm volume per step under DES-LOC.
    Ref: Section 4.1 — Ring-AllReduce: 2*(N-1)/N * param_bytes per op.
    DES-LOC: freq = 1/Kx + 1/Ku + 1/Kv. DDP: freq = 3."""
    if num_workers <= 1:
        return {'desloc_bytes': 0, 'ddp_bytes': 0, 'reduction': 0.0}
    ring = 2.0 * (num_workers - 1) / num_workers
    single_ar = ring * param_bytes
    desloc_freq = 1.0/max(1,Kx) + 1.0/max(1,Ku) + 1.0/max(1,Kv)
    return {
        'desloc_bytes': int(single_ar * desloc_freq),
        'ddp_bytes': int(single_ar * 3.0),
        'reduction': round(1.0 - desloc_freq / 3.0, 6),
    }


def desloc_convergence_bound(T, M, sigma2, Kx, Ku, beta1, eta):
    """Evaluate convergence bound from Theorem 1.
    E[||grad f||^2] <= O(1/sqrt(MT)) + O(psi/T).
    Ref: Section 3 — linear speedup in M workers preserved."""
    psi = desloc_psi_factor(Kx, Ku, beta1)
    leading = _math.sqrt(sigma2 / (M * T))
    higher = psi / T
    return {
        'leading': round(leading, 8),
        'higher_order': round(higher, 8),
        'total': round(leading + higher, 8),
        'psi': round(psi, 6),
        'ratio': round(higher / leading, 6) if leading > 0 else float('inf'),
    }


def desloc_model_flops(num_params, seq_len, batch_size):
    """Estimate FLOPs per training step: C ≈ 6*N*S*B.
    Ref: Kaplan et al. — 2N per fwd, 4N per bwd."""
    return 6 * num_params * seq_len * batch_size


class DeslocStepTracker:
    """Track DES-LOC sync boundaries. Ref: Algorithm 1.
    Determines which states should sync at each step."""

    def __init__(self, Kx=1, Ku=3, Kv=6, enabled=False):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.enabled = enabled
        self.step = 0
        self.param_syncs = 0
        self.momentum_syncs = 0
        self.variance_syncs = 0

    def advance(self):
        """Advance one step. Returns sync decisions."""
        self.step += 1
        if not self.enabled:
            return {'param': True, 'momentum': True, 'variance': True}
        sp = (self.step % self.Kx) == 0
        sm = (self.step % self.Ku) == 0
        sv = (self.step % self.Kv) == 0
        if sp: self.param_syncs += 1
        if sm: self.momentum_syncs += 1
        if sv: self.variance_syncs += 1
        return {'param': sp, 'momentum': sm, 'variance': sv, 'step': self.step}

    def should_sync(self, tier='param'):
        if not self.enabled:
            return True
        period = {'param': self.Kx, 'momentum': self.Ku, 'variance': self.Kv}.get(tier, 1)
        return (self.step % period) == 0

    def get_stats(self):
        t = max(1, self.step)
        return {
            'total_steps': self.step,
            'param_syncs': self.param_syncs,
            'momentum_syncs': self.momentum_syncs,
            'variance_syncs': self.variance_syncs,
            'param_skip_ratio': round(1.0 - self.param_syncs / t, 4),
            'momentum_skip_ratio': round(1.0 - self.momentum_syncs / t, 4),
            'variance_skip_ratio': round(1.0 - self.variance_syncs / t, 4),
        }

    def state_dict(self):
        return {'step': self.step, 'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
                'enabled': self.enabled, 'ps': self.param_syncs,
                'ms': self.momentum_syncs, 'vs': self.variance_syncs}

    def load_state_dict(self, sd):
        self.step = sd.get('step', 0)
        self.Kx = sd.get('Kx', self.Kx)
        self.Ku = sd.get('Ku', self.Ku)
        self.Kv = sd.get('Kv', self.Kv)
        self.enabled = sd.get('enabled', self.enabled)
        self.param_syncs = sd.get('ps', 0)
        self.momentum_syncs = sd.get('ms', 0)
        self.variance_syncs = sd.get('vs', 0)


class DeslocGradientClipper:
    """Per-coordinate gradient clipping for DES-LOC.
    Ref: Algorithm 1 line 12 — g_hat = clip(g, rho).
    Each gradient component satisfies |g_i| <= rho.
    This differs from global norm clipping (standard DeepSpeed)."""

    def __init__(self, clip_rho=1.0, enabled=False):
        self.clip_rho = clip_rho
        self.enabled = enabled
        self.total_clipped = 0
        self.total_elements = 0

    def clip(self, parameters):
        """Apply per-coordinate clipping. Returns clip fraction.
        Ref: Assumption 2 — bounded gradient ensures convergence."""
        if not self.enabled or self.clip_rho <= 0:
            return 0.0
        clipped = 0
        total = 0
        for p in parameters:
            if p.grad is None:
                continue
            g = p.grad.data
            total += g.numel()
            mask = g.abs() > self.clip_rho
            clipped += mask.sum().item()
            g.clamp_(-self.clip_rho, self.clip_rho)
        self.total_clipped += clipped
        self.total_elements += total
        return clipped / max(1, total)

    @property
    def clip_ratio(self):
        return self.total_clipped / max(1, self.total_elements)

    def reset(self):
        self.total_clipped = 0
        self.total_elements = 0


class DeslocWarmupTracker:
    """Track warmup phase. Ref: Section A.1 — TWARM=512 steps.
    During warmup: Kx=1 (full DDP sync to establish good initial states).
    After warmup: transition to configured Kx."""

    def __init__(self, warmup_steps=512, Kx_train=32):
        self.warmup_steps = warmup_steps
        self.Kx_train = Kx_train
        self.step = 0

    def get_current_Kx(self):
        if self.step < self.warmup_steps:
            return 1
        return self.Kx_train

    def advance(self):
        self.step += 1
        return self.get_current_Kx()

    def is_warmup(self):
        return self.step < self.warmup_steps

    def state_dict(self):
        return {'step': self.step, 'warmup': self.warmup_steps, 'Kx': self.Kx_train}

    def load_state_dict(self, sd):
        self.step = sd.get('step', 0)
        self.warmup_steps = sd.get('warmup', self.warmup_steps)
        self.Kx_train = sd.get('Kx', self.Kx_train)


class DeslocScalingLaw:
    """DES-LOC-aware scaling law predictor.
    Ref: Chinchilla — L(N,D) = a/N^alpha + b/D^beta + L_inf.
    DES-LOC adds comm penalty: L_desloc = L_base + c * psi(Kx,Ku) / T.

    Nick Joseph: 'scaling laws — predict with considerable accuracy:
    when you increase compute/data/params, loss decreases as power law.'"""

    def __init__(self, a=406.4, alpha=0.34, b=410.7, beta_exp=0.283, L_inf=1.69):
        self.a = a
        self.alpha = alpha
        self.b = b
        self.beta_exp = beta_exp
        self.L_inf = L_inf

    def predict_loss(self, N, D, Kx=1, Ku=3, beta1=0.9, T=None):
        """Predict loss for model size N and data tokens D."""
        base = self.a / (N ** self.alpha) + self.b / (D ** self.beta_exp) + self.L_inf
        if Kx > 1 and T is not None and T > 0:
            psi = desloc_psi_factor(Kx, Ku, beta1)
            base += psi / T * 0.01
        return round(base, 6)

    def compute_optimal(self, flops_budget):
        """Find optimal N, D for FLOPS budget. C ≈ 6*N*D."""
        best = None
        for log_n in range(7, 12):
            N = 10 ** log_n
            D = flops_budget / (6 * N)
            if D < N:
                continue
            loss = self.predict_loss(N, D)
            if best is None or loss < best['loss']:
                best = {'N': N, 'D': int(D), 'loss': loss, 'ratio': round(D/N, 2)}
        return best


MODEL_CONFIGS = {
    '125M': {'layers': 12, 'heads': 12, 'dim': 768, 'params': 125e6},
    '350M': {'layers': 24, 'heads': 16, 'dim': 1024, 'params': 350e6},
    '1B':   {'layers': 24, 'heads': 16, 'dim': 2048, 'params': 1e9},
    '7B':   {'layers': 32, 'heads': 32, 'dim': 4096, 'params': 7e9},
}


class DeslocConvergenceEstimator:
    """Estimate convergence rate from training loss trajectory.
    Ref: Theorem 1 — E[||grad f||^2] <= O(1/sqrt(T)) + O(psi/T).
    Fits loss ~ alpha * T^(-beta) + L_inf to verify convergence rate."""

    def __init__(self):
        self.loss_history = []
        self.step_history = []
        self._fitted = False
        self._alpha = 0.0
        self._beta_exp = 0.5
        self._L_inf = 0.0

    def record(self, step, loss):
        self.loss_history.append(float(loss))
        self.step_history.append(int(step))
        self._fitted = False

    def fit_power_law(self):
        """Fit loss ~ alpha * T^(-beta) + L_inf via log-linear regression.
        Ref: Nick Joseph — 'when curve departs from power law, something is wrong.'"""
        if len(self.loss_history) < 10:
            return None
        L_inf = min(self.loss_history) * 0.95
        log_s, log_l = [], []
        for s, l in zip(self.step_history, self.loss_history):
            if s > 0 and l > L_inf:
                log_s.append(_math.log(s))
                log_l.append(_math.log(l - L_inf))
        if len(log_s) < 5:
            return None
        n = len(log_s)
        sx = sum(log_s)
        sy = sum(log_l)
        sxy = sum(x*y for x, y in zip(log_s, log_l))
        sxx = sum(x*x for x in log_s)
        d = n * sxx - sx * sx
        if abs(d) < 1e-15:
            return None
        slope = (n * sxy - sx * sy) / d
        intercept = (sy - slope * sx) / n
        self._beta_exp = -slope
        self._alpha = _math.exp(intercept)
        self._L_inf = L_inf
        self._fitted = True
        y_pred = [slope * x + intercept for x in log_s]
        ss_res = sum((y - yp)**2 for y, yp in zip(log_l, y_pred))
        y_mean = sy / n
        ss_tot = sum((y - y_mean)**2 for y in log_l)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {
            'exponent': round(self._beta_exp, 6),
            'coefficient': round(self._alpha, 6),
            'L_inf': round(self._L_inf, 6),
            'r_squared': round(r2, 6),
            'n_points': n,
            'expected_exponent': 0.5,
        }

    def predict_loss(self, target_step):
        if not self._fitted:
            self.fit_power_law()
        if not self._fitted:
            return None
        return round(self._alpha * (target_step ** (-self._beta_exp)) + self._L_inf, 6)

    def compute_psi(self, Kx, Ku, beta1):
        """Compute psi and check if higher-order term is negligible."""
        psi = desloc_psi_factor(Kx, Ku, beta1)
        T = max(self.step_history) if self.step_history else 10000
        leading = 1.0 / _math.sqrt(T)
        higher = psi / T
        return {
            'psi': round(psi, 6),
            'ratio': round(higher / leading, 6) if leading > 0 else float('inf'),
            'negligible': (higher / leading < 0.01) if leading > 0 else False,
        }

    def estimate_kx_impact(self, kx_values, beta1=0.9, T=10000):
        """Estimate convergence impact for different Kx values."""
        results = {}
        for kx in kx_values:
            ku = max(1, kx * 3)
            psi = desloc_psi_factor(kx, ku, beta1)
            leading = 1.0 / _math.sqrt(T)
            higher = psi / T
            results[kx] = {
                'psi': round(psi, 6),
                'ratio': round(higher / leading, 6) if leading > 0 else 0,
                'negligible': higher < 0.01 * leading,
            }
        return results


def desloc_training_time_estimate(num_params, total_tokens, peak_tflops, mfu=0.45):
    """Estimate wall-clock training time in seconds.
    Ref: time = 6*N*D / (peak * 1e12 * mfu)."""
    return 6 * num_params * total_tokens / (peak_tflops * 1e12 * mfu)


def desloc_param_bytes(num_params, dtype_bytes=2):
    """Parameter size in bytes. BF16=2, FP32=4.
    With Adam: 3 states (param + m1 + m2) x dtype_bytes.
    DES-LOC reduces sync of m1/m2 by Ku/Kv factors."""
    return num_params * dtype_bytes


def desloc_format_training_report(step, loss, lr, Kx, Ku, Kv, is_sync,
                                   throughput=None, mfu=None):
    """Format a training step report in NKI-FA log format.
    Ref: NKI-FA draw_plot.py — 'metric: value' lines for parsing.
    All floats formatted to >= 6 digits."""
    lines = [
        f'step: {step}',
        f'loss: {loss:.6f}',
        f'lr: {lr:.8f}',
        f'Kx: {Kx}',
        f'Ku: {Ku}',
        f'Kv: {Kv}',
        f'is_param_sync: {int(is_sync)}',
    ]
    if throughput is not None:
        lines.append(f'throughput_samples_sec: {throughput:.4f}')
    if mfu is not None:
        lines.append(f'mfu: {mfu:.6f}')
    return '\n'.join(lines)


def desloc_validate_precision(value, name='value', min_sig_digits=4):
    """Validate that a number has sufficient significant digits.
    Ref: 'Data must not be 1, 11, 0.9 — NeurIPS reviewers will reject.'"""
    if not isinstance(value, float) or value == 0:
        return True
    s = f'{value:.10g}'.lstrip('-0').replace('.', '')
    sig = len(s.rstrip('0'))
    if sig < min_sig_digits:
        return False
    return True


# DES-LOC: end of M169 integration
