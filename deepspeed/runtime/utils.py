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

# =========================================================================
# DES-LOC Training Utilities (Section 2 + Section 3)
# =========================================================================

import math as _math


def desloc_half_life(beta):
    """Half-life of EMA state. Ref: Section 2, tau_0.5(beta) = ln(0.5)/ln(beta)."""
    if beta >= 1.0:
        return float('inf')
    if beta <= 0.0:
        return 0.0
    return _math.log(0.5) / _math.log(beta)


def desloc_recommend_periods(beta1, beta2, Kx):
    """Recommend Ku, Kv from half-life ratios.
    Ref: Section 5.2 Takeaway - set Ku, Kv proportional to half-life."""
    hl1 = desloc_half_life(beta1)
    hl2 = desloc_half_life(beta2)
    ratio = hl2 / hl1 if hl1 > 0 else 6.0
    Ku = max(1, round(ratio / 100.0)) * Kx
    Kv = max(Ku, round(ratio / 50.0) * Kx)
    return Kx, Ku, Kv


def desloc_psi_factor(Kx, Ku, beta1):
    """Compute psi factor from Theorem 1.
    psi = 4(1-px)/px^2 * (1-beta)(1-pu) / (6*(1-(1-pu)*beta))
    where px=1/Kx, pu=1/Ku."""
    px = 1.0 / max(Kx, 1)
    pu = 1.0 / max(Ku, 1)
    numer = 4.0 * (1.0 - px) * (1.0 - beta1) * (1.0 - pu)
    denom = px * px * 6.0 * (1.0 - (1.0 - pu) * beta1)
    return numer / max(denom, 1e-12)


def desloc_comm_volume_per_step(param_bytes, num_workers, Kx, Ku, Kv):
    """Estimate comm bytes/step under DES-LOC vs DDP.
    DDP: 2*(M-1)/M * param_bytes per allreduce, 3x for params+2 momenta.
    DES-LOC: same but at rates 1/Kx, 1/Ku, 1/Kv."""
    ar_bytes = 2.0 * (num_workers - 1) / max(num_workers, 1) * param_bytes
    ddp_per_step = ar_bytes * 3.0
    desloc_per_step = ar_bytes * (1.0/Kx + 1.0/Ku + 1.0/Kv)
    return {'ddp': ddp_per_step, 'desloc': desloc_per_step,
            'reduction': 1.0 - desloc_per_step / max(ddp_per_step, 1e-12)}


def desloc_format_log_line(step, loss, lr, Kx, is_sync, throughput=None):
    """Format one log line for experiment parsing. Ref: NKI-FA draw_plot.py."""
    parts = [f'step: {step}', f'loss: {loss:.6f}', f'lr: {lr:.8f}',
             f'Kx: {Kx}', f'is_sync: {int(is_sync)}']
    if throughput is not None:
        parts.append(f'throughput: {throughput:.2f}')
    return ' | '.join(parts)


def desloc_validate_precision(value, min_sig=4):
    """Check a float has enough significant digits for NeurIPS.
    Ref: NKI-FA commit da964f3 - all data points have >= 4 sig digits."""
    if value == 0:
        return True
    s = f'{value:.10g}'.lstrip('-').replace('.', '').lstrip('0')
    return len(s.rstrip('0')) >= min_sig

    s = f'{value:.10g}'.lstrip('-0').replace('.', '')
    sig = len(s.rstrip('0'))
    if sig < min_sig_digits:
        return False
    return True


# =============================================================================
# =============================================================================
# M258 (Claude-17): Compact Communication Reduction + Log Parsing + Aggregation
#
# Replaces bloated DeslocCommReductionCalculator (271 lines deleted)
# with 5 focused functions totaling ~120 lines.
# Adds: log file parser, multi-seed aggregator, precision validator,
#       power-law fitter, and CSV/JSON exporters.
#
# Ref: Section 5.3 — DES-LOC halves comm vs LocalAdam at matching loss
# Ref: NKI-FA da964f3 — log format ### config ### \n metric: value
# Ref: Megatron-LM megatron/core/optimizer/clip_grads.py — norm computation
# =============================================================================
import re as _desloc_re


def desloc_comm_reduction_ratio(Kx, Ku, Kv, total_steps):
    """Compute communication reduction ratio vs DDP.

    DDP baseline: 3 AllReduce ops per step (params + 2 momenta).
    DES-LOC: sync_x = T/Kx, sync_u = T/Ku, sync_v = T/Kv.
    Reduction = 3T / (T/Kx + T/Ku + T/Kv) = 3 / (1/Kx + 1/Ku + 1/Kv).

    Ref: Section 5.3, Table 2.
    """
    desloc_freq = (1.0 / max(1, Kx) + 1.0 / max(1, Ku) + 1.0 / max(1, Kv))
    if desloc_freq <= 0:
        return float('inf')
    return 3.0 / desloc_freq


def desloc_comm_bytes(n_params, Kx, Ku, Kv, total_steps, dtype_bytes=4):
    """Compute total communication bytes for DES-LOC vs DDP.

    Returns: dict with desloc_bytes, ddp_bytes, reduction_x, savings_pct.
    """
    msg = n_params * dtype_bytes
    ops_x = total_steps // max(1, Kx)
    ops_u = total_steps // max(1, Ku)
    ops_v = total_steps // max(1, Kv)
    desloc_total = (ops_x + ops_u + ops_v) * msg
    ddp_total = total_steps * 3 * msg
    reduction = ddp_total / max(1, desloc_total)
    savings = 100.0 * (1.0 - desloc_total / max(1, ddp_total))
    return {
        'desloc_bytes': desloc_total,
        'ddp_bytes': ddp_total,
        'reduction_x': round(reduction, 4),
        'savings_pct': round(savings, 2),
        'ops_x': ops_x, 'ops_u': ops_u, 'ops_v': ops_v,
    }


def desloc_local_adam_comm_bytes(n_params, Kx, total_steps, dtype_bytes=4):
    """LocalAdam communication: syncs ALL 3 states every Kx steps.
    Ref: Section 1 — 'Local Adam requires synchronizing momenta alongside
    model parameters, tripling communication costs.'
    """
    msg = n_params * dtype_bytes
    ops = (total_steps // max(1, Kx)) * 3
    return ops * msg


def desloc_parse_nkifa_logfile(filepath):
    """Parse a NKI-FA format log file into structured experiment records.

    Handles both NKI-FA original format:
        ### headdim = 128, causal = False, seqlen = 16384 ###
        Fav3 bwd: 18.034ms, 609.7 TFLOPS

    And DES-LOC extended format:
        ### model = 125M, Kx = 32, seed = 42 ###
        DES-LOC step 0: loss=10.8356, mfu=0.0312, comm_bytes=0

    Returns: list of dicts, each with 'config' and 'steps' keys.
    """
    config_re = _desloc_re.compile(r'###\s*(.+?)\s*###')
    step_re = _desloc_re.compile(
        r'DES-LOC step (\d+):\s*(.+)')
    nkifa_re = _desloc_re.compile(
        r'(\w+)\s+(fwd|bwd):\s+([\d.]+)\s*ms,\s+([\d.]+)\s*TFLOPS')
    summary_re = _desloc_re.compile(r'^(\w+):\s+(.+)$')

    experiments = []
    current = None
    in_summary = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') and not line.startswith('###'):
                if line == '--- SUMMARY ---':
                    in_summary = True
                continue

            # Config header
            m = config_re.match(line)
            if m:
                if current is not None:
                    experiments.append(current)
                config_str = m.group(1)
                config = {}
                for pair in config_str.split(','):
                    pair = pair.strip()
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        k, v = k.strip(), v.strip()
                        try:
                            v = int(v)
                        except ValueError:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                        config[k] = v
                current = {'config': config, 'steps': [], 'summary': {}}
                in_summary = False
                continue

            if current is None:
                continue

            # DES-LOC step data
            m = step_re.match(line)
            if m:
                step_num = int(m.group(1))
                metrics_str = m.group(2)
                metrics = {'step': step_num}
                for pair in metrics_str.split(','):
                    pair = pair.strip()
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        k, v = k.strip(), v.strip()
                        try:
                            metrics[k] = float(v)
                        except ValueError:
                            try:
                                metrics[k] = int(v)
                            except ValueError:
                                metrics[k] = v
                current['steps'].append(metrics)
                continue

            # NKI-FA format (Fav3 bwd: 18.034ms, 609.7 TFLOPS)
            m = nkifa_re.match(line)
            if m:
                current['steps'].append({
                    'method': m.group(1),
                    'direction': m.group(2),
                    'time_ms': float(m.group(3)),
                    'tflops': float(m.group(4)),
                })
                continue

            # Summary section
            if in_summary:
                m = summary_re.match(line)
                if m:
                    k, v = m.group(1), m.group(2)
                    try:
                        v = float(v)
                    except ValueError:
                        try:
                            v = int(v)
                        except ValueError:
                            pass
                    current['summary'][k] = v

    if current is not None:
        experiments.append(current)

    return experiments


def desloc_aggregate_experiments(experiments, group_keys=None):
    """Aggregate experiments by config (excluding seed), compute mean±std.

    Args:
        experiments: list from desloc_parse_nkifa_logfile()
        group_keys: keys to group by. Default: ['rq','model','Kx','Ku','Kv']

    Returns: list of dicts with mean/std of final_loss, comm_reduction, etc.
    """
    import math as _m
    if group_keys is None:
        group_keys = ['rq', 'model', 'Kx', 'Ku', 'Kv', 'outer', 'inner',
                       'beta2', 'mode']

    groups = {}
    for exp in experiments:
        cfg = exp.get('config', {})
        key = tuple((k, cfg.get(k)) for k in group_keys if k in cfg)
        groups.setdefault(key, []).append(exp)

    aggregated = []
    for key, exps in groups.items():
        config_base = dict(key)
        n = len(exps)

        # Extract final losses
        final_losses = []
        for exp in exps:
            steps = exp.get('steps', [])
            if steps:
                last = steps[-1]
                if 'loss' in last:
                    final_losses.append(last['loss'])
            summary = exp.get('summary', {})
            if 'final_loss' in summary:
                final_losses.append(summary['final_loss'])

        if not final_losses:
            continue

        mean_loss = sum(final_losses) / len(final_losses)
        if len(final_losses) > 1:
            std_loss = _m.sqrt(sum((x - mean_loss)**2
                                   for x in final_losses) / (n - 1))
        else:
            std_loss = 0.0

        # Comm reduction from config
        kx = config_base.get('Kx', 1)
        ku = config_base.get('Ku', 1)
        kv = config_base.get('Kv', 1)
        reduction = desloc_comm_reduction_ratio(kx, ku, kv, 1)

        aggregated.append({
            **config_base,
            'n_seeds': n,
            'final_loss_mean': round(mean_loss, 6),
            'final_loss_std': round(std_loss, 6),
            'comm_reduction_x': round(reduction, 4),
            'seeds': [exp.get('config', {}).get('seed', 0) for exp in exps],
        })

    return aggregated


def desloc_power_law_fit_simple(x_vals, y_vals):
    """Fit y = a · x^b using log-log linear regression.

    No numpy — pure Python implementation.
    Ref: Kaplan et al. scaling laws, Nick Joseph on power-law loss curves.

    Returns: (a, b, r_squared) or None if fit fails.
    """
    import math as _m
    if len(x_vals) < 3 or len(y_vals) < 3:
        return None

    # Filter positive values for log
    pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x > 0 and y > 0]
    if len(pairs) < 3:
        return None

    log_x = [_m.log(p[0]) for p in pairs]
    log_y = [_m.log(p[1]) for p in pairs]
    n = len(pairs)

    # Linear regression in log-log space
    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xx = sum(lx * lx for lx in log_x)
    sum_xy = sum(lx * ly for lx, ly in zip(log_x, log_y))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-15:
        return None

    b = (n * sum_xy - sum_x * sum_y) / denom
    log_a = (sum_y - b * sum_x) / n
    a = _m.exp(log_a)

    # R² in log space
    mean_y = sum_y / n
    ss_tot = sum((ly - mean_y)**2 for ly in log_y)
    ss_res = sum((ly - (log_a + b * lx))**2
                 for lx, ly in zip(log_x, log_y))
    r_sq = 1.0 - ss_res / max(1e-15, ss_tot)

    return (round(a, 8), round(b, 6), round(r_sq, 6))


def desloc_export_aggregated_csv(aggregated, filepath):
    """Export aggregated results to CSV. All floats ≥4 decimal places."""
    header = ('rq,model,Kx,Ku,Kv,beta2,outer,inner,mode,'
              'n_seeds,final_loss_mean,final_loss_std,'
              'comm_reduction_x\n')
    with open(filepath, 'w') as f:
        f.write(header)
        for g in aggregated:
            f.write(f"{g.get('rq','')},{g.get('model','')},"
                    f"{g.get('Kx','')},{g.get('Ku','')},{g.get('Kv','')},"
                    f"{g.get('beta2','')},{g.get('outer','')},"
                    f"{g.get('inner','')},{g.get('mode','')},"
                    f"{g['n_seeds']},"
                    f"{g['final_loss_mean']:.6f},"
                    f"{g['final_loss_std']:.6f},"
                    f"{g['comm_reduction_x']:.4f}\n")


def desloc_scan_log_directory(log_dir):
    """Scan a directory for all .log files, parse each, return merged list."""
    import os
    all_experiments = []
    if not os.path.isdir(log_dir):
        return all_experiments
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith('.log'):
            path = os.path.join(log_dir, fname)
            try:
                exps = desloc_parse_nkifa_logfile(path)
                all_experiments.extend(exps)
            except Exception:
                pass
    return all_experiments
# Ref: Section 5.3 RQ3 — DES-LOC halves comm vs Local Adam
# Ref: NKI-FA da964f3 — data from logs, not hardcoded
# Ref: Megatron-LM megatron/core/distributed — comm volume tracking
# Ref: NCCL src/include/profiler — byte-level event tracking
# =============================================================================



def desloc_comm_reduction_sweep(kx_values, num_params, total_steps,
                                 beta1=0.9, beta2=0.999, dtype_bytes=2):
    """Compute Figure 2 data across multiple Kx values.

    Returns list of Figure 2 data dicts, one per Kx.
    For each Kx, simulates sync decisions using DES-LOC rules
    (deterministic, no randomness — pure modular arithmetic).

    Ref: Algorithm 1 — sync when step % K == 0
    Ref: Section 5.3 — Ku=3Kx, Kv=6Kx heuristic

    Args:
        kx_values: list of Kx values to sweep
        num_params: model parameter count
        total_steps: number of training steps
        beta1: Adam beta1 (for half-life heuristic)
        beta2: Adam beta2
        dtype_bytes: 2 for BF16, 4 for FP32
    """
    results = []
    for kx in kx_values:
        ku = 3 * kx
        kv = 6 * kx
        calc = DeslocCommReductionCalculator(
            Kx=kx, Ku=ku, Kv=kv,
            num_params=num_params, dtype_bytes=dtype_bytes)
        # Simulate training steps — deterministic sync decisions
        warmup = min(512, total_steps // 4)
        for step in range(total_steps):
            if step < warmup:
                # During warmup: sync everything (Kx=1 effective)
                calc.record_step(step, is_sync_x=True,
                                 is_sync_u=True, is_sync_v=True)
            else:
                is_x = (step % kx) == 0
                is_u = (step % ku) == 0
                is_v = (step % kv) == 0
                calc.record_step(step, is_sync_x=is_x,
                                 is_sync_u=is_u, is_sync_v=is_v)
        results.append(calc.compute_figure2_data())
    return results


def desloc_format_figure2_table(sweep_results, fmt='md'):
    """Format Figure 2 sweep results as a table.

    Ref: NKI-FA — annotations with exact values, not placeholders.

    Args:
        sweep_results: list from desloc_comm_reduction_sweep()
        fmt: 'md' for markdown, 'latex' for NeurIPS submission
    """
    lines = []
    if fmt == 'latex':
        lines.append(r'\begin{tabular}{cccccc}')
        lines.append(r'\toprule')
        lines.append(r'$K_x$ & $K_u$ & $K_v$ & '
                     r'DDP (GB) & DES-LOC (GB) & Reduction \\')
        lines.append(r'\midrule')
    else:
        lines.append('| Kx | Ku | Kv | DDP (GB) | DES-LOC (GB) | '
                     'vs DDP | vs Local Adam |')
        lines.append('|---|---|---|---|---|---|---|')

    for r in sweep_results:
        kx, ku, kv = r['Kx'], r['Ku'], r['Kv']
        ddp_gb = r['methods']['DDP']['total_gb']
        dl_gb = r['methods']['DES-LOC']['total_gb']
        r_ddp = r['reduction_vs_ddp']
        r_la = r['reduction_vs_local_adam']
        if fmt == 'latex':
            lines.append(
                f'{kx} & {ku} & {kv} & {ddp_gb:.3f} & '
                f'{dl_gb:.3f} & {r_ddp:.1f}$\\times$ \\\\')
        else:
            lines.append(
                f'| {kx} | {ku} | {kv} | {ddp_gb:.3f} | '
                f'{dl_gb:.3f} | {r_ddp:.1f}× | {r_la:.1f}× |')

    if fmt == 'latex':
        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
    return '\n'.join(lines)


# DES-LOC: end of M229 integration

# DES-LOC: end of M169 integration


# =====================================================================
# M243 — Claude-16: DES-LOC Advanced Training Utilities
# Cross-scale Kx extrapolation, power-law regression, MFU prediction,
# comm-compute overlap, adaptive Kx, step time estimation
# Ref: Chinchilla scaling laws, DES-LOC Theorem 1, NKI-FA benchmark
# =====================================================================

import math as _m243_math


def desloc_extrapolate_Kx(small_N, small_loss_at_Kx, target_N,
                          beta1=0.9, beta2=0.999, alpha=0.34):
    """Extrapolate optimal Kx from small model to large model.

    The key insight: DES-LOC's psi penalty scales with 1/T,
    while larger models train for more steps T. So larger models
    can tolerate larger Kx.

    Args:
        small_N: small model param count (e.g. 125M)
        small_loss_at_Kx: dict of {Kx: final_loss} from small model
        target_N: target model param count (e.g. 1B)
        beta1, beta2: optimizer betas
        alpha: scaling law exponent

    Returns:
        dict with recommended Kx and predicted losses
    """
    if not small_loss_at_Kx:
        return {'recommended_Kx': 32, 'confidence': 'low', 'reason': 'no data'}

    # Scale factor: larger model trains longer → psi penalty smaller
    scale_ratio = target_N / max(1, small_N)
    step_ratio = scale_ratio ** 0.5  # Chinchilla: T ∝ √(N*D) ∝ N^0.5

    results = {}
    for kx, loss in sorted(small_loss_at_Kx.items()):
        # Extrapolate: at target_N, the psi penalty is divided by step_ratio
        # because psi appears in O((1+psi)/T) and T is larger
        psi = desloc_psi_factor(kx, max(1, kx * 3), beta1)
        adjusted_penalty = psi / step_ratio
        results[kx] = {
            'small_loss': round(loss, 6),
            'psi': round(psi, 6),
            'adjusted_psi': round(adjusted_penalty, 6),
        }

    # Find largest Kx where adjusted psi < 1.0 (manageable overhead)
    recommended = 1
    for kx in sorted(results.keys(), reverse=True):
        if results[kx]['adjusted_psi'] < 1.0:
            recommended = kx
            break

    return {
        'recommended_Kx': recommended,
        'target_N': target_N,
        'scale_ratio': round(scale_ratio, 4),
        'step_ratio': round(step_ratio, 4),
        'per_Kx_analysis': results,
        'confidence': 'medium' if len(small_loss_at_Kx) >= 4 else 'low',
    }


def desloc_power_law_fit(compute_loss_pairs):
    """Fit power law: loss = A * compute^(-alpha) + E.

    Uses pure Python (no numpy) for log-linear regression.

    Args:
        compute_loss_pairs: list of (compute_flops, loss) tuples

    Returns:
        dict with fitted A, alpha, E, r_squared
    """
    if len(compute_loss_pairs) < 3:
        return {'A': 406.4, 'alpha': 0.34, 'E': 1.69,
                'r_squared': 0.0, 'fitted': False}

    best_r2 = -1e9
    best_params = None

    for e_100x in range(100, 260, 5):  # E from 1.0 to 2.55
        e_try = e_100x / 100.0

        # Filter valid points
        valid = [(c, l - e_try) for c, l in compute_loss_pairs
                 if c > 0 and l > e_try]
        if len(valid) < 2:
            continue

        n = len(valid)
        sum_x = sum_y = sum_x2 = sum_xy = sum_y2 = 0.0

        for c, l_adj in valid:
            x = _m243_math.log(c)
            y = _m243_math.log(l_adj)
            sum_x += x
            sum_y += y
            sum_x2 += x * x
            sum_xy += x * y
            sum_y2 += y * y

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-15:
            continue

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_mean = sum_y / n
        ss_tot = sum_y2 - n * y_mean * y_mean
        ss_res = 0.0
        for c, l_adj in valid:
            x = _m243_math.log(c)
            y = _m243_math.log(l_adj)
            pred = intercept + slope * x
            ss_res += (y - pred) ** 2

        r2 = 1.0 - ss_res / max(1e-15, ss_tot)

        if r2 > best_r2:
            best_r2 = r2
            best_params = (
                _m243_math.exp(intercept),  # A
                -slope,                      # alpha (negative slope = positive alpha)
                e_try,                       # E
            )

    if best_params is None:
        return {'A': 406.4, 'alpha': 0.34, 'E': 1.69,
                'r_squared': 0.0, 'fitted': False}

    A, alpha, E = best_params
    return {
        'A': round(A, 6),
        'alpha': round(alpha, 6),
        'E': round(E, 6),
        'r_squared': round(best_r2, 6),
        'fitted': True,
        'n_points': len(compute_loss_pairs),
    }


def desloc_predict_mfu(model_params, batch_tokens, step_time_s,
                       Kx=1, hardware_tflops=989.5, dtype_bytes=2):
    """Predict Model FLOPS Utilization (MFU) under DES-LOC.

    MFU = actual_FLOPS / peak_FLOPS
    actual_FLOPS ≈ 6 * N * batch_tokens / step_time  (forward + backward)
    peak_FLOPS = hardware_tflops * 1e12

    DES-LOC effect: reduces comm time, so step_time decreases,
    so MFU increases.

    Args:
        model_params: N (number of parameters)
        batch_tokens: tokens per step (micro_batch * seq_len * grad_accum)
        step_time_s: measured wall-clock time per step
        Kx: parameter sync period
        hardware_tflops: peak hardware TFLOPS (BF16)
        dtype_bytes: bytes per parameter element

    Returns:
        dict with MFU breakdown
    """
    if step_time_s <= 0 or model_params <= 0:
        return {'mfu': 0.0, 'compute_tflops': 0.0}

    # 6*N*D FLOPS per step (approx: 2 for fwd, 4 for bwd)
    flops_per_step = 6.0 * model_params * batch_tokens
    actual_tflops = flops_per_step / step_time_s / 1e12

    peak = hardware_tflops
    mfu = actual_tflops / peak if peak > 0 else 0.0

    # Estimate comm fraction
    # AllReduce: 2 * model_params * dtype_bytes / bandwidth
    # Assume 400 GB/s effective bandwidth (H100 NVLink)
    effective_bw = 400e9  # bytes/sec
    ar_time = 2.0 * model_params * dtype_bytes / effective_bw
    comm_fraction = ar_time / max(1e-15, step_time_s)

    # DES-LOC: comm only 1/Kx of the time
    desloc_comm_fraction = comm_fraction / max(1, Kx)
    desloc_mfu_gain = comm_fraction - desloc_comm_fraction

    return {
        'mfu': round(mfu, 6),
        'compute_tflops': round(actual_tflops, 4),
        'peak_tflops': peak,
        'flops_per_step': int(flops_per_step),
        'comm_fraction': round(comm_fraction, 6),
        'desloc_comm_fraction': round(desloc_comm_fraction, 6),
        'desloc_mfu_gain': round(desloc_mfu_gain, 6),
        'Kx': Kx,
    }


def desloc_comm_compute_overlap_ratio(model_params, batch_tokens,
                                       compute_tflops, bw_gbps,
                                       Kx=1, dtype_bytes=2):
    """Calculate comm-compute overlap ratio.

    If AllReduce can be fully hidden behind compute → overlap = 1.0
    If AllReduce dominates → overlap → 0.0

    Ref: Megatron-LM — uses overlapped AllReduce with gradient bucketing
    Ref: NCCL — async collective launch enables overlap
    """
    if compute_tflops <= 0 or bw_gbps <= 0:
        return {'overlap_ratio': 0.0, 'bottleneck': 'unknown'}

    # Compute time per step
    flops = 6.0 * model_params * batch_tokens
    compute_time = flops / (compute_tflops * 1e12)

    # Communication time per DES-LOC sync step
    comm_bytes = 2.0 * model_params * dtype_bytes  # AllReduce
    comm_time = comm_bytes / (bw_gbps * 1e9 / 8)   # convert Gbps to bytes/s

    # With Kx gating: amortized comm per step
    amortized_comm = comm_time / max(1, Kx)

    if compute_time >= amortized_comm:
        overlap = 1.0
        bottleneck = 'compute-bound'
    else:
        overlap = compute_time / max(1e-15, amortized_comm)
        bottleneck = 'comm-bound'

    return {
        'overlap_ratio': round(overlap, 6),
        'compute_time_ms': round(compute_time * 1000, 4),
        'comm_time_ms': round(comm_time * 1000, 4),
        'amortized_comm_ms': round(amortized_comm * 1000, 4),
        'bottleneck': bottleneck,
        'Kx': Kx,
    }


def desloc_adaptive_Kx(current_mfu, target_mfu=0.55, current_Kx=32,
                        min_Kx=1, max_Kx=256, step_factor=2):
    """Adaptively adjust Kx to reach target MFU.

    If current MFU < target: reduce Kx (more syncs, but maybe we're
    diverging due to stale params)
    If current MFU > target: increase Kx (less syncs, save comm)

    This is a simple multiplicative adjustment, not a control loop.
    """
    if current_mfu <= 0:
        return current_Kx

    if current_mfu < target_mfu * 0.9:
        # MFU too low → might be diverging, reduce Kx
        new_Kx = max(min_Kx, current_Kx // step_factor)
        action = 'decrease'
    elif current_mfu > target_mfu * 1.1:
        # MFU already high → try increasing Kx for more savings
        new_Kx = min(max_Kx, current_Kx * step_factor)
        action = 'increase'
    else:
        new_Kx = current_Kx
        action = 'hold'

    return {
        'new_Kx': new_Kx,
        'old_Kx': current_Kx,
        'action': action,
        'current_mfu': round(current_mfu, 6),
        'target_mfu': round(target_mfu, 6),
    }


def desloc_estimate_step_time(model_params, batch_tokens,
                               hardware_tflops=989.5, mfu_estimate=0.45,
                               Kx=1, bw_gbps=3200, dtype_bytes=2):
    """Estimate wall-clock step time before training starts.

    Useful for planning Kx and total training duration.

    Args:
        model_params: N
        batch_tokens: tokens per step
        hardware_tflops: peak TFLOPS
        mfu_estimate: expected MFU (0.4-0.6 typical)
        Kx: planned sync period
        bw_gbps: network bandwidth in Gbps
    """
    if hardware_tflops <= 0 or mfu_estimate <= 0:
        return {'step_time_ms': 0.0}

    # Compute time
    flops = 6.0 * model_params * batch_tokens
    effective_tflops = hardware_tflops * mfu_estimate
    compute_time = flops / (effective_tflops * 1e12)

    # Comm time (amortized by Kx)
    comm_bytes = 2.0 * model_params * dtype_bytes
    comm_time = comm_bytes / (bw_gbps * 1e9 / 8)
    amortized_comm = comm_time / max(1, Kx)

    # Step time = max(compute, comm) if no overlap, sum if sequential
    # Assume partial overlap (50%)
    step_time = compute_time + amortized_comm * 0.5

    return {
        'step_time_ms': round(step_time * 1000, 4),
        'compute_time_ms': round(compute_time * 1000, 4),
        'comm_time_ms': round(comm_time * 1000, 4),
        'amortized_comm_ms': round(amortized_comm * 1000, 4),
        'Kx': Kx,
        'estimated_mfu': round(mfu_estimate, 4),
    }


def desloc_nkifa_log_format(config_dict, metrics_dict):
    """Format experiment results in NKI-FA commit da964f3 style.

    Output format:
    ### key1 = val1, key2 = val2, ... ###
    metric1: value1
    metric2: value2

    All values must have >= 4 significant digits.
    """
    # Config line
    config_parts = [f'{k} = {v}' for k, v in sorted(config_dict.items())]
    header = '### ' + ', '.join(config_parts) + ' ###'

    # Metric lines
    metric_lines = []
    for k, v in sorted(metrics_dict.items()):
        if isinstance(v, float):
            metric_lines.append(f'{k}: {v:.6f}')
        elif isinstance(v, int):
            metric_lines.append(f'{k}: {v}')
        else:
            metric_lines.append(f'{k}: {v}')

    return header + '\n' + '\n'.join(metric_lines)


def desloc_parse_nkifa_log(log_text):
    """Parse NKI-FA formatted log back to structured data.

    Inverse of desloc_nkifa_log_format().
    Returns list of (config_dict, metrics_dict) pairs.
    """
    results = []
    current_config = None
    current_metrics = {}

    for line in log_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('###') and line.endswith('###'):
            # Save previous record
            if current_config is not None:
                results.append((current_config, current_metrics))

            # Parse new config
            inner = line[3:-3].strip()
            current_config = {}
            for part in inner.split(','):
                part = part.strip()
                if '=' in part:
                    k, v = part.split('=', 1)
                    k, v = k.strip(), v.strip()
                    # Try numeric conversion
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    current_config[k] = v
            current_metrics = {}

        elif ':' in line and current_config is not None:
            k, v = line.split(':', 1)
            k, v = k.strip(), v.strip()
            try:
                v = float(v) if '.' in v else int(v)
            except ValueError:
                pass
            current_metrics[k] = v

    # Don't forget last record
    if current_config is not None:
        results.append((current_config, current_metrics))

    return results


# M243: end of Claude-16 utils integration
