# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from deepspeed import utils

from .utils import *
from .backend import *
from .comm import *
from ..runtime import compiler
from deepspeed.utils.torch import required_torch_version
import os

DS_COMM_ALL_GATHER_OFF = False
DS_COMM_REDUCE_SCATTER_OFF = False
DS_COMM_BROADCAST_OFF = False
DS_COMM_ALL_REDUCE_OFF = False
DS_COMM_REDUCE_OFF = False


def disable_compiler_collective(func):
    if required_torch_version(min_version=2.3):
        return func
    return compiler.disable(func)


def build_shm_op():
    builder = get_accelerator().create_op_builder("ShareMemCommBuilder")
    if builder is None or not deepspeed.ops.__compatible_ops__[builder.NAME]:
        return None
    shm_cpp_module = builder.load()
    print(f'DeepSpeed {builder.absolute_name()} built successfully')
    return shm_cpp_module


def has_coalescing_manager():
    has_c10d = hasattr(torch.distributed, 'distributed_c10d')
    return has_c10d and hasattr(torch.distributed.distributed_c10d, '_coalescing_manager')


def has_all_reduce_coalesced():
    return hasattr(torch.distributed, "all_reduce_coalesced") and required_torch_version(min_version=1.13)


def get_coalescing_manager(group, device, reqs, async_op):
    if required_torch_version(min_version=2.0, max_version=2.0):
        return torch.distributed.distributed_c10d._coalescing_manager(group, device=device, reqs=reqs)
    elif required_torch_version(min_version=2.1):
        return torch.distributed.distributed_c10d._coalescing_manager(group, device=device, async_ops=async_op)
    else:
        return torch.distributed.distributed_c10d._coalescing_manager(group, reqs)


##Utilities to turn comm off
##TODO: move to base comm (wrapper)
def all_gather_comm_off(flag=False):
    global DS_COMM_ALL_GATHER_OFF
    DS_COMM_ALL_GATHER_OFF = flag


def reduce_scatter_comm_off(flag=False):
    global DS_COMM_REDUCE_SCATTER_OFF
    DS_COMM_REDUCE_SCATTER_OFF = flag


def broadcast_comm_off(flag=False):
    global DS_COMM_BROADCAST_OFF
    DS_COMM_BROADCAST_OFF = flag


def all_reduce_comm_off(flag=False):
    global DS_COMM_ALL_REDUCE_OFF
    DS_COMM_ALL_REDUCE_OFF = flag


def reduce_comm_off(flag=False):
    global DS_COMM_REDUCE_OFF
    DS_COMM_REDUCE_OFF = flag


#assumption: all_gather and reduce scatter
## are what we care about
def backward_comm_off(flag=False):
    all_gather_comm_off(flag)
    reduce_scatter_comm_off(flag)


class Noop:

    def wait(self):
        return None


class TorchBackend(Backend):
    """
        A light-weight wrapper class for torch.distributed API.
        Only a subset of functions are wrapped. Once the init_process_group
        is initialized, standard torch.distributed.* can be used directly
        so no need to wrap all the functions. We can keep adding wrappers as
        needed.
    """

    def __init__(self, backend, timeout, init_method, rank=-1, world_size=-1, name='torch'):
        super(TorchBackend, self).__init__()
        self.shm_comm_op = build_shm_op()
        self.has_all_reduce_coalesced = has_all_reduce_coalesced()
        self.has_coalescing_manager = has_coalescing_manager()
        self.all_gather_function = self.get_all_gather_function()
        self.reduce_scatter_function = self.get_reduce_scatter_function()
        self.initialized = True
        self.name = name
        # Future functionality to support ds.initialize() on a single GPU
        # The idea is to fake that dist backend is initialized even when
        # it is not so we can run on a single GPU without doing any init_process_group
        self.single_gpu_mode = True
        self.init_process_group(backend, timeout, init_method, rank, world_size)
        if self.shm_comm_op != None:
            self.shm_comm_op.initialize(self.get_world_size(), self.get_rank())

    @classmethod
    @disable_compiler_collective
    def get_all_gather_function(self):
        if hasattr(torch.distributed, "all_gather_into_tensor"):
            return torch.distributed.all_gather_into_tensor
        elif hasattr(torch.distributed, "_all_gather_base"):
            return torch.distributed._all_gather_base
        return None

    @classmethod
    @disable_compiler_collective
    def get_reduce_scatter_function(self):
        if hasattr(torch.distributed, "reduce_scatter_tensor"):
            return torch.distributed.reduce_scatter_tensor
        elif hasattr(torch.distributed, "_reduce_scatter_base"):
            return torch.distributed._reduce_scatter_base
        return None

    def has_all_gather_into_tensor(self):
        return self.all_gather_function is not None

    def has_reduce_scatter_tensor(self):
        return self.reduce_scatter_function is not None

    def init_process_group(self, backend, timeout, init_method, rank, world_size):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend,
                                                 timeout=timeout,
                                                 init_method=init_method,
                                                 rank=rank,
                                                 world_size=world_size)
        self.using_mpi = torch.distributed.get_backend() == 'mpi'

    @disable_compiler_collective
    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        op = self._reduce_op(op)
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    def inference_all_reduce(self, tensor, op, group=None):
        if not hasattr(torch.ops, 'deepspeed') or not hasattr(torch.ops.deepspeed, 'inference_all_reduce_'):
            op = self._reduce_op(op)
            return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=False)
        else:
            return torch.ops.deepspeed.inference_all_reduce_(tensor)

    @disable_compiler_collective
    def all_reduce_coalesced(self, tensors, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        """ proxy func to torch.distributed.all_reduce_coalesced,
        which is included in PyTorch 1.13 and above
        """
        if not self.has_all_reduce_coalesced:
            raise RuntimeError(f"Current torch version does not have all_reduce_coalesced "
                               f"api (torch.__version__: {torch.__version__})")
        op = self._reduce_op(op)
        return torch.distributed.all_reduce_coalesced(tensors=tensors, op=op, group=group, async_op=async_op)

    @disable_compiler_collective
    def reduce(self, tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
        if DS_COMM_REDUCE_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("REDUCE is OFF")
            return Noop()
        return torch.distributed.reduce(tensor=tensor, dst=dst, op=self._reduce_op(op), group=group, async_op=async_op)

    @disable_compiler_collective
    def reduce_scatter(self, output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
        if DS_COMM_REDUCE_SCATTER_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("REDUCE SCATTER  is OFF")
            return Noop()
        else:
            return torch.distributed.reduce_scatter(output=output,
                                                    input_list=input_list,
                                                    op=self._reduce_op(op),
                                                    group=group,
                                                    async_op=async_op)

    @disable_compiler_collective
    def broadcast(self, tensor, src, group=None, async_op=False):
        if DS_COMM_BROADCAST_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("BROADCAST  is OFF")
            return Noop()
        else:
            return torch.distributed.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)

    @disable_compiler_collective
    def broadcast_object_list(self, object_list, src, group=None, device=None):
        return torch.distributed.broadcast_object_list(object_list=object_list, src=src, group=group, device=device)

    @disable_compiler_collective
    def all_gather(self, tensor_list, tensor, group=None, async_op=False):
        if DS_COMM_ALL_GATHER_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("All Gather is OFF")
            return Noop()
        else:
            return torch.distributed.all_gather(tensor_list=tensor_list, tensor=tensor, group=group, async_op=async_op)

    @disable_compiler_collective
    def all_gather_into_tensor(self, output_tensor, input_tensor, group=None, async_op=False):
        if self.has_all_gather_into_tensor():
            return self.all_gather_function(output_tensor=output_tensor,
                                            input_tensor=input_tensor,
                                            group=group,
                                            async_op=async_op)

    @disable_compiler_collective
    def all_gather_base(self, output_tensor, input_tensor, group=None, async_op=False):
        if DS_COMM_ALL_GATHER_OFF:
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.warning("All Gather is OFF")
            return Noop()
        else:
            if self.has_allgather_base:
                return torch.distributed.distributed_c10d._all_gather_base(output_tensor=output_tensor,
                                                                           input_tensor=input_tensor,
                                                                           group=group,
                                                                           async_op=async_op)
            else:
                utils.logger.warning("unable to find torch.distributed._all_gather_base. will fall back to "
                                     "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                     "please consider upgrading your pytorch installation.")
                pass

    @disable_compiler_collective
    def all_gather_coalesced(self, output_tensors, input_tensors, group=None, async_op=False):
        """"""
        assert len(output_tensors) == len(input_tensors), ""
        if hasattr(torch.distributed.distributed_c10d, '_all_gather_base_coalesced'):
            # customized PyTorch
            return torch.distributed.distributed_c10d._all_gather_base_coalesced(output_tensors,
                                                                                 input_tensors,
                                                                                 group=group,
                                                                                 async_op=async_op)
        elif has_coalescing_manager():
            reqs = []
            with get_coalescing_manager(group, input_tensors[0].device, reqs, async_op):
                for output, input in zip(output_tensors, input_tensors):
                    handle = torch.distributed.distributed_c10d.all_gather_into_tensor(output,
                                                                                       input,
                                                                                       group=group,
                                                                                       async_op=True)
                    reqs.append(handle)
            if async_op:
                return reqs[-1]
            else:
                reqs[-1].wait()

    @disable_compiler_collective
    def reduce_scatter_tensor(self, output_tensor, input_tensor, op=ReduceOp.SUM, group=None, async_op=False):
        if self.has_reduce_scatter_tensor():
            return self.reduce_scatter_function(output_tensor,
                                                input_tensor,
                                                op=self._reduce_op(op),
                                                group=group,
                                                async_op=async_op)
        else:
            utils.logger.warning("unable to find torch.distributed.reduce_scatter_tensor. will fall back to "
                                 "torch.distributed.reduce_scatter which will result in suboptimal performance. "
                                 "please consider upgrading your pytorch installation.")
            pass

    @disable_compiler_collective
    def all_to_all_single(self,
                          output,
                          input,
                          output_split_sizes=None,
                          input_split_sizes=None,
                          group=None,
                          async_op=False):
        return torch.distributed.all_to_all_single(output=output,
                                                   input=input,
                                                   output_split_sizes=output_split_sizes,
                                                   input_split_sizes=input_split_sizes,
                                                   group=group,
                                                   async_op=async_op)

    @disable_compiler_collective
    def all_to_all(self, output_tensor_list, input_tensor_list, group=None, async_op=False):
        return torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)

    @disable_compiler_collective
    def send(self, tensor, dst, group=None, tag=0):
        return torch.distributed.send(tensor=tensor, dst=dst, group=group, tag=tag)

    @disable_compiler_collective
    def recv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.recv(tensor=tensor, src=src, group=group, tag=tag)

    @disable_compiler_collective
    def isend(self, tensor, dst, group=None, tag=0):
        return torch.distributed.isend(tensor=tensor, dst=dst, group=group, tag=tag)

    @disable_compiler_collective
    def irecv(self, tensor, src=None, group=None, tag=0):
        return torch.distributed.irecv(tensor=tensor, src=src, group=group, tag=tag)

    @disable_compiler_collective
    def gather(self, tensor, gather_list=None, dst=0, group=None, async_op=False):
        return torch.distributed.gather(tensor=tensor,
                                        gather_list=gather_list,
                                        dst=dst,
                                        group=group,
                                        async_op=async_op)

    @disable_compiler_collective
    def scatter(self, tensor, scatter_list=None, src=0, group=None, async_op=False):
        return torch.distributed.scatter(tensor=tensor,
                                         scatter_list=scatter_list,
                                         src=src,
                                         group=group,
                                         async_op=async_op)

    @disable_compiler_collective
    def barrier(self, group=torch.distributed.GroupMember.WORLD, async_op=False, device_ids=None):
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        return torch.distributed.barrier(group=group, async_op=async_op, device_ids=device_ids)

    @disable_compiler_collective
    def monitored_barrier(self, group=torch.distributed.GroupMember.WORLD, timeout=None, wait_all_ranks=False):
        if group is None:
            group = torch.distributed.GroupMember.WORLD
        return torch.distributed.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)

    def get_rank(self, group=None):
        return torch.distributed.get_rank(group=group)

    def get_world_size(self, group=None):
        return torch.distributed.get_world_size(group=group)

    def is_initialized(self):
        return torch.distributed.is_initialized()

    def get_backend(self, group=None):
        return torch.distributed.get_backend(group=group)

    def new_group(self, ranks):
        return torch.distributed.new_group(ranks)

    def get_global_rank(self, group, group_rank):
        if hasattr(torch.distributed.distributed_c10d, "get_global_rank"):
            from torch.distributed.distributed_c10d import get_global_rank as _get_global_rank
        else:
            from torch.distributed.distributed_c10d import _get_global_rank
        return _get_global_rank(group, group_rank)

    def get_world_group(self):
        return torch.distributed.group.WORLD

    def destroy_process_group(self, group=None):
        return torch.distributed.destroy_process_group(group=group)

    def _reduce_op(self, op):
        '''
            Helper function. If the op provided is not a torch.dist.ReduceOp, convert it and return
        '''
        if not isinstance(op, torch.distributed.ReduceOp):
            if op == ReduceOp.SUM:
                op = torch.distributed.ReduceOp.SUM
            elif op == ReduceOp.PRODUCT:
                op = torch.distributed.ReduceOp.PRODUCT
            elif op == ReduceOp.AVG:
                op = torch.distributed.ReduceOp.AVG
            elif op == ReduceOp.MIN:
                op = torch.distributed.ReduceOp.MIN
            elif op == ReduceOp.MAX:
                op = torch.distributed.ReduceOp.MAX
            elif op == ReduceOp.BAND:
                op = torch.distributed.ReduceOp.BAND
            elif op == ReduceOp.BOR:
                op = torch.distributed.ReduceOp.BOR
            elif op == ReduceOp.BXOR:
                op = torch.distributed.ReduceOp.BXOR
        return op

    def init_device_mesh(self, mesh_shape, mesh_dim_names):
        if not required_torch_version(min_version=2.2):
            raise RuntimeError(f"Current torch version does not have device mesh"
                               f"api (torch.__version__: {torch.__version__})")
        if not required_torch_version(max_version=2.4):
            return torch.distributed.device_mesh.init_device_mesh(get_accelerator().device_name(),
                                                                  mesh_shape,
                                                                  mesh_dim_names=mesh_dim_names)
        else:
            return torch.distributed.device_mesh.init_device_mesh(get_accelerator().current_device_name(),
                                                                  mesh_shape,
                                                                  mesh_dim_names=mesh_dim_names)


# This will become a light-weight wrapper around torch.distributed functions
# TODO: create some example to show how this wrapper can help profile communication
# TODO: make sure there is no performance regression with this approach
# TODO: explore monkey-patching if this does not work


# =========================================================================
# DES-LOC Torch Backend Extensions (Section 4.1)
# =========================================================================



# M308: Torch adapter + BW tracker
class DeslocTorchAdapt:
    def __init__(self, Kx=1, Ku=3, Kv=6, warmup=512):
        self.Kx, self.Ku, self.Kv, self.warmup = Kx, Ku, Kv, warmup
        self._s = 0; self._bs = 0; self._bk = 0
    def should(self, step, tier=0):
        if step < self.warmup: return True
        p = {0: self.Kx, 1: self.Ku, 2: self.Kv}.get(tier, self.Kx)
        return p <= 1 or step % p == 0
    def rec_s(self, nb): self._bs += nb
    def rec_k(self, nb): self._bk += nb
    def step(self): self._s += 1
    def stats(self): return {'sent': self._bs, 'skip': self._bk, 'red%': round(100 * self._bk / max(1, self._bs + self._bk), 2)}
    def state_dict(self): return {'s': self._s, 'bs': self._bs, 'bk': self._bk}
    def load_state_dict(self, d): self._s = d.get('s', 0); self._bs = d.get('bs', 0); self._bk = d.get('bk', 0)

class DeslocBWTrack:
    def __init__(self, w=100): self._w = w; self._r = []
    def record(self, nb, us, step=0):
        bw = nb / max(1, us) * 1e6 / 1e9; self._r.append({'step': step, 'gbps': round(bw, 4)})
        if len(self._r) > self._w: self._r.pop(0)
    def avg(self): return sum(r['gbps'] for r in self._r) / max(1, len(self._r)) if self._r else 0
    def peak(self): return max((r['gbps'] for r in self._r), default=0)
# --- End M308 ---


# =========================================================================
# M325: DES-LOC Tier-Aware Torch Backend Extensions
# Reference: Megatron reduce_scatter_with_fp32_accumulation.py,
#            neuronx-distributed parallel_layers/comm.py,
#            veScale dtensor/_redistribute.py
# =========================================================================

import torch
import math
import time
import logging
from collections import OrderedDict
from enum import IntEnum

_desloc_torch_logger = logging.getLogger("desloc.torch")


class DeslocTier(IntEnum):
    """Parameter tier classification for DES-LOC communication scheduling.

    Tier 0 (GRADIENT): synced every Kx steps — loss-critical params (embed, LM head)
    Tier 1 (MOMENTUM): synced every Ku steps — attention projections
    Tier 2 (VARIANCE): synced every Kv steps — FFN, layernorm (least comm-sensitive)
    """
    GRADIENT = 0
    MOMENTUM = 1
    VARIANCE = 2


def _desloc_classify_param_tier(name, shape, total_params):
    """Classify parameter into DES-LOC tier by name and shape heuristics.

    Follows Megatron DistributedDataParallel bucket assignment logic,
    adapted for DES-LOC 3-tier scheme:
      - Embeddings and LM head → tier 0 (sync every Kx)
      - Attention QKV/out projections → tier 1 (sync every Ku)
      - FFN and layernorm → tier 2 (sync every Kv)

    Args:
        name: parameter name string
        shape: parameter shape tuple
        total_params: total model parameter count for relative sizing

    Returns:
        DeslocTier enum value
    """
    name_lower = name.lower()
    # Embedding and output head are loss-critical
    if any(k in name_lower for k in ('embed', 'lm_head', 'wte', 'wpe',
                                      'position', 'token_type')):
        return DeslocTier.GRADIENT
    # Attention projections: medium sensitivity
    if any(k in name_lower for k in ('attn', 'attention', 'self_attn',
                                      'q_proj', 'k_proj', 'v_proj',
                                      'query', 'key', 'value',
                                      'out_proj', 'o_proj')):
        return DeslocTier.MOMENTUM
    # Everything else: FFN, layernorm, biases
    return DeslocTier.VARIANCE


class DeslocFP32AccumHandle:
    """Async work handle for FP32-accumulated reduce-scatter.

    Mirrors Megatron's _ReduceScatterWithFP32AccumulationWorkHandle
    to track in-flight reduce-scatter ops with FP32 intermediate buffers.
    """

    def __init__(self, work_handle, output_tensor, fp32_buffer):
        self._work = work_handle
        self._output = output_tensor
        self._fp32_buf = fp32_buffer
        self._completed = False

    def wait(self):
        """Block until the reduce-scatter completes, then downcast result."""
        if self._completed:
            return
        if self._work is not None:
            self._work.wait()
        # Downcast FP32 accumulation buffer back to output dtype
        if self._fp32_buf is not None and self._output is not None:
            self._output.copy_(self._fp32_buf.to(self._output.dtype))
        self._completed = True

    @property
    def is_completed(self):
        if self._completed:
            return True
        if self._work is not None and hasattr(self._work, 'is_completed'):
            return self._work.is_completed()
        return False


def desloc_reduce_scatter_fp32(output, input_tensor, group=None,
                                async_op=False, op=None):
    """Reduce-scatter with FP32 intermediate accumulation.

    Reference: Megatron-LM reduce_scatter_with_fp32_accumulation.py
    Prevents precision loss during gradient reduction across workers,
    which is critical for DES-LOC where gradients accumulate over Kx steps
    and small numerical errors compound.

    Args:
        output: pre-allocated output tensor (any dtype)
        input_tensor: input tensor to reduce-scatter (any dtype)
        group: process group (default: WORLD)
        async_op: if True, returns DeslocFP32AccumHandle
        op: reduce operation (default: SUM)

    Returns:
        DeslocFP32AccumHandle if async_op else None
    """
    if op is None:
        op = torch.distributed.ReduceOp.SUM

    needs_upcast = input_tensor.dtype != torch.float32
    if needs_upcast:
        fp32_input = input_tensor.float()
        fp32_output = torch.empty(output.shape, dtype=torch.float32,
                                  device=output.device)
    else:
        fp32_input = input_tensor
        fp32_output = output

    # Use reduce_scatter_tensor if available (PyTorch 2.0+)
    if hasattr(torch.distributed, 'reduce_scatter_tensor'):
        work = torch.distributed.reduce_scatter_tensor(
            fp32_output, fp32_input, op=op, group=group, async_op=True
        )
    else:
        # Fallback: chunk input manually
        world_size = torch.distributed.get_world_size(group=group)
        input_list = list(fp32_input.chunk(world_size))
        work = torch.distributed.reduce_scatter(
            fp32_output, input_list, op=op, group=group, async_op=True
        )

    handle = DeslocFP32AccumHandle(
        work, output if needs_upcast else None,
        fp32_output if needs_upcast else None
    )

    if not async_op:
        handle.wait()
        return None
    return handle


class DeslocTieredAllReduceTorch:
    """Tier-aware AllReduce scheduler for torch.distributed backend.

    Extends TorchBackend.all_reduce with DES-LOC's 3-tier communication
    gating: each parameter tier has an independent sync period (Kx, Ku, Kv).
    At non-sync steps, AllReduce is skipped and gradients accumulate locally.

    Reference: neuronx-distributed parallel_layers/comm.py all_reduce(),
               veScale dtensor/_collective_utils.py mesh_scatter_ragged()

    Attributes:
        Kx: gradient sync period (tier 0)
        Ku: momentum sync period (tier 1)
        Kv: variance sync period (tier 2)
        warmup_steps: number of initial steps with full sync (Kx=1)
    """

    def __init__(self, Kx=32, Ku=96, Kv=192, warmup_steps=5,
                 group=None, fp32_reduce=True):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.warmup_steps = warmup_steps
        self.group = group
        self.fp32_reduce = fp32_reduce
        self._step = 0
        self._param_tier_map = {}
        self._comm_log = []
        self._skipped_bytes = 0
        self._sent_bytes = 0
        self._pending_handles = []
        self._bucket_buffers = OrderedDict()

    def register_param_tiers(self, model, custom_map=None):
        """Build parameter → tier mapping from model.

        Args:
            model: nn.Module to classify parameters for
            custom_map: optional dict {param_name: DeslocTier} overrides
        """
        total_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if custom_map and name in custom_map:
                tier = DeslocTier(custom_map[name])
            else:
                tier = _desloc_classify_param_tier(
                    name, param.shape, total_params
                )
            self._param_tier_map[name] = tier
            param._desloc_tier = tier
            param._desloc_name = name

    def _get_period(self, tier):
        """Return sync period for the given tier."""
        if tier == DeslocTier.GRADIENT:
            return self.Kx
        elif tier == DeslocTier.MOMENTUM:
            return self.Ku
        else:
            return self.Kv

    def should_sync(self, step, tier):
        """Determine whether to perform AllReduce at this step for this tier.

        During warmup, always sync (Kx=1 equivalent).
        After warmup, sync only at period boundaries.

        Args:
            step: current training step
            tier: DeslocTier enum value

        Returns:
            bool: True if AllReduce should proceed
        """
        if step < self.warmup_steps:
            return True
        period = self._get_period(tier)
        if period <= 1:
            return True
        return step % period == 0

    def tiered_all_reduce(self, tensor, param_name=None, tier=None,
                          step=None, op=None, async_op=False):
        """Perform tier-gated AllReduce on a gradient tensor.

        If the current step is not a sync boundary for this tier,
        the AllReduce is skipped entirely (gradient accumulates locally).

        Args:
            tensor: gradient tensor to reduce
            param_name: parameter name for tier lookup
            tier: explicit DeslocTier override
            step: training step (uses internal counter if None)
            op: reduce operation (default: SUM)
            async_op: non-blocking operation

        Returns:
            work handle if async_op and sync performed, else None
        """
        if step is None:
            step = self._step
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        if tier is None:
            tier = self._param_tier_map.get(
                param_name, DeslocTier.GRADIENT
            )

        nbytes = tensor.nelement() * tensor.element_size()

        if not self.should_sync(step, tier):
            self._skipped_bytes += nbytes
            self._comm_log.append({
                'step': step, 'tier': int(tier), 'action': 'skip',
                'bytes': nbytes, 'param': param_name or ''
            })
            return None

        # Perform the AllReduce
        self._sent_bytes += nbytes

        # For FP32 reduce: upcast, reduce, downcast
        if self.fp32_reduce and tensor.dtype != torch.float32:
            fp32_tensor = tensor.float()
            work = torch.distributed.all_reduce(
                fp32_tensor, op=op, group=self.group, async_op=True
            )
            if async_op:
                handle = _DeslocFP32ARHandle(work, tensor, fp32_tensor)
                self._pending_handles.append(handle)
                return handle
            else:
                work.wait()
                tensor.copy_(fp32_tensor.to(tensor.dtype))
        else:
            work = torch.distributed.all_reduce(
                tensor, op=op, group=self.group, async_op=async_op
            )
            if async_op:
                self._pending_handles.append(work)
            return work

        self._comm_log.append({
            'step': step, 'tier': int(tier), 'action': 'sync',
            'bytes': nbytes, 'param': param_name or ''
        })
        return None

    def flush_pending(self):
        """Wait for all pending async AllReduce operations."""
        for h in self._pending_handles:
            if h is not None:
                h.wait()
        self._pending_handles.clear()

    def step(self):
        """Advance internal step counter and flush pending ops."""
        self.flush_pending()
        self._step += 1

    def get_comm_stats(self):
        """Return communication reduction statistics."""
        total = self._sent_bytes + self._skipped_bytes
        return {
            'step': self._step,
            'sent_bytes': self._sent_bytes,
            'skipped_bytes': self._skipped_bytes,
            'total_bytes': total,
            'reduction_pct': round(
                100.0 * self._skipped_bytes / max(1, total), 2
            ),
            'tier_periods': {
                'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv
            }
        }

    def get_comm_log(self, last_n=100):
        """Return recent communication log entries."""
        return self._comm_log[-last_n:]

    def state_dict(self):
        """Serialize scheduler state for checkpointing."""
        return {
            'step': self._step,
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'warmup_steps': self.warmup_steps,
            'sent_bytes': self._sent_bytes,
            'skipped_bytes': self._skipped_bytes,
            'param_tier_map': {
                k: int(v) for k, v in self._param_tier_map.items()
            }
        }

    def load_state_dict(self, d):
        """Restore scheduler state from checkpoint."""
        self._step = d.get('step', 0)
        self.Kx = d.get('Kx', self.Kx)
        self.Ku = d.get('Ku', self.Ku)
        self.Kv = d.get('Kv', self.Kv)
        self.warmup_steps = d.get('warmup_steps', self.warmup_steps)
        self._sent_bytes = d.get('sent_bytes', 0)
        self._skipped_bytes = d.get('skipped_bytes', 0)
        raw_map = d.get('param_tier_map', {})
        self._param_tier_map = {
            k: DeslocTier(v) for k, v in raw_map.items()
        }


class _DeslocFP32ARHandle:
    """Internal handle for FP32-upcast AllReduce with deferred downcast."""

    def __init__(self, work, orig_tensor, fp32_tensor):
        self._work = work
        self._orig = orig_tensor
        self._fp32 = fp32_tensor
        self._done = False

    def wait(self):
        if self._done:
            return
        if self._work is not None:
            self._work.wait()
        self._orig.copy_(self._fp32.to(self._orig.dtype))
        self._done = True

    def is_completed(self):
        if self._done:
            return True
        if self._work is not None and hasattr(self._work, 'is_completed'):
            return self._work.is_completed()
        return False


class DeslocRedistribute(torch.autograd.Function):
    """Autograd-safe tensor redistribute for DES-LOC heterogeneous workers.

    Reference: veScale DTensor Redistribute (torch.autograd.Function)

    In hetero GPU setups (e.g. 2xA6000+H100), tensors may need to be
    redistributed with non-uniform chunk sizes. This function preserves
    autograd graph connectivity so that backward pass correctly routes
    gradients through the redistribute operation.

    Forward: all-to-all with potentially non-uniform splits
    Backward: transpose all-to-all (reverse the split mapping)
    """

    @staticmethod
    def forward(ctx, input_tensor, output_splits, input_splits, group):
        """Redistribute tensor across workers with non-uniform splits.

        Args:
            input_tensor: local tensor shard
            output_splits: list of output chunk sizes per rank
            input_splits: list of input chunk sizes per rank
            group: process group
        """
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        world_size = torch.distributed.get_world_size(group=group)
        total_out = sum(output_splits) if output_splits else (
            input_tensor.numel() // world_size * world_size
        )
        output = torch.empty(
            total_out, dtype=input_tensor.dtype,
            device=input_tensor.device
        )

        torch.distributed.all_to_all_single(
            output, input_tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Reverse redistribute: swap input/output splits."""
        grad_input = torch.empty(
            sum(ctx.input_splits) if ctx.input_splits else grad_output.numel(),
            dtype=grad_output.dtype, device=grad_output.device
        )
        # Backward = forward with swapped splits
        torch.distributed.all_to_all_single(
            grad_input, grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
            group=ctx.group
        )
        return grad_input, None, None, None


def desloc_redistribute(tensor, output_splits, input_splits, group=None):
    """Convenience wrapper for autograd-safe redistribute.

    Args:
        tensor: input tensor to redistribute
        output_splits: per-rank output sizes
        input_splits: per-rank input sizes
        group: process group

    Returns:
        redistributed tensor with autograd support
    """
    return DeslocRedistribute.apply(tensor, output_splits, input_splits, group)


class DeslocAsyncBucketManager:
    """Manages bucket-level async AllReduce with Kx gating.

    Reference: Megatron _ParamAndGradBucketGroup

    Groups parameters into fixed-size buckets and performs AllReduce
    at bucket granularity. Combined with DES-LOC tier gating, this
    enables per-bucket communication decisions.

    Each bucket tracks:
      - Parameters it contains and their tier
      - Accumulated gradient buffer
      - Whether it's ready for reduction
      - Pending async handles

    Attributes:
        bucket_size: max elements per bucket
        buckets: list of DeslocBucket instances
    """

    def __init__(self, model, bucket_size=40_000_000, tiered_ar=None,
                 group=None, pad_alignment=128):
        self.bucket_size = bucket_size
        self.group = group
        self.tiered_ar = tiered_ar
        self.pad_alignment = pad_alignment
        self.buckets = []
        self._param_to_bucket = {}
        self._build_buckets(model)

    def _build_buckets(self, model):
        """Partition parameters into gradient reduction buckets.

        Follows Megatron's strategy: iterate parameters in reverse order
        (matching typical backward pass order), filling buckets up to
        bucket_size elements. Each bucket gets assigned the highest
        (most frequent sync) tier among its parameters.

        Args:
            model: nn.Module to partition
        """
        params = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad]
        # Reverse order to match backward execution order
        params.reverse()

        current_bucket_params = []
        current_bucket_elems = 0
        bucket_idx = 0

        for name, param in params:
            numel = param.numel()
            # Pad to alignment boundary
            padded = ((numel + self.pad_alignment - 1)
                      // self.pad_alignment * self.pad_alignment)

            if (current_bucket_elems + padded > self.bucket_size
                    and current_bucket_params):
                # Flush current bucket
                self._create_bucket(bucket_idx, current_bucket_params)
                bucket_idx += 1
                current_bucket_params = []
                current_bucket_elems = 0

            current_bucket_params.append((name, param, padded))
            current_bucket_elems += padded

        # Final bucket
        if current_bucket_params:
            self._create_bucket(bucket_idx, current_bucket_params)

    def _create_bucket(self, idx, param_list):
        """Create a gradient reduction bucket from parameter list.

        Args:
            idx: bucket index
            param_list: list of (name, param, padded_numel) tuples
        """
        total_elems = sum(p[2] for p in param_list)
        # Determine bucket tier: use the minimum tier (highest priority)
        tiers = []
        for name, param, _ in param_list:
            t = getattr(param, '_desloc_tier', DeslocTier.VARIANCE)
            tiers.append(t)
        bucket_tier = min(tiers) if tiers else DeslocTier.VARIANCE

        bucket = DeslocBucket(
            idx=idx,
            tier=bucket_tier,
            total_elems=total_elems,
            param_names=[p[0] for p in param_list],
            dtype=param_list[0][1].dtype if param_list else torch.float32,
            device=param_list[0][1].device if param_list else 'cpu'
        )
        self.buckets.append(bucket)

        for name, param, _ in param_list:
            self._param_to_bucket[name] = bucket

    def reduce_bucket(self, bucket_idx, step, async_op=True):
        """Reduce a specific bucket with tier-aware gating.

        Args:
            bucket_idx: index into self.buckets
            step: current training step
            async_op: non-blocking reduce

        Returns:
            async handle if sync performed, None if skipped
        """
        if bucket_idx >= len(self.buckets):
            return None
        bucket = self.buckets[bucket_idx]

        if self.tiered_ar is not None:
            if not self.tiered_ar.should_sync(step, bucket.tier):
                bucket.mark_skipped(step)
                return None

        # Perform AllReduce on bucket buffer
        if bucket.grad_buffer is not None:
            handle = torch.distributed.all_reduce(
                bucket.grad_buffer,
                op=torch.distributed.ReduceOp.SUM,
                group=self.group,
                async_op=async_op
            )
            bucket.mark_synced(step)
            if async_op:
                bucket.pending_handle = handle
            return handle
        return None

    def reduce_all_buckets(self, step, async_op=True):
        """Reduce all buckets that are due for sync at this step."""
        handles = []
        for i, bucket in enumerate(self.buckets):
            h = self.reduce_bucket(i, step, async_op=async_op)
            if h is not None:
                handles.append(h)
        return handles

    def flush(self):
        """Wait for all pending bucket reductions."""
        for bucket in self.buckets:
            if bucket.pending_handle is not None:
                bucket.pending_handle.wait()
                bucket.pending_handle = None

    def get_bucket_stats(self):
        """Return per-bucket sync/skip statistics."""
        stats = []
        for b in self.buckets:
            stats.append({
                'idx': b.idx, 'tier': int(b.tier),
                'elems': b.total_elems,
                'synced': b.sync_count,
                'skipped': b.skip_count,
                'params': len(b.param_names)
            })
        return stats

    def state_dict(self):
        return {
            'bucket_size': self.bucket_size,
            'n_buckets': len(self.buckets),
            'stats': self.get_bucket_stats()
        }


class DeslocBucket:
    """Single gradient reduction bucket.

    Holds metadata and optional pre-allocated gradient buffer for
    a group of parameters that will be reduced together.
    """

    def __init__(self, idx, tier, total_elems, param_names,
                 dtype=torch.float32, device='cpu'):
        self.idx = idx
        self.tier = tier
        self.total_elems = total_elems
        self.param_names = param_names
        self.dtype = dtype
        self.device = device
        self.grad_buffer = None  # Allocated lazily on first backward
        self.pending_handle = None
        self.sync_count = 0
        self.skip_count = 0
        self.last_sync_step = -1
        self.last_skip_step = -1

    def allocate_buffer(self):
        """Lazily allocate contiguous gradient buffer."""
        if self.grad_buffer is None:
            self.grad_buffer = torch.zeros(
                self.total_elems, dtype=self.dtype, device=self.device
            )
        return self.grad_buffer

    def mark_synced(self, step):
        self.sync_count += 1
        self.last_sync_step = step

    def mark_skipped(self, step):
        self.skip_count += 1
        self.last_skip_step = step

    def reset_stats(self):
        self.sync_count = 0
        self.skip_count = 0


class DeslocCommOverlap:
    """Communication-computation overlap manager for DES-LOC.

    Reference: TransformerEngine userbuffers_forward_linear.py

    Manages overlapping AllReduce with forward/backward computation
    by streaming bucket reductions during backward pass. This maximizes
    GPU utilization by hiding communication latency behind compute.

    The overlap strategy:
      1. Register backward hooks on bucket boundary parameters
      2. When a bucket's last parameter computes its gradient, launch
         async AllReduce on that bucket
      3. Forward compute of next micro-batch can overlap with AllReduce

    Attributes:
        bucket_mgr: DeslocAsyncBucketManager instance
        enabled: whether overlap is active
    """

    def __init__(self, bucket_mgr, enabled=True):
        self.bucket_mgr = bucket_mgr
        self.enabled = enabled
        self._hooks = []
        self._step = 0
        self._overlap_events = []
        self._compute_stream = None
        self._comm_stream = None

    def setup_streams(self, device):
        """Create dedicated CUDA streams for comm/compute overlap.

        Args:
            device: torch device for stream creation
        """
        if device.type != 'cuda':
            self.enabled = False
            return
        self._compute_stream = torch.cuda.current_stream(device)
        self._comm_stream = torch.cuda.Stream(device=device)

    def register_hooks(self, model, step_fn=None):
        """Register backward hooks for overlapped bucket reduction.

        Args:
            model: nn.Module to hook
            step_fn: optional callback after each bucket reduces
        """
        if not self.enabled:
            return

        # Map each parameter to its bucket
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            bucket = self.bucket_mgr._param_to_bucket.get(name)
            if bucket is None:
                continue

            def _make_hook(bkt, pname):
                def hook(grad):
                    # Launch async reduce when grad is ready
                    if self._comm_stream is not None:
                        event = torch.cuda.Event()
                        event.record(self._compute_stream)
                        self._comm_stream.wait_event(event)
                        with torch.cuda.stream(self._comm_stream):
                            self.bucket_mgr.reduce_bucket(
                                bkt.idx, self._step, async_op=True
                            )
                    else:
                        self.bucket_mgr.reduce_bucket(
                            bkt.idx, self._step, async_op=True
                        )
                    return grad
                return hook

            h = param.register_hook(_make_hook(bucket, name))
            self._hooks.append(h)

    def step(self):
        """Advance step counter and flush pending communications."""
        self.bucket_mgr.flush()
        self._step += 1

    def remove_hooks(self):
        """Remove all registered backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def state_dict(self):
        return {
            'step': self._step,
            'enabled': self.enabled,
            'bucket_stats': self.bucket_mgr.state_dict()
        }


class DeslocSequenceParallelComm:
    """Communication primitives for AutoSP + DES-LOC integration.

    Handles the orthogonal composition of:
      - AutoSP: splits input along sequence dimension
      - DES-LOC: gates AllReduce along worker dimension with Kx

    The key insight is that these are orthogonal:
      - AutoSP partitions the sequence axis within each forward/backward
      - DES-LOC controls the frequency of cross-worker gradient sync
      - Both operate at ZeRO Stage 0, so no conflict in optimizer state

    This class provides the scatter/gather primitives for the sequence
    dimension, respecting DES-LOC's Kx boundaries.

    Args:
        seq_group: process group for sequence parallel
        dp_group: process group for data parallel (DES-LOC domain)
        Kx: DES-LOC gradient sync period
    """

    def __init__(self, seq_group=None, dp_group=None, Kx=32):
        self.seq_group = seq_group
        self.dp_group = dp_group
        self.Kx = Kx
        self._step = 0

    def scatter_along_seq(self, input_tensor, dim=1):
        """Scatter input tensor along sequence dimension.

        Splits the input evenly across workers in seq_group.

        Args:
            input_tensor: [batch, seq_len, hidden] tensor
            dim: dimension to scatter (default: 1 = sequence)

        Returns:
            local shard of shape [batch, seq_len // world_size, hidden]
        """
        if self.seq_group is None:
            return input_tensor
        world_size = torch.distributed.get_world_size(group=self.seq_group)
        if world_size <= 1:
            return input_tensor
        rank = torch.distributed.get_rank(group=self.seq_group)
        chunks = input_tensor.chunk(world_size, dim=dim)
        return chunks[rank].contiguous()

    def gather_along_seq(self, input_tensor, dim=1):
        """Gather tensor shards along sequence dimension.

        Args:
            input_tensor: local shard [batch, local_seq, hidden]
            dim: dimension to gather

        Returns:
            full tensor [batch, seq_len, hidden]
        """
        if self.seq_group is None:
            return input_tensor
        world_size = torch.distributed.get_world_size(group=self.seq_group)
        if world_size <= 1:
            return input_tensor

        gather_list = [torch.empty_like(input_tensor)
                       for _ in range(world_size)]
        torch.distributed.all_gather(
            gather_list, input_tensor, group=self.seq_group
        )
        return torch.cat(gather_list, dim=dim)

    def reduce_scatter_seq(self, input_tensor, dim=1, op=None):
        """Reduce-scatter along sequence dimension.

        Combines reduce and scatter: each worker gets the reduced
        result for its local sequence shard.

        Args:
            input_tensor: full-sequence gradient tensor
            dim: sequence dimension
            op: reduce operation

        Returns:
            local reduced shard
        """
        if self.seq_group is None:
            return input_tensor
        if op is None:
            op = torch.distributed.ReduceOp.SUM
        world_size = torch.distributed.get_world_size(group=self.seq_group)
        if world_size <= 1:
            return input_tensor

        chunk_size = input_tensor.size(dim) // world_size
        rank = torch.distributed.get_rank(group=self.seq_group)
        output_shape = list(input_tensor.shape)
        output_shape[dim] = chunk_size
        output = torch.empty(output_shape, dtype=input_tensor.dtype,
                             device=input_tensor.device)

        if hasattr(torch.distributed, 'reduce_scatter_tensor'):
            # Flatten for reduce_scatter_tensor API
            flat_input = input_tensor.contiguous().view(-1)
            flat_output = output.contiguous().view(-1)
            torch.distributed.reduce_scatter_tensor(
                flat_output, flat_input, op=op, group=self.seq_group
            )
            return flat_output.view(output_shape)
        else:
            # Fallback: manual chunk + reduce_scatter
            input_list = list(input_tensor.chunk(world_size, dim=dim))
            input_list = [c.contiguous() for c in input_list]
            torch.distributed.reduce_scatter(
                output, input_list, op=op, group=self.seq_group
            )
            return output

    def dp_gated_allreduce(self, tensor, step=None):
        """Data-parallel AllReduce gated by DES-LOC Kx period.

        Only performs cross-worker AllReduce at Kx boundaries.
        Called after sequence-parallel reduce-scatter completes.

        Args:
            tensor: gradient tensor already reduced along seq dim
            step: training step for Kx gating

        Returns:
            tensor (modified in-place if sync performed)
        """
        if step is None:
            step = self._step
        if self.dp_group is None:
            return tensor
        if self.Kx <= 1 or step % self.Kx == 0:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.SUM,
                group=self.dp_group
            )
        return tensor

    def step(self):
        self._step += 1


class DeslocHeteroBalancer:
    """Load balancer for heterogeneous GPU configurations.

    Computes optimal micro-batch sizes and sequence partition ratios
    for mixed GPU setups (e.g., A6000 + H100 in the same job).

    Each GPU reports its compute capability (TFLOPS) and memory,
    and the balancer assigns proportional work shares.

    Args:
        gpu_specs: list of dicts with 'name', 'tflops', 'mem_gb'
    """

    # Reference GPU specs database (from M302 cuda_accelerator.py)
    GPU_DB = {
        'A6000': {'tflops_bf16': 38.7, 'mem_gb': 48, 'nvlink': False},
        'A100-40G': {'tflops_bf16': 312.0, 'mem_gb': 40, 'nvlink': True},
        'A100-80G': {'tflops_bf16': 312.0, 'mem_gb': 80, 'nvlink': True},
        'H100-SXM': {'tflops_bf16': 989.5, 'mem_gb': 80, 'nvlink': True},
        'H100-PCIe': {'tflops_bf16': 756.0, 'mem_gb': 80, 'nvlink': False},
        'L40S': {'tflops_bf16': 362.0, 'mem_gb': 48, 'nvlink': False},
        'RTX4090': {'tflops_bf16': 165.2, 'mem_gb': 24, 'nvlink': False},
        'V100-32G': {'tflops_bf16': 125.0, 'mem_gb': 32, 'nvlink': True},
    }

    def __init__(self, gpu_specs=None):
        self.gpu_specs = gpu_specs or []
        self._ratios = None

    def detect_gpus(self):
        """Auto-detect GPU specs from CUDA device properties."""
        if not torch.cuda.is_available():
            return
        self.gpu_specs = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            name = props.name
            mem_gb = props.total_mem / (1024 ** 3)
            # Estimate BF16 TFLOPS from SMs and clock
            sm_count = props.multi_processor_count
            clock_ghz = props.clock_rate / 1e6
            # Rough heuristic: each SM does ~256 BF16 ops/cycle (Ampere+)
            tflops_est = sm_count * clock_ghz * 256 / 1e3
            self.gpu_specs.append({
                'idx': i, 'name': name,
                'tflops_bf16': round(tflops_est, 1),
                'mem_gb': round(mem_gb, 1)
            })

    def compute_ratios(self):
        """Compute proportional work ratios based on GPU TFLOPS.

        Returns:
            list of float ratios (sum to 1.0)
        """
        if not self.gpu_specs:
            self._ratios = [1.0]
            return self._ratios
        total_tflops = sum(g['tflops_bf16'] for g in self.gpu_specs)
        if total_tflops <= 0:
            n = len(self.gpu_specs)
            self._ratios = [1.0 / n] * n
            return self._ratios
        self._ratios = [
            g['tflops_bf16'] / total_tflops for g in self.gpu_specs
        ]
        return self._ratios

    def micro_batch_sizes(self, global_batch, min_mb=1):
        """Compute per-GPU micro-batch sizes proportional to TFLOPS.

        Args:
            global_batch: total batch size across all GPUs
            min_mb: minimum micro-batch per GPU

        Returns:
            list of per-GPU micro-batch sizes
        """
        if self._ratios is None:
            self.compute_ratios()
        n = len(self._ratios)
        raw = [max(min_mb, int(global_batch * r)) for r in self._ratios]
        # Adjust to match global_batch exactly
        diff = global_batch - sum(raw)
        if diff > 0:
            # Add remainder to fastest GPU
            fastest_idx = max(range(n), key=lambda i: self._ratios[i])
            raw[fastest_idx] += diff
        elif diff < 0:
            # Remove from slowest GPU
            slowest_idx = min(range(n), key=lambda i: self._ratios[i])
            raw[slowest_idx] = max(min_mb, raw[slowest_idx] + diff)
        return raw

    def seq_partition_sizes(self, seq_len):
        """Compute per-GPU sequence partition sizes for AutoSP.

        Args:
            seq_len: total sequence length

        Returns:
            list of per-GPU sequence lengths (sum = seq_len)
        """
        if self._ratios is None:
            self.compute_ratios()
        n = len(self._ratios)
        # Ensure divisible by 8 for tensor core alignment
        align = 8
        raw = [max(align, int(seq_len * r) // align * align)
               for r in self._ratios]
        diff = seq_len - sum(raw)
        if diff != 0:
            idx = max(range(n), key=lambda i: self._ratios[i])
            raw[idx] += diff
        return raw

    def optimal_Kx_per_gpu(self, base_Kx=32):
        """Suggest per-GPU Kx values based on interconnect topology.

        Faster GPUs can tolerate larger Kx (more local steps)
        because they contribute more compute per unit time.
        Slower GPUs should sync more often to stay aligned.

        Args:
            base_Kx: baseline Kx for the fastest GPU

        Returns:
            list of per-GPU Kx values
        """
        if self._ratios is None:
            self.compute_ratios()
        max_ratio = max(self._ratios) if self._ratios else 1.0
        kx_list = []
        for r in self._ratios:
            # Faster GPU → higher Kx (more local steps)
            scale = r / max_ratio
            kx = max(1, int(base_Kx * scale))
            kx_list.append(kx)
        return kx_list

    def report(self):
        """Generate human-readable load balance report."""
        if self._ratios is None:
            self.compute_ratios()
        lines = ["DES-LOC Heterogeneous GPU Load Balance Report",
                 "=" * 50]
        for i, (spec, ratio) in enumerate(
                zip(self.gpu_specs, self._ratios)):
            lines.append(
                f"  GPU {spec.get('idx', i)}: {spec['name']}  "
                f"BF16={spec['tflops_bf16']} TFLOPS  "
                f"MEM={spec['mem_gb']} GB  "
                f"share={ratio:.1%}"
            )
        return '\n'.join(lines)


# =========================================================================
# Module-level convenience functions
# =========================================================================

_desloc_tiered_ar_torch = None
_desloc_bucket_mgr = None
_desloc_comm_overlap = None
_desloc_seq_par = None


def init_desloc_torch_extensions(model, Kx=32, Ku=96, Kv=192,
                                  warmup_steps=5, bucket_size=40_000_000,
                                  group=None, enable_overlap=True,
                                  seq_group=None, custom_tier_map=None):
    """Initialize all DES-LOC torch backend extensions.

    This is the single entry point called from engine.py after model init.

    Args:
        model: nn.Module
        Kx, Ku, Kv: tier sync periods
        warmup_steps: steps with full sync at start
        bucket_size: gradient bucket size in elements
        group: data-parallel process group
        enable_overlap: enable comm-compute overlap
        seq_group: sequence-parallel process group (for AutoSP)
        custom_tier_map: optional {param_name: tier} overrides
    """
    global _desloc_tiered_ar_torch, _desloc_bucket_mgr
    global _desloc_comm_overlap, _desloc_seq_par

    # 1. Create tiered AllReduce scheduler
    _desloc_tiered_ar_torch = DeslocTieredAllReduceTorch(
        Kx=Kx, Ku=Ku, Kv=Kv, warmup_steps=warmup_steps,
        group=group, fp32_reduce=True
    )
    _desloc_tiered_ar_torch.register_param_tiers(
        model, custom_map=custom_tier_map
    )

    # 2. Build bucket manager
    _desloc_bucket_mgr = DeslocAsyncBucketManager(
        model, bucket_size=bucket_size,
        tiered_ar=_desloc_tiered_ar_torch, group=group
    )

    # 3. Setup comm-compute overlap
    if enable_overlap:
        _desloc_comm_overlap = DeslocCommOverlap(
            _desloc_bucket_mgr, enabled=True
        )
        device = next(model.parameters()).device
        _desloc_comm_overlap.setup_streams(device)
        _desloc_comm_overlap.register_hooks(model)
    else:
        _desloc_comm_overlap = None

    # 4. AutoSP integration (if seq_group provided)
    if seq_group is not None:
        _desloc_seq_par = DeslocSequenceParallelComm(
            seq_group=seq_group, dp_group=group, Kx=Kx
        )
    else:
        _desloc_seq_par = None

    _desloc_torch_logger.info(
        f"DES-LOC torch extensions initialized: "
        f"Kx={Kx} Ku={Ku} Kv={Kv} warmup={warmup_steps} "
        f"buckets={len(_desloc_bucket_mgr.buckets)} "
        f"overlap={enable_overlap} seq_par={seq_group is not None}"
    )


def get_desloc_tiered_ar_torch():
    return _desloc_tiered_ar_torch


def get_desloc_bucket_mgr():
    return _desloc_bucket_mgr


def get_desloc_comm_overlap():
    return _desloc_comm_overlap


def get_desloc_seq_par():
    return _desloc_seq_par


def desloc_torch_step():
    """Advance all DES-LOC torch extension step counters."""
    if _desloc_tiered_ar_torch is not None:
        _desloc_tiered_ar_torch.step()
    if _desloc_comm_overlap is not None:
        _desloc_comm_overlap.step()
    if _desloc_seq_par is not None:
        _desloc_seq_par.step()


def desloc_torch_state_dict():
    """Collect state dicts from all DES-LOC torch extensions."""
    sd = {}
    if _desloc_tiered_ar_torch is not None:
        sd['tiered_ar'] = _desloc_tiered_ar_torch.state_dict()
    if _desloc_bucket_mgr is not None:
        sd['bucket_mgr'] = _desloc_bucket_mgr.state_dict()
    if _desloc_comm_overlap is not None:
        sd['comm_overlap'] = _desloc_comm_overlap.state_dict()
    return sd


def desloc_torch_load_state_dict(sd):
    """Restore state dicts for all DES-LOC torch extensions."""
    if _desloc_tiered_ar_torch is not None and 'tiered_ar' in sd:
        _desloc_tiered_ar_torch.load_state_dict(sd['tiered_ar'])
# --- End M325 ---
