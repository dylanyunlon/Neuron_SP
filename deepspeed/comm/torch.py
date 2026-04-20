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

class DeslocTorchCommTracker:
    """Track DES-LOC communication volume per tier on torch backend.
    Ref: Algorithm 1 - independent sync periods for params/momenta."""

    def __init__(self):
        self._tier_bytes = {0: 0, 1: 0, 2: 0}
        self._tier_ops = {0: 0, 1: 0, 2: 0}

    def record(self, tier, num_bytes):
        """Record a communication event."""
        t = min(tier, 2)
        self._tier_bytes[t] += num_bytes
        self._tier_ops[t] += 1

    def total_bytes(self):
        return sum(self._tier_bytes.values())

    def tier_summary(self):
        return {f'tier{t}': {'bytes': b, 'ops': self._tier_ops[t]}
                for t, b in self._tier_bytes.items()}

    def reset(self):
        self._tier_bytes = {0: 0, 1: 0, 2: 0}
        self._tier_ops = {0: 0, 1: 0, 2: 0}


# =========================================================================
# M213: DES-LOC TorchBackend Extensions (Algorithm 1 + Section 4.1)
# Tier-aware AllReduce, NCCL group management, async comm overlap
# =========================================================================

import math
import time
from collections import defaultdict, deque


class DeslocTieredAllReduce:
    """Tier-aware AllReduce that respects DES-LOC sync periods.

    Wraps torch.distributed collective operations to gate them by tier.
    Tier 0 (params) syncs at Kx, Tier 1 (momentum) at Ku, Tier 2 (variance) at Kv.

    Ref: Algorithm 1 lines 14-19 — conditional sync based on t mod K.
    Ref: Section 4.1 — 'Ring-AllReduce algorithm scaling linearly with model size.'

    Key invariant: when Kx=1, this is identical to standard DDP AllReduce.
    """

    def __init__(self, scheduler):
        """Args:
            scheduler: DeslocCommScheduler instance that tracks step/period.
        """
        self._sched = scheduler
        self._pending_ops = []
        self._tier_buffers = {0: [], 1: [], 2: []}
        self._comm_log = deque(maxlen=1000)

    def maybe_allreduce(self, tensor, tier=0, group=None, async_op=False,
                        is_safety_critical=False):
        """Conditionally AllReduce tensor based on DES-LOC schedule.

        Args:
            tensor: torch.Tensor to reduce.
            tier: 0=param, 1=momentum, 2=variance.
            group: process group (None = default world group).
            async_op: if True, return a handle for later wait().
            is_safety_critical: if True, always sync (MoE capacity, overflow).

        Returns:
            handle if async_op and sync happened, None otherwise.
        """
        should = self._sched.should_sync(tier, is_safety_critical)
        num_bytes = tensor.numel() * tensor.element_size()

        self._sched.record_comm(tier, num_bytes, should)

        if not should:
            return None

        try:
            import torch.distributed as dist
            handle = dist.all_reduce(tensor, group=group, async_op=async_op)
            self._comm_log.append({
                'step': self._sched.step,
                'tier': tier,
                'bytes': num_bytes,
                'async': async_op,
                'ts': time.monotonic(),
            })
            return handle
        except Exception:
            return None

    def maybe_allreduce_with_average(self, tensor, tier=0, group=None,
                                     world_size=None, is_safety_critical=False):
        """AllReduce + divide by world_size (averaging).
        This is what Algorithm 1 line 15 does: E_m[s] = average across workers."""
        handle = self.maybe_allreduce(tensor, tier, group,
                                      is_safety_critical=is_safety_critical)
        if handle is not None:
            if world_size is not None and world_size > 1:
                tensor.div_(world_size)
        elif self._sched.should_sync(tier, is_safety_critical):
            if world_size is not None and world_size > 1:
                tensor.div_(world_size)
        return handle

    def flush_pending(self):
        """Wait for all pending async operations."""
        for op in self._pending_ops:
            if op is not None:
                op.wait()
        self._pending_ops.clear()

    def recent_comm_bytes(self, last_n=100):
        """Total bytes communicated in last N logged operations."""
        recent = list(self._comm_log)[-last_n:]
        return sum(r['bytes'] for r in recent)


class DeslocNCCLGroupManager:
    """Manage NCCL process groups for DES-LOC tier-separated communication.

    In standard DDP, one process group handles all AllReduce.
    DES-LOC can optionally use separate groups per tier for isolation,
    but defaults to sharing the data-parallel group.

    Ref: Section 4.1 — 'bandwidth-optimal Ring-AllReduce'
    uses the same group topology but at different frequencies.
    """

    def __init__(self):
        self._tier_groups = {}
        self._default_group = None
        self._world_size = 1

    def set_default_group(self, group, world_size):
        """Set the default data-parallel group."""
        self._default_group = group
        self._world_size = world_size

    def set_tier_group(self, tier, group):
        """Optionally assign a dedicated group to a tier."""
        self._tier_groups[int(tier)] = group

    def get_group(self, tier):
        """Get process group for tier (falls back to default)."""
        return self._tier_groups.get(int(tier), self._default_group)

    @property
    def world_size(self):
        return self._world_size


class DeslocAsyncOverlapManager:
    """Overlap DES-LOC communication with computation.

    Strategy: on non-sync steps, compute runs unblocked.
    On sync steps, we overlap the AllReduce with the next micro-batch's forward.

    Ref: Section 4.1 — DES-LOC sync naturally provides overlap opportunities
    because non-sync steps are pure compute.
    """

    def __init__(self):
        self._pending_handles = {}
        self._overlap_stats = {'overlapped': 0, 'waited': 0}

    def submit(self, tier, handle):
        """Submit an async AllReduce handle."""
        if handle is not None:
            self._pending_handles[tier] = handle

    def wait(self, tier):
        """Wait for a specific tier's pending AllReduce."""
        h = self._pending_handles.pop(tier, None)
        if h is not None:
            h.wait()
            self._overlap_stats['waited'] += 1

    def wait_all(self):
        """Wait for all pending AllReduces."""
        for tier in list(self._pending_handles.keys()):
            self.wait(tier)

    def has_pending(self, tier):
        return tier in self._pending_handles

    def stats(self):
        return dict(self._overlap_stats)


class DeslocGradientBucketer:
    """Bucket gradients by DES-LOC tier for efficient AllReduce.

    Standard DDP buckets all gradients together.
    DES-LOC separates them by tier so each tier can be AllReduced independently.

    Ref: Algorithm 1 — parameters (tier 0) and optimizer states (tiers 1,2)
    have independent sync schedules.
    """

    def __init__(self, bucket_size_mb=25.0):
        self._bucket_bytes = int(bucket_size_mb * 1024 * 1024)
        self._tier_buckets = {0: [], 1: [], 2: []}
        self._tier_sizes = {0: 0, 1: 0, 2: 0}

    def assign_param_to_tier(self, param_name, param):
        """Assign a parameter to a DES-LOC tier based on its role.
        Parameters → tier 0, exp_avg → tier 1, exp_avg_sq → tier 2."""
        name_lower = param_name.lower()
        if 'exp_avg_sq' in name_lower or 'v_hat' in name_lower:
            tier = 2
        elif 'exp_avg' in name_lower or 'momentum_buffer' in name_lower:
            tier = 1
        else:
            tier = 0
        nbytes = param.numel() * param.element_size()
        self._tier_buckets[tier].append({
            'name': param_name,
            'numel': param.numel(),
            'bytes': nbytes,
        })
        self._tier_sizes[tier] += nbytes
        return tier

    def tier_total_bytes(self, tier):
        return self._tier_sizes.get(int(tier), 0)

    def num_buckets_for_tier(self, tier):
        """How many AllReduce buckets needed for this tier."""
        total = self._tier_sizes.get(int(tier), 0)
        if total == 0:
            return 0
        return max(1, int(math.ceil(total / self._bucket_bytes)))

    def summary(self):
        return {
            f'tier{t}': {
                'params': len(self._tier_buckets[t]),
                'bytes': self._tier_sizes[t],
                'buckets': self.num_buckets_for_tier(t),
            }
            for t in range(3)
        }


class DeslocEFABandwidthEstimator:
    """Estimate available bandwidth on AWS EFA (Elastic Fabric Adapter).

    Anthropic uses AWS infrastructure (per Nick Joseph interview).
    EFA provides 100-400 Gbps inter-node bandwidth via SRD protocol.
    Intra-node uses NVLink (600 GB/s on H100) or PCIe.

    This estimates effective bandwidth for DES-LOC comm scheduling.
    Ref: Section 4.1 — '100Gb/s links' (Table 1).
    """

    # Known bandwidth tiers (Gbps)
    NVLINK_H100 = 900.0
    NVLINK_A100 = 600.0
    PCIE_GEN4 = 32.0
    PCIE_GEN5 = 64.0
    EFA_V1 = 100.0
    EFA_V2 = 400.0

    def __init__(self, intra_node_gbps=None, inter_node_gbps=None):
        self._intra = intra_node_gbps or self.NVLINK_A100
        self._inter = inter_node_gbps or self.EFA_V1

    def effective_bandwidth(self, is_intra_node):
        """Return bandwidth in Gbps."""
        return self._intra if is_intra_node else self._inter

    def allreduce_time_estimate(self, msg_bytes, num_workers, is_intra):
        """Estimate Ring-AllReduce time in seconds."""
        bw = self.effective_bandwidth(is_intra) * 1e9 / 8.0  # bytes/sec
        coeff = 2.0 * (num_workers - 1) / max(num_workers, 1)
        return coeff * msg_bytes / max(bw, 1.0)

    def recommend_Kx(self, param_bytes, compute_time_sec, num_workers,
                     is_intra=False):
        """Recommend minimum Kx to make training compute-bound.
        Kx should be large enough that comm_time/Kx < compute_time.
        Ref: Section 5.4 — 'set Kx for sufficient throughput based on bandwidth.'"""
        ar_time = self.allreduce_time_estimate(param_bytes, num_workers, is_intra)
        if compute_time_sec <= 0 or ar_time <= 0:
            return 1
        Kx = int(math.ceil(ar_time / compute_time_sec))
        # Round up to power of 2 for cleaner scheduling
        p = 1
        while p < Kx:
            p *= 2
        return max(1, p)


class DeslocPerCoordinateClipper:
    """Per-coordinate gradient clipping as specified in DES-LOC Algorithm 1.

    Ref: Section 2 — 'With coordinate-wise clipping, each gradient component
    satisfies |(g_t)_i| <= rho.'
    Algorithm 1 line 12: g_hat_t^m <- clip(g_t^m, rho)

    Standard gradient clipping clips by NORM (L2).
    DES-LOC uses COORDINATE-WISE clipping: clamp each element independently.
    This is crucial for the convergence guarantees in Theorem 1.
    """

    def __init__(self, rho=1.0):
        self._rho = max(float(rho), 0.0)
        self._clip_count = 0
        self._total_count = 0

    @property
    def rho(self):
        return self._rho

    def clip(self, gradient_tensor):
        """Apply per-coordinate clipping: clip(X,rho)_i = sgn(X_i)*min(|X_i|, rho).
        This is equivalent to torch.clamp(tensor, -rho, rho)."""
        if self._rho <= 0:
            return gradient_tensor
        self._total_count += 1
        max_val = gradient_tensor.abs().max().item()
        if max_val > self._rho:
            self._clip_count += 1
            gradient_tensor.clamp_(-self._rho, self._rho)
        return gradient_tensor

    def clip_fraction(self):
        """Fraction of steps where any coordinate was clipped."""
        if self._total_count == 0:
            return 0.0
        return self._clip_count / self._total_count

    def stats(self):
        return {
            'rho': self._rho,
            'total': self._total_count,
            'clipped': self._clip_count,
            'clip_fraction': round(self.clip_fraction(), 4),
        }


class DeslocNesterovOuterOptimizer:
    """Nesterov momentum outer optimizer for parameter averaging.

    Ref: Section 5.5 — 'using Nesterov as the outer optimizer improves
    performance over averaging by ~0.03 eval loss.'

    At each Kx sync boundary:
      v_{k+1} = mu * v_k + (x_avg - x_prev)
      x_{k+1} = x_avg + mu * v_{k+1}

    With mu=0 this degrades to simple averaging (standard DES-LOC).
    """

    def __init__(self, momentum=0.9, lr=1.0):
        self._mu = momentum
        self._lr = lr
        self._velocity = {}
        self._prev_params = {}
        self._apply_count = 0

    def apply(self, param_name, averaged_param, param_tensor):
        """Apply Nesterov outer step after AllReduce averaging.

        Args:
            param_name: unique key for this parameter.
            averaged_param: result of AllReduce average (x_avg).
            param_tensor: the tensor to update in-place.
        """
        pid = param_name
        if pid not in self._velocity:
            self._velocity[pid] = param_tensor.new_zeros(param_tensor.shape)
            self._prev_params[pid] = averaged_param.clone()
            param_tensor.copy_(averaged_param)
            return

        delta = averaged_param.float() - self._prev_params[pid].float()
        v = self._velocity[pid]
        v.mul_(self._mu).add_(delta.to(v.dtype), alpha=self._lr)
        param_tensor.copy_(averaged_param)
        param_tensor.add_(v.to(param_tensor.dtype), alpha=self._mu)
        self._prev_params[pid].copy_(param_tensor)
        self._apply_count += 1

    def state_dict(self):
        return {'mu': self._mu, 'lr': self._lr, 'count': self._apply_count}

    def load_state_dict(self, sd):
        self._mu = sd.get('mu', self._mu)
        self._lr = sd.get('lr', self._lr)
