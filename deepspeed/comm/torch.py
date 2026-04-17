# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
from deepspeed import utils
from packaging import version
import inspect

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
            kwargs = dict(timeout=timeout, init_method=init_method, rank=rank, world_size=world_size)

            # 1. device_id arg was added in torch==2.3
            # 2. setting device_id leads to hanging in 2.6.0<torch<2.7.1 https://github.com/pytorch/pytorch/issues/153960
            # 3. device_id works and is needed for `cuda`, other accelerators may have issues at the moment. Therefore only do it for the `cuda` accelerator.
            if ('device_id' in inspect.signature(torch.distributed.init_process_group).parameters
                    and not (version.parse("2.6.0") < version.parse(torch.__version__) < version.parse("2.7.1"))
                    and get_accelerator().device_name() == 'cuda'):
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                kwargs.update(device_id=get_accelerator().device(local_rank))
            torch.distributed.init_process_group(backend, **kwargs)

        self.using_mpi = torch.distributed.get_backend() == 'mpi'

    @disable_compiler_collective
    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        op = self._reduce_op(op)
        return torch.distributed.all_reduce(tensor=tensor, op=op, group=group, async_op=async_op)

    def inference_all_reduce(self, tensor, op, group=None):
        use_ds_op = hasattr(torch.ops, 'deepspeed') and hasattr(torch.ops.deepspeed, 'inference_all_reduce_')
        world_size = torch.distributed.get_world_size(group=group)
        if world_size <= 1:
            return tensor
        if not use_ds_op:
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
    def all_gather_object(self, object_list, obj, group=None):
        return torch.distributed.all_gather_object(object_list=object_list, obj=obj, group=group)

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

    def enable_symm_mem_for_group(self, group_name):
        if not required_torch_version(min_version=2.5):
            raise RuntimeError(f"Torch version must be 2.5 or higher to use symmetric memory. "
                               f"Current version: {torch.__version__}")
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        return enable_symm_mem_for_group(group_name)


# This will become a light-weight wrapper around torch.distributed functions
# TODO: create some example to show how this wrapper can help profile communication
# TODO: make sure there is no performance regression with this approach
# TODO: explore monkey-patching if this does not work


# =====================================================================
# M057: DES-LOC NCCL Communication Hooks (400 lines)
# =====================================================================
# Instruments the TorchBackend with DES-LOC-aware communication
# tracking, bandwidth measurement, and conditional allreduce.
# Architecture reference: CCCL c/parallel/src/util/context.cpp
# =====================================================================

import time as _time
import json as _json
from collections import deque as _deque


class DESLOCCommInstrumentation:
    """
    Instrumentation layer for DES-LOC communication.
    Wraps TorchBackend to measure real NCCL bandwidth and latency.
    All data from actual torch.distributed calls — no simulation.
    """

    def __init__(self, backend, world_size=1):
        self._backend = backend
        self._world_size = world_size
        self._enabled = True
        self._op_count = 0
        self._total_bytes = 0
        self._total_time_s = 0
        self._allreduce_history = _deque(maxlen=5000)
        self._broadcast_history = _deque(maxlen=1000)
        self._reduce_scatter_history = _deque(maxlen=1000)
        self._allgather_history = _deque(maxlen=1000)
        self._skipped_ops = 0
        self._step = 0

    def advance_step(self):
        """Advance the DES-LOC step counter."""
        self._step += 1

    def record_allreduce(self, tensor_numel, element_size, elapsed_ms,
                          op_name='all_reduce', was_skipped=False):
        """Record a single allreduce operation."""
        if not self._enabled:
            return
        self._op_count += 1
        tensor_bytes = tensor_numel * element_size

        if was_skipped:
            self._skipped_ops += 1
            self._allreduce_history.append({
                'step': self._step,
                'op': op_name,
                'bytes': tensor_bytes,
                'ms': 0,
                'bw_gbps': 0,
                'skipped': True,
            })
            return

        self._total_bytes += tensor_bytes
        self._total_time_s += elapsed_ms / 1000

        bw_gbps = 0
        if elapsed_ms > 0:
            # Ring allreduce transfers 2*(N-1)/N * data
            algo_bytes = tensor_bytes * 2 * (self._world_size - 1) / max(self._world_size, 1)
            bw_gbps = (algo_bytes / 1e9) / (elapsed_ms / 1000)

        self._allreduce_history.append({
            'step': self._step,
            'op': op_name,
            'bytes': tensor_bytes,
            'ms': round(elapsed_ms, 4),
            'bw_gbps': round(bw_gbps, 2),
            'skipped': False,
        })

    def record_broadcast(self, tensor_bytes, elapsed_ms):
        """Record a broadcast operation."""
        self._broadcast_history.append({
            'step': self._step,
            'bytes': tensor_bytes,
            'ms': round(elapsed_ms, 4),
        })

    def get_allreduce_stats(self):
        """Return allreduce statistics."""
        if not self._allreduce_history:
            return {'total_ops': 0}

        active = [h for h in self._allreduce_history if not h['skipped']]
        bws = [h['bw_gbps'] for h in active if h['bw_gbps'] > 0]
        times = [h['ms'] for h in active if h['ms'] > 0]

        return {
            'total_ops': self._op_count,
            'active_ops': len(active),
            'skipped_ops': self._skipped_ops,
            'skip_ratio': round(self._skipped_ops / max(self._op_count, 1), 4),
            'total_bytes': self._total_bytes,
            'total_gb': round(self._total_bytes / 1e9, 4),
            'total_time_s': round(self._total_time_s, 3),
            'avg_bw_gbps': round(sum(bws) / len(bws), 2) if bws else 0,
            'max_bw_gbps': round(max(bws), 2) if bws else 0,
            'avg_latency_ms': round(sum(times) / len(times), 3) if times else 0,
            'p95_latency_ms': round(
                sorted(times)[int(len(times) * 0.95)] if times else 0, 3),
        }

    def get_full_stats(self):
        """Return all communication statistics."""
        stats = {
            'allreduce': self.get_allreduce_stats(),
            'world_size': self._world_size,
            'total_steps': self._step,
        }
        if self._broadcast_history:
            stats['broadcasts'] = len(self._broadcast_history)
        return stats

    def save_log(self, path):
        """Save communication log to JSON."""
        output = {
            'stats': self.get_full_stats(),
            'allreduce_log': list(self._allreduce_history)[-2000:],
        }
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            _json.dump(output, f, indent=2)


class DESLOCTorchAllReduce:
    """
    DES-LOC-aware allreduce wrapper.

    Intercepts torch.distributed.all_reduce calls and applies
    the DES-LOC communication schedule (sync only at Kx/Ku/Kv steps).

    Usage:
        desloc_ar = DESLOCTorchAllReduce(Kx=32, Ku=96, Kv=192, world_size=8)
        # In training loop:
        desloc_ar.advance_step()
        desloc_ar.conditional_allreduce(param_tensor, state_type='x')
        desloc_ar.conditional_allreduce(exp_avg_tensor, state_type='u')
        desloc_ar.conditional_allreduce(exp_avg_sq_tensor, state_type='v')
    """

    def __init__(self, Kx=32, Ku=96, Kv=192, world_size=1,
                 clip_radius=1.0, instrumentation=None):
        self._Kx = Kx
        self._Ku = Ku
        self._Kv = Kv
        self._world_size = world_size
        self._clip_radius = clip_radius
        self._step = 0
        self._instrument = instrumentation or DESLOCCommInstrumentation(
            None, world_size)

    def advance_step(self):
        """Advance the step counter."""
        self._step += 1
        self._instrument.advance_step()

    def should_sync(self, state_type):
        """Check if a state should be synced at the current step."""
        if state_type == 'x':
            return self._step % self._Kx == 0
        elif state_type == 'u':
            return self._step % self._Ku == 0
        elif state_type == 'v':
            return self._step % self._Kv == 0
        return True  # unknown: always sync

    def conditional_allreduce(self, tensor, state_type='x',
                               op=None, group=None, async_op=False):
        """Perform allreduce only if the DES-LOC schedule says to.

        Returns True if allreduce was performed, False if skipped.
        """
        if self._world_size <= 1:
            return False

        if not self.should_sync(state_type):
            # Skip: record as skipped
            self._instrument.record_allreduce(
                tensor.numel(), tensor.element_size(), 0,
                op_name=f'allreduce_{state_type}', was_skipped=True)
            return False

        # Perform actual allreduce
        t0 = _time.time()
        if op is None:
            op = torch.distributed.ReduceOp.AVG
        torch.distributed.all_reduce(tensor, op=op, group=group,
                                      async_op=async_op)
        if not async_op:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        elapsed_ms = (_time.time() - t0) * 1000

        self._instrument.record_allreduce(
            tensor.numel(), tensor.element_size(), elapsed_ms,
            op_name=f'allreduce_{state_type}', was_skipped=False)
        return True

    def clip_and_allreduce(self, grad_tensor, state_type='x',
                            op=None, group=None):
        """Apply per-coordinate clipping then conditional allreduce.

        Combines Algorithm 1 line 168 (clipping) with the sync schedule.
        """
        # Per-coordinate clipping
        if self._clip_radius > 0:
            grad_tensor.clamp_(-self._clip_radius, self._clip_radius)

        return self.conditional_allreduce(
            grad_tensor, state_type=state_type, op=op, group=group)

    def get_comm_reduction(self):
        """Compute communication reduction vs DDP."""
        n = max(self._step, 1)
        ddp_syncs = n * 3  # DDP syncs x, u, v every step
        desloc_syncs = n // self._Kx + n // self._Ku + n // self._Kv
        return {
            'step': self._step,
            'ddp_syncs': ddp_syncs,
            'desloc_syncs': max(desloc_syncs, 1),
            'reduction': round(ddp_syncs / max(desloc_syncs, 1), 2),
        }

    def get_stats(self):
        """Return full DES-LOC communication stats."""
        stats = self._instrument.get_full_stats()
        stats['schedule'] = {
            'Kx': self._Kx,
            'Ku': self._Ku,
            'Kv': self._Kv,
            'clip_radius': self._clip_radius,
        }
        stats['reduction'] = self.get_comm_reduction()
        return stats


# Global instrumentation instance (created lazily)
_global_desloc_instrument = None
_global_desloc_torch_ar = None


def init_desloc_torch_comm(Kx=32, Ku=96, Kv=192, world_size=1,
                            clip_radius=1.0):
    """Initialize global DES-LOC torch communication layer."""
    global _global_desloc_instrument, _global_desloc_torch_ar
    _global_desloc_instrument = DESLOCCommInstrumentation(None, world_size)
    _global_desloc_torch_ar = DESLOCTorchAllReduce(
        Kx=Kx, Ku=Ku, Kv=Kv, world_size=world_size,
        clip_radius=clip_radius, instrumentation=_global_desloc_instrument)
    return _global_desloc_torch_ar


def get_desloc_torch_allreduce():
    """Get the global DES-LOC torch allreduce wrapper."""
    return _global_desloc_torch_ar


def get_desloc_torch_stats():
    """Get global DES-LOC torch communication statistics."""
    if _global_desloc_torch_ar is not None:
        return _global_desloc_torch_ar.get_stats()
    return {'initialized': False}


# =================================================================
# M065: Ring-AllReduce Instrumented Ops + Nesterov Outer (400 lines)
# =================================================================
# Extends torch comm layer with:
# 1. Bandwidth-optimal Ring-AllReduce timing hooks
# 2. Nesterov outer optimizer integration (Section 5.5)
# 3. Per-state allreduce routing for DES-LOC schedule
#
# Reference: Section 4.1 "bandwidth-optimal Ring-AllReduce"
# Reference: template_extraction_from_latex.txt Ⅱ=Ring-AllReduce
# =================================================================

import time as _time
import math as _math


class DESLOCRingAllReduceProfiler:
    """Profile Ring-AllReduce operations for DES-LOC.

    Section 4.1: "bandwidth-optimal Ring-AllReduce"
    Measures actual bandwidth utilization per sync event to
    generate wallclock comparison figures.
    """

    def __init__(self, world_size=1):
        self.world_size = world_size
        self.op_log = []
        self.total_bytes_sent = 0
        self.total_time_ms = 0.0
        self.op_count = 0

    def start_op(self):
        """Mark the start of an allreduce operation."""
        return _time.monotonic()

    def end_op(self, start_time, num_bytes, state_type='x',
               op_type='allreduce'):
        """Record completion of an allreduce operation.

        Args:
            start_time: from start_op()
            num_bytes: bytes communicated
            state_type: 'x', 'u', or 'v'
            op_type: 'allreduce', 'broadcast', 'reduce_scatter'
        """
        elapsed = (_time.monotonic() - start_time) * 1000.0
        self.total_bytes_sent += num_bytes
        self.total_time_ms += elapsed
        self.op_count += 1

        # Ring-AllReduce: effective bandwidth = data_size * 2(N-1)/N / time
        if self.world_size > 1 and elapsed > 0:
            ring_factor = 2.0 * (self.world_size - 1) / self.world_size
            effective_bw_gbps = (
                num_bytes * ring_factor / (elapsed / 1000.0) / 1e9)
        else:
            effective_bw_gbps = 0.0

        entry = {
            'op_index': self.op_count,
            'state_type': state_type,
            'op_type': op_type,
            'num_bytes': num_bytes,
            'elapsed_ms': round(elapsed, 4),
            'effective_bw_gbps': round(effective_bw_gbps, 2),
        }
        self.op_log.append(entry)
        return entry

    def get_avg_bandwidth_gbps(self):
        """Get average effective bandwidth across all ops."""
        if not self.op_log:
            return 0.0
        total_bw = sum(e['effective_bw_gbps'] for e in self.op_log)
        return total_bw / len(self.op_log)

    def get_bandwidth_by_state(self):
        """Get bandwidth breakdown by state type."""
        result = {}
        for state in ('x', 'u', 'v'):
            ops = [e for e in self.op_log
                   if e['state_type'] == state]
            if ops:
                result[state] = {
                    'count': len(ops),
                    'total_bytes': sum(e['num_bytes'] for e in ops),
                    'total_ms': sum(e['elapsed_ms'] for e in ops),
                    'avg_bw_gbps': (
                        sum(e['effective_bw_gbps'] for e in ops) /
                        len(ops)),
                }
        return result

    def format_log(self):
        """Format profiler log in NKI-FA structured format."""
        lines = [
            f"### Ring-AllReduce Profile "
            f"(world_size={self.world_size}, "
            f"ops={self.op_count}) ###"
        ]
        by_state = self.get_bandwidth_by_state()
        for state, stats in by_state.items():
            lines.append(
                f"State {state}: {stats['count']} ops, "
                f"{stats['total_bytes']/1e6:.1f}MB, "
                f"{stats['total_ms']:.1f}ms, "
                f"avg {stats['avg_bw_gbps']:.1f} Gbps")
        lines.append(
            f"Overall avg bandwidth: "
            f"{self.get_avg_bandwidth_gbps():.1f} Gbps")
        return "\n".join(lines)


class DESLOCNesterovOuterOptimizer:
    """Nesterov momentum outer optimizer for DES-LOC.

    Section 5.5: "using Nesterov as the outer optimizer improves
    performance over averaging by ≈ 0.3%"

    Applied at parameter sync points (when t mod Kx == 0).
    Replaces simple E_m[x] averaging with Nesterov momentum.

    Reference: template_extraction_section5.txt CXLV = Nesterov
    """

    def __init__(self, momentum=0.9, outer_lr=1.0):
        self.momentum = momentum
        self.outer_lr = outer_lr
        self.velocity_buffers = {}
        self.sync_count = 0

    def apply(self, params, dp_group=None):
        """Apply Nesterov outer step after parameter averaging.

        For each parameter p:
          1. Compute averaged params: p_avg = E_m[p]
          2. v_{t+1} = momentum * v_t + (p_avg - p_local_before_avg)
          3. p = p_avg + momentum * v_{t+1} (Nesterov lookahead)

        But since dist.all_reduce(AVG) modifies p in-place,
        we need to save pre-avg values.
        """
        import torch.distributed as _dist

        pre_avg_snapshots = {}
        for p in params:
            if p.requires_grad:
                pre_avg_snapshots[id(p)] = p.data.clone()

        # Step 1: AllReduce average
        for p in params:
            if p.requires_grad:
                _dist.all_reduce(p.data, op=_dist.ReduceOp.AVG,
                                 group=dp_group)

        # Step 2-3: Nesterov momentum update
        for p in params:
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid not in self.velocity_buffers:
                self.velocity_buffers[pid] = p.data.new_zeros(
                    p.data.shape)

            v = self.velocity_buffers[pid]
            delta = p.data - pre_avg_snapshots[pid]

            # v = momentum * v + delta
            v.mul_(self.momentum).add_(delta)

            # Nesterov lookahead: p += momentum * v * outer_lr
            p.data.add_(v, alpha=self.momentum * self.outer_lr)

        self.sync_count += 1

    def reset(self):
        """Reset velocity buffers (e.g., after checkpoint load)."""
        self.velocity_buffers.clear()
        self.sync_count = 0

    def get_stats(self):
        """Get outer optimizer statistics."""
        total_v_norm = 0.0
        count = 0
        for v in self.velocity_buffers.values():
            total_v_norm += v.norm().item()
            count += 1
        return {
            'sync_count': self.sync_count,
            'momentum': self.momentum,
            'outer_lr': self.outer_lr,
            'num_params_tracked': count,
            'avg_velocity_norm': (
                total_v_norm / max(count, 1)),
        }


class DESLOCStateRouter:
    """Route optimizer states to correct sync handler.

    For Adam/ADOPT: routes x, u (exp_avg), v (exp_avg_sq)
    For Muon: routes x and momentum_buffer only
    For SGDM: routes x and momentum_buffer only

    Section 5.6: "the relevant synchronization periods reduce
    to parameters (Kx) and momentum (Ku)" for Muon
    """

    def __init__(self, optimizer_type='adam', Kx=32, Ku=96,
                 Kv=192, profiler=None):
        self.optimizer_type = optimizer_type
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        self.profiler = profiler
        self.step = 0

    def get_states_to_sync(self, step=None):
        """Determine which states need syncing at this step.

        Returns list of (state_key, state_type) tuples.
        """
        if step is None:
            step = self.step
        states = []

        if step % self.Kx == 0:
            states.append(('params', 'x'))

        if self.optimizer_type in ('adam', 'adopt'):
            if step % self.Ku == 0:
                states.append(('exp_avg', 'u'))
            if step % self.Kv == 0:
                states.append(('exp_avg_sq', 'v'))
        elif self.optimizer_type in ('muon', 'sgdm'):
            if step % self.Ku == 0:
                states.append(('momentum_buffer', 'u'))
                # Also check exp_avg for Muon variants
                states.append(('exp_avg', 'u'))

        return states

    def sync_optimizer_states(self, optimizer, dp_group=None):
        """Execute state synchronization for current step.

        Performs allreduce only for states that need syncing
        according to DES-LOC schedule.
        """
        import torch.distributed as _dist

        states_to_sync = self.get_states_to_sync()
        if not states_to_sync:
            self.step += 1
            return {'synced': [], 'bytes': 0}

        synced = []
        total_bytes = 0

        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p.grad is None or p not in optimizer.state:
                    continue
                state = optimizer.state[p]

                for state_key, state_type in states_to_sync:
                    if state_key == 'params':
                        continue  # params handled separately
                    if state_key not in state:
                        continue

                    tensor = state[state_key]
                    start = None
                    if self.profiler is not None:
                        start = self.profiler.start_op()

                    _dist.all_reduce(tensor, op=_dist.ReduceOp.AVG,
                                     group=dp_group)
                    num_bytes = tensor.numel() * tensor.element_size()
                    total_bytes += num_bytes

                    if self.profiler is not None and start is not None:
                        self.profiler.end_op(
                            start, num_bytes, state_type)

        synced_types = [s[1] for s in states_to_sync]
        self.step += 1
        return {'synced': synced_types, 'bytes': total_bytes}

    def get_sync_summary(self):
        """Get summary of sync decisions made."""
        total_steps = max(self.step, 1)
        x_syncs = total_steps // max(self.Kx, 1)
        u_syncs = total_steps // max(self.Ku, 1)
        summary = {
            'optimizer_type': self.optimizer_type,
            'total_steps': total_steps,
            'x_syncs': x_syncs,
            'u_syncs': u_syncs,
        }
        if self.optimizer_type in ('adam', 'adopt'):
            v_syncs = total_steps // max(self.Kv, 1)
            summary['v_syncs'] = v_syncs
            summary['total_syncs'] = x_syncs + u_syncs + v_syncs
            ddp_syncs = total_steps * 3
        else:
            summary['total_syncs'] = x_syncs + u_syncs
            ddp_syncs = total_steps * 2
        summary['ddp_equivalent_syncs'] = ddp_syncs
        summary['reduction_factor'] = (
            ddp_syncs / max(summary['total_syncs'], 1))
        return summary


class DESLOCGradientClipper:
    """Per-coordinate gradient clipping from Algorithm 1.

    Section 2: "clip(g, ρ)_i = sign(g_i) * min(|g_i|, ρ)"

    Records clipping statistics for experiment logs.
    """

    def __init__(self, clip_radius=1.0):
        self.clip_radius = clip_radius
        self.total_elements = 0
        self.clipped_elements = 0
        self.clip_count = 0

    def clip(self, parameters):
        """Apply per-coordinate clipping to all parameter gradients.

        Returns:
            dict with clipping statistics
        """
        total = 0
        clipped = 0
        for p in parameters:
            if p.grad is None:
                continue
            grad = p.grad.data
            total += grad.numel()
            mask = grad.abs() > self.clip_radius
            clipped += mask.sum().item()
            grad.clamp_(-self.clip_radius, self.clip_radius)

        self.total_elements += total
        self.clipped_elements += clipped
        self.clip_count += 1

        clip_ratio = clipped / max(total, 1)
        return {
            'clip_radius': self.clip_radius,
            'total_elements': total,
            'clipped_elements': clipped,
            'clip_ratio': clip_ratio,
        }

    def get_cumulative_stats(self):
        """Get cumulative clipping statistics."""
        return {
            'clip_radius': self.clip_radius,
            'total_clips': self.clip_count,
            'cumulative_elements': self.total_elements,
            'cumulative_clipped': self.clipped_elements,
            'cumulative_clip_ratio': (
                self.clipped_elements /
                max(self.total_elements, 1)),
        }


# Global profiler and router instances
_desloc_ring_profiler = None
_desloc_nesterov_outer = None
_desloc_state_router = None
_desloc_grad_clipper = None


def init_desloc_ring_profiler(world_size=1):
    """Initialize global Ring-AllReduce profiler."""
    global _desloc_ring_profiler
    _desloc_ring_profiler = DESLOCRingAllReduceProfiler(world_size)
    return _desloc_ring_profiler


def init_desloc_nesterov_outer(momentum=0.9, outer_lr=1.0):
    """Initialize global Nesterov outer optimizer."""
    global _desloc_nesterov_outer
    _desloc_nesterov_outer = DESLOCNesterovOuterOptimizer(
        momentum, outer_lr)
    return _desloc_nesterov_outer


def init_desloc_state_router(optimizer_type='adam', Kx=32,
                              Ku=96, Kv=192):
    """Initialize global state router."""
    global _desloc_state_router
    _desloc_state_router = DESLOCStateRouter(
        optimizer_type, Kx, Ku, Kv, _desloc_ring_profiler)
    return _desloc_state_router


def init_desloc_grad_clipper(clip_radius=1.0):
    """Initialize global gradient clipper."""
    global _desloc_grad_clipper
    _desloc_grad_clipper = DESLOCGradientClipper(clip_radius)
    return _desloc_grad_clipper


# =================================================================
# End M065
# =================================================================
