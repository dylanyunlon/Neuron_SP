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



# M291 — Claude-19: NCCLStream + AsyncReducer + LatencyModel + Profiler
class DeslocNCCLStream:
    __slots__=('stream','pending','on','st')
    def __init__(s,on=True):s.stream=None;s.pending=[];s.on=on;s.st={'ov':0,'sr':0,'b':0}
    def init(s):
        try:
            import torch
            if torch.cuda.is_available():s.stream=torch.cuda.Stream()
        except:pass
    def ar(s,t,tier='x'):
        import torch
        if not s.stream or not s.on:s.st['sr']+=1;s.st['b']+=t.numel()*t.element_size();return None
        eb,ea=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);eb.record()
        with torch.cuda.stream(s.stream):
            import torch.distributed as dist
            if dist.is_initialized():dist.all_reduce(t)
            ea.record()
        h={'eb':eb,'ea':ea,'tier':tier,'b':t.numel()*t.element_size()};s.pending.append(h);s.st['ov']+=1;s.st['b']+=h['b'];return h
    def wait(s):
        for h in s.pending:h['ea'].synchronize()
        s.pending.clear()
class DeslocAsyncRed:
    __slots__=('Kx','sm','gb','rb','st','wu','stats')
    def __init__(s,Kx,wu=512):s.Kx=max(1,Kx);s.wu=wu;s.sm=DeslocNCCLStream();s.gb={};s.rb={};s.st=0;s.stats={'a':0,'s':0}
    def init(s):s.sm.init()
    def acc(s,n,g):s.gb[n]=g.clone()if n not in s.gb else s.gb[n].add_(g)
    def reduce(s,st):
        s.st=st
        if st%s.Kx!=0:return
        if st<s.wu:
            import torch.distributed as dist
            if dist.is_initialized():
                w=dist.get_world_size()
                for n,b in s.gb.items():dist.all_reduce(b);b.div_(w);s.rb[n]=b
            s.gb.clear();s.stats['s']+=1;return
        for n,b in s.gb.items():s.sm.ar(b);s.rb[n]=b
        s.gb.clear();s.stats['a']+=1
    def sync(s):s.sm.wait()
class DeslocLatModel:
    __slots__=('a','b','nw','cal','r2')
    def __init__(s,nw=2):s.nw=nw;s.a=.05;s.b=.04;s.cal=False;s.r2=0.
    def predict(s,sz,op='ar'):mb=sz/1048576;rf=2.*(s.nw-1)/max(1,s.nw)if op=='ar'else 1.;return s.a+s.b*mb*rf
    def calibrate(s,ms):
        if len(ms)<3:return
        xs=[z/1048576 for z,_ in ms];ys=[t for _,t in ms];n=len(xs);sx,sy=sum(xs),sum(ys);sxy=sum(x*y for x,y in zip(xs,ys));sx2=sum(x*x for x in xs)
        d=n*sx2-sx*sx
        if abs(d)<1e-12:return
        s.b=(n*sxy-sx*sy)/d;s.a=(sy-s.b*sx)/n;my=sy/n;sst=sum((y-my)**2 for y in ys);ssr=sum((y-(s.a+s.b*x))**2 for x,y in zip(xs,ys));s.r2=1.-ssr/(sst+1e-12);s.cal=True
class DeslocTProf:
    __slots__=('log','tt','er','mx')
    def __init__(s,mx=50000):s.log=[];s.mx=mx;s.er=0;s.tt={t:{'b':0,'t':0,'o':0}for t in('x','u','v')}
    def rec(s,st,t,sz,ms,ok=True):
        s.log.append((st,t,sz,ms,ok));s.log=s.log[-s.mx//2:]if len(s.log)>s.mx else s.log
        if ok and t in s.tt:s.tt[t]['b']+=sz;s.tt[t]['t']+=ms;s.tt[t]['o']+=1
        elif not ok:s.er+=1
# M291: end
