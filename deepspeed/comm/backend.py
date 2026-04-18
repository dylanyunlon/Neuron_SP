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


# =========================================================================
# DES-LOC Backend Extensions
# Ref: Algorithm 1 — per-tier communication primitives
# =========================================================================

class DeslocBackendMixin:
    """Mixin for DES-LOC tier awareness in any comm backend.

    Any backend (Torch, NCCL, Gloo) can use this to get:
    - Per-tier byte accounting
    - Sync period gating
    - Communication reduction tracking
    """

    def init_desloc(self, Kx=1, Ku=3, Kv=6):
        self._desloc_Kx = Kx
        self._desloc_Ku = Ku
        self._desloc_Kv = Kv
        self._desloc_step = 0
        self._desloc_bytes = {0: 0, 1: 0, 2: 0}
        self._desloc_skipped = {0: 0, 1: 0, 2: 0}

    def desloc_should_comm(self, tier):
        period = {0: self._desloc_Kx, 1: self._desloc_Ku,
                  2: self._desloc_Kv}.get(tier, 1)
        return (self._desloc_step % max(1, period)) == 0

    def desloc_record(self, tier, num_bytes):
        self._desloc_bytes[tier] = self._desloc_bytes.get(tier, 0) + num_bytes

    def desloc_record_skip(self, tier):
        self._desloc_skipped[tier] = self._desloc_skipped.get(tier, 0) + 1

    def desloc_advance(self):
        self._desloc_step += 1

    def desloc_report(self):
        return {
            'step': self._desloc_step,
            'bytes': dict(self._desloc_bytes),
            'skipped': dict(self._desloc_skipped),
            'Kx': self._desloc_Kx, 'Ku': self._desloc_Ku, 'Kv': self._desloc_Kv,
        }


class DeslocTopologyDetector:
    """Detect GPU interconnect for Kx recommendation.
    Ref: Nick Joseph — 'ran clustering algorithm for chip locations.'"""

    @staticmethod
    def detect():
        try:
            import torch
            if not torch.cuda.is_available():
                return {'type': 'cpu', 'bw': 0}
            count = torch.cuda.device_count()
            if count <= 1:
                return {'type': 'single', 'bw': 0, 'n': 1}
            name = torch.cuda.get_device_name(0)
            nvl = any(g in name for g in ('H100', 'A100', 'H200'))
            bw = (600 if 'H100' in name else 300) if nvl else 32
            return {'type': 'nvlink' if nvl else 'pcie', 'bw': bw, 'n': count, 'gpu': name}
        except Exception:
            return {'type': 'unknown', 'bw': 0}

    @staticmethod
    def detect_heterogeneous():
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
                return {'hetero': False}
            names = set(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            return {'hetero': len(names) > 1, 'gpus': list(names)}
        except Exception:
            return {'hetero': False}


class DeslocInterconnectProfiler:
    """Profile interconnect at runtime for dynamic Kx adjustment.

    Periodically measures allreduce bandwidth and adjusts Kx.
    This handles cases where network conditions change during training
    (e.g. shared cluster with varying load).

    Ref: Nick Joseph — 'latency differed based on datacenter room.'
    """

    def __init__(self, profile_interval=1000, min_Kx=1, max_Kx=256):
        self.profile_interval = profile_interval
        self.min_Kx = min_Kx
        self.max_Kx = max_Kx
        self.step = 0
        self.bandwidth_history = []
        self.current_Kx = 32

    def should_profile(self):
        return self.step % self.profile_interval == 0

    def update_bandwidth(self, bw_gbps):
        self.bandwidth_history.append(bw_gbps)
        if len(self.bandwidth_history) > 10:
            self.bandwidth_history = self.bandwidth_history[-10:]

    def get_recommended_Kx(self, model_params, compute_time_s):
        if not self.bandwidth_history:
            return self.current_Kx
        import math
        avg_bw = sum(self.bandwidth_history) / len(self.bandwidth_history)
        if avg_bw <= 0:
            return self.max_Kx
        ar_time = model_params * 2 * 2 / (avg_bw * 1e9)
        if ar_time <= compute_time_s:
            kx = self.min_Kx
        else:
            kx = 2 ** int(math.ceil(math.log2(ar_time / compute_time_s)))
        return max(self.min_Kx, min(self.max_Kx, kx))

    def advance(self):
        self.step += 1

    def state_dict(self):
        return {
            'step': self.step, 'kx': self.current_Kx,
            'bw_history': list(self.bandwidth_history),
        }

    def load_state_dict(self, sd):
        self.step = sd.get('step', 0)
        self.current_Kx = sd.get('kx', 32)
        self.bandwidth_history = sd.get('bw_history', [])


# =========================================================================
# DES-LOC Backend Extensions — Interconnect Profiling + Dynamic Kx
# Ref: Nick Joseph — 'understand physical layout of hardware'
# =========================================================================

class DeslocInterconnectProfiler:
    """Profile interconnect at runtime for dynamic Kx adjustment.
    Periodically measures bandwidth and adjusts Kx.
    Handles varying network load in shared clusters."""

    def __init__(self, profile_interval=1000, min_Kx=1, max_Kx=256):
        self.profile_interval = profile_interval
        self.min_Kx = min_Kx
        self.max_Kx = max_Kx
        self.step = 0
        self.bw_history = []
        self.current_Kx = 32

    def should_profile(self):
        return self.step % self.profile_interval == 0

    def update_bandwidth(self, bw_gbps):
        self.bw_history.append(bw_gbps)
        if len(self.bw_history) > 20:
            self.bw_history = self.bw_history[-20:]

    def get_recommended_Kx(self, model_params, compute_s):
        if not self.bw_history:
            return self.current_Kx
        import math
        avg_bw = sum(self.bw_history) / len(self.bw_history)
        if avg_bw <= 0:
            return self.max_Kx
        ar_s = model_params * 2 * 2 / (avg_bw * 1e9)
        if ar_s <= compute_s:
            return self.min_Kx
        kx = 2 ** int(math.ceil(math.log2(ar_s / compute_s)))
        return max(self.min_Kx, min(self.max_Kx, kx))

    def advance(self):
        self.step += 1

    def state_dict(self):
        return {'step': self.step, 'kx': self.current_Kx, 'bw': list(self.bw_history)}

    def load_state_dict(self, sd):
        self.step = sd.get('step', 0)
        self.current_Kx = sd.get('kx', 32)
        self.bw_history = sd.get('bw', [])


class DeslocBackendStats:
    """Aggregate communication statistics across all backends.
    Provides unified view of DES-LOC comm efficiency."""

    def __init__(self):
        self.total_bytes = 0
        self.total_ops = 0
        self.skipped_ops = 0
        self.tier_stats = {0: {'bytes': 0, 'ops': 0, 'lat_ms': 0},
                          1: {'bytes': 0, 'ops': 0, 'lat_ms': 0},
                          2: {'bytes': 0, 'ops': 0, 'lat_ms': 0}}

    def record(self, tier, num_bytes, latency_ms=0):
        self.total_bytes += num_bytes
        self.total_ops += 1
        if tier in self.tier_stats:
            self.tier_stats[tier]['bytes'] += num_bytes
            self.tier_stats[tier]['ops'] += 1
            self.tier_stats[tier]['lat_ms'] += latency_ms

    def record_skip(self):
        self.skipped_ops += 1

    def get_summary(self):
        total = self.total_ops + self.skipped_ops
        return {
            'total_bytes': self.total_bytes,
            'total_ops': self.total_ops,
            'skipped_ops': self.skipped_ops,
            'skip_ratio': round(self.skipped_ops / max(1, total), 4),
            'tiers': {t: dict(s) for t, s in self.tier_stats.items() if s['ops'] > 0},
        }

    def reset(self):
        self.total_bytes = 0
        self.total_ops = 0
        self.skipped_ops = 0
        for t in self.tier_stats:
            self.tier_stats[t] = {'bytes': 0, 'ops': 0, 'lat_ms': 0}


class DeslocMultiBackendRouter:
    """Route DES-LOC communication ops to appropriate backend.
    In multi-backend setups (e.g. NCCL for GPU, Gloo for CPU),
    route param sync via NCCL and momentum sync via either.

    Ref: DeepSpeed supports multiple backends simultaneously."""

    def __init__(self, gpu_backend='nccl', cpu_backend='gloo'):
        self.gpu_backend = gpu_backend
        self.cpu_backend = cpu_backend
        self.routing_table = {
            0: gpu_backend,  # param sync → GPU
            1: gpu_backend,  # momentum sync → GPU (or CPU for offload)
            2: gpu_backend,  # variance sync → GPU (or CPU for offload)
        }

    def set_offload_tiers(self, offload_momentum=False, offload_variance=False):
        """Route momentum/variance sync to CPU backend when offloaded."""
        if offload_momentum:
            self.routing_table[1] = self.cpu_backend
        if offload_variance:
            self.routing_table[2] = self.cpu_backend

    def get_backend_for_tier(self, tier):
        return self.routing_table.get(tier, self.gpu_backend)

    def summary(self):
        return {t: b for t, b in self.routing_table.items()}


class DeslocCommBudget:
    """Communication budget tracker for DES-LOC.
    Sets a per-step byte budget and tracks utilization.
    When budget is exceeded, automatically increase Kx next step.

    Ref: Network-aware training — adapt to available bandwidth."""

    def __init__(self, budget_bytes_per_step=None, param_bytes=0):
        self.budget = budget_bytes_per_step
        self.param_bytes = param_bytes
        self.step = 0
        self.overbudget_count = 0

    def check_budget(self, bytes_this_step):
        self.step += 1
        if self.budget is None:
            return True
        if bytes_this_step > self.budget:
            self.overbudget_count += 1
            return False
        return True

    def get_utilization(self, bytes_this_step):
        if self.budget is None or self.budget <= 0:
            return 0.0
        return round(bytes_this_step / self.budget, 4)

    def recommend_Kx_for_budget(self, current_Kx):
        """If frequently over budget, suggest larger Kx."""
        if self.step == 0:
            return current_Kx
        ratio = self.overbudget_count / self.step
        if ratio > 0.1:
            return min(256, current_Kx * 2)
        return current_Kx

    def state_dict(self):
        return {'budget': self.budget, 'step': self.step,
                'overbudget': self.overbudget_count}


# --- DES-LOC Backend Extensions (M146) ---
import time as _b146;import math as _b146m;from collections import defaultdict as _b146dd
class DeslocConvergenceMonitor:
    def __init__(self,w=100,th=0.5): self.w,self.th=w,th;self._n=[];self._a=0
    def record(self,norm): self._n.append(norm);self._n=self._n[-self.w*2:]
    def cv(self):
        r=self._n[-self.w:]
        if len(r)<10: return 0.0
        m=sum(r)/len(r);return _b146m.sqrt(sum((x-m)**2 for x in r)/len(r))/max(m,1e-8)
    def is_stable(self): c=self.cv();s=c<self.th;self._a+=0 if s else 1;return s
    def recommend_Kx(self,cur):
        c=self.cv()
        if c>self.th*1.5: return max(1,cur//2)
        if c>self.th: return max(1,int(cur*0.75))
        if c<self.th*0.3 and cur<256: return min(256,int(cur*1.25))
        return cur
    def state_dict(self): return {'a':self._a,'n':self._n[-self.w:]}
    def load_state_dict(self,sd): self._a=sd.get('a',0);self._n=sd.get('n',[])
class DeslocCollectiveOps:
    def __init__(self,Kx=32,Ku=96,Kv=192,clip=1.0):
        self.Kx,self.Ku,self.Kv,self.clip=Kx,Ku,Kv,clip;self._step=0;self._bytes=_b146dd(int);self._skips=_b146dd(int)
    def should_sync(self,tier='x'):
        K={'x':self.Kx,'u':self.Ku,'v':self.Kv}.get(tier,1);return K<=1 or self._step%K==0
    def advance(self): self._step+=1
    def all_reduce(self,tensor,tier='x',op=None,group=None,async_op=False):
        if not self.should_sync(tier): self._skips[tier]+=1;return None,False
        import torch.distributed as td
        if op is None:
            try: op=td.ReduceOp.AVG
            except: op=td.ReduceOp.SUM
        nb=tensor.numel()*tensor.element_size();h=td.all_reduce(tensor,op=op,group=group,async_op=async_op);self._bytes[tier]+=nb;return h,True
    def clip_grads(self,params):
        if self.clip<=0: return
        for p in params:
            if p.grad is not None: p.grad.data.clamp_(-self.clip,self.clip)
    def state_dict(self): return {'step':self._step,'Kx':self.Kx,'Ku':self.Ku,'Kv':self.Kv}
class DeslocServerOptimizer:
    def __init__(self,mu=0.9): self.mu=mu;self._v={};self._p={}
    def step(self,named_params,synced):
        if not synced or self.mu<=0: return
        for n,p in named_params:
            if not p.requires_grad: continue
            pid=id(p)
            if pid not in self._v: self._v[pid]=p.data.new_zeros(p.data.shape);self._p[pid]=p.data.clone();continue
            d=p.data.float()-self._p[pid].float();v=self._v[pid];v.mul_(self.mu).add_(d)
            p.data.add_(v.to(p.dtype),alpha=self.mu);self._p[pid].copy_(p.data)
class DeslocCommProfiler:
    def __init__(self,interval=100): self.interval=interval;self._r=[];self._step=0
    def record(self,Kx,synced,nb=0,ms=0):
        self._r.append({'s':self._step,'Kx':Kx,'sx':'x' in synced,'b':nb,'ms':ms});self._step+=1
        if self._step%self.interval==0:
            r=self._r[-self.interval:];print(f"### DES-LOC step={self._step} ### {sum(1 for x in r if x['sx'])}/{len(r)} syncs")
    def export_csv(self,path):
        lines=['step,Kx,sx,bytes,ms']
        for r in self._r: lines.append(f"{r['s']},{r['Kx']},{int(r['sx'])},{r['b']},{r['ms']:.3f}")
        with open(path,'w') as f: f.write('\n'.join(lines))
def create_desloc_stack(Kx=32,Ku=96,Kv=192,clip=1.0,mu=0.9):
    return {'collective':DeslocCollectiveOps(Kx,Ku,Kv,clip),'server_opt':DeslocServerOptimizer(mu),'monitor':DeslocConvergenceMonitor(),'profiler':DeslocCommProfiler()}
