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
# DES-LOC Backend Mixin (Algorithm 1 communication tier awareness)
# =========================================================================

class DeslocBackendMixin:
    """Mixin adding DES-LOC tier awareness to any comm backend.
    Ref: Section 4.1 - Ring-AllReduce with independent sync periods."""

    def init_desloc(self, Kx=1, Ku=3, Kv=6):
        self._desloc_Kx = Kx
        self._desloc_Ku = Ku
        self._desloc_Kv = Kv
        self._desloc_step = 0
        self._desloc_bytes_sent = 0
        self._desloc_bytes_skipped = 0

    def desloc_should_comm(self, tier):
        """Check if communication should happen for given tier at current step."""
        period = {0: self._desloc_Kx, 1: self._desloc_Ku,
                  2: self._desloc_Kv}.get(tier, 1)
        if period <= 1:
            return True
        return self._desloc_step % period == 0

    def desloc_advance(self):
        self._desloc_step += 1

    def desloc_stats(self):
        return {
            'step': self._desloc_step,
            'bytes_sent': self._desloc_bytes_sent,
            'bytes_skipped': self._desloc_bytes_skipped,
        }


# =========================================================================
# M212: DES-LOC Communication Scheduler (Algorithm 1, lines 9-21)
# =========================================================================

import math
import time
from collections import defaultdict, deque


class DeslocCommScheduler:
    """Schedule AllReduce operations according to DES-LOC Algorithm 1.

    Core logic: at each training step t, decide which tiers synchronize:
      - Tier 0 (parameters x): sync when t % Kx == 0
      - Tier 1 (1st momentum u): sync when t % Ku == 0
      - Tier 2 (2nd momentum v): sync when t % Kv == 0

    Ref: Algorithm 1 lines 14-19:
      if t mod Kj = 0 then
        s_t^{j,m} <- UPDATE^j(E_m[s_{t-1}^j], g_t^m)
      else
        s_t^{j,m} <- UPDATE^j(s_{t-1}^{j,m}, g_t^m)

    Design: Kx=1 degrades to standard DDP (every step syncs everything).
    MoE capacity allreduces are NEVER gated (is_safety_critical=True bypasses).
    """

    TIER_PARAM = 0
    TIER_MOMENTUM = 1
    TIER_VARIANCE = 2
    TIER_NAMES = {0: 'param', 1: 'momentum', 2: 'variance'}

    def __init__(self, Kx=1, Ku=3, Kv=6, warmup_steps=512):
        self._Kx = max(int(Kx), 1)
        self._Ku = max(int(Ku), 1)
        self._Kv = max(int(Kv), 1)
        self._warmup_steps = max(int(warmup_steps), 0)
        self._step = 0
        self._sync_count = {0: 0, 1: 0, 2: 0}
        self._skip_count = {0: 0, 1: 0, 2: 0}
        self._bytes_sent = {0: 0, 1: 0, 2: 0}
        self._bytes_skipped = {0: 0, 1: 0, 2: 0}
        self._overflow_force_sync = False
        self._last_sync_step = {0: 0, 1: 0, 2: 0}

    @property
    def Kx(self):
        return self._Kx

    @property
    def Ku(self):
        return self._Ku

    @property
    def Kv(self):
        return self._Kv

    @property
    def step(self):
        return self._step

    def period_for_tier(self, tier):
        """Return sync period for given tier."""
        return {0: self._Kx, 1: self._Ku, 2: self._Kv}.get(int(tier), 1)

    def is_warmup(self):
        """During warmup, ALL tiers sync every step (like DDP)."""
        return self._step < self._warmup_steps

    def should_sync(self, tier, is_safety_critical=False):
        """Decide if tier should sync at current step.

        Args:
            tier: 0=param, 1=momentum, 2=variance
            is_safety_critical: True for MoE capacity, overflow detection, etc.
                                These ALWAYS sync regardless of Kx.
        Returns:
            bool: True if this tier should AllReduce at this step.
        """
        if is_safety_critical:
            return True
        if self._overflow_force_sync:
            return True
        if self.is_warmup():
            return True
        period = self.period_for_tier(tier)
        if period <= 1:
            return True
        return self._step % period == 0

    def record_comm(self, tier, num_bytes, did_sync):
        """Record a communication decision for profiling."""
        tier = int(tier)
        if did_sync:
            self._sync_count[tier] = self._sync_count.get(tier, 0) + 1
            self._bytes_sent[tier] = self._bytes_sent.get(tier, 0) + num_bytes
            self._last_sync_step[tier] = self._step
        else:
            self._skip_count[tier] = self._skip_count.get(tier, 0) + 1
            self._bytes_skipped[tier] = self._bytes_skipped.get(tier, 0) + num_bytes

    def advance(self):
        """Advance to next step. Call once per training step."""
        self._step += 1
        self._overflow_force_sync = False

    def force_sync_on_overflow(self):
        """Force all tiers to sync next step due to gradient overflow."""
        self._overflow_force_sync = True

    def steps_since_last_sync(self, tier):
        """How many steps since tier last synced."""
        return self._step - self._last_sync_step.get(int(tier), 0)

    def comm_reduction_ratio(self):
        """Compute actual communication reduction vs DDP.
        DDP syncs all 3 tiers every step.
        DES-LOC syncs at rates 1/Kx + 1/Ku + 1/Kv.
        Ref: Section 5.3 — 'halves communication versus Local Adam.'"""
        total_sent = sum(self._bytes_sent.values())
        total_possible = total_sent + sum(self._bytes_skipped.values())
        if total_possible == 0:
            return 0.0
        return 1.0 - total_sent / total_possible

    def theoretical_reduction(self):
        """Theoretical reduction based on periods alone.
        Ref: DDP does 3 allreduces/step; DES-LOC does 1/Kx + 1/Ku + 1/Kv."""
        ddp_rate = 3.0
        desloc_rate = 1.0 / self._Kx + 1.0 / self._Ku + 1.0 / self._Kv
        return 1.0 - desloc_rate / ddp_rate

    def state_dict(self):
        """Serialize for checkpointing."""
        return {
            'Kx': self._Kx, 'Ku': self._Ku, 'Kv': self._Kv,
            'step': self._step, 'warmup': self._warmup_steps,
            'sync_count': dict(self._sync_count),
            'skip_count': dict(self._skip_count),
            'bytes_sent': dict(self._bytes_sent),
            'bytes_skipped': dict(self._bytes_skipped),
            'last_sync': dict(self._last_sync_step),
        }

    def load_state_dict(self, sd):
        """Restore from checkpoint."""
        self._Kx = sd.get('Kx', self._Kx)
        self._Ku = sd.get('Ku', self._Ku)
        self._Kv = sd.get('Kv', self._Kv)
        self._step = sd.get('step', 0)
        self._warmup_steps = sd.get('warmup', self._warmup_steps)
        self._sync_count = sd.get('sync_count', self._sync_count)
        self._skip_count = sd.get('skip_count', self._skip_count)
        self._bytes_sent = sd.get('bytes_sent', self._bytes_sent)
        self._bytes_skipped = sd.get('bytes_skipped', self._bytes_skipped)
        self._last_sync_step = sd.get('last_sync', self._last_sync_step)

    def summary_string(self):
        """One-line summary for logging."""
        red = self.comm_reduction_ratio()
        return (f'DES-LOC step={self._step} Kx={self._Kx} Ku={self._Ku} '
                f'Kv={self._Kv} reduction={red:.1%}')


class DeslocHalfLifeCalculator:
    """Compute half-lives and recommend sync periods.

    Ref: Section 2, Eq.(1-2):
      tau_psi(beta) = ln(psi) / ln(beta)
      tau_0.5(0.9) ≈ 6.6 steps
      tau_0.5(0.95) ≈ 13.5 steps
      tau_0.5(0.999) ≈ 692.8 steps

    The key insight: second momentum (beta2=0.999) changes ~100x slower
    than first momentum (beta1=0.9), so Kv can be ~100x larger than Ku.
    """

    @staticmethod
    def half_life(beta):
        """Compute tau_0.5(beta) = ln(0.5) / ln(beta).
        Returns infinity for beta >= 1.0."""
        if beta >= 1.0:
            return float('inf')
        if beta <= 0.0:
            return 0.0
        return math.log(0.5) / math.log(beta)

    @staticmethod
    def decay_after_k_steps(beta, k):
        """Fraction of original value remaining after k steps: beta^k."""
        return beta ** k

    @staticmethod
    def max_deviation(rho, beta, K):
        """Upper bound on state deviation after K steps without sync.
        Ref: Section 2, Eq.(2): ||u_{t+K} - u_t||_inf <= 2*rho*(1-beta^K).
        Where rho is the gradient clipping radius."""
        return 2.0 * rho * (1.0 - beta ** K)

    @staticmethod
    def recommend_periods(beta1=0.9, beta2=0.999, base_Kx=32):
        """Recommend Ku, Kv based on half-life ratios.

        Ref: Section 5.2 Takeaway — 'Momentum synchronization periods
        matter empirically only when chosen near their half-lives.'

        Strategy:
          Ku = base_Kx * ceil(hl1 / base_Kx) — sync 1st momentum ~once per half-life
          Kv = base_Kx * ceil(hl2 / base_Kx) — sync 2nd momentum ~once per half-life
          But cap Ku at 3*Kx and Kv at 6*Kx per paper's heuristic (Section 5.3).
        """
        hl1 = DeslocHalfLifeCalculator.half_life(beta1)
        hl2 = DeslocHalfLifeCalculator.half_life(beta2)

        if hl1 == float('inf'):
            Ku = base_Kx
        else:
            Ku = max(base_Kx, int(math.ceil(hl1 / base_Kx)) * base_Kx)
            Ku = min(Ku, 3 * base_Kx)

        if hl2 == float('inf'):
            Kv = Ku
        else:
            Kv = max(Ku, int(math.ceil(hl2 / base_Kx)) * base_Kx)
            Kv = min(Kv, 6 * base_Kx)

        return base_Kx, Ku, Kv

    @staticmethod
    def half_life_ratio(beta1, beta2):
        """Ratio of 2nd to 1st momentum half-lives.
        For beta1=0.9, beta2=0.999: ratio ≈ 104.8."""
        hl1 = DeslocHalfLifeCalculator.half_life(beta1)
        hl2 = DeslocHalfLifeCalculator.half_life(beta2)
        if hl1 <= 0 or hl1 == float('inf'):
            return 1.0
        return hl2 / hl1


class DeslocPsiCalculator:
    """Compute psi factor from Theorem 1 for convergence analysis.

    Ref: Theorem 1, Eq.(4):
      psi = 4*(1-px)/px^2 * (1-beta)*(1-pu) / (6*(1-(1-pu)*beta))
    where px = 1/Kx, pu = 1/Ku.

    The psi factor appears in the step-size restriction:
      eta_0 = 1/(4L) * min(1-beta, 1/sqrt(psi * max(1, B^2-1)))
    and in the higher-order term: O((1+psi)/T).
    """

    @staticmethod
    def compute(Kx, Ku, beta1):
        """Compute psi given sync periods and momentum decay."""
        px = 1.0 / max(Kx, 1)
        pu = 1.0 / max(Ku, 1)
        beta = beta1

        numer = 4.0 * (1.0 - px) * (1.0 - beta) * (1.0 - pu)
        denom = px * px * 6.0 * (1.0 - (1.0 - pu) * beta)

        if abs(denom) < 1e-15:
            return float('inf')
        return numer / denom

    @staticmethod
    def step_size_bound(L, beta1, Kx, Ku, B_sq=1.0):
        """Compute step-size upper bound eta_0 from Theorem 1.
        eta_0 = 1/(4L) * min(1-beta, 1/sqrt(psi * max(1, B^2-1)))"""
        psi = DeslocPsiCalculator.compute(Kx, Ku, beta1)
        term1 = 1.0 - beta1
        inner = psi * max(1.0, B_sq - 1.0)
        term2 = 1.0 / math.sqrt(max(inner, 1e-15))
        return min(term1, term2) / (4.0 * max(L, 1e-15))


class DeslocBandwidthModel:
    """Model communication time for DES-LOC configurations.

    Ref: Appendix G.1 — Wall-clock time modeling.
    Ring-AllReduce time = 2*(M-1)/M * S / bandwidth
    where S = message size in bytes, M = number of workers.

    For DES-LOC: effective comm time per step =
      T_comm = (1/Kx + 1/Ku + 1/Kv) * T_allreduce(S)
    vs DDP:
      T_comm = 3 * T_allreduce(S)
    """

    def __init__(self, num_workers, bandwidth_gbps=100.0, latency_us=5.0):
        self._M = max(num_workers, 1)
        self._bw_bytes_per_sec = bandwidth_gbps * 1e9 / 8.0
        self._latency_sec = latency_us * 1e-6

    def allreduce_time_sec(self, message_bytes):
        """Time for one Ring-AllReduce.
        Ref: Section 4.1 — bandwidth-optimal Ring-AllReduce."""
        coeff = 2.0 * (self._M - 1) / self._M
        return self._latency_sec + coeff * message_bytes / self._bw_bytes_per_sec

    def ddp_comm_time(self, param_bytes):
        """DDP: 3 allreduces per step (params + 2 momenta)."""
        return 3.0 * self.allreduce_time_sec(param_bytes)

    def desloc_comm_time(self, param_bytes, Kx, Ku, Kv):
        """DES-LOC: allreduces at rates 1/Kx, 1/Ku, 1/Kv per step."""
        t_ar = self.allreduce_time_sec(param_bytes)
        return (1.0/Kx + 1.0/Ku + 1.0/Kv) * t_ar

    def speedup_over_ddp(self, param_bytes, Kx, Ku, Kv, compute_time_sec=0.0):
        """Compute wall-clock speedup of DES-LOC over DDP.
        Total time = max(compute, comm) due to overlap potential."""
        ddp_total = max(compute_time_sec, self.ddp_comm_time(param_bytes))
        desloc_total = max(compute_time_sec, self.desloc_comm_time(
            param_bytes, Kx, Ku, Kv))
        if desloc_total <= 0:
            return 1.0
        return ddp_total / desloc_total

    def comm_bound_threshold(self, param_bytes, compute_time_sec):
        """Find the Kx threshold below which training is comm-bound.
        Returns Kx such that comm_time(Kx) ≈ compute_time."""
        if compute_time_sec <= 0:
            return 1
        t_ar = self.allreduce_time_sec(param_bytes)
        if t_ar <= 0:
            return 1
        Kx_float = t_ar / compute_time_sec
        return max(1, int(math.ceil(Kx_float)))


class DeslocCommProfiler:
    """Profile actual communication patterns during training.

    Tracks per-step timing with a sliding window for real-time metrics.
    Output format compatible with NKI-FA draw_plot.py parsing.
    Ref: NKI-FA commit da964f3 — structured log output.
    """

    def __init__(self, window_size=100):
        self._window = window_size
        self._history = deque(maxlen=window_size)
        self._total_comm_sec = 0.0
        self._total_compute_sec = 0.0
        self._total_steps = 0

    def record_step(self, comm_sec, compute_sec, tier, did_sync):
        """Record timing for one step."""
        self._history.append({
            'comm_sec': comm_sec,
            'compute_sec': compute_sec,
            'tier': tier,
            'did_sync': did_sync,
            'timestamp': time.monotonic(),
        })
        self._total_comm_sec += comm_sec
        self._total_compute_sec += compute_sec
        self._total_steps += 1

    def avg_comm_time(self):
        """Average comm time over recent window."""
        if not self._history:
            return 0.0
        return sum(r['comm_sec'] for r in self._history) / len(self._history)

    def avg_compute_time(self):
        if not self._history:
            return 0.0
        return sum(r['compute_sec'] for r in self._history) / len(self._history)

    def comm_compute_ratio(self):
        """Ratio of comm to compute time (< 1 means compute-bound)."""
        c = self.avg_compute_time()
        if c <= 0:
            return float('inf')
        return self.avg_comm_time() / c

    def is_comm_bound(self):
        """Is training currently communication-bound?"""
        return self.comm_compute_ratio() > 1.0

    def sync_rate(self):
        """Fraction of recent steps that actually synced."""
        if not self._history:
            return 1.0
        synced = sum(1 for r in self._history if r['did_sync'])
        return synced / len(self._history)

    def format_log_line(self, step, loss=None, lr=None, throughput=None):
        """Format a log line parseable by DeslocLogParser.
        Ref: NKI-FA draw_plot.py — 'metric: value' format."""
        parts = [f'step: {step}']
        if loss is not None:
            parts.append(f'loss: {loss:.6f}')
        if lr is not None:
            parts.append(f'lr: {lr:.8f}')
        if throughput is not None:
            parts.append(f'throughput: {throughput:.4f}')
        parts.append(f'comm_ratio: {self.comm_compute_ratio():.4f}')
        parts.append(f'sync_rate: {self.sync_rate():.4f}')
        return ' | '.join(parts)

    def summary(self):
        return {
            'total_steps': self._total_steps,
            'total_comm_sec': round(self._total_comm_sec, 4),
            'total_compute_sec': round(self._total_compute_sec, 4),
            'avg_comm_ms': round(self.avg_comm_time() * 1000, 4),
            'avg_compute_ms': round(self.avg_compute_time() * 1000, 4),
            'is_comm_bound': self.is_comm_bound(),
        }

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


# M247 — Claude-16: DES-LOC Backend Transport Abstraction
# Ref: NCCL transport.cc — canConnect/setup/connect pattern

class DeslocTransportSelector:
    """Select optimal transport for DES-LOC AllReduce.
    Models NCCL's P2P→SHM→NET fallback chain."""
    TRANSPORTS = ('p2p', 'shm', 'net')
    def __init__(self):
        self._selected = {}
        self._fallback_chain = list(self.TRANSPORTS)
    def select(self, src_rank, dst_rank, msg_size_bytes):
        key = (src_rank, dst_rank)
        if key in self._selected:
            return self._selected[key]
        # Heuristic: same-node (rank diff <=7) → P2P, else NET
        if abs(src_rank - dst_rank) <= 7:
            transport = 'p2p' if msg_size_bytes < 64 * 1024 * 1024 else 'shm'
        else:
            transport = 'net'
        self._selected[key] = transport
        return transport
    def get_bandwidth_estimate(self, transport):
        bw_map = {'p2p': 600.0, 'shm': 200.0, 'net': 50.0}
        return bw_map.get(transport, 10.0)
    def compute_allreduce_time_ms(self, model_params, world_size,
                                   transport='p2p', dtype_bytes=2):
        bw = self.get_bandwidth_estimate(transport)
        msg = 2 * model_params * dtype_bytes
        ring_factor = 2.0 * (world_size - 1) / max(1, world_size)
        return round(msg * ring_factor / (bw * 1e6), 4)

class DeslocBackendHealthChecker:
    """Monitor backend health for DES-LOC training.
    Detects: NCCL timeouts, NaN in comm buffers, bandwidth degradation.
    Ref: Nick Joseph — 'sometimes the computer is wrong, GPU is broken'"""
    def __init__(self, timeout_s=300.0, bw_degradation_threshold=0.5):
        self._timeout_s = timeout_s
        self._bw_threshold = bw_degradation_threshold
        self._baseline_bw = None
        self._consecutive_timeouts = 0
        self._nan_count = 0
        self._health_log = []
    def record_comm(self, duration_s, nbytes, had_nan=False):
        if had_nan:
            self._nan_count += 1
        if duration_s > self._timeout_s:
            self._consecutive_timeouts += 1
        else:
            self._consecutive_timeouts = 0
        if duration_s > 0 and nbytes > 0:
            bw = (nbytes * 8) / (duration_s * 1e9)
            if self._baseline_bw is None:
                self._baseline_bw = bw
            self._health_log.append({
                'bw_gbps': round(bw, 4),
                'duration_s': round(duration_s, 6),
                'nan': had_nan,
            })
            if len(self._health_log) > 1000:
                self._health_log = self._health_log[-1000:]
    def is_healthy(self):
        if self._consecutive_timeouts >= 3:
            return False, 'consecutive_timeouts'
        if self._nan_count > 10:
            return False, 'excessive_nans'
        if (self._baseline_bw and self._health_log and
            len(self._health_log) >= 10):
            recent_bw = sum(h['bw_gbps'] for h in self._health_log[-10:]) / 10
            if recent_bw < self._baseline_bw * self._bw_threshold:
                return False, 'bandwidth_degradation'
        return True, 'healthy'
    def should_adjust_Kx(self, current_Kx):
        healthy, reason = self.is_healthy()
        if not healthy and reason == 'bandwidth_degradation':
            return max(1, current_Kx * 2)  # Double Kx to reduce comm
        if not healthy and reason == 'consecutive_timeouts':
            return max(1, current_Kx * 4)  # Aggressively reduce comm
        return current_Kx
    def reset(self):
        self._consecutive_timeouts = 0
        self._nan_count = 0
        self._health_log.clear()
        self._baseline_bw = None
    def stats(self):
        healthy, reason = self.is_healthy()
        return {
            'healthy': healthy,
            'reason': reason,
            'consecutive_timeouts': self._consecutive_timeouts,
            'nan_count': self._nan_count,
            'baseline_bw_gbps': round(self._baseline_bw, 4) if self._baseline_bw else None,
            'n_records': len(self._health_log),
        }

class DeslocRingAllReducePlanner:
    """Plan optimal Ring-AllReduce for DES-LOC.
    Ref: NCCL Ring-AllReduce: O(2(N-1)/N * D) bandwidth-optimal."""
    def __init__(self, world_size, ring_order=None):
        self._world_size = world_size
        self._ring_order = ring_order or list(range(world_size))
    def compute_optimal_chunks(self, tensor_size, max_chunks=256):
        chunk_size = tensor_size // self._world_size
        if chunk_size < 1024:
            return 1
        n_chunks = min(max_chunks, tensor_size // (1024 * 1024))
        return max(1, n_chunks)
    def bandwidth_optimal_time(self, tensor_bytes, bw_gbps):
        factor = 2.0 * (self._world_size - 1) / max(1, self._world_size)
        return round(tensor_bytes * factor / (bw_gbps * 1e9 / 8) * 1000, 4)
    def plan_tiered_allreduce(self, step, Kx, Ku, Kv,
                               param_bytes, mom1_bytes, mom2_bytes,
                               bw_gbps=400.0):
        plan = []
        if step % Kx == 0:
            t = self.bandwidth_optimal_time(param_bytes, bw_gbps)
            plan.append(('x', param_bytes, t))
        if step % Ku == 0:
            t = self.bandwidth_optimal_time(mom1_bytes, bw_gbps)
            plan.append(('u', mom1_bytes, t))
        if step % Kv == 0:
            t = self.bandwidth_optimal_time(mom2_bytes, bw_gbps)
            plan.append(('v', mom2_bytes, t))
        total_bytes = sum(p[1] for p in plan)
        total_time = sum(p[2] for p in plan)
        return {
            'step': step,
            'reductions': plan,
            'total_bytes': total_bytes,
            'total_time_ms': round(total_time, 4),
            'n_reductions': len(plan),
        }

# M247: end


# =====================================================================
# M247T — Claude-16
# =====================================================================

import math as _m247t_math

class DeslocBackendDiagnostics:
    """Full backend diagnostics for DES-LOC troubleshooting."""

    def __init__(self):
        self._errors = []
        self._warnings = []
        self._comm_events = 0
        self._failed_events = 0

    def record_success(self):
        self._comm_events += 1

    def record_failure(self, error_msg):
        self._failed_events += 1
        self._errors.append(error_msg)
        if len(self._errors) > 500:
            self._errors = self._errors[-500:]

    def failure_rate(self):
        total = self._comm_events + self._failed_events
        if total == 0:
            return 0.0
        return round(self._failed_events / total, 6)

    def report(self):
        return {"comm_events": self._comm_events,
                "failed": self._failed_events,
                "failure_rate": self.failure_rate(),
                "recent_errors": self._errors[-5:]}

class DeslocBackendBenchmark:
    """Micro-benchmark backend comm primitives."""

    def __init__(self):
        self._results = {}

    def record_benchmark(self, op, size_bytes, time_ms):
        key = (op, size_bytes)
        if key not in self._results:
            self._results[key] = []
        self._results[key].append(time_ms)
        if len(self._results[key]) > 100:
            self._results[key] = self._results[key][-100:]

    def get_avg_time(self, op, size_bytes):
        key = (op, size_bytes)
        vals = self._results.get(key, [])
        if not vals:
            return 0.0
        return round(sum(vals) / len(vals), 4)

    def get_bandwidth(self, op, size_bytes):
        avg_ms = self.get_avg_time(op, size_bytes)
        if avg_ms <= 0:
            return 0.0
        return round((size_bytes * 8) / (avg_ms * 1e6), 4)


# M247T: end
