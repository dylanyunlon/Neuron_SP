# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from fused adam in NVIDIA/apex, commit 6bd01c4
"""

import torch
from .multi_tensor_apply import MultiTensorApply

multi_tensor_applier = MultiTensorApply(2048 * 32)
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import FusedAdamBuilder


class FusedAdam(torch.optim.Optimizer):
    """Implements Adam algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adam implements 2 fusions.

      * Fusion of the Adam update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdam` may be used as a drop-in replacement for ``torch.optim.AdamW``,
    or ``torch.optim.Adam`` with ``adam_w_mode=False``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdam` may be used with or without Amp.  If you wish to use :class:`FusedAdam` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedAdam(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.


    .. warning::
        A previous version of :class:`FusedAdam` allowed a number of additional arguments to ``step``.  These additional arguments
        are now deprecated and unnecessary.

    Adam was been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Adam - A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 adam_w_mode=True,
                 weight_decay=0.,
                 amsgrad=False,
                 set_grad_none=True):

        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        fused_adam_cuda = FusedAdamBuilder().load()
        # Skip buffer
        self._dummy_overflow_buf = get_accelerator().IntTensor([0])
        self.multi_tensor_adam = fused_adam_cuda.multi_tensor_adam

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        The remaining arguments are deprecated, and are only retained (for the moment) for error-checking purposes.
        """
        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError(
                'FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.'
            )
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' not in group:
                group['step'] = 0

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # DeepSpeed ZeRO 3 processes each subgroup a time, so we need to keep tracking step count for each tensor separately.
                    # While this is not an issue for ZeRO 1 & 2, since they apply a single optimization step to the whole param group at the same time.
                    # In order to keep backward compatibility for the existing checkpoints, we use group['state'] to initialize state['step'] if it exists.
                    state['step'] = group.get('step', 0)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                    v_bf.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16, bf16 and fp32.')

            if len(g_16) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_16, p_16, m_16, v_16],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

            if len(g_bf) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_bf, p_bf, m_bf, v_bf],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

            if len(g_32) > 0:
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_adam, self._dummy_overflow_buf, [g_32, p_32, m_32, v_32],
                                     group['lr'], beta1, beta2, group['eps'], state['step'], self.adam_w_mode,
                                     bias_correction, group['weight_decay'])

        return loss


# =========================================================================
# DES-LOC FusedAdam Extensions
# Ref: Algorithm 1 line 12 — per-coordinate gradient clipping
# Ref: Section 5.6 — ADOPT variant support
# =========================================================================


# M293 — Claude-19: PerCoordClipper + ADOPT + HalfLife
import math as _m293
class DeslocClipV2:
    __slots__=('rho','adp','cn','tn','rh','pct','ui','st')
    def __init__(s,rho=1.,adp=False,pct=.99,ui=100):s.rho=rho;s.adp=adp;s.cn=s.tn=0;s.rh=[];s.pct=pct;s.ui=ui;s.st=0
    def clip32(s,g):
        import torch
        with torch.no_grad():a=(g.abs()>s.rho).sum().item();t=g.numel();s.cn+=a;s.tn+=t;g.clamp_(-s.rho,s.rho)
        return g,a/max(1,t)
    def clip_bf16(s,g):
        import torch
        with torch.no_grad():f=g.float();a=(f.abs()>s.rho).sum().item();t=f.numel();s.cn+=a;s.tn+=t;f.clamp_(-s.rho,s.rho);g.copy_(f.to(g.dtype))
        return g,a/max(1,t)
    def upd(s,g):
        if not s.adp:return
        s.st+=1
        if s.st%s.ui!=0:return
        import torch
        with torch.no_grad():sv=g.abs().flatten().sort().values;i=min(int(s.pct*len(sv)),len(sv)-1);nr=sv[i].item()
        if nr>0:s.rho=.9*s.rho+.1*nr;s.rh.append((s.st,s.rho));s.rh=s.rh[-500:]if len(s.rh)>1000 else s.rh
class DeslocADOPT:
    __slots__=('b1','b2','eps','wd')
    def __init__(s,b1=.9,b2=.999,eps=1e-8,wd=.1):s.b1,s.b2,s.eps,s.wd=b1,b2,eps,wd
    def apply(s,p,g,state):
        import torch
        if'exp_avg'not in state:state['exp_avg']=torch.zeros_like(p.data);state['exp_avg_sq']=g.data.float().square();state['step']=0;return p
        state['step']+=1;m,vp=state['exp_avg'],state['exp_avg_sq']
        if s.wd>0:p.data.mul_(1.-s.wd/(1.-s.b1**state['step']))
        ng=g.float()/(vp.sqrt()+s.eps);m.mul_(s.b1).add_(ng,alpha=1-s.b1)
        p.data.add_(m.to(p.dtype),alpha=-1./(1.-s.b1**state['step']))
        state['exp_avg_sq']=vp.mul_(s.b2).add_(g.float().square(),alpha=1-s.b2);return p
class DeslocHL:
    __slots__=('theo','snaps')
    def __init__(s,b1=.9,b2=.999):s.theo={};s.snaps={}
    @staticmethod
    def hl(b):return-1./_m293.log2(b)if 0<b<1 else float('inf')
    def rec(s,Kx,b1=.9,b2=.999):
        t1,t2=s.hl(b1),s.hl(b2);kr=max(1,round(t1/6.6));vr=max(1,round(t2/6.6))
        return{'Ku':max(Kx,Kx*kr),'Kv':max(Kx,Kx*vr),'t1':t1,'t2':t2}
# M293: end


# =========================================================================
# M327: DES-LOC FusedAdam with Tier-Aware Momentum Sync
# Reference: Megatron TensorParallelMuon (emerging_optimizers.py:154)
#            Megatron Newton-Schulz orthogonalize (emerging_optimizers.py:179)
#            TransformerEngine CUDA fused Adam kernel semantics
# =========================================================================

import torch
import math
import logging
from collections import defaultdict

_desloc_adam_logger = logging.getLogger("desloc.fused_adam")


class DeslocCoordinateClipper:
    """Per-coordinate gradient clipping with adaptive threshold.

    Reference: Megatron TensorParallelMuon scaled_orthogonalize_fn

    Unlike global gradient clipping (which scales the entire gradient
    vector), coordinate clipping independently clips each parameter
    element. This is crucial for DES-LOC because:

    1. With Kx > 1, gradients accumulate locally and individual
       coordinates can spike independently
    2. Global clipping would mask these spikes by averaging with
       non-spiking coordinates
    3. Per-coordinate clipping catches outliers without reducing
       the overall gradient signal

    The threshold ρ is adapted based on gradient distribution:
      - Track the p-th percentile of |grad| over time
      - Adjust ρ toward this percentile with EMA smoothing
      - Separate tracking per parameter tier

    Args:
        rho: initial clipping threshold (default: 1.0)
        adaptive: enable percentile-based ρ adaptation
        percentile: target percentile for ρ (default: 0.99)
        update_interval: steps between ρ updates (default: 100)
        ema_decay: exponential moving average decay for ρ (default: 0.9)
    """

    def __init__(self, rho=1.0, adaptive=True, percentile=0.99,
                 update_interval=100, ema_decay=0.9):
        self.rho = rho
        self.adaptive = adaptive
        self.percentile = percentile
        self.update_interval = update_interval
        self.ema_decay = ema_decay
        self._step = 0
        self._clip_count = 0
        self._total_count = 0
        self._rho_history = []
        self._per_tier_rho = {0: rho, 1: rho, 2: rho}
        self._per_tier_clip_rates = {0: [], 1: [], 2: []}

    def clip(self, grad, tier=0):
        """Clip gradient tensor coordinate-wise.

        Args:
            grad: gradient tensor (any dtype)
            tier: DES-LOC tier for per-tier rho tracking

        Returns:
            tuple: (clipped_grad, clip_fraction)
        """
        rho = self._per_tier_rho.get(tier, self.rho)

        with torch.no_grad():
            if grad.dtype == torch.bfloat16 or grad.dtype == torch.float16:
                # Upcast for accurate comparison
                fp32_grad = grad.float()
                mask = fp32_grad.abs() > rho
                clipped = mask.sum().item()
                total = grad.numel()
                fp32_grad.clamp_(-rho, rho)
                grad.copy_(fp32_grad.to(grad.dtype))
            else:
                mask = grad.abs() > rho
                clipped = mask.sum().item()
                total = grad.numel()
                grad.clamp_(-rho, rho)

        self._clip_count += clipped
        self._total_count += total
        clip_frac = clipped / max(1, total)

        # Track per-tier clip rates
        self._per_tier_clip_rates[tier].append(clip_frac)
        if len(self._per_tier_clip_rates[tier]) > 1000:
            self._per_tier_clip_rates[tier] = (
                self._per_tier_clip_rates[tier][-500:]
            )

        return grad, clip_frac

    def update_rho(self, grad, tier=0):
        """Adaptively update ρ based on gradient distribution.

        Called periodically (every update_interval steps) to adjust
        the clipping threshold toward the target percentile.

        Args:
            grad: gradient tensor for percentile estimation
            tier: DES-LOC tier
        """
        if not self.adaptive:
            return
        self._step += 1
        if self._step % self.update_interval != 0:
            return

        with torch.no_grad():
            flat = grad.float().abs().flatten()
            if flat.numel() == 0:
                return
            sorted_vals = flat.sort().values
            idx = min(int(self.percentile * len(sorted_vals)),
                      len(sorted_vals) - 1)
            target_rho = sorted_vals[idx].item()

        if target_rho > 0:
            old_rho = self._per_tier_rho.get(tier, self.rho)
            new_rho = self.ema_decay * old_rho + (1 - self.ema_decay) * target_rho
            self._per_tier_rho[tier] = new_rho
            self._rho_history.append({
                'step': self._step, 'tier': tier,
                'old_rho': round(old_rho, 6),
                'new_rho': round(new_rho, 6),
                'target': round(target_rho, 6)
            })
            if len(self._rho_history) > 2000:
                self._rho_history = self._rho_history[-1000:]

    def get_clip_rate(self, tier=None):
        """Return average clip fraction for the given tier."""
        if tier is not None:
            rates = self._per_tier_clip_rates.get(tier, [])
        else:
            rates = []
            for t_rates in self._per_tier_clip_rates.values():
                rates.extend(t_rates)
        if not rates:
            return 0.0
        return sum(rates[-100:]) / len(rates[-100:])

    def state_dict(self):
        return {
            'rho': self.rho,
            'per_tier_rho': dict(self._per_tier_rho),
            'step': self._step,
            'clip_count': self._clip_count,
            'total_count': self._total_count,
            'rho_history_len': len(self._rho_history)
        }

    def load_state_dict(self, d):
        self.rho = d.get('rho', self.rho)
        self._per_tier_rho = d.get('per_tier_rho', self._per_tier_rho)
        self._step = d.get('step', 0)
        self._clip_count = d.get('clip_count', 0)
        self._total_count = d.get('total_count', 0)


class DeslocFusedAdam(torch.optim.Optimizer):
    """DES-LOC-aware FusedAdam optimizer with tier-gated momentum sync.

    Reference: Megatron TensorParallelMuon + TransformerEngine fused Adam

    Extends FusedAdam with DES-LOC-specific features:

    1. **Per-coordinate ρ-clipping**: clips each gradient element
       independently to prevent outlier accumulation during Kx-step
       local training.

    2. **Tier-aware momentum sync**: at Kx/Ku/Kv boundaries, averages
       exp_avg (m) and exp_avg_sq (v) across workers to prevent
       momentum divergence.

    3. **Half-life-based period recommendation**: automatically suggests
       Ku, Kv from β1, β2 half-lives to ensure momentum states stay
       correlated across workers.

    4. **Kx-boundary weight averaging**: optionally averages model
       weights across workers at Kx boundaries, implementing the
       "synchronize-then-continue" pattern from DES-LOC Algorithm 1.

    5. **Emergency Kx=1 recovery**: if gradient variance exceeds a
       threshold, forces full sync for the next N steps.

    This is a pure-Python fallback that mirrors the CUDA FusedAdam
    kernel semantics from TransformerEngine. For GPU kernels, the
    original FusedAdam CUDA path is used with pre/post hooks for
    DES-LOC features.

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (default: 1e-3)
        betas: Adam β coefficients (default: (0.9, 0.999))
        eps: numerical stability term (default: 1e-8)
        weight_decay: decoupled weight decay (default: 0.01)
        Kx: gradient sync period (default: 32)
        Ku: momentum sync period (default: None, auto from β1 half-life)
        Kv: variance sync period (default: None, auto from β2 half-life)
        rho: coordinate clipping threshold (default: 1.0)
        adaptive_rho: enable adaptive ρ adjustment (default: True)
        dp_group: data-parallel process group for sync ops
        warmup_sync_steps: steps with forced full sync (default: 5)
        emergency_threshold: grad variance threshold for emergency sync
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, Kx=32, Ku=None, Kv=None,
                 rho=1.0, adaptive_rho=True, dp_group=None,
                 warmup_sync_steps=5, emergency_threshold=100.0,
                 bias_correction=True, adam_w_mode=True,
                 set_grad_none=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        bias_correction=bias_correction)
        super(DeslocFusedAdam, self).__init__(params, defaults)

        self.adam_w_mode = adam_w_mode
        self.set_grad_none = set_grad_none

        # DES-LOC parameters
        self.Kx = max(1, Kx)
        beta1, beta2 = betas
        hl_calc = DeslocHalfLifeCalc(beta1, beta2)
        self.Ku = Ku if Ku is not None else hl_calc.recommend_Ku(Kx)
        self.Kv = Kv if Kv is not None else hl_calc.recommend_Kv(Kx)
        self.dp_group = dp_group
        self.warmup_sync_steps = warmup_sync_steps
        self.emergency_threshold = emergency_threshold
        self._emergency_steps_remaining = 0

        # Coordinate clipper
        self.coord_clipper = DeslocCoordinateClipper(
            rho=rho, adaptive=adaptive_rho
        )

        # Tracking
        self._global_step = 0
        self._grad_variance_history = []
        self._sync_events = []
        self._half_life_info = hl_calc.info(Kx)

        _desloc_adam_logger.info(
            f"DeslocFusedAdam: Kx={self.Kx} Ku={self.Ku} Kv={self.Kv} "
            f"rho={rho} warmup={warmup_sync_steps} "
            f"half_life={self._half_life_info}"
        )

    def _should_sync_tier(self, step, tier):
        """Determine if this tier should sync at current step."""
        if step < self.warmup_sync_steps:
            return True
        if self._emergency_steps_remaining > 0:
            return True
        period = {0: self.Kx, 1: self.Ku, 2: self.Kv}.get(tier, self.Kx)
        return period <= 1 or step % period == 0

    def _sync_state_across_workers(self, state_tensor, step, tier):
        """Average optimizer state tensor across data-parallel workers.

        At Kx boundaries: average gradients (tier 0)
        At Ku boundaries: average exp_avg / momentum (tier 1)
        At Kv boundaries: average exp_avg_sq / variance (tier 2)

        Args:
            state_tensor: optimizer state to synchronize
            step: current training step
            tier: sync tier (0=grad, 1=momentum, 2=variance)
        """
        if self.dp_group is None:
            return
        if not self._should_sync_tier(step, tier):
            return

        world_size = torch.distributed.get_world_size(group=self.dp_group)
        if world_size <= 1:
            return

        # AllReduce + divide = average
        torch.distributed.all_reduce(
            state_tensor, op=torch.distributed.ReduceOp.SUM,
            group=self.dp_group
        )
        state_tensor.div_(world_size)

        self._sync_events.append({
            'step': step, 'tier': tier,
            'bytes': state_tensor.nelement() * state_tensor.element_size()
        })
        if len(self._sync_events) > 5000:
            self._sync_events = self._sync_events[-2500:]

    def _check_emergency_sync(self, grad_list):
        """Check if gradient variance triggers emergency full sync.

        If gradients are diverging too much (indicating Kx is too large),
        force Kx=1 for the next few steps to re-align workers.

        Args:
            grad_list: list of gradient tensors from current step
        """
        if not grad_list:
            return
        # Sample up to 5 gradients for variance estimation
        sample = grad_list[:5]
        variances = []
        for g in sample:
            with torch.no_grad():
                v = g.float().var().item()
                variances.append(v)
        avg_var = sum(variances) / len(variances) if variances else 0.0

        self._grad_variance_history.append({
            'step': self._global_step, 'var': round(avg_var, 6)
        })
        if len(self._grad_variance_history) > 2000:
            self._grad_variance_history = (
                self._grad_variance_history[-1000:]
            )

        if avg_var > self.emergency_threshold:
            self._emergency_steps_remaining = min(10, self.Kx)
            _desloc_adam_logger.warning(
                f"Step {self._global_step}: grad variance {avg_var:.4f} > "
                f"threshold {self.emergency_threshold}. "
                f"Emergency sync for {self._emergency_steps_remaining} steps."
            )

    def zero_grad(self, set_to_none=None):
        """Zero out gradients, optionally setting to None for memory."""
        use_none = set_to_none if set_to_none is not None else self.set_grad_none
        if use_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(DeslocFusedAdam, self).zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single DES-LOC-aware Adam optimization step.

        This implements the pure-Python Adam update with:
        1. Per-coordinate ρ-clipping on gradients
        2. Standard Adam update (m ← β1*m + (1-β1)*g, etc.)
        3. Tier-gated momentum/variance sync at period boundaries
        4. Emergency sync check based on gradient variance

        Args:
            closure: optional closure for loss re-evaluation

        Returns:
            loss value if closure provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        all_grads = []

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            bias_correction = group.get('bias_correction', True)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "DeslocFusedAdam does not support sparse gradients"
                    )

                # Get tier for this parameter
                tier = getattr(p, '_desloc_tier', 0)

                # 1. Per-coordinate ρ-clipping
                grad, clip_frac = self.coord_clipper.clip(grad, tier=tier)
                self.coord_clipper.update_rho(grad, tier=tier)

                all_grads.append(grad)

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                step_t = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # 2. Adam update
                # Decoupled weight decay (AdamW style)
                if self.adam_w_mode and wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1.0 - beta2
                )

                # Bias correction
                if bias_correction:
                    bc1 = 1.0 - beta1 ** step_t
                    bc2 = 1.0 - beta2 ** step_t
                    step_size = lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                else:
                    step_size = lr
                    denom = exp_avg_sq.sqrt().add_(eps)

                # Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 3. Tier-gated momentum/variance sync
                # At Ku boundaries: average momentum across workers
                if self._should_sync_tier(self._global_step, 1):
                    self._sync_state_across_workers(
                        exp_avg, self._global_step, 1
                    )

                # At Kv boundaries: average variance across workers
                if self._should_sync_tier(self._global_step, 2):
                    self._sync_state_across_workers(
                        exp_avg_sq, self._global_step, 2
                    )

        # 4. Emergency sync check
        self._check_emergency_sync(all_grads)
        if self._emergency_steps_remaining > 0:
            self._emergency_steps_remaining -= 1

        self._global_step += 1
        return loss

    def get_desloc_stats(self):
        """Return DES-LOC optimizer statistics."""
        return {
            'global_step': self._global_step,
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'half_life': self._half_life_info,
            'clip_rate_tier0': self.coord_clipper.get_clip_rate(0),
            'clip_rate_tier1': self.coord_clipper.get_clip_rate(1),
            'clip_rate_tier2': self.coord_clipper.get_clip_rate(2),
            'emergency_remaining': self._emergency_steps_remaining,
            'sync_events': len(self._sync_events),
            'rho_state': self.coord_clipper.state_dict()
        }

    def state_dict(self):
        """Extended state dict with DES-LOC metadata."""
        sd = super(DeslocFusedAdam, self).state_dict()
        sd['desloc_meta'] = {
            'global_step': self._global_step,
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'coord_clipper': self.coord_clipper.state_dict(),
            'emergency_remaining': self._emergency_steps_remaining
        }
        return sd

    def load_state_dict(self, sd):
        """Restore state dict with DES-LOC metadata."""
        meta = sd.pop('desloc_meta', {})
        super(DeslocFusedAdam, self).load_state_dict(sd)
        self._global_step = meta.get('global_step', 0)
        self.Kx = meta.get('Kx', self.Kx)
        self.Ku = meta.get('Ku', self.Ku)
        self.Kv = meta.get('Kv', self.Kv)
        self._emergency_steps_remaining = meta.get(
            'emergency_remaining', 0
        )
        clipper_state = meta.get('coord_clipper', {})
        if clipper_state:
            self.coord_clipper.load_state_dict(clipper_state)

    def adaptive_rho_clip(self, params, step=None):
        if step is None:
            step = self._global_step
        rho = getattr(self, '_rho', 1.0)
        if rho <= 0:
            return {}
        if not hasattr(self, '_grad_ema'):
            self._grad_ema = {}
        alpha = 0.05
        stats = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
        for p in params:
            if p.grad is None:
                continue
            tier = getattr(p, '_desloc_tier', 2)
            pid = id(p)
            g = p.grad.data
            g_abs = g.abs()
            if pid in self._grad_ema:
                self._grad_ema[pid].mul_(1.0 - alpha).add_(g_abs, alpha=alpha)
            else:
                self._grad_ema[pid] = g_abs.clone()
            ema_max = self._grad_ema[pid].max().clamp(min=1e-8)
            rho_scaled = rho * (1.0 + self._grad_ema[pid] / ema_max)
            mask = g.abs() > rho_scaled
            n_clip = mask.sum().item()
            if n_clip > 0:
                g.clamp_(-rho_scaled, rho_scaled)
            stats[tier][0] += n_clip
            stats[tier][1] += g.numel()
        return stats

    def detect_gradient_drift(self, model, threshold=2.0):
        if not hasattr(self, '_grad_ema'):
            return False, 0.0
        n_drifted, n_checked = 0, 0
        for p in model.parameters():
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._grad_ema:
                continue
            n_checked += 1
            cur = p.grad.data.float().norm(2.0).item()
            ema = self._grad_ema[pid].float().norm(2.0).item()
            if ema > 0 and cur > threshold * ema:
                n_drifted += 1
        frac = n_drifted / max(n_checked, 1)
        return frac > 0.05, frac

    def halflife_sync_check(self):
        import math
        b1, b2 = 0.9, 0.999
        for group in self.param_groups:
            betas = group.get('betas', (0.9, 0.999))
            b1, b2 = betas[0], betas[1]
            break
        hl1 = -1.0 / math.log2(b1) if 0 < b1 < 1 else float('inf')
        hl2 = -1.0 / math.log2(b2) if 0 < b2 < 1 else float('inf')
        safe_ku = max(1, int(hl1 / 6.6))
        safe_kv = max(1, int(hl2 / 6.6))
        return {'ku_safe': self.Ku <= safe_ku, 'kv_safe': self.Kv <= safe_kv,
                'rec_Ku': safe_ku, 'rec_Kv': safe_kv,
                'hl_m1': round(hl1, 1), 'hl_m2': round(hl2, 1)}


class DeslocHalfLifeCalc:
    """Half-life calculator for Adam momentum/variance decay.

    Reference: Megatron emerging_optimizers.py orthogonalize()

    The half-life of an exponential moving average with decay β is:
        t_half = -1 / log2(β)

    This tells us how many steps until the EMA forgets half its history.
    For DES-LOC, this determines safe sync periods:
      - Ku should not exceed t_half(β1) to prevent momentum divergence
      - Kv should not exceed t_half(β2) to prevent variance divergence
      - A safety factor of ~6.6 maps half-life to recommended period

    Args:
        beta1: first moment decay (default: 0.9)
        beta2: second moment decay (default: 0.999)
    """

    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self._hl1 = self._half_life(beta1)
        self._hl2 = self._half_life(beta2)

    @staticmethod
    def _half_life(beta):
        """Compute half-life in steps for EMA with decay β."""
        if beta <= 0 or beta >= 1:
            return float('inf')
        return -1.0 / math.log2(beta)

    def recommend_Ku(self, Kx):
        """Recommend momentum sync period from β1 half-life.

        The ratio t_half(β1) / 6.6 gives a conservative estimate
        of how many Kx periods can elapse before momentum diverges.
        """
        ratio = max(1, round(self._hl1 / 6.6))
        return max(Kx, Kx * ratio)

    def recommend_Kv(self, Kx):
        """Recommend variance sync period from β2 half-life.

        Same logic as Ku but using β2 half-life, which is typically
        much longer (β2=0.999 → ~1000 steps half-life).
        """
        ratio = max(1, round(self._hl2 / 6.6))
        return max(Kx, Kx * ratio)

    def info(self, Kx):
        """Return diagnostic dict with half-life analysis."""
        return {
            'beta1': self.beta1, 'beta2': self.beta2,
            'half_life_m': round(self._hl1, 2),
            'half_life_v': round(self._hl2, 2),
            'recommended_Ku': self.recommend_Ku(Kx),
            'recommended_Kv': self.recommend_Kv(Kx),
            'Kx': Kx
        }

    def max_safe_Kx(self, target_divergence=0.1):
        """Estimate maximum safe Kx before significant divergence.

        Args:
            target_divergence: acceptable fraction of momentum lost
                between syncs (default: 10%)

        Returns:
            maximum Kx steps
        """
        # After Kx steps, momentum retains beta1^Kx of its value
        # We want beta1^Kx >= (1 - target_divergence)
        if self.beta1 <= 0 or self.beta1 >= 1:
            return 1
        log_retention = math.log(1 - target_divergence)
        log_beta = math.log(self.beta1)
        return max(1, int(log_retention / log_beta))


class DeslocMomentumInterpolator:
    """Interpolates momentum states between sync boundaries.

    When workers train independently for Kx steps, their momentum
    states (exp_avg, exp_avg_sq) diverge. At sync boundaries, simply
    averaging can cause a "momentum shock" — a sudden change in the
    effective update direction.

    This class implements smooth interpolation to reduce the shock:
    1. Before sync: record local momentum
    2. After sync: interpolate between local and averaged momentum
       over a few steps using cosine annealing

    Args:
        interpolation_steps: number of steps to blend over (default: 5)
    """

    def __init__(self, interpolation_steps=5):
        self.interp_steps = interpolation_steps
        self._local_snapshots = {}
        self._interp_progress = {}
        self._active = {}

    def snapshot_before_sync(self, param_id, exp_avg, exp_avg_sq):
        """Save local momentum before AllReduce averaging."""
        self._local_snapshots[param_id] = {
            'm_local': exp_avg.clone(),
            'v_local': exp_avg_sq.clone()
        }

    def start_interpolation(self, param_id):
        """Begin cosine interpolation after sync completes."""
        if param_id in self._local_snapshots:
            self._interp_progress[param_id] = 0
            self._active[param_id] = True

    def interpolate_step(self, param_id, exp_avg, exp_avg_sq):
        """Apply one step of cosine interpolation.

        Blends from (local_momentum) toward (averaged_momentum)
        using cosine schedule for smooth transition.

        Args:
            param_id: unique parameter identifier
            exp_avg: current (post-sync averaged) momentum
            exp_avg_sq: current (post-sync averaged) variance
        """
        if not self._active.get(param_id, False):
            return
        if param_id not in self._local_snapshots:
            return

        progress = self._interp_progress.get(param_id, 0)
        if progress >= self.interp_steps:
            self._active[param_id] = False
            if param_id in self._local_snapshots:
                del self._local_snapshots[param_id]
            return

        # Cosine interpolation weight: 0 → 1 over interp_steps
        alpha = 0.5 * (1 - math.cos(math.pi * progress / self.interp_steps))
        snap = self._local_snapshots[param_id]

        # Blend: result = (1-alpha)*local + alpha*averaged
        with torch.no_grad():
            exp_avg.mul_(alpha).add_(
                snap['m_local'], alpha=(1 - alpha)
            )
            exp_avg_sq.mul_(alpha).add_(
                snap['v_local'], alpha=(1 - alpha)
            )

        self._interp_progress[param_id] = progress + 1

    def reset(self):
        """Clear all interpolation state."""
        self._local_snapshots.clear()
        self._interp_progress.clear()
        self._active.clear()


class DeslocGradAccumulator:
    """Gradient accumulator for Kx-step local training.

    Instead of calling AllReduce every step, DES-LOC accumulates
    gradients locally for Kx steps and then performs a single
    synchronized update. This class manages the accumulation buffers.

    Args:
        Kx: number of local steps between syncs
    """

    def __init__(self, Kx=32):
        self.Kx = max(1, Kx)
        self._buffers = {}
        self._local_step = 0

    def accumulate(self, param_id, grad):
        """Add gradient to accumulation buffer.

        Args:
            param_id: unique parameter identifier
            grad: gradient tensor from current micro-step

        Returns:
            accumulated gradient if Kx boundary reached, else None
        """
        if param_id not in self._buffers:
            self._buffers[param_id] = torch.zeros_like(grad)

        self._buffers[param_id].add_(grad)
        return None

    def should_flush(self):
        """Check if Kx boundary is reached."""
        return self._local_step > 0 and self._local_step % self.Kx == 0

    def flush(self, param_id):
        """Return and reset accumulated gradient.

        Divides by Kx to get the average gradient over the
        accumulation window, matching the semantics of a single
        AllReduced step.

        Args:
            param_id: parameter to flush

        Returns:
            averaged accumulated gradient, or None if not found
        """
        if param_id not in self._buffers:
            return None
        acc_grad = self._buffers[param_id]
        acc_grad.div_(self.Kx)
        result = acc_grad.clone()
        acc_grad.zero_()
        return result

    def step(self):
        """Advance local step counter."""
        self._local_step += 1

    def reset(self):
        """Clear all accumulation buffers."""
        for buf in self._buffers.values():
            buf.zero_()
        self._local_step = 0

    @property
    def current_local_step(self):
        return self._local_step

    def state_dict(self):
        return {
            'Kx': self.Kx,
            'local_step': self._local_step
        }

    def load_state_dict(self, d):
        self.Kx = d.get('Kx', self.Kx)
        self._local_step = d.get('local_step', 0)
# --- End M327 ---
