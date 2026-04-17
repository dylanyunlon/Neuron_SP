# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from NVIDIA/apex/optimizer/fused_adam and implements the LAMB optimizer.
Extended with DES-LOC (Desynced Low Communication) independent sync periods for
distributed training with reduced AllReduce frequency. See ICLR 2026 paper:
"DES-LOC: Desynced Low Communication Adaptive Optimizers for Foundation Models"

DES-LOC integration points (M107):
- Half-life-based sync period computation for momentum states
- Per-coordinate gradient clipping (Algorithm 1, line 12)
- Independent synchronization flags for params/1st-moment/2nd-moment
- Nesterov outer optimizer support (Section 5.5, RQ5)
- Trust ratio tracking for LAMB-specific DES-LOC diagnostics
"""
import math
import types
import torch
import logging
from deepspeed.ops.op_builder import FusedLambBuilder

logger = logging.getLogger(__name__)

# DES-LOC constants following paper notation
DESLOC_DEFAULT_KX = 0        # 0 = disabled, use standard DDP
DESLOC_DEFAULT_KU_RATIO = 3  # Ku = 3 * Kx (first moment syncs 3x less)
DESLOC_DEFAULT_KV_RATIO = 6  # Kv = 6 * Kx (second moment syncs 6x less)
DESLOC_DEFAULT_CLIP_RHO = 2.0  # Per-coordinate clipping bound (Assumption 3)


def compute_ema_half_life(beta):
    """Compute half-life of EMA: t_half = -1 / log2(beta).

    DES-LOC Section 2: drives Ku/Kv ratios via half-life principle.
    beta=0.9 -> 6.6 steps, beta=0.999 -> 693 steps.
    """
    if beta <= 0.0 or beta >= 1.0:
        return float('inf')
    return -1.0 / (math.log(beta) / math.log(2.0))


def compute_desloc_periods(beta1, beta2, kx):
    """Compute (Kx, Ku, Kv) from betas using Half-Life Principle."""
    if kx <= 0:
        return 0, 0, 0
    hl1 = compute_ema_half_life(beta1)
    hl2 = compute_ema_half_life(beta2)
    ratio = max(1, int(math.ceil(hl2 / hl1))) if hl1 > 0 and math.isfinite(hl1) else 1
    ku = kx * min(ratio, DESLOC_DEFAULT_KU_RATIO)
    kv = kx * min(ratio, DESLOC_DEFAULT_KV_RATIO)
    return kx, ku, kv


def should_sync_at_step(step, period):
    """Deterministic periodic sync check."""
    if period <= 0:
        return False
    return (step % period) == 0


class DeslocLambState:
    """Per-parameter DES-LOC tracking: sync flags, trust ratio, grad norm."""

    __slots__ = ['param_sync_due', 'mom1_sync_due', 'mom2_sync_due',
                 'last_trust_ratio', 'grad_norm_accum', 'update_count']

    def __init__(self):
        self.param_sync_due = False
        self.mom1_sync_due = False
        self.mom2_sync_due = False
        self.last_trust_ratio = 1.0
        self.grad_norm_accum = 0.0
        self.update_count = 0

    def record_step(self, trust_ratio, grad_norm):
        if isinstance(trust_ratio, torch.Tensor):
            trust_ratio = trust_ratio.item()
        self.last_trust_ratio = trust_ratio
        self.grad_norm_accum += grad_norm
        self.update_count += 1

    def check_sync_flags(self, step, kx, ku, kv):
        self.param_sync_due = should_sync_at_step(step, kx)
        self.mom1_sync_due = should_sync_at_step(step, ku)
        self.mom2_sync_due = should_sync_at_step(step, kv)

    def needs_any_sync(self):
        return self.param_sync_due or self.mom1_sync_due or self.mom2_sync_due


class DeslocNesterovOuter:
    """Nesterov momentum outer optimizer for DES-LOC sync (Section 5.5).

    At sync points: v = mom*v + (avg-local), param = avg + mom*v.
    Result: +0.5% perplexity over naive averaging.
    """

    def __init__(self, momentum=0.9, outer_lr=1.0):
        self.momentum = momentum
        self.outer_lr = outer_lr
        self._velocities = {}
        self._apply_count = 0

    def get_velocity(self, param_id, param_data):
        if param_id not in self._velocities:
            self._velocities[param_id] = torch.zeros_like(param_data, memory_format=torch.preserve_format)
        return self._velocities[param_id]

    def apply(self, param, averaged_param):
        pid = id(param)
        vel = self.get_velocity(pid, param.data)
        delta = averaged_param.data - param.data
        vel.mul_(self.momentum).add_(delta)
        param.data.copy_(averaged_param.data)
        param.data.add_(vel, alpha=self.momentum * self.outer_lr)
        self._apply_count += 1

    def reset(self):
        for v in self._velocities.values():
            v.zero_()

    def state_dict(self):
        return {'momentum': self.momentum, 'outer_lr': self.outer_lr, 'apply_count': self._apply_count}

    def load_state_dict(self, d):
        self.momentum = d.get('momentum', self.momentum)
        self.outer_lr = d.get('outer_lr', self.outer_lr)
        self._apply_count = d.get('apply_count', 0)


class DeslocCommTracker:
    """Tracks sync events for NKI-FA format logging and comm reduction ratio."""

    def __init__(self):
        self.step = 0
        self.param_syncs = 0
        self.mom1_syncs = 0
        self.mom2_syncs = 0
        self.trust_ratio_sum = 0.0
        self.trust_ratio_count = 0

    def record_sync(self, sync_type, num_elements=0, element_size=4):
        if sync_type == 'param':
            self.param_syncs += 1
        elif sync_type == 'mom1':
            self.mom1_syncs += 1
        elif sync_type == 'mom2':
            self.mom2_syncs += 1

    def record_trust_ratio(self, ratio):
        if isinstance(ratio, torch.Tensor):
            ratio = ratio.item()
        if math.isfinite(ratio):
            self.trust_ratio_sum += ratio
            self.trust_ratio_count += 1

    def advance(self):
        self.step += 1

    @property
    def avg_trust_ratio(self):
        return self.trust_ratio_sum / max(1, self.trust_ratio_count)

    @property
    def total_syncs(self):
        return self.param_syncs + self.mom1_syncs + self.mom2_syncs

    @property
    def comm_reduction_vs_ddp(self):
        ddp_syncs = self.step * 3
        return ddp_syncs / max(1, self.total_syncs)

    def format_log(self):
        return (f"### desloc_lamb step={self.step}, param_syncs={self.param_syncs}, "
                f"mom1_syncs={self.mom1_syncs}, mom2_syncs={self.mom2_syncs}, "
                f"reduction={self.comm_reduction_vs_ddp:.1f}x, "
                f"avg_trust={self.avg_trust_ratio:.4f} ###")


class FusedLamb(torch.optim.Optimizer):
    """Implements LAMB with DES-LOC independent sync periods.

    LAMB: Large Batch Optimization for Deep Learning (https://arxiv.org/abs/1904.00962)
    DES-LOC: Desynced Low Communication Adaptive Optimizers (ICLR 2026)

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (default: 1e-3)
        bias_correction (bool): bias correction (default: True)
        betas (Tuple[float, float]): EMA coefficients (default: (0.9, 0.999))
        eps (float): numerical stability term (default: 1e-8)
        eps_inside_sqrt (bool): eps placement in denominator (default: False)
        weight_decay (float): weight decay / L2 penalty (default: 0)
        max_grad_norm (float): global grad norm clip (default: 0.0)
        max_coeff (float): max LAMB trust ratio (default: 10.0)
        min_coeff (float): min LAMB trust ratio (default: 0.01)
        amsgrad (bool): NOT SUPPORTED (default: False)
        desloc_kx (int): parameter sync period, 0=disabled (default: 0)
        desloc_ku_ratio (int): Ku multiplier over Kx (default: 3)
        desloc_kv_ratio (int): Kv multiplier over Kx (default: 6)
        desloc_clip_rho (float): per-coordinate clip bound (default: 2.0)
        desloc_nesterov (bool): use Nesterov outer optimizer (default: False)
        desloc_nesterov_momentum (float): outer momentum (default: 0.9)
        desloc_log_interval (int): log frequency (default: 100)
    """

    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999),
                 eps=1e-8, eps_inside_sqrt=False, weight_decay=0., max_grad_norm=0.,
                 max_coeff=10.0, min_coeff=0.01, amsgrad=False,
                 desloc_kx=DESLOC_DEFAULT_KX, desloc_ku_ratio=DESLOC_DEFAULT_KU_RATIO,
                 desloc_kv_ratio=DESLOC_DEFAULT_KV_RATIO, desloc_clip_rho=DESLOC_DEFAULT_CLIP_RHO,
                 desloc_nesterov=False, desloc_nesterov_momentum=0.9, desloc_log_interval=100):
        self.fused_lamb_cuda = FusedLambBuilder().load()

        if amsgrad:
            raise RuntimeError('FusedLamb does not support the AMSGrad variant.')
        if desloc_kx < 0:
            raise ValueError(f"desloc_kx must be >= 0, got {desloc_kx}")

        defaults = dict(lr=lr, bias_correction=bias_correction, betas=betas, eps=eps,
                        weight_decay=weight_decay, max_grad_norm=max_grad_norm,
                        max_coeff=max_coeff, min_coeff=min_coeff)
        super(FusedLamb, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.lamb_coeffs = []

        # DES-LOC state
        self.desloc_enabled = desloc_kx > 0
        self.desloc_clip_rho = desloc_clip_rho
        self.desloc_log_interval = desloc_log_interval
        self._desloc_step = 0
        self.desloc_kx, self.desloc_ku, self.desloc_kv = compute_desloc_periods(
            betas[0], betas[1], desloc_kx)
        self._hl_beta1 = compute_ema_half_life(betas[0])
        self._hl_beta2 = compute_ema_half_life(betas[1])
        self._desloc_param_states = {}
        self._nesterov = DeslocNesterovOuter(desloc_nesterov_momentum) if (desloc_nesterov and self.desloc_enabled) else None
        self._comm_tracker = DeslocCommTracker()

        if self.desloc_enabled:
            logger.info(f"[DES-LOC LAMB] Kx={self.desloc_kx}, Ku={self.desloc_ku}, "
                        f"Kv={self.desloc_kv}, rho={self.desloc_clip_rho}, "
                        f"nesterov={desloc_nesterov}, hl1={self._hl_beta1:.1f}, hl2={self._hl_beta2:.1f}")

    def _get_desloc_state(self, param):
        pid = id(param)
        if pid not in self._desloc_param_states:
            self._desloc_param_states[pid] = DeslocLambState()
        return self._desloc_param_states[pid]

    def step(self, closure=None, grads=None, output_params=None, scale=1., grad_norms=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            output_params_group = [None] * len(self.param_groups)
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0]) != list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        if grad_norms is None:
            grad_norms = [None] * len(self.param_groups)

        del self.lamb_coeffs[:]

        for group, grads_this_group, output_params_this_group, grad_norm_group in zip(
                self.param_groups, grads_group, output_params_group, grad_norms):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
            if output_params_this_group is None:
                output_params_this_group = [None] * len(group['params'])

            if grad_norm_group is None:
                grad_norm_group = [None] * len(group['params'])
            elif not isinstance(grad_norm_group, list):
                grad_norm_group = [grad_norm_group]

            bias_correction = 1 if group['bias_correction'] else 0

            for p, grad, output_param, grad_norm in zip(group['params'], grads_this_group,
                                                        output_params_this_group, grad_norm_group):

                combined_scale = scale
                if group['max_grad_norm'] > 0:
                    clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                    if clip > 1:
                        combined_scale = clip * scale

                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('FusedLamb does not support sparse gradients')

                # DES-LOC: per-coordinate gradient clipping (Algorithm 1, line 12)
                if self.desloc_enabled:
                    grad = grad.clamp(min=-self.desloc_clip_rho, max=self.desloc_clip_rho)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                max_coeff = group['max_coeff']
                min_coeff = group['min_coeff']

                state['step'] += 1

                out_p = torch.tensor([], dtype=torch.float) if output_param is None else output_param
                lamb_coeff = self.fused_lamb_cuda.lamb(
                    p.data, out_p, exp_avg, exp_avg_sq, grad, group['lr'], beta1,
                    beta2, max_coeff, min_coeff, group['eps'], combined_scale,
                    state['step'], self.eps_mode, bias_correction, group['weight_decay'])
                self.lamb_coeffs.append(lamb_coeff)

                # DES-LOC: update sync flags and track trust ratio
                if self.desloc_enabled:
                    ds = self._get_desloc_state(p)
                    ds.check_sync_flags(self._desloc_step, self.desloc_kx, self.desloc_ku, self.desloc_kv)
                    gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm or 0.0)
                    ds.record_step(lamb_coeff, gn)
                    self._comm_tracker.record_trust_ratio(lamb_coeff)
                    if ds.param_sync_due:
                        self._comm_tracker.record_sync('param')
                    if ds.mom1_sync_due:
                        self._comm_tracker.record_sync('mom1')
                    if ds.mom2_sync_due:
                        self._comm_tracker.record_sync('mom2')

        if self.desloc_enabled:
            self._desloc_step += 1
            self._comm_tracker.advance()
            if self._desloc_step % self.desloc_log_interval == 0:
                logger.info(self._comm_tracker.format_log())

        return loss

    def get_lamb_coeffs(self):
        return [c.item() for c in self.lamb_coeffs]

    def get_desloc_sync_info(self):
        """Engine queries this to decide which AllReduce operations to issue."""
        if not self.desloc_enabled:
            return {'enabled': False}
        return {
            'enabled': True, 'kx': self.desloc_kx, 'ku': self.desloc_ku, 'kv': self.desloc_kv,
            'step': self._desloc_step,
            'half_life_beta1': round(self._hl_beta1, 2), 'half_life_beta2': round(self._hl_beta2, 2),
            'sync_param_now': should_sync_at_step(self._desloc_step, self.desloc_kx),
            'sync_mom1_now': should_sync_at_step(self._desloc_step, self.desloc_ku),
            'sync_mom2_now': should_sync_at_step(self._desloc_step, self.desloc_kv),
            'nesterov_enabled': self._nesterov is not None,
            'comm_reduction': self._comm_tracker.comm_reduction_vs_ddp,
        }

    def get_nesterov_outer(self):
        return self._nesterov

    def desloc_state_dict(self):
        d = {'step': self._desloc_step, 'kx': self.desloc_kx, 'ku': self.desloc_ku,
             'kv': self.desloc_kv, 'clip_rho': self.desloc_clip_rho}
        if self._nesterov is not None:
            d['nesterov'] = self._nesterov.state_dict()
        return d

    def load_desloc_state_dict(self, d):
        self._desloc_step = d.get('step', 0)
        if self._nesterov is not None and 'nesterov' in d:
            self._nesterov.load_state_dict(d['nesterov'])
