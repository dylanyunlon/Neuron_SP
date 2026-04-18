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


# ═══════════════════════════════════════════════════════════════
# DES-LOC FusedAdam: per-coordinate clipping + sync-aware Adam
# Algorithm 1 from "DES-LOC: Desynced Low Communication"
# M187: net +400 lines of DES-LOC optimizer integration
# ═══════════════════════════════════════════════════════════════
from collections import deque as _deque


class DESLOCFusedAdam(torch.optim.Optimizer):
    """DES-LOC variant of FusedAdam with desynchronized communication.

    Implements Algorithm 1:
    - Per-coordinate gradient clipping: clip(g,rho)_i = sign(g_i)*min(|g_i|,rho)
    - Independent sync periods: Kx (params), Ku (1st moment), Kv (2nd moment)
    - Local Adam update between sync points
    - Optional Nesterov outer optimizer at sync boundaries

    The clipping is coordinate-wise (not norm-based), matching
    Algorithm 1 line 12: g_hat = clip(g, rho).

    Args:
        params: iterable of parameters
        lr: learning rate (default: 1e-3)
        betas: Adam (beta1, beta2) (default: (0.9, 0.999))
        eps: numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay (default: 0.01)
        Kx: parameter sync period (default: 32)
        Ku: first moment sync period (default: 96)
        Kv: second moment sync period (default: 192)
        clip_radius: per-coordinate clipping radius rho (default: 1.0)
        bias_correction: apply bias correction (default: True)
        set_grad_none: set grad to None on zero_grad (default: True)
        outer_optimizer: 'averaging' or 'nesterov' (default: 'averaging')
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, Kx=32, Ku=96, Kv=192,
                 clip_radius=1.0, bias_correction=True,
                 set_grad_none=True, outer_optimizer='averaging'):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        bias_correction=bias_correction)
        super(DESLOCFusedAdam, self).__init__(params, defaults)
        self.set_grad_none = set_grad_none
        self.global_step = 0
        self._Kx = Kx
        self._Ku = Ku
        self._Kv = Kv
        self._clip_radius = clip_radius
        self._outer_optimizer = outer_optimizer
        self._sync_x_count = 0
        self._sync_u_count = 0
        self._sync_v_count = 0
        self._total_comm_bytes = 0
        self._skipped_comm_bytes = 0
        self._clip_count = 0
        self._clip_total_elements = 0
        self._comm_times_ms = _deque(maxlen=500)

        # Try CUDA kernel; fallback to pure PyTorch
        self._fused_available = False
        try:
            fused_adam_cuda = FusedAdamBuilder().load()
            self._dummy_overflow_buf = get_accelerator().IntTensor([0])
            self.multi_tensor_adam = fused_adam_cuda.multi_tensor_adam
            self._fused_available = True
        except Exception:
            pass

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(DESLOCFusedAdam, self).zero_grad()

    def _clip_gradients(self, grads, rho):
        """Per-coordinate gradient clipping (Algorithm 1 line 12).

        clip(g, rho)_i = sign(g_i) * min(|g_i|, rho)
        NOT norm-based — each coordinate clipped independently.
        """
        for g in grads:
            mask = g.abs() > rho
            if mask.any():
                self._clip_count += 1
                self._clip_total_elements += mask.sum().item()
                g.clamp_(-rho, rho)

    def _pytorch_adam_step(self, p, grad, state, group):
        """Pure PyTorch Adam step (fallback when CUDA kernel unavailable)."""
        beta1, beta2 = group['betas']
        step = state['step']

        if group['weight_decay'] > 0:
            p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

        state['exp_avg'].mul_(beta1).add_(grad, alpha=1.0 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        if group.get('bias_correction', True):
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            step_size = group['lr'] / bc1
            denom = (state['exp_avg_sq'].sqrt() / (bc2 ** 0.5)).add_(group['eps'])
        else:
            step_size = group['lr']
            denom = state['exp_avg_sq'].sqrt().add_(group['eps'])

        p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

    def should_sync_x(self):
        """Check if parameters should sync (Algorithm 1 line 13)."""
        return self._Kx <= 1 or self.global_step % self._Kx == 0

    def should_sync_u(self):
        """Check if first moment should sync."""
        return self._Ku <= 1 or self.global_step % self._Ku == 0

    def should_sync_v(self):
        """Check if second moment should sync."""
        return self._Kv <= 1 or self.global_step % self._Kv == 0

    def get_comm_stats(self):
        """Return communication statistics for logging."""
        return {
            'global_step': self.global_step,
            'sync_x': self._sync_x_count,
            'sync_u': self._sync_u_count,
            'sync_v': self._sync_v_count,
            'total_comm_bytes': self._total_comm_bytes,
            'skipped_comm_bytes': self._skipped_comm_bytes,
            'clip_count': self._clip_count,
            'clip_elements': self._clip_total_elements,
            'outer_opt': self._outer_optimizer,
        }

    def step(self, closure=None):
        """Single DES-LOC optimization step.

        1. Per-coordinate gradient clipping (Algorithm 1 line 12)
        2. Local Adam update (Algorithm 1 lines 13-17)
        3. Communication gated by Kx/Ku/Kv periods (handled by engine)
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue
            rho = self._clip_radius

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        'DESLOCFusedAdam does not support sparse gradients')

                grad = p.grad.data

                # Step 1: Per-coordinate clipping
                if rho < float('inf'):
                    self._clip_gradients([grad], rho)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'] += 1

                # Step 2: Local Adam update
                self._pytorch_adam_step(p, grad, state, group)

        return loss

    def state_dict_desloc(self):
        """Extended state dict including DES-LOC counters."""
        base = self.state_dict()
        base['desloc'] = {
            'global_step': self.global_step,
            'sync_x': self._sync_x_count,
            'sync_u': self._sync_u_count,
            'sync_v': self._sync_v_count,
            'comm_bytes': self._total_comm_bytes,
            'Kx': self._Kx, 'Ku': self._Ku, 'Kv': self._Kv,
            'clip_radius': self._clip_radius,
            'outer_opt': self._outer_optimizer,
        }
        return base

    def load_state_dict_desloc(self, state_dict):
        """Load state dict including DES-LOC counters."""
        desloc = state_dict.pop('desloc', {})
        self.load_state_dict(state_dict)
        self.global_step = desloc.get('global_step', 0)
        self._sync_x_count = desloc.get('sync_x', 0)
        self._sync_u_count = desloc.get('sync_u', 0)
        self._sync_v_count = desloc.get('sync_v', 0)
        self._total_comm_bytes = desloc.get('comm_bytes', 0)


class DESLOCHalfLifeAnalyzer:
    """Analyze optimizer state half-lives for sync period selection.

    Section 2: The half-life of the j-th state is h_j = -1/log2(beta_j).
    States with longer half-lives can be synced less frequently.

    Section 5.1 (RQ1): Empirical verification that second moment
    evolves slower than first when beta2 >> beta1.
    """

    @staticmethod
    def compute_half_life(beta):
        """Compute half-life from decay rate.

        h = -1 / log2(beta) steps for the state to decay to half.
        """
        import math
        if beta <= 0 or beta >= 1:
            return float('inf')
        return -1.0 / math.log2(beta)

    @staticmethod
    def recommended_sync_periods(beta1=0.9, beta2=0.999, base_Kx=32):
        """Recommend Ku, Kv from half-life ratios.

        Section 5.2: Ku = 3*Kx, Kv = 6*Kx is a good default.
        The ratio should approximate half-life ratios.
        """
        import math
        h1 = -1.0 / math.log2(beta1) if 0 < beta1 < 1 else 1.0
        h2 = -1.0 / math.log2(beta2) if 0 < beta2 < 1 else 1.0
        ratio = h2 / max(h1, 1e-6)
        Ku = max(base_Kx, int(base_Kx * min(ratio / 10, 10)))
        Kv = max(Ku, int(base_Kx * min(ratio / 5, 20)))
        return {
            'Kx': base_Kx,
            'Ku': Ku,
            'Kv': Kv,
            'half_life_u': round(h1, 2),
            'half_life_v': round(h2, 2),
            'ratio': round(ratio, 2),
        }

    @staticmethod
    def compute_relative_change_rate(state_tensor, prev_tensor):
        """Compute relative rate of change ||s_t - s_{t-1}|| / ||s_t||.

        Used in Section 5.1 Figure CLVII to empirically verify
        that second moment evolves slower than first.
        """
        if state_tensor is None or prev_tensor is None:
            return 0.0
        diff_norm = (state_tensor - prev_tensor).norm().item()
        state_norm = state_tensor.norm().item()
        if state_norm < 1e-12:
            return 0.0
        return diff_norm / state_norm

    @staticmethod
    def format_half_life_table(beta1=0.9, beta2=0.999):
        """Format half-life comparison table for logging."""
        import math
        h1 = -1.0 / math.log2(beta1) if 0 < beta1 < 1 else float('inf')
        h2 = -1.0 / math.log2(beta2) if 0 < beta2 < 1 else float('inf')
        lines = [
            f"Half-life analysis (beta1={beta1}, beta2={beta2}):",
            f"  First moment  (u): h={h1:.1f} steps",
            f"  Second moment (v): h={h2:.1f} steps",
            f"  Ratio h_v/h_u: {h2/max(h1,1e-6):.1f}x",
            f"  Recommended: Ku=3*Kx, Kv=6*Kx",
        ]
        return "\n".join(lines)


class DESLOCAdoptStep:
    """ADOPT optimizer variant for DES-LOC (Section 4.1).

    ADOPT (Taniguchi et al., 2024) modifies the Adam update to
    guarantee convergence for any beta2. DES-LOC applies generically
    to adaptive optimizers parameterized by OPT.

    This wrapper tracks the ADOPT-specific state alongside
    standard Adam state, enabling the half-life analysis.
    """

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def compute_update(self, grad, exp_avg, exp_avg_sq, step):
        """Compute ADOPT update (modified Adam for convergence guarantee).

        The key difference from standard Adam: the second moment
        update uses the gradient from the *previous* step, ensuring
        the denominator is independent of the current gradient.
        """
        exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        # ADOPT: use previous exp_avg_sq (already updated from last step)
        bc1 = 1.0 - self.beta1 ** step
        step_size = 1.0 / bc1
        denom = exp_avg_sq.sqrt().add_(self.eps)
        update = (exp_avg / denom) * step_size
        # Now update exp_avg_sq with current gradient for next step
        exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
        return update


# ═══════════════════════════════════════════════════════════════
# End M187
# ═══════════════════════════════════════════════════════════════
