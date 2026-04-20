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

class DeslocAdamConfig:
    """Config for DES-LOC-aware Adam.
    1. Per-coordinate clipping: |g_i| <= rho (not global norm)
    2. Independent state sync: m1 at Ku, m2 at Kv
    Ref: Section 2 — ADOPT removes beta1 < sqrt(beta2) constraint."""

    def __init__(self, clip_rho=1.0, use_adopt=False):
        self.clip_rho = clip_rho
        self.use_adopt = use_adopt
        self.clip_count = 0
        self.total_count = 0

    def per_coordinate_clip(self, grad):
        """Clamp each element: [-rho, rho]. Ref: Algorithm 1 line 12."""
        if self.clip_rho <= 0:
            return grad
        self.total_count += grad.numel()
        mask = grad.abs() > self.clip_rho
        self.clip_count += mask.sum().item()
        return grad.clamp(-self.clip_rho, self.clip_rho)

    @property
    def clip_ratio(self):
        return self.clip_count / max(1, self.total_count)

    def summary(self):
        return {'rho': self.clip_rho, 'adopt': self.use_adopt,
                'clip_ratio': round(self.clip_ratio, 6)}


class DeslocAdoptStep:
    """ADOPT optimizer step. Ref: Section 4.1.
    Key difference from Adam: v_t uses g_{t-1} (not g_t).
    This guarantees convergence for any beta2."""

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._prev_grads = {}

    def step(self, name, param, grad, state, lr):
        if 'exp_avg' not in state:
            state['exp_avg'] = param.data.new_zeros(param.data.shape)
            state['exp_avg_sq'] = param.data.new_zeros(param.data.shape)
        m = state['exp_avg']
        v = state['exp_avg_sq']
        m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        prev = self._prev_grads.get(name)
        if prev is not None:
            v.mul_(self.beta2).addcmul_(prev, prev, value=1 - self.beta2)
        self._prev_grads[name] = grad.clone()
        denom = v.sqrt().add_(self.eps)
        param.data.addcdiv_(m, denom, value=-lr)


class DeslocMomentumStateTracker:
    """Track optimizer state staleness for DES-LOC.
    Between sync points, each worker's m1/m2 diverge from global average.
    This tracker measures divergence magnitude.

    Ref: Theorem 1 — convergence bound includes psi term that captures
    the cost of stale momentum states."""

    def __init__(self):
        self.local_updates_since_sync = {'m1': 0, 'm2': 0}
        self.max_staleness = {'m1': 0, 'm2': 0}

    def record_local_update(self, state_name='m1'):
        self.local_updates_since_sync[state_name] = \
            self.local_updates_since_sync.get(state_name, 0) + 1
        self.max_staleness[state_name] = max(
            self.max_staleness.get(state_name, 0),
            self.local_updates_since_sync[state_name])

    def record_sync(self, state_name='m1'):
        self.local_updates_since_sync[state_name] = 0

    def get_staleness(self):
        return {
            'current': dict(self.local_updates_since_sync),
            'max_ever': dict(self.max_staleness),
        }

    def reset(self):
        self.local_updates_since_sync = {'m1': 0, 'm2': 0}
        self.max_staleness = {'m1': 0, 'm2': 0}


class DeslocGradientAccumulator:
    """Accumulate gradients between Kx sync boundaries.

    Between sync points, gradients are accumulated locally.
    At Kx boundaries, accumulated gradients are allreduced and applied.

    This is the gradient-level view of Algorithm 1:
    - Non-sync step: g_local += grad; apply local Adam step
    - Sync step: g_avg = allreduce(params); apply averaging
    """

    def __init__(self, Kx=1):
        self.Kx = max(1, Kx)
        self.accumulated_steps = 0

    def should_flush(self):
        return self.accumulated_steps >= self.Kx

    def accumulate(self):
        self.accumulated_steps += 1
        return self.should_flush()

    def flush(self):
        steps = self.accumulated_steps
        self.accumulated_steps = 0
        return steps

    def state_dict(self):
        return {'Kx': self.Kx, 'acc': self.accumulated_steps}


class DeslocMomentumStateTracker:
    """Track optimizer state staleness for DES-LOC.
    Between sync points, each worker's m1/m2 diverge from global average.
    This measures divergence for convergence monitoring.
    Ref: Theorem 1 — psi captures cost of stale momentum."""

    def __init__(self):
        self.local_updates = {'m1': 0, 'm2': 0}
        self.max_staleness = {'m1': 0, 'm2': 0}

    def record_local_update(self, state='m1'):
        self.local_updates[state] = self.local_updates.get(state, 0) + 1
        self.max_staleness[state] = max(self.max_staleness.get(state, 0),
                                        self.local_updates[state])

    def record_sync(self, state='m1'):
        self.local_updates[state] = 0

    def get_staleness(self):
        return {'current': dict(self.local_updates), 'max': dict(self.max_staleness)}

    def reset(self):
        self.local_updates = {'m1': 0, 'm2': 0}
        self.max_staleness = {'m1': 0, 'm2': 0}


class DeslocGradientAccumulator:
    """Accumulate gradients between Kx sync boundaries.
    Non-sync step: accumulate locally + local Adam step.
    Sync step: allreduce + average outer optimizer step.
    Ref: Algorithm 1 — local optimization between sync points."""

    def __init__(self, Kx=1):
        self.Kx = max(1, Kx)
        self.accumulated = 0

    def should_flush(self):
        return self.accumulated >= self.Kx

    def accumulate(self):
        self.accumulated += 1
        return self.should_flush()

    def flush(self):
        n = self.accumulated
        self.accumulated = 0
        return n

    def state_dict(self):
        return {'Kx': self.Kx, 'acc': self.accumulated}


class DeslocAdamHyperparamScaler:
    """Scale Adam hyperparameters with DES-LOC Kx.
    Ref: Theorem 1 — learning rate should scale as 1/sqrt(psi).
    Larger Kx → larger psi → smaller learning rate for convergence.
    Also: beta1 effective change due to local updates between syncs."""

    def __init__(self, base_lr, base_beta1=0.9, base_beta2=0.999):
        self.base_lr = base_lr
        self.base_beta1 = base_beta1
        self.base_beta2 = base_beta2

    def scale_lr(self, Kx, Ku=None):
        """Scale learning rate for given Kx."""
        import math
        if Kx <= 1:
            return self.base_lr
        ku = Ku if Ku else max(1, Kx * 3)
        px = 1.0 / Kx
        pu = 1.0 / ku
        num = 4 * (1 - px) * (1 - self.base_beta1) * (1 - pu)
        den = px * px * 6 * (1 - (1 - pu) * self.base_beta1)
        psi = num / den if abs(den) > 1e-15 else 1
        scale = 1.0 / math.sqrt(max(1, psi / 10))
        return round(self.base_lr * scale, 8)

    def get_lr_table(self, kx_values=None):
        if kx_values is None:
            kx_values = [1, 4, 8, 16, 32, 64, 128]
        return {kx: self.scale_lr(kx) for kx in kx_values}

    def effective_beta1(self, Kx):
        """Effective beta1 accounting for local updates between syncs.
        After Kx local updates, the effective momentum decay is beta1^Kx."""
        return round(self.base_beta1 ** Kx, 6)

    def summary(self):
        return {
            'base_lr': self.base_lr,
            'base_beta1': self.base_beta1,
            'base_beta2': self.base_beta2,
            'lr_table': self.get_lr_table(),
        }


# =============================================================================
# M234 (Claude-15): Adam vs ADOPT loss comparison data for Figure 6
# Ref: Section 5.6 — DES-LOC with Adam vs ADOPT vs Muon
# Ref: Algorithm 2 — ADOPT modifies update to guarantee convergence for any β2
# Ref: NKI-FA da964f3 — structured comparison data
# =============================================================================


class DeslocOptimizerComparisonLogger:
    """Log optimizer comparison data for Figure 6.

    Tracks per-step metrics for Adam, ADOPT, and Muon variants
    under the same DES-LOC configuration, enabling side-by-side
    loss curve comparison.

    From Section 5.6: DES-LOC is inner-optimizer-agnostic.
    ADOPT guarantees convergence for any β2, while Adam requires β1 < √β2.
    Muon uses SVD preconditioning instead of second moments.

    From NKI-FA: each optimizer variant produces its own log block
    with '### optimizer = adam, Kx = 32 ###' header.

    Usage:
        logger = DeslocOptimizerComparisonLogger()
        logger.begin_experiment('adam', Kx=32, Ku=96, Kv=192)
        for step in training:
            logger.record(step, loss=loss.item(), grad_norm=gn)
        logger.end_experiment()
    """

    def __init__(self):
        self._experiments = {}
        self._current_key = None
        self._current_data = None

    def begin_experiment(self, optimizer_name, Kx=32, Ku=96, Kv=192,
                         model_size='125M', seed=42):
        """Start recording for one optimizer variant.

        Args:
            optimizer_name: 'adam', 'adopt', 'muon'
            Kx, Ku, Kv: DES-LOC sync periods
            model_size: model parameter count label
            seed: random seed
        """
        key = (optimizer_name, Kx, model_size, seed)
        self._current_key = key
        self._current_data = {
            'optimizer': optimizer_name,
            'Kx': Kx, 'Ku': Ku, 'Kv': Kv,
            'model_size': model_size, 'seed': seed,
            'steps': [], 'losses': [], 'grad_norms': [],
            'lr_values': [], 'step_times_ms': [],
        }

    def record(self, step, loss=None, grad_norm=None, lr=None,
               step_time_ms=None):
        """Record one training step."""
        if self._current_data is None:
            return
        self._current_data['steps'].append(step)
        if loss is not None:
            self._current_data['losses'].append(float(loss))
        if grad_norm is not None:
            self._current_data['grad_norms'].append(float(grad_norm))
        if lr is not None:
            self._current_data['lr_values'].append(float(lr))
        if step_time_ms is not None:
            self._current_data['step_times_ms'].append(float(step_time_ms))

    def end_experiment(self):
        """Finalize current experiment and store."""
        if self._current_key is not None and self._current_data is not None:
            self._experiments[self._current_key] = self._current_data
        self._current_key = None
        self._current_data = None

    def get_figure6_data(self):
        """Extract Figure 6 comparison data.

        Returns: dict {optimizer_label: {
            'final_loss': float,
            'min_loss': float,
            'convergence_step': int,  # step where loss < threshold
            'loss_curve': [(step, loss)],
            'avg_step_ms': float,
        }}
        """
        result = {}
        for key, data in self._experiments.items():
            opt_name = data['optimizer']
            kx = data['Kx']
            label = f'{opt_name.upper()} (Kx={kx})'
            losses = data['losses']
            if not losses:
                continue
            # Find convergence step (loss within 5% of minimum)
            min_loss = min(losses)
            threshold = min_loss * 1.05
            conv_step = len(losses)
            for i, l in enumerate(losses):
                if l <= threshold:
                    conv_step = data['steps'][i] if i < len(data['steps']) else i
                    break
            avg_ms = 0.0
            if data['step_times_ms']:
                avg_ms = sum(data['step_times_ms']) / len(data['step_times_ms'])
            result[label] = {
                'final_loss': round(losses[-1], 6),
                'min_loss': round(min_loss, 6),
                'convergence_step': conv_step,
                'loss_curve': list(zip(data['steps'][:len(losses)], losses)),
                'avg_step_ms': round(avg_ms, 3),
                'n_steps': len(losses),
                'optimizer': opt_name,
                'Kx': kx,
            }
        return result

    def emit_nkifa_log(self, output_path=None):
        """Write all experiments in NKI-FA format.

        Each experiment block:
            ### optimizer = adam, Kx = 32, model = 125M ###
            step: 0 | loss: 10.234567
            ...
        """
        blocks = []
        for key, data in sorted(self._experiments.items()):
            lines = [f'### optimizer = {data["optimizer"]}, '
                     f'Kx = {data["Kx"]}, Ku = {data["Ku"]}, '
                     f'Kv = {data["Kv"]}, '
                     f'model = {data["model_size"]} ###']
            for i, step in enumerate(data['steps']):
                parts = [f'step: {step}']
                if i < len(data['losses']):
                    parts.append(f'loss: {data["losses"][i]:.6f}')
                if i < len(data['grad_norms']):
                    parts.append(f'grad_norm: {data["grad_norms"][i]:.6f}')
                lines.append(' | '.join(parts))
            # Summary
            if data['losses']:
                lines.append('')
                lines.append('--- summary ---')
                lines.append(f'final_loss: {data["losses"][-1]:.6f}')
                lines.append(f'min_loss: {min(data["losses"]):.6f}')
                lines.append(f'total_steps: {len(data["losses"])}')
                lines.append('--- end summary ---')
            blocks.append('\n'.join(lines))
        content = '\n\n'.join(blocks)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        return content

    def get_comparison_table(self, fmt='md'):
        """Generate comparison table across optimizers.

        Ref: NKI-FA — exact value annotations.
        """
        fig6 = self.get_figure6_data()
        if fmt == 'latex':
            lines = [r'\begin{tabular}{lccccc}', r'\toprule',
                     r'Optimizer & $K_x$ & Final Loss & Min Loss & Conv Step & Step (ms) \\',
                     r'\midrule']
            for label, d in sorted(fig6.items()):
                lines.append(f'{d["optimizer"]} & {d["Kx"]} & '
                             f'{d["final_loss"]:.6f} & {d["min_loss"]:.6f} & '
                             f'{d["convergence_step"]} & {d["avg_step_ms"]:.3f} \\\\')
            lines.extend([r'\bottomrule', r'\end{tabular}'])
        else:
            lines = ['| Optimizer | Kx | Final Loss | Min Loss | Conv Step | Step (ms) |',
                     '|---|---|---|---|---|---|']
            for label, d in sorted(fig6.items()):
                lines.append(f'| {d["optimizer"]} | {d["Kx"]} | '
                             f'{d["final_loss"]:.6f} | {d["min_loss"]:.6f} | '
                             f'{d["convergence_step"]} | {d["avg_step_ms"]:.3f} |')
        return '\n'.join(lines)
