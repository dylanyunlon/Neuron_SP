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


# =====================================================================
# M054: DES-LOC FusedAdam (400 lines)
# =====================================================================
# Extends FusedAdam with DES-LOC per-coordinate clipping and
# independent optimizer state synchronization schedule.
# Architecture reference: CCCL cub/cub/agent/agent_reduce.cuh
#
# This is the production kernel-fused DES-LOC optimizer.
# Per-coordinate clipping: clip(g, rho)_i = sign(g_i)*min(|g_i|, rho)
# Reference: des_loc_reconstructed.tex line 168, Algorithm 1
# =====================================================================

import math
import os
import time as _time
from collections import deque as _deque


class DESLOCFusedAdam(torch.optim.Optimizer):
    """
    DES-LOC Fused Adam — CUDA-fused AdamW with desynchronized communication.

    Implements the full DES-LOC Algorithm 1 from the paper:
    - Per-coordinate gradient clipping: clip(g, rho)
    - Independent sync periods: Kx (params), Ku (first moment), Kv (second moment)
    - Local Adam update between sync points

    Uses the same CUDA kernel as FusedAdam for the Adam update,
    but wraps it with DES-LOC's communication schedule.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for running averages (default: (0.9, 0.999))
        eps: term for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay (default: 0.01)
        Kx: parameter sync period (default: 32)
        Ku: first moment sync period (default: 96)
        Kv: second moment sync period (default: 192)
        clip_radius: per-coordinate clipping radius rho (default: 1.0)
        bias_correction: apply bias correction (default: True)
        set_grad_none: set grad to None on zero_grad (default: True)
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.01,
                 Kx=32,
                 Ku=96,
                 Kv=192,
                 clip_radius=1.0,
                 bias_correction=True,
                 set_grad_none=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay,
                         bias_correction=bias_correction,
                         Kx=Kx, Ku=Ku, Kv=Kv,
                         clip_radius=clip_radius)
        super(DESLOCFusedAdam, self).__init__(params, defaults)
        self.set_grad_none = set_grad_none
        self.global_step = 0
        self._Kx = Kx
        self._Ku = Ku
        self._Kv = Kv
        self._clip_radius = clip_radius

        # Communication tracking
        self._sync_x_count = 0
        self._sync_u_count = 0
        self._sync_v_count = 0
        self._total_comm_bytes = 0
        self._skipped_comm_bytes = 0
        self._clip_count = 0
        self._clip_total_elements = 0
        self._comm_times_ms = _deque(maxlen=500)

        # Try to load CUDA kernel; fallback to pure PyTorch
        self._fused_available = False
        try:
            fused_adam_cuda = FusedAdamBuilder().load()
            self._dummy_overflow_buf = get_accelerator().IntTensor([0])
            self.multi_tensor_adam = fused_adam_cuda.multi_tensor_adam
            self._fused_available = True
        except Exception:
            pass  # Will use pure-torch fallback

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(DESLOCFusedAdam, self).zero_grad()

    def _clip_gradients(self, grads, rho):
        """Apply per-coordinate gradient clipping.

        clip(g, rho)_i = sign(g_i) * min(|g_i|, rho)

        This is NOT norm-based clipping — it clips each coordinate
        independently, matching Algorithm 1 line 168.
        """
        for g in grads:
            mask = g.abs() > rho
            if mask.any():
                self._clip_count += 1
                self._clip_total_elements += mask.sum().item()
                g.clamp_(-rho, rho)

    def _pytorch_adam_step(self, p, grad, state, group):
        """Pure PyTorch Adam update (fallback when CUDA kernel unavailable)."""
        beta1, beta2 = group['betas']
        eps = group['eps']

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        state['step'] += 1

        # Decoupled weight decay
        if group['weight_decay'] != 0:
            p.data.mul_(1 - group['lr'] * group['weight_decay'])

        # Adam update
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if group.get('bias_correction', True):
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            step_size = group['lr']
            denom = exp_avg_sq.sqrt().add_(eps)

        p.data.addcdiv_(exp_avg, denom, value=-step_size)

    def step(self, closure=None):
        """Performs a single DES-LOC optimization step.

        1. Apply per-coordinate clipping to gradients
        2. Perform local Adam update
        3. Communication is handled separately by sync_if_needed()
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            if len(group['params']) == 0:
                continue

            rho = group.get('clip_radius', self._clip_radius)
            bias_correction = 1 if group.get('bias_correction', True) else 0
            beta1, beta2 = group['betas']

            # Collect gradients for clipping
            all_grads = []
            for p in group['params']:
                if p.grad is not None and not p.grad.data.is_sparse:
                    all_grads.append(p.grad.data)

            # DES-LOC per-coordinate clipping (Algorithm 1 line 168)
            if rho > 0:
                self._clip_gradients(all_grads, rho)

            if self._fused_available:
                # Use CUDA kernel path
                g_16, p_16, m_16, v_16 = [], [], [], []
                g_bf, p_bf, m_bf, v_bf = [], [], [], []
                g_32, p_32, m_32, v_32 = [], [], [], []

                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
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

                step_val = self.global_step

                if len(g_16) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam, self._dummy_overflow_buf,
                        [g_16, p_16, m_16, v_16],
                        group['lr'], beta1, beta2, group['eps'],
                        step_val, 1, bias_correction, group['weight_decay'])

                if len(g_bf) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam, self._dummy_overflow_buf,
                        [g_bf, p_bf, m_bf, v_bf],
                        group['lr'], beta1, beta2, group['eps'],
                        step_val, 1, bias_correction, group['weight_decay'])

                if len(g_32) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_adam, self._dummy_overflow_buf,
                        [g_32, p_32, m_32, v_32],
                        group['lr'], beta1, beta2, group['eps'],
                        step_val, 1, bias_correction, group['weight_decay'])
            else:
                # Pure PyTorch fallback
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    self._pytorch_adam_step(p, p.grad.data, state, group)

        return loss

    def sync_if_needed(self, world_size):
        """Sync optimizer states based on DES-LOC schedule.

        Called AFTER step() to average parameters and optimizer states
        across workers at the schedule-defined intervals.

        Algorithm 1 lines 170-180:
        - if t mod Kj == 0: sync state j (allreduce + average)
        """
        if world_size <= 1:
            return {'sync_x': False, 'sync_u': False, 'sync_v': False}

        sync_x = (self.global_step % self._Kx == 0)
        sync_u = (self.global_step % self._Ku == 0)
        sync_v = (self.global_step % self._Kv == 0)

        if not sync_x and not sync_u and not sync_v:
            return {'sync_x': False, 'sync_u': False, 'sync_v': False}

        try:
            import torch.distributed as torch_dist
        except ImportError:
            return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}

        t0 = _time.time()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                bytes_per_elem = p.data.element_size()
                numel = p.data.numel()

                # Sync parameters (x)
                if sync_x:
                    torch_dist.all_reduce(p.data, op=torch_dist.ReduceOp.AVG)
                    self._total_comm_bytes += numel * bytes_per_elem * 2

                # Sync first moment (u / exp_avg)
                if sync_u and 'exp_avg' in state:
                    torch_dist.all_reduce(state['exp_avg'],
                                          op=torch_dist.ReduceOp.AVG)
                    self._total_comm_bytes += numel * bytes_per_elem * 2

                # Sync second moment (v / exp_avg_sq)
                if sync_v and 'exp_avg_sq' in state:
                    torch_dist.all_reduce(state['exp_avg_sq'],
                                          op=torch_dist.ReduceOp.AVG)
                    self._total_comm_bytes += numel * bytes_per_elem * 2

        elapsed_ms = (_time.time() - t0) * 1000
        self._comm_times_ms.append(elapsed_ms)

        if sync_x:
            self._sync_x_count += 1
        if sync_u:
            self._sync_u_count += 1
        if sync_v:
            self._sync_v_count += 1

        return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}

    def get_comm_stats(self):
        """Return DES-LOC communication statistics."""
        return {
            'global_step': self.global_step,
            'Kx': self._Kx,
            'Ku': self._Ku,
            'Kv': self._Kv,
            'clip_radius': self._clip_radius,
            'sync_x_count': self._sync_x_count,
            'sync_u_count': self._sync_u_count,
            'sync_v_count': self._sync_v_count,
            'total_comm_bytes': self._total_comm_bytes,
            'clip_events': self._clip_count,
            'clipped_elements': self._clip_total_elements,
            'avg_comm_time_ms': (
                sum(self._comm_times_ms) / len(self._comm_times_ms)
                if self._comm_times_ms else 0),
            'fused_kernel': self._fused_available,
        }
