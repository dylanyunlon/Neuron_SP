# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from cpuinfo import get_cpu_info
from deepspeed.utils import logger
from deepspeed.utils.logging import should_log_le
from deepspeed.ops.op_builder import CPUAdamBuilder


class DeepSpeedCPUAdam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True,
                 fp32_optimizer_states=True):
        """Fast vectorized implementation of two variations of Adam optimizer on CPU:

        * Adam: A Method for Stochastic Optimization: (https://arxiv.org/abs/1412.6980);
        * AdamW: Fixing Weight Decay Regularization in Adam (https://arxiv.org/abs/1711.05101)

        DeepSpeed CPU Adam(W) provides between 5x to 7x speedup over torch.optim.adam(W).
        In order to apply this optimizer, the model requires to have its master parameter (in FP32)
        reside on the CPU memory.

        To train on a heterogeneous system, such as coordinating CPU and GPU, DeepSpeed offers
        the ZeRO-Offload technology which efficiently offloads the optimizer states into CPU memory,
        with minimal impact on training throughput. DeepSpeedCPUAdam plays an important role to minimize
        the overhead of the optimizer's latency on CPU. Please refer to ZeRO-Offload tutorial
        (https://www.deepspeed.ai/tutorials/zero-offload/) for more information on how to enable this technology.

        For calling step function, there are two options available: (1) update optimizer's states and (2) update
        optimizer's states and copy the parameters back to GPU at the same time. We have seen that the second
        option can bring 30% higher throughput than the doing the copy separately using option one.


        .. note::
                We recommend using our `config
                <https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`_
                to allow :meth:`deepspeed.initialize` to build this optimizer
                for you.


        Arguments:
            model_params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups.
            lr (float, optional): learning rate. (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square. (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability. (default: 1e-8)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                algorithm from the paper `On the Convergence of Adam and Beyond`_
                (default: False) NOT SUPPORTED in DeepSpeed CPUAdam!
            adamw_mode: select between Adam and AdamW implementations (default: AdamW)
            fp32_optimizer_states: creates momentum and variance in full precision regardless of
                        the precision of the parameters (default: True)
        """

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)
        super(DeepSpeedCPUAdam, self).__init__(model_params, default_args)

        cpu_info = get_cpu_info()
        self.cpu_vendor = cpu_info["vendor_id_raw"].lower() if "vendor_id_raw" in cpu_info else "unknown"
        if "amd" in self.cpu_vendor:
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group['params']):
                    if p.dtype == torch.half:
                        logger.warning("FP16 params for CPUAdam may not work on AMD CPUs")
                        break
                else:
                    continue
                break

        self.opt_id = DeepSpeedCPUAdam.optimizer_id
        DeepSpeedCPUAdam.optimizer_id = DeepSpeedCPUAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self.ds_opt_adam = CPUAdamBuilder().load()

        self.ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode,
                                     should_log_le("info"))

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        self.ds_opt_adam.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(DeepSpeedCPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Update the model parameters.

        .. note::
            This method will be called internally by ZeRO-Offload. DeepSpeed
            users should still use ``engine.step()`` as shown in the
            `Getting Started
            <https://www.deepspeed.ai/getting-started/#training>`_ guide.

        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # intended device for step
        device = torch.device('cpu')

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0

                    #use full precision by default unless self.fp32_optimizer_states is off
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    #memory_format=torch.preserve_format)

                state['step'] += 1
                beta1, beta2 = group['betas']

                self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                             group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                             state['exp_avg'], state['exp_avg_sq'])
        return loss

    @torch.no_grad()
    def step_subgroup(self, subgroup_id: int, closure=None):
        """Update the model parameters in a single subgroup (by index)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Intended device for step
        device = torch.device('cpu')

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue

                assert p.device == device, f"CPUAdam param is on {p.device} and must be 'cpu', make " \
                        "sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config."

                state = self.state[subgroup_id]

                if len(state) == 0:
                    state['step'] = 0

                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                    state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

                state['step'] += 1
                beta1, beta2 = group['betas']
                self.ds_opt_adam.adam_update(self.opt_id, state['step'], group['lr'], beta1, beta2, group['eps'],
                                             group['weight_decay'], group['bias_correction'], p.data, p.grad.data,
                                             state['exp_avg'], state['exp_avg_sq'])
        return loss

    @torch.no_grad()
    def rollback_subgroup(self, sub_group_id: int, closure=None):
        """
        Rollback the optimizer state for a specific subgroup.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Intended device for step
        device = torch.device('cpu')

        # Validate subgroup state exists and is initialized
        if sub_group_id not in self.state or len(self.state[sub_group_id]) == 0:
            raise RuntimeError(f"Cannot rollback optimizer state for sub_group_id {sub_group_id} "
                               f"as it has not been initialized.")

        subgroup_state = self.state[sub_group_id]

        # Check if we can rollback (step count must be > 0)
        if subgroup_state.get('step', 0) <= 0:
            raise RuntimeError(f"Cannot rollback sub_group_id {sub_group_id}: "
                               f"step count is {subgroup_state.get('step', 0)}")

        for _, group in enumerate(self.param_groups):
            for _, param in enumerate(group['params']):
                if param.grad is None:
                    continue

                assert param.device == device, (
                    f"CPUAdam param is on {param.device} and must be 'cpu', "
                    f"make sure you enabled 'offload_optimizer': 'cpu' in your ZeRO config.")

                beta1, beta2 = group['betas']

                self.ds_opt_adam.adam_rollback(self.opt_id, subgroup_state['step'], group['lr'], beta1, beta2,
                                               group['eps'], group['weight_decay'], group['bias_correction'],
                                               param.data, param.grad.data, subgroup_state['exp_avg'],
                                               subgroup_state['exp_avg_sq'])

                subgroup_state['step'] -= 1
        return loss


# =====================================================================
# M055: DES-LOC CPU Adam (400 lines)
# =====================================================================
# CPU-side DES-LOC AdamW for ZeRO-Offload integration.
# When optimizer states are offloaded to CPU, the DES-LOC sync
# schedule still applies — but sync happens via CPU↔GPU transfers
# rather than GPU↔GPU NCCL allreduce.
#
# Per-coordinate clipping: clip(g, rho) as in Algorithm 1 line 168
# Independent sync: Kx, Ku, Kv periods
# Architecture reference: CCCL cub/cub/device/dispatch/dispatch_reduce.cuh
# =====================================================================

import math
import time as _time
from collections import deque as _deque


class DESLOCCPUAdam(torch.optim.Optimizer):
    """
    DES-LOC CPU Adam — for ZeRO-Offload with desynchronized communication.

    When ZeRO-Offload places optimizer states on CPU, DES-LOC's Kx/Ku/Kv
    sync schedule controls when those states are gathered across workers.
    Between sync points, each worker runs Adam locally on CPU.

    Per-coordinate clipping matches Algorithm 1 exactly.

    Args:
        model_params: iterable of parameters
        lr: learning rate (default: 1e-3)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay (default: 0.01)
        Kx: parameter sync period (default: 32)
        Ku: first moment sync period (default: 96)
        Kv: second moment sync period (default: 192)
        clip_radius: per-coordinate clipping radius (default: 1.0)
        adamw_mode: use AdamW (True) or Adam (False) (default: True)
        fp32_optimizer_states: full precision states (default: True)
    """

    optimizer_id = 10000  # offset from DeepSpeedCPUAdam

    def __init__(self,
                 model_params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0.01,
                 Kx=32,
                 Ku=96,
                 Kv=192,
                 clip_radius=1.0,
                 bias_correction=True,
                 adamw_mode=True,
                 fp32_optimizer_states=True):
        default_args = dict(lr=lr, betas=betas, eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction)
        super(DESLOCCPUAdam, self).__init__(model_params, default_args)

        self.adamw_mode = adamw_mode
        self.fp32_optimizer_states = fp32_optimizer_states
        self._Kx = Kx
        self._Ku = Ku
        self._Kv = Kv
        self._clip_radius = clip_radius
        self.global_step = 0

        # Communication tracking
        self._sync_x_count = 0
        self._sync_u_count = 0
        self._sync_v_count = 0
        self._total_comm_bytes = 0
        self._clip_count = 0
        self._clip_elements = 0
        self._comm_times_ms = _deque(maxlen=500)
        self._step_times_ms = _deque(maxlen=500)

        # Try to use C++ kernel; fallback to pure Python
        self._cpp_available = False
        self.opt_id = DESLOCCPUAdam.optimizer_id
        DESLOCCPUAdam.optimizer_id += 1
        try:
            self.ds_opt_adam = CPUAdamBuilder().load()
            self.ds_opt_adam.create_adam(
                self.opt_id, lr, betas[0], betas[1], eps,
                weight_decay, adamw_mode, False)
            self._cpp_available = True
        except Exception:
            self.ds_opt_adam = None

    def __del__(self):
        if self._cpp_available and self.ds_opt_adam is not None:
            try:
                self.ds_opt_adam.destroy_adam(self.opt_id)
            except Exception:
                pass

    def _clip_coordinate_wise(self, grad, rho):
        """Per-coordinate clipping: clip(g, rho)_i = sign(g_i)*min(|g_i|, rho)."""
        mask = grad.abs() > rho
        if mask.any():
            self._clip_count += 1
            self._clip_elements += mask.sum().item()
            grad.clamp_(-rho, rho)
        return grad

    def _python_adam_update(self, p, grad, state, group):
        """Pure Python Adam update for CPU tensors."""
        beta1, beta2 = group['betas']
        eps = group['eps']
        lr = group['lr']

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        state['step'] += 1
        step = state['step']

        # Decoupled weight decay (AdamW)
        if self.adamw_mode and group['weight_decay'] != 0:
            p.data.mul_(1 - lr * group['weight_decay'])
        elif not self.adamw_mode and group['weight_decay'] != 0:
            grad = grad.add(p.data, alpha=group['weight_decay'])

        # Moment updates
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if group.get('bias_correction', True):
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step
            step_size = lr / bc1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
        else:
            step_size = lr
            denom = exp_avg_sq.sqrt().add_(eps)

        p.data.addcdiv_(exp_avg, denom, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        """DES-LOC CPU Adam step.

        1. Per-coordinate clip gradients
        2. Local Adam update on CPU
        3. Sync handled separately by sync_if_needed()
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        t0 = _time.time()
        self.global_step += 1
        device = torch.device('cpu')
        rho = self._clip_radius

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # DES-LOC per-coordinate clipping
                if rho > 0:
                    grad = self._clip_coordinate_wise(grad, rho)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state_dtype = torch.float if self.fp32_optimizer_states else p.dtype
                    state['exp_avg'] = torch.zeros_like(
                        p.data, dtype=state_dtype, device=device)
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data, dtype=state_dtype, device=device)

                if self._cpp_available and p.device == device:
                    state['step'] += 1
                    beta1, beta2 = group['betas']
                    self.ds_opt_adam.adam_update(
                        self.opt_id, state['step'], group['lr'],
                        beta1, beta2, group['eps'],
                        group['weight_decay'], group['bias_correction'],
                        p.data, grad, state['exp_avg'], state['exp_avg_sq'])
                else:
                    self._python_adam_update(p, grad, state, group)

        elapsed = (_time.time() - t0) * 1000
        self._step_times_ms.append(elapsed)
        return loss

    def sync_if_needed(self, world_size):
        """Sync optimizer states via DES-LOC schedule.

        For CPU offload, sync means:
        1. Gather states from all workers (CPU→GPU→NCCL→GPU→CPU)
        2. Average them
        3. Scatter back

        Only syncs at Kx/Ku/Kv intervals.
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
            if not torch_dist.is_initialized():
                return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}
        except ImportError:
            return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}

        t0 = _time.time()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                elem_size = p.data.element_size()
                numel = p.data.numel()

                if sync_x:
                    # For CPU params: move to GPU, allreduce, move back
                    if p.data.device.type == 'cpu':
                        gpu_tensor = p.data.cuda()
                        torch_dist.all_reduce(gpu_tensor, op=torch_dist.ReduceOp.AVG)
                        p.data.copy_(gpu_tensor.cpu())
                        del gpu_tensor
                    else:
                        torch_dist.all_reduce(p.data, op=torch_dist.ReduceOp.AVG)
                    self._total_comm_bytes += numel * elem_size * 2

                if sync_u and 'exp_avg' in state:
                    ea = state['exp_avg']
                    if ea.device.type == 'cpu':
                        gpu_ea = ea.cuda()
                        torch_dist.all_reduce(gpu_ea, op=torch_dist.ReduceOp.AVG)
                        ea.copy_(gpu_ea.cpu())
                        del gpu_ea
                    else:
                        torch_dist.all_reduce(ea, op=torch_dist.ReduceOp.AVG)
                    self._total_comm_bytes += numel * elem_size * 2

                if sync_v and 'exp_avg_sq' in state:
                    easq = state['exp_avg_sq']
                    if easq.device.type == 'cpu':
                        gpu_easq = easq.cuda()
                        torch_dist.all_reduce(gpu_easq, op=torch_dist.ReduceOp.AVG)
                        easq.copy_(gpu_easq.cpu())
                        del gpu_easq
                    else:
                        torch_dist.all_reduce(easq, op=torch_dist.ReduceOp.AVG)
                    self._total_comm_bytes += numel * elem_size * 2

        elapsed_ms = (_time.time() - t0) * 1000
        self._comm_times_ms.append(elapsed_ms)

        if sync_x:
            self._sync_x_count += 1
        if sync_u:
            self._sync_u_count += 1
        if sync_v:
            self._sync_v_count += 1

        # Clear GPU cache after CPU↔GPU transfers
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}

    def get_desloc_stats(self):
        """Return DES-LOC statistics for this optimizer."""
        n = max(self.global_step, 1)
        return {
            'optimizer': 'DESLOCCPUAdam',
            'global_step': self.global_step,
            'Kx': self._Kx,
            'Ku': self._Ku,
            'Kv': self._Kv,
            'clip_radius': self._clip_radius,
            'sync_x_count': self._sync_x_count,
            'sync_u_count': self._sync_u_count,
            'sync_v_count': self._sync_v_count,
            'total_comm_bytes': self._total_comm_bytes,
            'total_comm_gb': round(self._total_comm_bytes / 1e9, 4),
            'clip_events': self._clip_count,
            'clipped_elements': self._clip_elements,
            'avg_step_ms': round(
                sum(self._step_times_ms) / len(self._step_times_ms), 3
            ) if self._step_times_ms else 0,
            'avg_comm_ms': round(
                sum(self._comm_times_ms) / len(self._comm_times_ms), 3
            ) if self._comm_times_ms else 0,
            'cpp_kernel': self._cpp_available,
            'x_sync_ratio': round(self._sync_x_count / n, 4),
            'u_sync_ratio': round(self._sync_u_count / n, 4),
            'v_sync_ratio': round(self._sync_v_count / n, 4),
            'comm_reduction_vs_ddp': round(
                (n * 3) / max(self._sync_x_count + self._sync_u_count +
                              self._sync_v_count, 1), 2),
        }
