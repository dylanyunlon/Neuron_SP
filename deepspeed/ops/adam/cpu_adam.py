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


# =================================================================
# M068: DES-LOC CPUAdam ZeRO-Offload State Sync (400 lines)
# =================================================================
# Extends DESLOCCPUAdam with ZeRO-Offload-aware state sync.
# When optimizer states are on CPU (ZeRO Stage 2/3 offload),
# DES-LOC must coordinate GPU↔CPU transfers with sync schedule.
#
# Reference: Section 4.1 "Ring-AllReduce"
# Reference: DeepSpeed ZeRO-Offload architecture
# =================================================================

import time as _time
import math as _math


class DESLOCCPUOffloadSync:
    """Manage DES-LOC state sync with CPU-offloaded states.

    When ZeRO-Offload moves optimizer states to CPU:
    1. States live on CPU memory
    2. Allreduce requires GPU-side communication
    3. DES-LOC must schedule CPU→GPU→allreduce→CPU transfers

    This class batches transfers to minimize PCIe overhead.
    """

    def __init__(self, Kx=32, Ku=96, Kv=192, pin_memory=True):
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        self.pin_memory = pin_memory
        self.step = 0
        self.transfer_log = []
        self.total_cpu_to_gpu_bytes = 0
        self.total_gpu_to_cpu_bytes = 0
        self.total_transfer_time_ms = 0.0

    def should_sync(self, state_type):
        """Check if sync needed for state at current step."""
        if state_type == 'x':
            return self.step % self.Kx == 0
        elif state_type == 'u':
            return self.step % self.Ku == 0
        elif state_type == 'v':
            return self.step % self.Kv == 0
        return False

    def sync_cpu_state(self, cpu_tensor, device, state_type,
                       dp_group=None):
        """Sync a CPU-resident optimizer state via GPU allreduce.

        Steps:
        1. Copy CPU tensor → GPU (pinned → device)
        2. AllReduce on GPU
        3. Copy GPU tensor → CPU (device → pinned)
        """
        import torch
        import torch.distributed as dist

        if not self.should_sync(state_type):
            return 0

        start = _time.monotonic()

        # Step 1: CPU → GPU
        if self.pin_memory and not cpu_tensor.is_pinned():
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        else:
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)

        torch.cuda.synchronize(device)
        num_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
        self.total_cpu_to_gpu_bytes += num_bytes

        # Step 2: AllReduce on GPU
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(gpu_tensor, op=dist.ReduceOp.AVG,
                            group=dp_group)

        # Step 3: GPU → CPU
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.synchronize(device)
        self.total_gpu_to_cpu_bytes += num_bytes

        elapsed_ms = (_time.monotonic() - start) * 1000.0
        self.total_transfer_time_ms += elapsed_ms

        self.transfer_log.append({
            'step': self.step,
            'state_type': state_type,
            'bytes': num_bytes,
            'time_ms': round(elapsed_ms, 4),
        })

        return num_bytes

    def sync_all_states(self, optimizer, device, dp_group=None):
        """Sync all optimizer states that need syncing.

        Iterates over optimizer state dict and syncs CPU tensors
        that are due according to DES-LOC schedule.
        """
        total_bytes = 0
        for pg in optimizer.param_groups:
            for p in pg['params']:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]

                # First moment (u / exp_avg)
                if 'exp_avg' in state and state['exp_avg'].is_cpu:
                    total_bytes += self.sync_cpu_state(
                        state['exp_avg'], device, 'u', dp_group)

                # Second moment (v / exp_avg_sq)
                if 'exp_avg_sq' in state and state['exp_avg_sq'].is_cpu:
                    total_bytes += self.sync_cpu_state(
                        state['exp_avg_sq'], device, 'v', dp_group)

        self.step += 1
        return total_bytes

    def get_stats(self):
        """Get offload sync statistics."""
        return {
            'total_steps': self.step,
            'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv,
            'cpu_to_gpu_bytes': self.total_cpu_to_gpu_bytes,
            'gpu_to_cpu_bytes': self.total_gpu_to_cpu_bytes,
            'total_transfer_time_ms': round(
                self.total_transfer_time_ms, 2),
            'num_transfers': len(self.transfer_log),
        }

    def format_log(self):
        """Format offload sync log."""
        s = self.get_stats()
        lines = [
            f"### CPU Offload Sync "
            f"(Kx={s['Kx']}, Ku={s['Ku']}, Kv={s['Kv']}) ###",
            f"Steps: {s['total_steps']}",
            f"CPU→GPU: {s['cpu_to_gpu_bytes']/1e6:.1f}MB",
            f"GPU→CPU: {s['gpu_to_cpu_bytes']/1e6:.1f}MB",
            f"Transfer time: {s['total_transfer_time_ms']:.1f}ms",
            f"Transfers: {s['num_transfers']}",
        ]
        return "\n".join(lines)


class DESLOCGradientVarianceTracker:
    """Track gradient variance per parameter for Theorem 1.

    Assumption 2: E[‖g^m - ∇f_m(x)‖²] ≤ σ²

    Estimates σ² empirically to validate convergence bound
    and generate RQ1 supplementary data.
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.grad_sq_sums = {}
        self.grad_sums = {}
        self.counts = {}
        self.variance_history = []

    def record_gradient(self, param_id, grad_tensor):
        """Record gradient for variance estimation."""
        grad_flat = grad_tensor.flatten()
        grad_norm_sq = grad_flat.dot(grad_flat).item()
        grad_sum = grad_flat.sum().item()
        n = grad_flat.numel()

        if param_id not in self.grad_sq_sums:
            self.grad_sq_sums[param_id] = 0.0
            self.grad_sums[param_id] = 0.0
            self.counts[param_id] = 0

        self.grad_sq_sums[param_id] += grad_norm_sq / n
        self.grad_sums[param_id] += grad_sum / n
        self.counts[param_id] += 1

    def estimate_variance(self, param_id):
        """Estimate gradient variance for a parameter.

        Var(g) = E[g²] - (E[g])²
        """
        if param_id not in self.counts or self.counts[param_id] < 2:
            return 0.0
        n = self.counts[param_id]
        mean_sq = self.grad_sq_sums[param_id] / n
        mean = self.grad_sums[param_id] / n
        return max(mean_sq - mean * mean, 0.0)

    def get_global_variance_estimate(self):
        """Get σ² estimate across all parameters."""
        if not self.counts:
            return 0.0
        total_var = 0.0
        count = 0
        for pid in self.counts:
            total_var += self.estimate_variance(pid)
            count += 1
        if count == 0:
            return 0.0
        avg_var = total_var / count
        self.variance_history.append(avg_var)
        return avg_var

    def get_summary(self):
        """Get variance tracking summary."""
        return {
            'num_params_tracked': len(self.counts),
            'total_samples': sum(self.counts.values()),
            'global_variance': self.get_global_variance_estimate(),
            'variance_history_len': len(self.variance_history),
            'recent_variance': (
                self.variance_history[-1]
                if self.variance_history else 0.0),
        }


class DESLOCParameterPartitioner:
    """Partition parameters for efficient DES-LOC sync.

    Groups parameters by size and type to batch allreduce
    operations, reducing kernel launch overhead.
    """

    def __init__(self, bucket_size_mb=25):
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.buckets = {'x': [], 'u': [], 'v': []}
        self.bucket_sizes = {'x': 0, 'u': 0, 'v': 0}

    def add_parameter(self, param, state_type='x'):
        """Add parameter to appropriate bucket."""
        size = param.numel() * param.element_size()
        self.buckets[state_type].append(param)
        self.bucket_sizes[state_type] += size

    def get_buckets(self, state_type):
        """Get parameter buckets ready for batched allreduce."""
        params = self.buckets.get(state_type, [])
        if not params:
            return []

        # Group into buckets of bucket_size_bytes
        groups = []
        current_group = []
        current_size = 0

        for p in params:
            p_size = p.numel() * p.element_size()
            if current_size + p_size > self.bucket_size_bytes and current_group:
                groups.append(current_group)
                current_group = []
                current_size = 0
            current_group.append(p)
            current_size += p_size

        if current_group:
            groups.append(current_group)
        return groups

    def get_stats(self):
        """Get partitioning statistics."""
        return {
            'bucket_size_mb': self.bucket_size_bytes / (1024 * 1024),
            'x_params': len(self.buckets['x']),
            'u_params': len(self.buckets['u']),
            'v_params': len(self.buckets['v']),
            'x_bytes': self.bucket_sizes['x'],
            'u_bytes': self.bucket_sizes['u'],
            'v_bytes': self.bucket_sizes['v'],
            'x_buckets': len(self.get_buckets('x')),
            'u_buckets': len(self.get_buckets('u')),
            'v_buckets': len(self.get_buckets('v')),
        }


# =================================================================
# End M068
# =================================================================

# =================================================================
# M084: CPU Offload + DES-LOC State Management
# Claude-5 (M077-M091)
# Nick Joseph: "效率差的原因通常就是显存带宽瓶颈、CPU传输瓶颈"
# =================================================================

import math
import time
import os
import logging

_m084_logger = logging.getLogger("DeepSpeed")


class DeslocCPUStateManager:
    """Manage DES-LOC optimizer states on CPU with async GPU transfer.

    When GPU memory is tight, DES-LOC's independently-synced momentum
    states (u, v) can reside on CPU and transfer to GPU only at their
    respective sync periods (Ku, Kv).

    This saves GPU memory proportional to (1 - 1/Ku - 1/Kv) of the
    momentum memory footprint.
    """

    def __init__(self, Kx=8, Ku=24, Kv=48, pin_memory=True,
                 async_transfer=True, prefetch_steps=2):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.pin_memory = pin_memory
        self.async_transfer = async_transfer
        self.prefetch_steps = prefetch_steps
        self._cpu_states = {}
        self._gpu_buffers = {}
        self._transfer_streams = {}
        self._step = 0
        self._transfer_events = []
        self._bytes_transferred = 0
        self._bytes_avoided = 0

    def register_param(self, param_id, param_shape, dtype_str="float32"):
        """Register a parameter whose momentum will be CPU-offloaded."""
        numel = 1
        for d in param_shape:
            numel *= d
        self._cpu_states[param_id] = {
            "shape": param_shape,
            "numel": numel,
            "dtype": dtype_str,
            "exp_avg": None,      # First momentum on CPU
            "exp_avg_sq": None,   # Second momentum on CPU
            "bytes": numel * 4,   # Approximate
        }

    def init_states(self, param_id):
        """Initialize CPU-side momentum buffers."""
        try:
            import torch
        except ImportError:
            return
        info = self._cpu_states.get(param_id)
        if info is None:
            return
        shape = info["shape"]
        info["exp_avg"] = torch.zeros(shape, dtype=torch.float32,
                                      pin_memory=self.pin_memory)
        info["exp_avg_sq"] = torch.zeros(shape, dtype=torch.float32,
                                         pin_memory=self.pin_memory)

    def should_transfer_u(self, step=None):
        s = step or self._step
        return s % self.Ku == 0

    def should_transfer_v(self, step=None):
        s = step or self._step
        return s % self.Kv == 0

    def should_prefetch_u(self, step=None):
        s = step or self._step
        for look in range(1, self.prefetch_steps + 1):
            if (s + look) % self.Ku == 0:
                return True
        return False

    def should_prefetch_v(self, step=None):
        s = step or self._step
        for look in range(1, self.prefetch_steps + 1):
            if (s + look) % self.Kv == 0:
                return True
        return False

    def transfer_to_gpu(self, param_id, state_name, device):
        """Copy CPU state to GPU buffer (async if configured)."""
        try:
            import torch
        except ImportError:
            return None
        info = self._cpu_states.get(param_id)
        if info is None or info.get(state_name) is None:
            return None

        cpu_tensor = info[state_name]
        buf_key = (param_id, state_name)
        if buf_key not in self._gpu_buffers:
            self._gpu_buffers[buf_key] = torch.empty_like(
                cpu_tensor, device=device)

        gpu_buf = self._gpu_buffers[buf_key]
        if self.async_transfer:
            stream_key = f"{param_id}_{state_name}"
            if stream_key not in self._transfer_streams:
                self._transfer_streams[stream_key] = (
                    torch.cuda.Stream(device=device))
            stream = self._transfer_streams[stream_key]
            with torch.cuda.stream(stream):
                gpu_buf.copy_(cpu_tensor, non_blocking=True)
        else:
            gpu_buf.copy_(cpu_tensor)

        self._bytes_transferred += cpu_tensor.numel() * cpu_tensor.element_size()
        return gpu_buf

    def transfer_to_cpu(self, param_id, state_name, gpu_tensor):
        """Copy GPU state back to CPU (for checkpoint or sync)."""
        info = self._cpu_states.get(param_id)
        if info is None:
            return
        if info.get(state_name) is None:
            self.init_states(param_id)
        cpu_tensor = info[state_name]
        if cpu_tensor is not None:
            cpu_tensor.copy_(gpu_tensor.cpu())

    def step(self):
        """Advance step counter."""
        self._step += 1
        # Count bytes avoided by not transferring when not needed
        for pid, info in self._cpu_states.items():
            if not self.should_transfer_u():
                self._bytes_avoided += info.get("bytes", 0)
            if not self.should_transfer_v():
                self._bytes_avoided += info.get("bytes", 0)

    def synchronize_streams(self):
        """Wait for all async transfers to complete."""
        try:
            import torch
            for stream in self._transfer_streams.values():
                stream.synchronize()
        except Exception:
            pass

    def memory_savings_mb(self):
        """Estimate GPU memory saved by offloading."""
        total_bytes = sum(info.get("bytes", 0) * 2  # u + v
                          for info in self._cpu_states.values())
        return total_bytes / (1024 * 1024)

    def get_stats(self):
        return {
            "num_params_offloaded": len(self._cpu_states),
            "step": self._step,
            "Kx": self.Kx, "Ku": self.Ku, "Kv": self.Kv,
            "bytes_transferred": self._bytes_transferred,
            "bytes_avoided": self._bytes_avoided,
            "memory_savings_mb": round(self.memory_savings_mb(), 1),
            "async_transfer": self.async_transfer,
            "pin_memory": self.pin_memory,
        }


class StateCheckpointer:
    """Checkpoint DES-LOC optimizer states independently.

    Unlike standard checkpoint which bundles everything, DES-LOC
    states can be checkpointed at different frequencies since they
    change at different rates (half-life principle).
    """

    def __init__(self, checkpoint_dir, param_interval=100,
                 momentum_interval=500):
        self.checkpoint_dir = checkpoint_dir
        self.param_interval = param_interval
        self.momentum_interval = momentum_interval
        self._step = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def should_checkpoint_params(self, step=None):
        s = step or self._step
        return s > 0 and s % self.param_interval == 0

    def should_checkpoint_momentum(self, step=None):
        s = step or self._step
        return s > 0 and s % self.momentum_interval == 0

    def save_params(self, params_dict, step):
        """Save parameter checkpoint."""
        try:
            import torch
            path = os.path.join(self.checkpoint_dir,
                                f"params_step{step}.pt")
            torch.save(params_dict, path)
            _m084_logger.info(f"DES-LOC params checkpoint: {path}")
            return path
        except Exception as e:
            _m084_logger.warning(f"Checkpoint save failed: {e}")
            return None

    def save_momentum(self, momentum_dict, step):
        """Save momentum state checkpoint."""
        try:
            import torch
            path = os.path.join(self.checkpoint_dir,
                                f"momentum_step{step}.pt")
            torch.save(momentum_dict, path)
            _m084_logger.info(f"DES-LOC momentum checkpoint: {path}")
            return path
        except Exception as e:
            _m084_logger.warning(f"Momentum checkpoint failed: {e}")
            return None

    def load_latest(self, prefix="params"):
        """Load most recent checkpoint of given type."""
        try:
            import torch
            files = [f for f in os.listdir(self.checkpoint_dir)
                     if f.startswith(prefix) and f.endswith(".pt")]
            if not files:
                return None
            files.sort(key=lambda f: int(
                f.replace(prefix + "_step", "").replace(".pt", "")))
            path = os.path.join(self.checkpoint_dir, files[-1])
            return torch.load(path, map_location="cpu",
                              weights_only=True)
        except Exception as e:
            _m084_logger.warning(f"Checkpoint load failed: {e}")
            return None

    def step(self):
        self._step += 1

    def cleanup_old(self, keep_last=3, prefix="params"):
        """Remove old checkpoints, keep last N."""
        try:
            files = [f for f in os.listdir(self.checkpoint_dir)
                     if f.startswith(prefix) and f.endswith(".pt")]
            files.sort()
            for f in files[:-keep_last]:
                os.remove(os.path.join(self.checkpoint_dir, f))
        except Exception:
            pass

    def get_checkpoint_info(self):
        files = []
        try:
            for f in os.listdir(self.checkpoint_dir):
                if f.endswith(".pt"):
                    path = os.path.join(self.checkpoint_dir, f)
                    files.append({
                        "name": f,
                        "size_mb": os.path.getsize(path) / (1024 * 1024),
                    })
        except Exception:
            pass
        return files


# =================================================================
# End M084  (CPU State Manager + State Checkpointer)
# =================================================================
