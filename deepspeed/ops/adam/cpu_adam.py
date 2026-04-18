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


# =========================================================================
# DES-LOC CPU Adam Extensions
# Ref: Section 4.1 — CPU-offloaded optimizer states with sync awareness
# =========================================================================

class DeslocCPUStateManager:
    """Manage CPU-offloaded optimizer states for DES-LOC.
    With ZeRO-Offload, m1/m2 live on CPU. DES-LOC sync periods
    mean GPU<->CPU transfer only at Ku/Kv boundaries, saving PCIe BW.

    Standard: transfer m1+m2 every step = 2*param_bytes/step
    DES-LOC: transfer m1 every Ku, m2 every Kv steps
    Ref: Nick Joseph — 'CPU transfer bottleneck'"""

    def __init__(self, Ku=3, Kv=6):
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.step = 0
        self.m1_transfers = 0
        self.m2_transfers = 0
        self.m1_skips = 0
        self.m2_skips = 0

    def should_transfer_m1(self):
        return (self.step % self.Ku) == 0

    def should_transfer_m2(self):
        return (self.step % self.Kv) == 0

    def advance(self):
        self.step += 1
        if self.should_transfer_m1():
            self.m1_transfers += 1
        else:
            self.m1_skips += 1
        if self.should_transfer_m2():
            self.m2_transfers += 1
        else:
            self.m2_skips += 1

    def get_pcie_savings(self, param_bytes):
        if self.step == 0:
            return {'saved_bytes': 0, 'ratio': 0.0}
        standard = 2 * param_bytes * self.step
        desloc = param_bytes * (self.m1_transfers + self.m2_transfers)
        return {
            'standard_bytes': standard,
            'desloc_bytes': desloc,
            'saved_bytes': standard - desloc,
            'ratio': round((standard - desloc) / max(1, standard), 4),
            'm1_ratio': round(self.m1_transfers / max(1, self.step), 4),
            'm2_ratio': round(self.m2_transfers / max(1, self.step), 4),
        }

    def state_dict(self):
        return {'step': self.step, 'Ku': self.Ku, 'Kv': self.Kv,
                'm1t': self.m1_transfers, 'm2t': self.m2_transfers}

    def load_state_dict(self, sd):
        self.step = sd.get('step', 0)
        self.Ku = sd.get('Ku', self.Ku)
        self.Kv = sd.get('Kv', self.Kv)
        self.m1_transfers = sd.get('m1t', 0)
        self.m2_transfers = sd.get('m2t', 0)


class DeslocCPUPinMemoryManager:
    """Manage pinned memory buffers for efficient CPU<->GPU transfer.
    Ref: ZeRO-Offload uses pinned memory for async CPU-GPU copies.
    DES-LOC reduces number of copies, but each copy should still be fast."""

    def __init__(self, buffer_size=0):
        self.buffer_size = buffer_size
        self._pinned_buffers = {}
        self.total_pins = 0
        self.total_copies = 0

    def get_buffer(self, key, size, dtype=None):
        import torch
        if key not in self._pinned_buffers or self._pinned_buffers[key].numel() < size:
            if dtype is None:
                dtype = torch.float32
            self._pinned_buffers[key] = torch.zeros(size, dtype=dtype, pin_memory=True)
            self.total_pins += 1
        return self._pinned_buffers[key][:size]

    def copy_to_gpu(self, key, gpu_tensor, stream=None):
        buf = self._pinned_buffers.get(key)
        if buf is not None:
            gpu_tensor.copy_(buf[:gpu_tensor.numel()], non_blocking=True)
            self.total_copies += 1

    def copy_from_gpu(self, key, gpu_tensor, stream=None):
        buf = self.get_buffer(key, gpu_tensor.numel(), gpu_tensor.dtype)
        buf[:gpu_tensor.numel()].copy_(gpu_tensor, non_blocking=True)
        self.total_copies += 1

    def get_stats(self):
        return {
            'num_buffers': len(self._pinned_buffers),
            'total_pins': self.total_pins,
            'total_copies': self.total_copies,
            'total_bytes': sum(b.numel() * b.element_size()
                              for b in self._pinned_buffers.values()),
        }

    def clear(self):
        self._pinned_buffers.clear()


class DeslocCPUAdamScheduler:
    """Schedule CPU Adam operations to align with DES-LOC sync.
    
    At non-sync steps: run local Adam entirely on GPU (fast)
    At sync steps: fetch states from CPU, sync, update, push back
    
    This minimizes PCIe transfers while maintaining correctness."""

    def __init__(self, Kx=1, Ku=3, Kv=6):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.step = 0

    def get_action(self):
        """Return what actions should happen at this step.
        Returns dict with booleans for each operation."""
        return {
            'fetch_m1': (self.step % self.Ku) == 0,
            'fetch_m2': (self.step % self.Kv) == 0,
            'sync_params': (self.step % self.Kx) == 0,
            'push_m1': (self.step % self.Ku) == 0,
            'push_m2': (self.step % self.Kv) == 0,
        }

    def advance(self):
        self.step += 1
        return self.get_action()


class DeslocCPUPinMemoryManager:
    """Manage pinned memory for efficient CPU<->GPU transfer.
    DES-LOC reduces transfer count but each should be fast.
    Ref: ZeRO-Offload pinned memory for async copies."""

    def __init__(self):
        self._buffers = {}
        self.pins = 0
        self.copies = 0

    def get_buffer(self, key, size, dtype=None):
        import torch
        if key not in self._buffers or self._buffers[key].numel() < size:
            if dtype is None: dtype = torch.float32
            self._buffers[key] = torch.zeros(size, dtype=dtype, pin_memory=True)
            self.pins += 1
        return self._buffers[key][:size]

    def copy_to_gpu(self, key, gpu_tensor):
        buf = self._buffers.get(key)
        if buf is not None:
            gpu_tensor.copy_(buf[:gpu_tensor.numel()], non_blocking=True)
            self.copies += 1

    def copy_from_gpu(self, key, gpu_tensor):
        buf = self.get_buffer(key, gpu_tensor.numel(), gpu_tensor.dtype)
        buf[:gpu_tensor.numel()].copy_(gpu_tensor, non_blocking=True)
        self.copies += 1

    def stats(self):
        return {'buffers': len(self._buffers), 'pins': self.pins, 'copies': self.copies,
                'bytes': sum(b.numel() * b.element_size() for b in self._buffers.values())}

    def clear(self):
        self._buffers.clear()


class DeslocCPUAdamScheduler:
    """Schedule CPU Adam operations aligned with DES-LOC sync.
    Non-sync: local Adam on GPU (fast, no PCIe transfer).
    Sync: fetch states from CPU, sync, update, push back.
    Ref: Section 4.1 — reduce PCIe transfers via Ku/Kv gating."""

    def __init__(self, Kx=1, Ku=3, Kv=6):
        self.Kx = max(1, Kx)
        self.Ku = max(1, Ku)
        self.Kv = max(1, Kv)
        self.step = 0

    def get_action(self):
        return {
            'fetch_m1': (self.step % self.Ku) == 0,
            'fetch_m2': (self.step % self.Kv) == 0,
            'sync_params': (self.step % self.Kx) == 0,
            'push_m1': (self.step % self.Ku) == 0,
            'push_m2': (self.step % self.Kv) == 0,
        }

    def advance(self):
        self.step += 1
        return self.get_action()

    def get_transfer_savings(self, param_bytes, total_steps=None):
        """Estimate PCIe transfer savings vs standard offload."""
        t = total_steps or self.step
        if t <= 0:
            return {'saved_ratio': 0}
        standard = 2 * t  # m1 + m2 every step
        desloc = t // self.Ku + t // self.Kv  # m1 every Ku, m2 every Kv
        saved = standard - desloc
        return {
            'standard_transfers': standard,
            'desloc_transfers': desloc,
            'saved_transfers': saved,
            'saved_ratio': round(saved / max(1, standard), 4),
            'saved_bytes': saved * param_bytes,
        }

    def state_dict(self):
        return {'step': self.step, 'Kx': self.Kx, 'Ku': self.Ku, 'Kv': self.Kv}


class DeslocCPUGradientBuffer:
    """CPU-side gradient buffer for ZeRO-Offload + DES-LOC.
    Accumulates gradients on CPU between Kx boundaries.
    At Kx boundary, the accumulated gradient is used for the
    CPU-side optimizer step, then params are pushed to GPU."""

    def __init__(self, Kx=1):
        self.Kx = max(1, Kx)
        self.step = 0
        self._buffers = {}

    def add_gradient(self, name, grad_cpu):
        """Accumulate gradient on CPU."""
        if name not in self._buffers:
            self._buffers[name] = grad_cpu.clone()
        else:
            self._buffers[name].add_(grad_cpu)
        self.step += 1

    def should_apply(self):
        return (self.step % self.Kx) == 0

    def get_and_clear(self, name):
        buf = self._buffers.pop(name, None)
        if buf is not None:
            buf.div_(self.Kx)  # average over accumulated steps
        return buf

    def clear_all(self):
        self._buffers.clear()
