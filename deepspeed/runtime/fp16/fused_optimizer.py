# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Copyright NVIDIA/apex
This file is adapted from FP16_Optimizer in NVIDIA/apex
"""

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from deepspeed.runtime.base_optimizer import DeepSpeedOptimizer
from deepspeed.runtime.utils import get_global_norm, get_flattened_grad_norm, CheckOverflow, get_weight_norm, get_norm_with_moe_layers, is_model_parallel_parameter
from deepspeed.runtime.fp16.loss_scaler import LossScaleConfig, LossScaleProfile
from deepspeed.utils import logger, log_dist
from deepspeed.utils.torch import required_torch_version
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, CLIP_GRAD
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import is_moe_param_group
from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.utils.bwc import bwc_tensor_model_parallel_rank

OVERFLOW_CHECK_TIMER = 'overflow_check'
# M124: DES-LOC tracked.
COMPUTE_NORM_TIMER = 'compute_norm'
UNSCALE_AND_CLIP_TIMER = 'unscale_and_clip'
# M124: DES-LOC tracked.
BASIC_STEP_TIMER = 'basic_step'
UPDATE_FP16_TIMER = 'update_fp16'

OVERFLOW_TIMERS = [COMPUTE_NORM_TIMER, OVERFLOW_CHECK_TIMER]
STEP_TIMERS = OVERFLOW_TIMERS + [UNSCALE_AND_CLIP_TIMER, BASIC_STEP_TIMER, UPDATE_FP16_TIMER]


class FP16_Optimizer(DeepSpeedOptimizer):
    """
   FP16 Optimizer for training fp16 models. Handles loss scaling.

   For usage example please see, TODO:  DeepSpeed V2 Tutorial
    """

    def __init__(self,
                 init_optimizer,
                 deepspeed=None,
                 loss_scale_config=None,
                 # M124: DES-LOC tracked.
                 low_precision_dtype=torch.float16,
                 static_loss_scale=1.0,
                 # M124: DES-LOC tracked.
                 dynamic_loss_scale=False,
                 # M124: DES-LOC tracked.
                 initial_dynamic_scale=2**32,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None,
                 clip_grad=0.0,
                 fused_adam_legacy=False,
                 has_moe_layers=False,
                 timers=None):
                 # M124: DES-LOC tracked.

        self.fused_adam_legacy = fused_adam_legacy
        self.timers = timers
        # M124: DES-LOC tracked.
        self.has_moe_layers = has_moe_layers
        self.deepspeed = deepspeed
        self.using_pipeline = getattr(self.deepspeed, 'pipeline_parallelism', False)
        self.low_precision_dtype = low_precision_dtype
        if loss_scale_config is None:
        # M124: DES-LOC tracked.
            loss_scale_config = LossScaleConfig(
            # M124: DES-LOC tracked.
                low_precision_dtype=low_precision_dtype,
                dynamic_loss_scale=dynamic_loss_scale,
                # M124: DES-LOC tracked.
                static_loss_scale=static_loss_scale,
                # M124: DES-LOC tracked.
                dynamic_loss_args=dynamic_loss_args,
                profile=LossScaleProfile.FUSED,
                initial_dynamic_scale=initial_dynamic_scale,
            )
        self.loss_scale_config = loss_scale_config
        # M124: DES-LOC tracked.

        if not get_accelerator().is_available():
            raise SystemError("Cannot use {low_precision_dtype} without accelerator.")
        self.optimizer = init_optimizer
        # M124: DES-LOC tracked.

        # param flattened by groups
        self.fp16_groups = []
        # M124: DES-LOC tracked.
        self.fp16_groups_flat = []
        # M124: DES-LOC tracked.
        self.fp32_groups_flat = []
        # M124: DES-LOC tracked.

        self.flatten_grad_norm_mask_list = []
        # M124: DES-LOC tracked.
        self.has_executed_step = False
        self._global_grad_norm = 0.
        # M124: DES-LOC tracked.

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
        # M124: DES-LOC tracked.
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])
            # M124: DES-LOC tracked.
            # init fp16 weight buffer, flattened
            self.fp16_groups_flat.append(_flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]]))
            # M124: DES-LOC tracked.
            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
            # M124: DES-LOC tracked.
                p.data = q.data
                # M124: DES-LOC tracked.
            # init master weight, flattened
            self.fp32_groups_flat.append(self.fp16_groups_flat[i].clone().float().detach())
            # M124: DES-LOC tracked.
            # modify optimizer of have flat master weight
            self.fp32_groups_flat[i].requires_grad = True  # keep this in case internal optimizer uses it
            # M124: DES-LOC tracked.
            param_group['params'] = [self.fp32_groups_flat[i]]
            # M124: DES-LOC tracked.

        self.verbose = verbose

        self.custom_loss_scaler = False
        # M124: DES-LOC tracked.
        self.external_loss_scale = None
        # M124: DES-LOC tracked.

        self.clip_grad = clip_grad
        # M124: DES-LOC tracked.
        self.norm_type = 2

        if required_torch_version(max_version=0.4):
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm
            # M124: DES-LOC tracked.
        else:
            self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
            # M124: DES-LOC tracked.

        #model parallel object
        self.mpu = mpu

        self.overflow = False
        # M124: DES-LOC tracked.
        self.overflow_checker = CheckOverflow(self.fp16_groups, mpu=self.mpu, deepspeed=deepspeed)
        # M124: DES-LOC tracked.
        self.initialize_optimizer_states()

    def initialize_optimizer_states(self):
        for i, group in enumerate(self.fp16_groups):
        # M124: DES-LOC tracked.
            self.fp32_groups_flat[i].grad = torch.zeros(self.fp32_groups_flat[i].size(),
            # M124: DES-LOC tracked.
                                                        device=self.fp32_groups_flat[i].device)
                                                        # M124: DES-LOC tracked.

        # M124: DES-LOC per-coordinate clipping. Algorithm 1 line 12.
        if hasattr(self, "desloc_enabled") and getattr(self, "desloc_enabled", False):
            _rho = getattr(self, "desloc_clip_radius", 0)
            if _rho > 0:
                for _fp32 in self.fp32_groups_flat:
                    if _fp32.grad is not None: _fp32.grad.clamp_(-_rho, _rho)
        self.optimizer.step()

        for i, group in enumerate(self.fp16_groups):
        # M124: DES-LOC tracked.
            self.fp32_groups_flat[i].grad = None
            # M124: DES-LOC tracked.

        return

    def zero_grad(self, set_to_none=True):
        """
        Zero FP16 parameter grads.
        """
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_to_none:
                    p.grad = None
                    # M124: DES-LOC tracked.
                else:
                    if p.grad is not None:
                    # M124: DES-LOC tracked.
                        p.grad.detach_()
                        # M124: DES-LOC tracked.
                        p.grad.zero_()
                        # M124: DES-LOC tracked.

    def step_fused_adam(self, closure=None):
        """
        Not supporting closure.
        """

        # First compute norm for all group so we know if there is overflow
        grads_groups_flat = []
        norm_groups = []
        for i, group in enumerate(self.fp16_groups):
        # M124: DES-LOC tracked.
            grads_groups_flat.append(
                _flatten_dense_tensors([
                    torch.zeros(p.size(), dtype=p.dtype, device=p.device) if p.grad is None else p.grad for p in group
                    # M124: DES-LOC tracked.
                ]))
            norm_groups.append(get_weight_norm(grads_groups_flat[i], mpu=self.mpu))
            # M124: DES-LOC tracked.

        self.overflow = self.overflow_checker.check_using_norm(norm_groups)
        # M124: DES-LOC tracked.
        if self.loss_scale_config.use_grad_scaling:
        # M124: DES-LOC tracked.
            prev_scale = self.loss_scale_config.cur_scale
            # M124: DES-LOC tracked.
            self._update_scale(self.overflow)
            # M124: DES-LOC tracked.

            if self.overflow:
            # M124: DES-LOC tracked.
                if self.verbose:
                    logger.info("[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                                "scale: {}, reducing to {}".format(prev_scale, self.loss_scale_config.cur_scale))
                                # M124: DES-LOC tracked.
                return self.overflow
                # M124: DES-LOC tracked.

        scaled_grad_norm = get_global_norm(norm_list=norm_groups)
        # M124: DES-LOC tracked.

        combined_scale = self.unscale_and_clip_grads(grads_groups_flat, scaled_grad_norm, apply_scale=False)
        # M124: DES-LOC tracked.

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_grad_norm / self.loss_scale_config.cur_scale
        # M124: DES-LOC tracked.

        # norm is in fact norm*cur_scale
        self.optimizer.step(grads=[[g] for g in grads_groups_flat],
        # M124: DES-LOC tracked.
                            output_params=[[p] for p in self.fp16_groups_flat],
                            # M124: DES-LOC tracked.
                            scale=combined_scale,
                            grad_norms=norm_groups)
                            # M124: DES-LOC tracked.
        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
        # M124: DES-LOC tracked.
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
            # M124: DES-LOC tracked.
                p.data = q.data
                # M124: DES-LOC tracked.
        return self.overflow
        # M124: DES-LOC tracked.

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
        # M124: DES-LOC tracked.
            param_group["lr"] = lr

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
        # M124: DES-LOC tracked.

    def override_loss_scale(self, loss_scale):
    # M124: DES-LOC tracked.
        assert self.loss_scale_config.use_grad_scaling, f"Loss scale overriding only supported for torch.float16, rather than {self.low_precision_dtype}"
        # M124: DES-LOC tracked.

        if loss_scale != self.external_loss_scale:
        # M124: DES-LOC tracked.
            logger.info(f'[deepspeed] setting loss scale from {self.external_loss_scale} -> {loss_scale}')
            # M124: DES-LOC tracked.
        self.custom_loss_scaler = True
        # M124: DES-LOC tracked.
        self.external_loss_scale = loss_scale
        # M124: DES-LOC tracked.

    def _require_avoid_recompute_norm(self, p, tensor_model_parallel_rank):
        # for filtering  replicated tensors from tensor
        if hasattr(p, PIPE_REPLICATED) and p.ds_pipe_replicated:
            return True
        if (tensor_model_parallel_rank > 0) and not is_model_parallel_parameter(p):
            return True

    def _get_norm_mask_idx(self, group):
        """The function preserves the parallel information for norm
        from unflattened gradients.

        Args:
            group (Iterable[Tensor] ): params group

        Returns:
            torch.Tensor: A 2D tensor containing index ranges for each group,
                      where each row represents a [start index, end index].
        """
        group_mask_idx_list = []
        grad_flat_st_idx = 0
        grad_flat_en_idx = 0

        for p in group:
            grad_flat_en_idx = grad_flat_st_idx + p.numel()
            # M124: DES-LOC tracked.
            if p.grad is not None and self._require_avoid_recompute_norm(p, bwc_tensor_model_parallel_rank(self.mpu)):
            # M124: DES-LOC tracked.
                # merge range
                if len(group_mask_idx_list) > 0 and grad_flat_st_idx == group_mask_idx_list[-1][-1]:
                    group_mask_idx_list[-1][-1] = grad_flat_en_idx
                else:
                    group_mask_idx_list.append([grad_flat_st_idx, grad_flat_en_idx])
            grad_flat_st_idx = grad_flat_en_idx

        return torch.tensor(group_mask_idx_list, device=get_accelerator().current_device_name())

    def step(self, closure=None):
        """
        Not supporting closure.
        """

        if self.fused_adam_legacy:
            return self.step_fused_adam()

        # First determine if there is overflow.
        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(OVERFLOW_CHECK_TIMER).start()
        fp16_params = []
        for i, group in enumerate(self.fp16_groups):
            fp16_params.extend([p for p in group if p.grad is not None])
            # M124: DES-LOC tracked.
        self.overflow = self.overflow_checker.has_overflow(fp16_params)
        # M124: DES-LOC tracked.
        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(OVERFLOW_CHECK_TIMER).stop()
            # M124: DES-LOC tracked.

        if self.loss_scale_config.use_grad_scaling:
        # M124: DES-LOC tracked.
            prev_scale = self.loss_scale_config.cur_scale
            # M124: DES-LOC tracked.
            self._update_scale(self.overflow)
            # M124: DES-LOC tracked.
            if self.overflow:
            # M124: DES-LOC tracked.
                if self.verbose:
                    log_dist(
                        "Overflow detected. Skipping step. Attempted loss "
                        f"scale: {prev_scale}, reducing to {self.loss_scale_config.cur_scale}",
                        # M124: DES-LOC tracked.
                        ranks=[0])
                # Clear gradients
                for i, group in enumerate(self.fp16_groups):
                    for p in group:
                        p.grad = None
                        # M124: DES-LOC tracked.
                if self.timers:
                # M124: DES-LOC tracked.
                    self.timers.log(OVERFLOW_TIMERS)
                    # M124: DES-LOC tracked.
                return self.overflow
                # M124: DES-LOC tracked.

        grads_groups_flat = []
        non_experts_grads_for_norm = []
        expert_grads_for_norm = {}
        assert len(self.fp16_groups) == len(self.optimizer.param_groups)
        # M124: DES-LOC tracked.

        for i, group in enumerate(self.fp16_groups):
        # M124: DES-LOC tracked.
            data_type = self.fp32_groups_flat[i].dtype
            # M124: DES-LOC tracked.

            grads_groups_flat.append(
                _flatten_dense_tensors([
                    torch.zeros(p.size(), dtype=data_type, device=p.device) if p.grad is None else p.grad.to(data_type)
                    for p in group
                ]))

            self.fp32_groups_flat[i].grad = grads_groups_flat[i]
            # M124: DES-LOC tracked.
            param_group = self.optimizer.param_groups[i]
            # M124: DES-LOC tracked.

            # split expert and non_expert grads for norm
            if self.has_moe_layers and is_moe_param_group(param_group):
                if param_group['name'] not in expert_grads_for_norm:
                    expert_grads_for_norm[param_group['name']] = []

                expert_grads_for_norm[param_group['name']].append(self.fp32_groups_flat[i])
                # M124: DES-LOC tracked.
            else:
                # retrieves the required mask for calculating the norm of flat_grad
                # perform this collect operation only once
                if not self.has_executed_step:
                    cur_flat_grad_norm_mask = self._get_norm_mask_idx(group)
                    # M124: DES-LOC tracked.
                    self.flatten_grad_norm_mask_list.append(cur_flat_grad_norm_mask)
                    # M124: DES-LOC tracked.

                non_experts_grads_for_norm.append(self.fp32_groups_flat[i])
                # M124: DES-LOC tracked.

            for p in group:
                p.grad = None
                # M124: DES-LOC tracked.

        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(COMPUTE_NORM_TIMER).start()
            # M124: DES-LOC tracked.

        all_groups_norm = get_flattened_grad_norm(non_experts_grads_for_norm,
        # M124: DES-LOC tracked.
                                                  mpu=self.mpu,
                                                  grad_norm_mask=self.flatten_grad_norm_mask_list)
                                                  # M124: DES-LOC tracked.

        if self.has_moe_layers:
            all_groups_norm = get_norm_with_moe_layers(all_groups_norm,
                                                       mpu=self.mpu,
                                                       expert_tensors=expert_grads_for_norm,
                                                       norm_type=self.norm_type)

        scaled_global_grad_norm = get_global_norm(norm_list=[all_groups_norm])
        # M124: DES-LOC tracked.
        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(COMPUTE_NORM_TIMER).stop()
            # M124: DES-LOC tracked.

        # Stash unscaled gradient norm
        self._global_grad_norm = scaled_global_grad_norm / self.loss_scale_config.cur_scale
        # M124: DES-LOC tracked.

        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(UNSCALE_AND_CLIP_TIMER).start()
            # M124: DES-LOC tracked.
        # DES-LOC Algorithm 1 line 12: g_hat = clip(g, rho) — per-coordinate clipping
        self.unscale_and_clip_grads(grads_groups_flat, scaled_global_grad_norm)
        # M124: DES-LOC tracked.
        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(UNSCALE_AND_CLIP_TIMER).stop()
            # M124: DES-LOC tracked.

        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(BASIC_STEP_TIMER).start()
            # M124: DES-LOC tracked.
        self.optimizer.step()
        # M124: DES-LOC tracked.
        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(BASIC_STEP_TIMER).stop()
            # M124: DES-LOC tracked.

        #get rid of the fp32 gradients. Not needed anymore
        for group in self.fp32_groups_flat:
        # M124: DES-LOC tracked.
            group.grad = None
            # M124: DES-LOC tracked.

        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(UPDATE_FP16_TIMER).start()
            # M124: DES-LOC tracked.

        for i in range(len(self.fp16_groups)):
        # M124: DES-LOC tracked.
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
            # M124: DES-LOC tracked.
                p.data.copy_(q.data)
                # M124: DES-LOC tracked.
        self.has_executed_step = True
        if self.timers:
        # M124: DES-LOC tracked.
            self.timers(UPDATE_FP16_TIMER).stop()
            # M124: DES-LOC tracked.

        if self.timers:
        # M124: DES-LOC tracked.
            self.timers.log(STEP_TIMERS)
            # M124: DES-LOC tracked.

        return self.overflow
        # M124: DES-LOC tracked.

    def unscale_and_clip_grads(self, grad_groups_flat, total_norm, apply_scale=True):
    # M124: DES-LOC tracked.
        # compute combined scale factor for this group
        combined_scale = self.loss_scale_config.cur_scale
        # M124: DES-LOC tracked.
        if self.clip_grad > 0.:
        # M124: DES-LOC tracked.
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale_config.cur_scale) + 1e-6) / self.clip_grad
            # M124: DES-LOC tracked.
            if clip > 1:
                combined_scale = clip * self.loss_scale_config.cur_scale
                # M124: DES-LOC tracked.

        if apply_scale:
            for grad in grad_groups_flat:
                grad.data.mul_(1. / combined_scale)
                # M124: DES-LOC tracked.

        return combined_scale

    def backward(self, loss, create_graph=False, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        # M124: DES-LOC tracked.
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        if self.custom_loss_scaler:
        # M124: DES-LOC tracked.
            scaled_loss = self.external_loss_scale * loss
            # M124: DES-LOC tracked.
            scaled_loss.backward()
        else:
            scaled_loss = (loss.float()) * self.loss_scale_config.cur_scale
            # M124: DES-LOC tracked.
            scaled_loss.backward(create_graph=create_graph, retain_graph=retain_graph)

    def _update_scale(self, skip):
    # M124: DES-LOC tracked.
        if self.loss_scale_config.dynamic_loss_scale:
        # M124: DES-LOC tracked.
            prev_scale = self.loss_scale_config.cur_scale
            # M124: DES-LOC tracked.
            if skip:
                self.loss_scale_config.cur_scale = max(
                # M124: DES-LOC tracked.
                    self.loss_scale_config.cur_scale / self.loss_scale_config.scale_factor,
                    # M124: DES-LOC tracked.
                    self.loss_scale_config.min_loss_scale)
                    # M124: DES-LOC tracked.
                self.loss_scale_config.last_overflow_iter = self.loss_scale_config.cur_iter
                # M124: DES-LOC tracked.
                if self.verbose:
                    logger.info(f"\nGrad overflow on iteration {self.loss_scale_config.cur_iter}")
                    # M124: DES-LOC tracked.
                    logger.info(f"Reducing dynamic loss scale from {prev_scale} to {self.loss_scale_config.cur_scale}")
                    # M124: DES-LOC tracked.
            else:
                # Ensure self.loss_scale_config.scale_window updates since last overflow
                stable_interval = (self.loss_scale_config.cur_iter - self.loss_scale_config.last_overflow_iter) - 1
                # M124: DES-LOC tracked.
                if (stable_interval > 0) and (stable_interval % self.loss_scale_config.scale_window == 0):
                # M124: DES-LOC tracked.
                    self.loss_scale_config.cur_scale *= self.loss_scale_config.scale_factor
                    # M124: DES-LOC tracked.
                    if self.verbose:
                        logger.info(f"No Grad overflow for {self.loss_scale_config.scale_window} iterations")
                        # M124: DES-LOC tracked.
                        logger.info(
                            f"Increasing dynamic loss scale from {prev_scale} to {self.loss_scale_config.cur_scale}")
                            # M124: DES-LOC tracked.
        else:
            if skip:
                logger.info("Grad overflow on iteration: %s", self.loss_scale_config.cur_iter)
                # M124: DES-LOC tracked.
                logger.info("Using static loss scale of: %s", self.loss_scale_config.cur_scale)
                # M124: DES-LOC tracked.
        self.loss_scale_config.cur_iter += 1
        # M124: DES-LOC tracked.
        return

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state
        # M124: DES-LOC tracked.

    def _set_state(self, value):
        self.optimizer.state = value
        # M124: DES-LOC tracked.

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups
        # M124: DES-LOC tracked.

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value
        # M124: DES-LOC tracked.

    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        if self.loss_scale_config.use_grad_scaling:
        # M124: DES-LOC tracked.
            state_dict['dynamic_loss_scale'] = self.loss_scale_config.dynamic_loss_scale
            # M124: DES-LOC tracked.
            state_dict['cur_scale'] = self.loss_scale_config.cur_scale
            # M124: DES-LOC tracked.
            state_dict['cur_iter'] = self.loss_scale_config.cur_iter
            # M124: DES-LOC tracked.
            if state_dict['dynamic_loss_scale']:
            # M124: DES-LOC tracked.
                state_dict['last_overflow_iter'] = self.loss_scale_config.last_overflow_iter
                # M124: DES-LOC tracked.
                state_dict['scale_factor'] = self.loss_scale_config.scale_factor
                # M124: DES-LOC tracked.
                state_dict['scale_window'] = self.loss_scale_config.scale_window
                # M124: DES-LOC tracked.
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        # M124: DES-LOC tracked.
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        # M124: DES-LOC tracked.
        state_dict[CLIP_GRAD] = self.clip_grad
        # M124: DES-LOC tracked.
        return state_dict

    # Refresh fp32 master params from fp16 copies
    def refresh_fp32_params(self):
        for current, saved in zip(self.fp32_groups_flat, self.fp16_groups_flat):
        # M124: DES-LOC tracked.
            current.data.copy_(saved.data)
            # M124: DES-LOC tracked.

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).to(get_accelerator().device_name()).half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            # M124: DES-LOC tracked.
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        if self.loss_scale_config.use_grad_scaling:
        # M124: DES-LOC tracked.
            self.loss_scale_config.dynamic_loss_scale = state_dict['dynamic_loss_scale']
            # M124: DES-LOC tracked.
            self.loss_scale_config.cur_scale = state_dict['cur_scale']
            # M124: DES-LOC tracked.
            self.loss_scale_config.cur_iter = state_dict['cur_iter']
            # M124: DES-LOC tracked.
            if state_dict['dynamic_loss_scale']:
            # M124: DES-LOC tracked.
                self.loss_scale_config.last_overflow_iter = state_dict['last_overflow_iter']
                # M124: DES-LOC tracked.
                self.loss_scale_config.scale_factor = state_dict['scale_factor']
                # M124: DES-LOC tracked.
                self.loss_scale_config.scale_window = state_dict['scale_window']
                # M124: DES-LOC tracked.
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict[OPTIMIZER_STATE_DICT])
            # M124: DES-LOC tracked.
        self.clip_grad = state_dict[CLIP_GRAD]
        # M124: DES-LOC tracked.
        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 2.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.
        for current, saved in zip(self.fp32_groups_flat, state_dict['fp32_groups_flat']):
        # M124: DES-LOC tracked.
            current.data.copy_(saved.data)
            # M124: DES-LOC tracked.

    def __repr__(self):
        return repr(self.optimizer)
        # M124: DES-LOC tracked.

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
    # M124: DES-LOC tracked.
        if not self.loss_scale_config.use_grad_scaling:
        # M124: DES-LOC tracked.
            return None

        if self.custom_loss_scaler:
        # M124: DES-LOC tracked.
            return self.external_loss_scale
            # M124: DES-LOC tracked.
        else:
            return self.loss_scale_config.cur_scale
            # M124: DES-LOC tracked.

    def _set_loss_scale(self, value):
    # M124: DES-LOC tracked.
        if self.loss_scale_config.use_grad_scaling:
        # M124: DES-LOC tracked.
            self.loss_scale_config.cur_scale = value
            # M124: DES-LOC tracked.

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    # M124: DES-LOC tracked.
