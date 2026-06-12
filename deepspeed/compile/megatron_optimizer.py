# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1141: Megatron 91f3579ef — cleanup.
# Source: megatron/optimizer/distrib_optimizer.py (NVIDIA/Megatron-LM commit 91f3579ef)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-03-24
#
# Mapping: megatron/optimizer/distrib_optimizer.py → deepspeed/compile/megatron_optimizer.py
#
# Changes ported from distrib_optimizer.py (diff vs parent):
#   1. __init__(): removed >>> / <<< markers around
#      assert use_contiguous_buffers_in_local_ddp; inlined from args.
#   2. _collect_main_grad_data_for_unscaling(): simplified to iterate
#      optimizer.param_groups directly; removed old get_main_grad() helper path.
#   3. Added _get_model_and_main_params_data_float16() to
#      Float16DistributedOptimizer (upstream adds it to DistributedOptimizer;
#      maps identically here).
#   (Other upstream changes — removing commented-out class declarations,
#   pax/print_seq blocks, dead >>> / <<< wrappers — were already clean in
#   this file from prior migrations.)
# ---------------------------------------------------------------------------

print('[M1141]')

# ---------------------------------------------------------------------------
# M1111: Megatron 2c1660e76 — cleaned distrib_optimizer.py.
# Source: megatron/optimizer/distrib_optimizer.py (NVIDIA/Megatron-LM commit 2c1660e76)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-03-14
#
# Mapping: megatron/optimizer/distrib_optimizer.py → deepspeed/compile/megatron_optimizer.py
#
# Changes ported from distrib_optimizer.py (diff vs parent):
#   1. Removed `from lutil import pax, tp` / DEBUG_ITERATION debug block at top.
#   2. Removed commented-out class declarations above DistributedOptimizer.
#   3. allocate_main_param_shards(): removed >>> / <<< markers and commented
#      torch.zeros alternative.
#   4. state_dict(): removed commented-out pax block; removed commented
#      params key; kept state_dict['groups'] line.
#   5. load_state_dict(): removed pax debug blocks; simplified copy loop.
#   6. zero_grad(): added "# Collect model params." comment; changed comment
#      from `** using contiguous buffer; don't set_to_none **` to
#      `Distributed optimizer requires contiguous buffer; don't set to None.`;
#      force set_to_none=False; removed commented alternative call.
#   7. gather_model_params(): removed ITERATION parameter.
#   8. _copy_model_grads_to_main_grads(): removed ITERATION parameter.
#   9. _copy_main_params_to_model_params(): removed ITERATION parameter;
#      removed post-copy isnan/pax debug block.
#   (In this file Float16DistributedOptimizer maps to DistributedOptimizer;
#   stub raise-Exception methods replaced with real state_dict/load_state_dict.)
# ---------------------------------------------------------------------------

print('[M1111]')

# ---------------------------------------------------------------------------
# M1098: Megatron 862d70fce — small fixes.
# Source: megatron/optimizer/optimizer.py + megatron/model/distributed.py +
#         megatron/training.py (NVIDIA/Megatron-LM commit 862d70fce)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-03-10
#
# Mapping: megatron/optimizer/optimizer.py → deepspeed/compile/megatron_optimizer.py
#          megatron/model/distributed.py   → (no equivalent in this repo)
#          megatron/training.py            → deepspeed/compile/megatron_training.py
#
# Changes in upstream 862d70fce:
#   optimizer.py:
#     1. DEBUG_ITERATION: 2 → 1
#     2. MixedPrecisionOptimizer.allreduce_embedding_grads(): added commented-out
#        `# return # ** .. TEMPORARY .. **` block (>>> / <<<).
#     3. MixedPrecisionOptimizer.debug_base/debug_model/debug_main(): uncommented
#        from block comment; debug_base prefix changed from "            + " to "".
#     4. MixedPrecisionOptimizer.step(): added commented-out debug_model/debug_main
#        calls after _copy_main_params_to_model_params() (>>> / <<<).
#   model/distributed.py:
#     1. Removed commented-out `from collections import defaultdict` and
#        `_grad_buffer_param_offsets` lines.
#   training.py:
#     1. In train_step(): debug_model call before reduce_grads changed from
#        "before reduce grads.", 0 → "before reduce grads.", 1 (still commented).
#     2. In train_step(): debug_model call "after gather params." uncommented.
#
# DeepSpeed adaptation:
#   MixedPrecisionOptimizer, DEBUG_ITERATION, allreduce_embedding_grads,
#   debug_base/debug_model/debug_main, and the training.py call sites do not
#   exist in this file or megatron_training.py; changes documented here only.
#   model/distributed.py has no equivalent; skipped.
# ---------------------------------------------------------------------------

print('[M1098]')

# ---------------------------------------------------------------------------
# M1062: Megatron c13c0a3e8 — debugging; localized issue to gather_params()
# Source: megatron/optimizer/optimizer.py (NVIDIA/Megatron-LM commit c13c0a3e8)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-22
#
# Mapping: megatron/optimizer/optimizer.py → deepspeed/compile/megatron_optimizer.py
#
# Changes ported from optimizer.py (diff vs parent c13c0a3e8):
#   1. BaseFloat16Optimizer._unscale_main_grads_and_check_for_nan():
#      all_reduce changed to use no explicit group (was get_model_parallel_group());
#      old group= form commented out with >>> / <<< markers.
#   2. BaseFloat16Optimizer.step(): signature changed to step(self, ITERATION).
#      _copy_model_grads_to_main_grads() and _copy_main_params_to_model_params()
#      now receive ITERATION.  found_inf_flag branch activates pax(0, {...}).
#      Post-copy-back commented pax block added.
#   3. Float16DistributedOptimizer.__init__(): _copy_model_params_to_main_params()
#      moved to after optimizer.load_state_dict() (was before).
#   4. Float16DistributedOptimizer: added has_nan_debug(), get_local_model_param_views(),
#      get_local_model_grad_views(), get_world_model_params(), get_main_params(),
#      get_main_grads(); get_main_param() refactored to call get_main_params()[].
#   5. Float16DistributedOptimizer._collect_main_grad_data_for_unscaling():
#      simplified to use get_main_grads().
#   6. Float16DistributedOptimizer._copy_model_grads_to_main_grads(ITERATION):
#      NaN detection block added (has_nan_debug → raise Exception on main NaN).
#   7. Float16DistributedOptimizer._copy_main_params_to_model_params(ITERATION):
#      isnan → not isfinite; post-copy commented pax block added.
#   8. gather_params(): active pax() call added inside _gather_params
#      (training.py changes tracked in megatron_training.py M1062 section).
#
# DeepSpeed adaptation: Float16DistributedOptimizer does not exist in this file;
# changes for BaseFloat16Optimizer / Float16OptimizerWithFloat16Params applied
# to the analogous Float16OptimizerWithFloat16Params class herein.
# ---------------------------------------------------------------------------

print('[M1062]')

# ---------------------------------------------------------------------------
# M1110: Megatron efa3cbcf0 — partially cleaned optimizer.py.
# Source: megatron/optimizer/optimizer.py (NVIDIA/Megatron-LM commit efa3cbcf0)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-03-14
#
# Mapping: megatron/optimizer/optimizer.py → deepspeed/compile/megatron_optimizer.py
#
# Changes ported from optimizer.py (diff vs parent):
#   1. Removed top-level debug imports: `from lutil import pax, tp` and
#      `DEBUG_ITERATION = 1`.
#   2. clip_grad_norm(self, clip_grad, ITERATION) → clip_grad_norm(self, clip_grad).
#   3. gather_model_params(self, args, timers, ITERATION) →
#      gather_model_params(self, args, timers).
#   4. Removed commented-out `# return` in allreduce_embedding_grads.
#   5. Added blank line after `self.grad_scaler = grad_scaler`; added inline
#      comment `# None grad scaler is only supported for bf16.`.
#   6. Removed `# pax(1, {"main_grads": ...})` comment in
#      _unscale_main_grads_and_check_for_nan.
#   7. Removed large commented pax block after found_inf_flag check.
#   8. step(self, args, timers, ITERATION) → step(self, args, timers).
#   9. _copy_model_grads_to_main_grads(self, ITERATION) →
#      _copy_model_grads_to_main_grads(self); removed NaN debug block.
#  10. _copy_main_params_to_model_params(self, ITERATION) →
#      _copy_main_params_to_model_params(self); removed isnan check block.
#  11. Removed debug() closure and all >>> / <<< commented debug blocks in
#      __init__ (pax introspection, optimizer state dumps).
#  12. Removed active `pax(0, {"param": param})` in fp32 branch of __init__.
#  13. Removed class rename comments for BaseFloat16Optimizer /
#      Float16OptimizerWithFloat16Params.
#  14. FP32Optimizer.step(self, args, timers, ITERATION) →
#      step(self, args, timers); removed ITERATION from clip_grad_norm call.
# ---------------------------------------------------------------------------

print('[M1110]')

# ---------------------------------------------------------------------------
# M1013: Megatron 7dc8c4759 — feb 9 alpha
# Source: megatron/optimizer/optimizer.py + megatron/optimizer/grad_scaler.py
#         (NVIDIA/Megatron-LM commit 7dc8c4759)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-09
#
# M1061: Megatron 9b7854e4b — more cleanup of main params/grads
# Source: megatron/optimizer/optimizer.py
#         (NVIDIA/Megatron-LM commit 9b7854e4b)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-22
#
# Mapping: megatron/optimizer/optimizer.py → deepspeed/compile/megatron_optimizer.py
#          megatron/optimizer/grad_scaler.py → (inlined here)
#          (project convention: megatron top-level → deepspeed/compile/)
#
# M1061 Changes (Float16DistributedOptimizer, first migration of this class):
#   1. Added get_main_param(group_index) and get_main_grad(group_index) helper
#      methods to Float16DistributedOptimizer (after optimizer.load_state_dict()).
#   2. _collect_main_grad_data_for_unscaling(): replaced direct
#      main_param_shards list access with get_main_grad(gi) call.
#   3. _copy_model_params_to_main_params(): replaced direct
#      optimizer.param_groups[group_index] access with get_main_param(group_index);
#      commented out pax(0,...) debug call.
#   4. _copy_model_grads_to_main_grads(): replaced main_param_shards[group_index].grad
#      with get_main_grad(group_index).
#   5. _copy_main_params_to_model_params(): replaced main_param_shards[group_index]
#      with get_main_param(group_index).
#
# Also adds (as prerequisites not previously in this file):
#   - import math, from megatron import get_args
#   - Shard helper class
#   - BaseFloat16Optimizer base class
#   - Float16DistributedOptimizer full class
#
# M1013 Changes ported from optimizer.py (diff vs parent):
#   1. Float16OptimizerWithFloat16Params.__init__(): after setting
#      param_group['params'][i] = main_param, added debug() closure with
#      lutil pax introspection of optimizer param_groups and param hashes,
#      plus two commented-out debug() call sites (>>> / <<< markers).
#   2. Float16OptimizerWithFloat16Params.__init__(): in the fp32 branch,
#      added active pax(0, {"param": param}) call (>>> / <<< markers).
#   3. Float16OptimizerWithFloat16Params.__init__(): after
#      optimizer.load_state_dict(), added large commented-out pax block
#      inspecting float16/fp32 groups and optimizer param_groups
#      (>>> / <<< markers).
#   4. Float16OptimizerWithFloat16Params.step(): after optimizer.step(),
#      added commented-out pax block inspecting optimizer state
#      (>>> / <<< markers).
# ---------------------------------------------------------------------------

print('[M1013]')
print('[M1061]')


from abc import ABC
from abc import abstractmethod

import torch

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from megatron import get_timers
from megatron import mpu
from megatron import print_rank_0

from .clip_grads import clip_grad_norm_fp32, count_zeros_fp32


# ---------------------------------------------------------------------------
# Grad scalers (inlined from megatron/optimizer/grad_scaler.py)
# ---------------------------------------------------------------------------

class MegatronGradScaler(ABC):

    def __init__(self, initial_scale):
        """Initialize scale value with the input initial scale."""
        assert initial_scale > 0.0
        self._scale = torch.cuda.FloatTensor([initial_scale])

    @property
    def scale(self):
        return self._scale

    @property
    def inv_scale(self):
        return self._scale.double().reciprocal().float()

    @abstractmethod
    def update(self, found_inf):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass


class ConstantGradScaler(MegatronGradScaler):

    def update(self, found_inf):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass


class DynamicGradScaler(MegatronGradScaler):

    def __init__(self, initial_scale, min_scale, growth_factor, backoff_factor, growth_interval,
                 hysteresis):
        """"Grad scaler with dynamic scale that gets adjusted during training."""
        super(DynamicGradScaler, self).__init__(initial_scale)

        assert min_scale > 0.0
        assert min_scale <= initial_scale
        self.min_scale = torch.cuda.FloatTensor([min_scale])
        assert growth_factor > 1.0
        self.growth_factor = torch.cuda.FloatTensor([growth_factor])
        assert backoff_factor < 1.0
        assert backoff_factor > 0.0
        self.backoff_factor = torch.cuda.FloatTensor([backoff_factor])
        assert growth_interval > 0
        self.growth_interval = growth_interval
        assert hysteresis > 0
        self.hysteresis = hysteresis

        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis

    def update(self, found_inf):
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor, self.min_scale)
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self.growth_interval:
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                self._scale = self._scale * self.growth_factor

    def state_dict(self):
        state_dict = {}
        state_dict['scale'] = self._scale
        state_dict['growth_tracker'] = self._growth_tracker
        state_dict['hysteresis_tracker'] = self._hysteresis_tracker
        return state_dict

    def load_state_dict(self, state_dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']


# ---------------------------------------------------------------------------
# Optimizer utilities
# ---------------------------------------------------------------------------

def _zero_grad_group_helper(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(this, that, overflow_buf=None):
    """Use multi-tensor-applier to copy values from one list to another.
    We don't have a blfoat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16."""
    if overflow_buf:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


# ---------------------------------------------------------------------------
# Base optimizer
# ---------------------------------------------------------------------------

class MegatronOptimizer(ABC):

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp):

        """Input optimizer is the base optimizer for example Adam."""
        self.optimizer = optimizer
        assert self.optimizer, 'no optimizer is provided.'
        self.clip_grad = clip_grad
        self.log_num_zeros_in_grad = log_num_zeros_in_grad
        self.params_have_main_grad = params_have_main_grad
        self.use_contiguous_buffers_in_local_ddp = use_contiguous_buffers_in_local_ddp

        if self.use_contiguous_buffers_in_local_ddp:
            assert self.params_have_main_grad, \
                "use of contiguous buffer requires that params have main grad"

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        return clip_grad_norm_fp32(params, clip_grad)

    def count_zeros(self):
        params = self.get_parameters()
        return count_zeros_fp32(params)

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        pass

    @abstractmethod
    def get_loss_scale(self):
        """The output should be a cuda tensor of size 1."""
        pass

    def scale_loss(self, loss):
        """Simple scaling."""
        return self.get_loss_scale() * loss

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reload_model_params(self):
        """Refreshes any internal state from the current model parameters.
        Call whenever the parameters are changed outside of the optimizer.
        For example, when we load a model from a checkpoint  without loading
        the optimizer, the model parameters are updated but for fp16 optimizer
        with main parameters, the main parameters need to also be updated."""
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)


# ---------------------------------------------------------------------------
# Float16 optimizer
# ---------------------------------------------------------------------------

class Float16OptimizerWithFloat16Params(MegatronOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
    """

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp, bf16, grad_scaler):

        super(Float16OptimizerWithFloat16Params, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
            use_contiguous_buffers_in_local_ddp)

        self.bf16 = bf16
        self.grad_scaler = grad_scaler

        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, 'fp16 expects a grad scaler.'

        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])

        # Three groups of parameters:
        #   float16_groups: original float16 parameters
        #   fp32_from_float16_groups: fp32 copy of float16 parameters
        #   fp32_from_fp32_groups: original fp32 parameters
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:

                    # float16 params:
                    if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                        float16_params_this_group.append(param)
                        # Create a copy
                        main_param = param.detach().clone().float()
                        # Copy tensor model parallel attributes.
                        mpu.copy_tensor_model_parallel_attributes(main_param, param)
                        if hasattr(param, 'shared'):
                            main_param.shared = param.shared
<

                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param

                    else:
                        raise TypeError('Wrapped parameters must be one of '
                                        'torch.cuda.FloatTensor,  '
                                        'torch.cuda.HalfTensor, or '
                                        'torch.cuda.BFloat16Tensor. '
                                        'Received {}'.format(param.type()))

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def zero_grad(self, set_to_none=True):
        """We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point."""
        for group in self.float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            _zero_grad_group_helper(group, set_to_none)

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def get_main_params(self):
        return [ p for g in self.fp32_from_float16_groups for p in g ] + \
               [ p for g in self.fp32_from_fp32_groups for p in g ]
    def get_main_grads(self):
        return [ p.grad for p in self.get_main_params() ]

    def _copy_model_grads_to_main_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if self.params_have_main_grad and hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None
                if self.params_have_main_grad and \
                   not self.use_contiguous_buffers_in_local_ddp:
                    model_param.main_grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        if self.params_have_main_grad:
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad

                    if not self.use_contiguous_buffers_in_local_ddp:
                        model_param.main_grad = None

    def _unscale_main_grads_and_check_for_nan(self):
        main_grads = []
        for main_group in self.fp32_from_float16_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        for main_group in self.fp32_from_fp32_groups:
            for main_param in main_group:
                if main_param.grad is not None:
                    main_grads.append(main_param.grad.data)
        self.found_inf.fill_(0.0)
        torch._amp_foreach_non_finite_check_and_unscale_(main_grads, self.found_inf,
                                                         self.grad_scaler.inv_scale)
        # >>>
        # torch.distributed.all_reduce(self.found_inf,
        #                              op=torch.distributed.ReduceOp.MAX,
        #                              group=mpu.get_model_parallel_group())
        # +++
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX)
        # <<<

        found_inf_flag = (self.found_inf.item() > 0)
        return found_inf_flag

    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_main_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self, args, timers):

        timers('optimizer-copy-to-main-grad').start()
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        if self.grad_scaler:

            timers('optimizer-unscale-and-check-inf').start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        timers('optimizer-clip-main-grad').start()
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None

        # Step the optimizer.
        self.optimizer.step()

        timers('optimizer-copy-main-to-model-params').start()
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from ' 'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        if 'grad_scaler' not in state_dict:
            print_rank_0('***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                print_rank_0('***WARNING*** fould the grad scaler in the '
                             'checkpoint but it is None in the class. '
                             'Skipping loading grad scaler ...')

        fp32_from_float16_params_key = 'fp32_from_fp16_params'
        if fp32_from_float16_params_key not in state_dict:
            fp32_from_float16_params_key = 'fp32_from_fp16'
        for current_group, saved_group in zip(self.fp32_from_float16_groups,
                                              state_dict[fp32_from_float16_params_key]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


# ---------------------------------------------------------------------------
# FP32 optimizer
# ---------------------------------------------------------------------------

class FP32Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp):

        super(FP32Optimizer, self).__init__(optimizer, clip_grad, log_num_zeros_in_grad,
                                            params_have_main_grad,
                                            use_contiguous_buffers_in_local_ddp)

        self._scale = torch.cuda.FloatTensor([1.0])

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            _zero_grad_group_helper(group['params'], set_to_none)

    def get_loss_scale(self):
        """FP32 optimizer does not do any scaling."""
        return self._scale

    @torch.no_grad()
    def step(self):
        """Clip gradients (if needed) and step the base optimizer.
        Always return successful since there is no overflow."""

        if self.params_have_main_grad:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad

                    if not self.use_contiguous_buffers_in_local_ddp:
                        param.main_grad = None

        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None

        self.optimizer.step()

        return True, grad_norm, num_zeros_in_grad

    def reload_model_params(self):
        pass

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# M1061 additions: import math, get_args, Shard, BaseFloat16Optimizer,
#                  Float16DistributedOptimizer
# ---------------------------------------------------------------------------

import math

from megatron import get_args


# class ShardIndex:
class Shard:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start
    def normalize(self, start=0):
        return Shard(start, start + self.size)
    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)


class BaseFloat16Optimizer(MegatronOptimizer):

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler,
                 models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp)

        # >>>
        self.models = models
        # <<<
        self.bf16 = bf16
        self.grad_scaler = grad_scaler
        # None grad scaler is only supported for bf16.
        if self.grad_scaler is None:
            assert self.bf16, 'fp16 expects a grad scaler.'

        # Tensor used to determine if a nan/if has happend.
        # Any non-zero value indicates inf/nan.
        # Note that we keep this for the cases that grad scaler is none.
        # We still record nan/inf if we have a bfloat16 with a grad scaler.
        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        if bf16:
            self._dummy_overflow_buf = None
        else:
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # In case grad scaler is not passed, define the unity scale.
        if self.grad_scaler is None:
            self._scale_one = torch.cuda.FloatTensor([1.0])

    def get_loss_scale(self):
        if self.grad_scaler is None:
            return self._scale_one
        return self.grad_scaler.scale

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    def _unscale_main_grads_and_check_for_nan(self):

        # Collect main grads.
        main_grads = self._collect_main_grad_data_for_unscaling()

        # Reset found inf.
        self.found_inf.fill_(0.0)

        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(
            main_grads, self.found_inf, self.grad_scaler.inv_scale)

        # Update across all model parallel instances.
        torch.distributed.all_reduce(self.found_inf,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=mpu.get_model_parallel_group())

        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)

        return found_inf_flag

    @torch.no_grad()
    def step(self):

        timers = get_timers()

        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad').start()
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf').start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                return False, None, None

        # Clip the main gradients.
        timers('optimizer-clip-main-grad').start()
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        # count the zeros in the grads
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None

        # Step the optimizer.
        self.optimizer.step()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params').start()
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad


class Float16DistributedOptimizer(BaseFloat16Optimizer):

    @classmethod
    def get_model_gbuf_param_shard_map(cls, model, dtype, gbuf_world_shard):

        # Param shard map.
        param_world_index_map = model._grad_buffer_param_index_map[dtype]
        param_shard_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Shard range.
            param_world_start, param_world_end = param_world_indexes
            param_local_start = max(
                0,
                param_world_start - gbuf_world_shard.start)
            param_local_end = min(
                gbuf_world_shard.size,
                param_world_end - gbuf_world_shard.start)

            # Add shard, if within range.
            if param_local_end > param_local_start:
                param_local_shard = Shard(param_local_start, param_local_end)
                param_world_shard = param_local_shard.normalize(param_world_start)
                sub_param_start = max(0, gbuf_world_shard.start - param_world_start)
                sub_param_shard = param_local_shard.normalize(sub_param_start)
                param_shard_map[param] = {
                    "gbuf_world": param_world_shard,
                    "gbuf_local": param_local_shard,
                    "param": sub_param_shard,
                }

        return param_shard_map

    @classmethod
    def get_model_gbuf_shard(cls, model, dtype):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_world_size = mpu.get_data_parallel_world_size()

        # Grad buffer shard.
        grad_buffer = model._grad_buffers[dtype]
        gbuf_size = grad_buffer.numel
        max_gbuf_shard_size = int(math.ceil(gbuf_size / data_parallel_world_size))

        gbuf_world_all_shards = []
        for r in range(data_parallel_world_size):
            gbuf_world_start = r * max_gbuf_shard_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_shard_size)
            gbuf_world_shard = Shard(gbuf_world_start, gbuf_world_end)
            gbuf_world_all_shards.append(gbuf_world_shard)
        gbuf_world_shard = gbuf_world_all_shards[data_parallel_rank]

        # Param shards.
        param_shard_map = cls.get_model_gbuf_param_shard_map(model,
                                                              dtype,
                                                              gbuf_world_shard)

        # Altogether.
        data = {
            "local": gbuf_world_shard.normalize(),
            "world": gbuf_world_shard,
            "world_all": gbuf_world_all_shards,
            "param_map": param_shard_map,
        }

        return data

    @classmethod
    def get_model_gbuf_shard_map(cls, model):
        return {
            dtype: cls.get_model_gbuf_shard(model, dtype)
            for dtype in model._grad_buffers
        }

    @classmethod
    def get_param_gbuf_map(cls, model_gbuf_shards):

        param_gbuf_map = {}
        for model_index, model_gbuf_shard_map in enumerate(model_gbuf_shards):
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param, param_shard_map in gbuf_shard_map["param_map"].items():
                    param_gbuf_map[param] = (model_index, dtype)

        return param_gbuf_map

    @classmethod
    def get_optimizer_group_shards(cls, param_groups, model_gbuf_shards):

        num_groups = len(param_groups)

        # Param group map.
        param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                param_group_map[param] = group_index

        # Optimizer group shards.
        group_shards = [{"size": 0, "param_map": {}} for _ in param_groups]
        for model_gbuf_shard_map in model_gbuf_shards:
            for dtype, gbuf_shard_map in model_gbuf_shard_map.items():
                for param in gbuf_shard_map["param_map"]:

                    group_index = param_group_map[param]
                    group_shard = group_shards[group_index]
                    param_size = gbuf_shard_map["param_map"][param]["param"].size

                    param_group_start = group_shard["size"]
                    param_group_end = param_group_start + param_size
                    param_group_shard = Shard(param_group_start, param_group_end)

                    group_shard["size"] += param_size
                    group_shard["param_map"][param] = param_group_shard

        # Squeeze zero-size group shards.
        for group_index, group_shard in enumerate(group_shards):
            group_shard["orig_group"] = param_groups[group_index]
        group_shards = [g for g in group_shards if g["size"] > 0]

        return group_shards

    @classmethod
    def allocate_main_param_shards(cls, opt_group_shards):

        allocate_shard = lambda shard_size, dtype: torch.empty(
            (shard_size,),
            dtype=dtype,
            device=torch.cuda.current_device(),
            requires_grad=True)

        # main_param_shards = []
        for group_index, group_shard in enumerate(opt_group_shards):

            group_size = group_shard["size"]
            assert group_size != 0, "temporary check ... remove me."

            main_param = allocate_shard(group_size, torch.float)
            main_param.grad = allocate_shard(group_size, torch.float)
            mpu.set_tensor_model_parallel_attributes(main_param, True, 0, 1)

            # main_param_shards.append(main_param)
            group_shard["orig_group"]["params"] = [main_param]

        # return main_param_shards

    def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 bf16, grad_scaler, models):

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            bf16, grad_scaler, models)

        assert use_contiguous_buffers_in_local_ddp

        # Model grad buffer shards.
        self.model_gbuf_shards = []
        for model_index, model in enumerate(self.models):
            self.model_gbuf_shards.append(self.get_model_gbuf_shard_map(model))
        self.param_gbuf_map = self.get_param_gbuf_map(self.model_gbuf_shards)

        # Optimizer shards.
        self.opt_group_shards = self.get_optimizer_group_shards(
            self.optimizer.param_groups,
            self.model_gbuf_shards)

        # Allocate main param shards.
        # self.main_param_shards = \
        #     self.allocate_main_param_shards(self.opt_group_shards)
        self.allocate_main_param_shards(self.opt_group_shards)

        # Initialize main params.
        self._copy_model_params_to_main_params()

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [g["orig_group"] for g in self.opt_group_shards]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    # M1061: added get_main_param / get_main_grad helpers
    def get_main_param(self, group_index):
        return self.optimizer.param_groups[group_index]["params"][0]
    def get_main_grad(self, group_index):
        return self.get_main_param(group_index).grad

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        state_dict['groups'] = [g['params'] for g in self.optimizer.param_groups]
        return state_dict

    def load_state_dict(self, state_dict):
        optimizer_key = 'optimizer'
        if optimizer_key not in state_dict:
            optimizer_key = 'optimizer_state_dict'
            print_rank_0('***WARNING*** loading optimizer from '
                         'an old checkpoint ...')
        self.optimizer.load_state_dict(state_dict[optimizer_key])

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            print_rank_0('***WARNING*** found an old checkpoint, will not '
                         'load grad scaler ...')
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])

        # Copy data for the main params.
        current_groups = [g["params"] for g in self.optimizer.param_groups]
        assert "groups" in state_dict, "key 'groups' not in state_dict."
        for current_group, saved_group in zip(current_groups, state_dict["groups"]):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)

    def zero_grad(self, set_to_none=True):

        # Collect model params.
        model_params = []
        for model in self.models:
            for dtype, param_map in model._grad_buffer_param_index_map.items():
                model_params.extend(param_map.keys())

        # Distributed optimizer requires contiguous buffer; don't set to None.
        _zero_grad_group_helper(model_params, set_to_none = False)

    def get_model_grad_buffer_dp_views(self):

        # >>>
        # ** only contiguous grad buffer supported, for now [ TEMPORARY ] **
        args = get_args()
        assert args.use_contiguous_buffers_in_local_ddp
        # <<<

        # Grad buffer views.
        gbuf_view_items = []
        for model_index, model in enumerate(self.models):
            for dtype, gbuf_shard in self.model_gbuf_shards[model_index].items():
                world_shards = gbuf_shard["world_all"]

                gbuf = model._grad_buffers[dtype]
                gbuf_views = []
                for shard in world_shards:
                    gbuf_views.append(gbuf.data[shard.start:shard.end])

                gbuf_view_items.append((model_index, dtype, gbuf_views))

        return gbuf_view_items

    def reduce_gradients(self, model):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sync word embedding params.
        # ... todo ...

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sync T5 position embedding params.
        # ... todo ...

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reduce-scatter.
        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        for model_index, dtype, gbuf_views in gbuf_view_items:
            torch.distributed.reduce_scatter(
                gbuf_views[data_parallel_rank],
                gbuf_views,
                group=data_parallel_group,
            )

    def gather_params(self):

        data_parallel_rank = mpu.get_data_parallel_rank()
        data_parallel_group = mpu.get_data_parallel_group()

        gbuf_view_items = self.get_model_grad_buffer_dp_views()

        # All-gather updated main params.
        for model_index, dtype, gbuf_views in gbuf_view_items:
            torch.distributed.all_gather(
                gbuf_views,
                gbuf_views[data_parallel_rank],
                group=data_parallel_group,
            )

        # Each model param now contains its updated values in it's
        # '.main_grad' field.
        for param in self.param_gbuf_map:
            param.detach().copy_(param.main_grad)

    def _collect_main_grad_data_for_unscaling(self):
        return [
            param.grad.data
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]


    def _get_model_and_main_params_data_float16(self):
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.shard_float16_groups,
                                           self.shard_fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data


    # M1061: replaced direct param_groups access and pax() with get_main_param()
    def _copy_model_params_to_main_params(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            # main_param = self.main_param_shards[group_index]
            # main_param = self.optimizer.param_groups[group_index]["params"][0]
            main_param = self.get_main_param(group_index)
            # if group_index > 0:
            #     pax({"main_param": tp(main_param)})
            for model_param, main_shard in group_shard["param_map"].items():

                # Model shard.
                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["param"]

                assert main_shard.size == model_shard.size

                # Copy shard data.
                main_view = main_param[main_shard.start:main_shard.end]
                model_view = model_param.view(-1)[model_shard.start:model_shard.end]
                main_view.detach().copy_(model_view)

    def _copy_model_grads_to_main_grads(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for model_param, main_shard in group_shard["param_map"].items():

                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["gbuf_world"]

                assert main_shard.size == model_shard.size

                # Copy from DDP's contiguous buffer to main shard's grad.
                model_grad = self.models[model_index]._grad_buffers[dtype].data
                # main_grad = self.main_param_shards[group_index].grad
                # M1061: use get_main_grad()
                main_grad = self.get_main_grad(group_index)

                # Copy sub-range within tensor.
                model_view = model_grad[model_shard.start:model_shard.end]
                main_view = main_grad[main_shard.start:main_shard.end]

                main_view.detach().copy_(model_view)

    # M1061: replaced main_param_shards[group_index] with get_main_param()
    def _copy_main_params_to_model_params(self):

        for group_index, group_shard in enumerate(self.opt_group_shards):
            for model_param, main_shard in group_shard["param_map"].items():

                model_index, dtype = self.param_gbuf_map[model_param]
                model_shard = self.model_gbuf_shards \
                    [model_index][dtype]["param_map"][model_param]["gbuf_world"]

                assert main_shard.size == model_shard.size

                # Use DDP's contiguous buffer to temporarily hold params.
                model_param = self.models[model_index]._grad_buffers[dtype].data
                # main_param = self.main_param_shards[group_index]
                main_param = self.get_main_param(group_index)

                # Copy sub-range within tensor.
                model_view = model_param[model_shard.start:model_shard.end]
                main_view = main_param[main_shard.start:main_shard.end]

                model_view.detach().copy_(main_view)
