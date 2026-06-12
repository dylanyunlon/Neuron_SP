# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

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
# M1013: Megatron 7dc8c4759 — feb 9 alpha
# Source: megatron/optimizer/optimizer.py + megatron/optimizer/grad_scaler.py
#         (NVIDIA/Megatron-LM commit 7dc8c4759)
# Author: Lawrence McAfee <lmcafee@nvidia.com>  Date: 2022-02-09
#
# Mapping: megatron/optimizer/optimizer.py → deepspeed/compile/megatron_optimizer.py
#          megatron/optimizer/grad_scaler.py → (inlined here)
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from optimizer.py (diff vs parent):
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
                        # Replace the optimizer params with the new fp32 copy.
                        param_group['params'][i] = main_param
                        # >>>
                        def debug():
                            from lutil import pax, tp
                            pax(0, {
                                "optimizer": optimizer,
                                # "optimizer / state" : optimizer.state,
                                "optimizer / pg / 0": optimizer.param_groups[0]["params"],
                                "optimizer / pg / 1": optimizer.param_groups[1]["params"],
                                "param": tp(param),
                                "param / hash": hash(param),
                                "main_param": tp(main_param),
                                "main_param / hash": hash(main_param),
                            })
                        # <<<
                        # >>>
                        # debug()
                        # <<<
                        fp32_from_float16_params_this_group.append(main_param)
                        # Reset existing state dict key to the new main param.
                        if param in self.optimizer.state:
                            self.optimizer.state[main_param] \
                                = self.optimizer.state.pop(param)
                        # >>>
                        # debug()
                        # <<<

                    # fp32 params.
                    elif param.type() == 'torch.cuda.FloatTensor':
                        # >>>
                        from lutil import pax
                        pax(0, {"param": param})
                        # <<<
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

        # >>>
        # from lutil import pax
        # pax(0, {
        #     # "float16_groups / len" : [ len(g) for g in self.float16_groups ],
        #     # "fp32_from_float16_groups / len" :
        #     # [ len(g) for g in self.fp32_from_float16_groups ],
        #     # "float16_groups / 0" : self.float16_groups[0],
        #     # "float16_groups / 1" : self.float16_groups[1],
        #     # "fp32_from_float16_groups / 0" : self.fp32_from_float16_groups[0],
        #     # "fp32_from_float16_groups / 1" : self.fp32_from_float16_groups[1],
        #     # "fp32_from_float32_groups" : self.fp32_from_fp32_groups,
        #     "optimizer" : self.optimizer,
        #     # "optimizer / sd" : self.optimizer.state_dict(),
        #     # "optimizer / state" : self.optimizer.state_dict()["state"],
        #     # "optimizer / pg" : self.optimizer.state_dict()["param_groups"],
        #     # "optimizer / pg / 0" : self.optimizer.state_dict()["param_groups"][0],
        #     # "optimizer / pg / 1" : self.optimizer.state_dict()["param_groups"][1],
        #     "optimizer -> pg" : optimizer.param_groups,
        #     "optimizer -> pg / 0" : optimizer.param_groups[0]["params"],
        #     "optimizer -> pg / 1" : optimizer.param_groups[1]["params"],
        # })
        # <<<

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

    def _copy_model_grads_to_main_grads(self, ITERATION):
        # >>>
        main_grads = [ p.grad for g in self.fp32_from_float16_groups for p in g
                       if p.grad is not None ]
        if main_grads:
            import torch as _t
            main_has_nan = any(
                (not _t.all(_t.isfinite(g)).item()) for g in main_grads
            )
            if main_has_nan:
                raise Exception("hi.")
        # <<<

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

    def _copy_main_params_to_model_params(self, ITERATION):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=main_data, that=model_data,
                                        overflow_buf=self._dummy_overflow_buf)
        # >>>
        import torch as _t
        for param_data in model_data:
            is_nan = not _t.all(_t.isfinite(param_data)).item()
            if is_nan:
                pax({
                    "param" : param_data,
                    "is_nan" : is_nan,
                })
        # <<<

    def _copy_model_params_to_main_params(self):
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
        _multi_tensor_copy_this_to_that(this=model_data, that=main_data,
                                        overflow_buf=self._dummy_overflow_buf)

    def reload_model_params(self):
        self._copy_model_params_to_main_params()

    @torch.no_grad()
    def step(self, ITERATION):

        timers = get_timers()

        timers('optimizer-copy-to-main-grad').start()
        self._copy_model_grads_to_main_grads(ITERATION)
        timers('optimizer-copy-to-main-grad').stop()

        if self.grad_scaler:

            timers('optimizer-unscale-and-check-inf').start()
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                pax(0, {
                    "main params" : self.get_main_params(),
                    "main grads" : self.get_main_grads(),
                    "found_inf_flag" : found_inf_flag,
                })
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

        # >>>
        # pax(0, {
        #     "main params" : self.get_main_params(),
        #     "main grads" : self.get_main_grads(),
        # })
        # <<<

        timers('optimizer-copy-main-to-model-params').start()
        self._copy_main_params_to_model_params(ITERATION)
        timers('optimizer-copy-main-to-model-params').stop()

        # >>>
        # pax(1, {
        #     "ITERATION" : ITERATION,
        #     "model_params" : [ p for p in self.get_parameters() ],
        # })
        # <<<

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
