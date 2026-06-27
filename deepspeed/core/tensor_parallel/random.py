# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CUDA RNG state tracking and activation checkpointing for tensor parallelism.

Adapted from Megatron-LM megatron/core/tensor_parallel/random.py.
All megatron.core.* imports have been replaced with deepspeed.core.* equivalents.

Key classes/functions:
  * CudaRNGStatesTracker       — tracks named CUDA RNG states for TP/EP/DP
  * get_cuda_rng_tracker       — returns (and lazily initialises) the global tracker
  * model_parallel_cuda_manual_seed — seeds all RNG states after dist init
  * checkpoint                 — gradient-checkpoint wrapper that saves/restores
                                 CUDA RNG states across the TP group
  * CheckpointWithoutOutput    — zero-copy variant that discards outputs
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Optional, TypeVar, Union

import torch
from torch import _C
from torch.cuda import _lazy_call, _lazy_init
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

# ---------------------------------------------------------------------------
# Optional: TransformerEngine support (mirrors Megatron's approach)
# ---------------------------------------------------------------------------
try:
    import transformer_engine  # noqa: F401
    from transformer_engine.pytorch.distributed import activation_recompute_forward
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
    HAVE_TE = True
except ModuleNotFoundError:
    HAVE_TE = False


# ---------------------------------------------------------------------------
# Named RNG tracker constants
# ---------------------------------------------------------------------------
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
_EXPERT_PARALLEL_RNG_TRACKER_NAME = 'expert-parallel-rng'
_DATA_PARALLEL_RNG_TRACKER_NAME = 'data-parallel-rng'


def get_expert_parallel_rng_tracker_name() -> str:
    """Return the expert-parallel RNG tracker name."""
    return _EXPERT_PARALLEL_RNG_TRACKER_NAME


def get_data_parallel_rng_tracker_name() -> str:
    """Return the data-parallel RNG tracker name."""
    return _DATA_PARALLEL_RNG_TRACKER_NAME


# ---------------------------------------------------------------------------
# Low-level CUDA RNG state get/set
# ---------------------------------------------------------------------------

def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda",
    clone: bool = False,
    graph_safe: bool = False,
) -> torch.Tensor:
    """Return the RNG state of the specified GPU.

    When *graph_safe* is True, uses the CUDA-graph-safe Generator API instead
    of the standard Tensor-based state.
    """
    if not graph_safe:
        return torch.cuda.random.get_rng_state(device=device)

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()

    default_generator = torch.cuda.default_generators[idx]
    if clone:
        return default_generator.clone_state()
    return default_generator.graphsafe_get_state()


def _set_cuda_rng_state(
    new_state: torch.Tensor,
    device: int = -1,
    graph_safe: bool = False,
) -> None:
    """Set the RNG state of the current (or specified) GPU.

    Avoids cloning *new_state* to prevent major performance issues on ≥4-GPU runs.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        if device == -1:
            device_ = torch.device('cuda')
        elif isinstance(device, str):
            device_ = torch.device(device)
        else:
            device_ = torch.device('cuda', device)

        def cb():
            idx = device_.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def convert_cuda_rng_state(
    state: Union[torch.Tensor, torch.Generator],
    to_graphable: bool = False,
) -> Union[torch.Tensor, torch.Generator]:
    """Convert between Tensor-based and graph-safe Generator RNG states."""
    if to_graphable:
        if isinstance(state, torch.Tensor):
            orig = _get_cuda_rng_state(graph_safe=False)
            _set_cuda_rng_state(state, graph_safe=False)
            graphable = _get_cuda_rng_state(clone=True, graph_safe=True)
            _set_cuda_rng_state(orig, graph_safe=False)
            return graphable
        elif isinstance(state, torch.Generator):
            return state  # already graphable
        else:
            raise ValueError(f"Invalid state type: {type(state)}")
    else:
        if isinstance(state, torch.Tensor):
            return state  # already non-graphable
        elif isinstance(state, torch.Generator):
            return state.get_state()
        else:
            raise ValueError(f"Invalid state type: {type(state)}")


# ---------------------------------------------------------------------------
# CudaRNGStatesTracker
# ---------------------------------------------------------------------------

class CudaRNGStatesTracker:
    """Tracker for named CUDA RNG states.

    Each name maps to an independently-seeded CUDA RNG state.  The ``fork``
    context manager temporarily switches the active state, allowing
    per-layer / per-rank deterministic dropout without polluting the default
    data-parallel RNG stream.
    """

    def __init__(
        self,
        use_cudagraphable_rng: bool = False,
        is_inference_rng_tracker: bool = False,
    ) -> None:
        self.use_cudagraphable_rng = use_cudagraphable_rng
        self.is_inference_rng_tracker = is_inference_rng_tracker

        if self.use_cudagraphable_rng:
            assert (
                hasattr(torch.cuda.CUDAGraph, "register_generator_state")
                and hasattr(torch.Generator, "graphsafe_set_state")
                and hasattr(torch.Generator, "graphsafe_get_state")
                and hasattr(torch.Generator, "clone_state")
            ), "Tried using cudagraphs with RNG, however not detected in pytorch!"

        self.reset()

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #

    def is_initialized(self) -> bool:
        """Return True once set_states() has been called."""
        return self._is_initialized

    def reset(self) -> None:
        """Clear all tracked states."""
        self._is_initialized = False
        self.states_: dict = {}
        self.seeds_: set = set()
        self._current_state_name: str = "default-rng"

    def get_states(self) -> dict:
        """Return a shallow copy of the state dict."""
        return dict(self.states_)

    def set_states(self, states: dict) -> None:
        """Overwrite the state dict."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """Create and track a new named RNG state seeded with *seed*."""
        self._is_initialized = True
        if seed in self.seeds_:
            raise Exception(f"seed {seed} already exists")
        self.seeds_.add(seed)
        if name in self.states_:
            raise Exception(f"cuda rng state {name} already exists")

        if self.use_cudagraphable_rng:
            new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
        else:
            orig = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(seed)
            self.states_[name] = torch.cuda.get_rng_state()
            _set_cuda_rng_state(orig)

    @contextlib.contextmanager
    def fork(self, name: str = _MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork into named RNG state, restore previous state on exit."""
        if name not in self.states_:
            raise Exception(f"cuda rng state {name} is not added")

        orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        orig_state_name = self._current_state_name
        if orig_state_name != "default-rng":
            self.states_[orig_state_name] = orig_cuda_rng_state

        _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
        self._current_state_name = name
        cpu_rng_state = torch.get_rng_state()

        try:
            yield
        finally:
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning(
                    "CPU RNG state changed within GPU RNG context"
                )
            if self._current_state_name != name:
                raise Exception(
                    f"current state name {self._current_state_name} is not "
                    f"the same as the desired state name {name}."
                )
            self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
            if orig_state_name != "default-rng":
                orig_cuda_rng_state = self.states_[orig_state_name]
            _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)
            self._current_state_name = orig_state_name


# ---------------------------------------------------------------------------
# Global tracker singleton
# ---------------------------------------------------------------------------

_CUDA_RNG_STATE_TRACKER: Optional[CudaRNGStatesTracker] = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED: bool = False


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
) -> None:
    """Create (or reset) the global CUDA RNG tracker.

    Args:
        use_te_rng_tracker:    Use TransformerEngine's tracker when available.
        inference_rng_tracker: Create a no-op tracker for inference.
        use_cudagraphable_rng: Use CUDA-graph-safe RNG APIs.
        force_reset:           Destroy and recreate the tracker even if already set.
    """
    global _CUDA_RNG_STATE_TRACKER, _CUDA_RNG_STATE_TRACKER_INITIALIZED

    if force_reset:
        _CUDA_RNG_STATE_TRACKER = None
        _CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    base_tracker = None
    if HAVE_TE and use_te_rng_tracker:
        try:
            from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
            base_tracker = TECudaRNGStatesTracker
            tracker_kwargs: dict = {"is_inference_rng_tracker": inference_rng_tracker}
        except ImportError:
            pass  # Fall through to DeepSpeed implementation

    if base_tracker is None:
        base_tracker = CudaRNGStatesTracker
        tracker_kwargs = {
            "use_cudagraphable_rng": use_cudagraphable_rng,
            "is_inference_rng_tracker": inference_rng_tracker,
        }

    if inference_rng_tracker:
        class InferenceCudaRNGStatesTracker(base_tracker):  # type: ignore[valid-type, misc]
            """No-op RNG tracker for inference (no state needed)."""

            def add(self, name, seed):
                pass

            def set_states(self, states):
                pass

            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                return contextlib.nullcontext()

        tracker_class = InferenceCudaRNGStatesTracker
    else:
        tracker_class = base_tracker

    _CUDA_RNG_STATE_TRACKER = tracker_class(**tracker_kwargs)
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
) -> CudaRNGStatesTracker:
    """Return the global CUDA RNG tracker, initialising it if needed."""
    initialize_rng_tracker(use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    return _CUDA_RNG_STATE_TRACKER  # type: ignore[return-value]


def get_all_rng_states() -> dict:
    """Return all named generator states from the current tracker."""
    assert _CUDA_RNG_STATE_TRACKER_INITIALIZED, (
        "Tried getting all rng states but RNG Tracker has not been initialised!"
    )
    if isinstance(_CUDA_RNG_STATE_TRACKER, CudaRNGStatesTracker):
        return _CUDA_RNG_STATE_TRACKER.states_
    return {}


def is_graph_safe_cuda_rng_tracker(cuda_rng_tracker) -> bool:
    """Return True when *cuda_rng_tracker* uses graph-safe RNG APIs."""
    if HAVE_TE:
        try:
            from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
            if isinstance(cuda_rng_tracker, TECudaRNGStatesTracker):
                return True
        except ImportError:
            pass
    return getattr(cuda_rng_tracker, "use_cudagraphable_rng", False)


# ---------------------------------------------------------------------------
# model_parallel_cuda_manual_seed
# ---------------------------------------------------------------------------

def model_parallel_cuda_manual_seed(
    seed: int,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    tp_rank: Optional[int] = None,
    ep_rank: Optional[int] = None,
    etp_rank: Optional[int] = None,
    force_reset_rng: bool = False,
) -> None:
    """Initialise model-parallel CUDA seeds.

    Must be called after parallel_state is initialised.  Seeds three
    independent RNG streams:

    * data-parallel (default): same within a TP group, different across DP groups.
    * tensor-model-parallel:   different across the TP group, same across DP.
    * expert-parallel:         different across EP/ETP ranks.

    Args:
        seed:                 Base random seed.
        te_rng_tracker:       Use TransformerEngine RNG tracker.
        inference_rng_tracker: Use inference (no-op) tracker.
        use_cudagraphable_rng: Use graph-safe RNG APIs.
        tp_rank:              Override TP rank (auto-detected if None).
        ep_rank:              Override EP rank (auto-detected if None).
        etp_rank:             Override expert-TP rank (auto-detected if None).
        force_reset_rng:      Force tracker reset before seeding.
    """
    try:
        from deepspeed.core.parallel_state import get_tensor_model_parallel_rank
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
    except (ImportError, AssertionError):
        if tp_rank is None:
            tp_rank = 0

    if ep_rank is None:
        ep_rank = 0
    if etp_rank is None:
        etp_rank = 0

    offset = seed + 2718
    tensor_model_parallel_seed = offset + tp_rank
    data_parallel_seed = seed

    initialize_rng_tracker(
        te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng,
        force_reset=force_reset_rng,
    )
    _CUDA_RNG_STATE_TRACKER.reset()

    torch.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)

    expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)


# ---------------------------------------------------------------------------
# RNG state helpers for checkpointing
# ---------------------------------------------------------------------------

def _get_all_rng_states():
    """Return (cpu_state, cuda_state, tracker_states) tuple."""
    graph_safe = is_graph_safe_cuda_rng_tracker(get_cuda_rng_tracker())
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = _get_cuda_rng_state(graph_safe=graph_safe)
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    return cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker


def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
    """Restore (cpu_state, cuda_state, tracker_states) tuple."""
    torch.set_rng_state(cpu_rng_state)
    graph_safe = is_graph_safe_cuda_rng_tracker(get_cuda_rng_tracker())
    _set_cuda_rng_state(cuda_rng_state, graph_safe=graph_safe)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)


@contextlib.contextmanager
def _fork_rng():
    """Fork all RNG states, restoring them on exit."""
    current_states = _get_all_rng_states()
    try:
        yield
    finally:
        _set_all_rng_states(*current_states)


# ---------------------------------------------------------------------------
# Checkpointing flags
# ---------------------------------------------------------------------------

IS_CHECKPOINTING = False


def _set_checkpointing():
    global IS_CHECKPOINTING
    IS_CHECKPOINTING = True


def _unset_checkpointing():
    global IS_CHECKPOINTING
    IS_CHECKPOINTING = False


def is_checkpointing() -> bool:
    """Return True when currently inside a checkpoint context."""
    return IS_CHECKPOINTING


# ---------------------------------------------------------------------------
# CheckpointFunction
# ---------------------------------------------------------------------------

_R = TypeVar('_R')


class CheckpointFunction(torch.autograd.Function):
    """Activation checkpoint that saves/restores CUDA RNG states.

    Adapted from torch.utils.checkpoint with two changes:
    1. ``torch.cuda.set_rng_state`` → ``_set_cuda_rng_state``
    2. Model-parallel tracker states are also saved/restored.
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        distribute_saved_activations: bool,
        *args,
    ):
        """Forward pass: run without grad, save inputs + RNG states."""
        _set_checkpointing()
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations
        ctx.rng_states = _get_all_rng_states()

        with torch.no_grad():
            outputs = run_function(*args)

        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            # Shard first input across TP ranks to save activation memory
            from deepspeed.core.tensor_parallel.mappings import (
                split_tensor_into_1d_equal_chunks,
            )
            args[0].data = split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)

        ctx.save_for_backward(*args)
        _unset_checkpointing()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        """Backward pass: recompute forward with saved RNG states."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        _set_checkpointing()
        inputs = ctx.saved_tensors

        if ctx.distribute_saved_activations:
            from deepspeed.core.tensor_parallel.mappings import gather_split_1d_tensor
            inputs[0].data = gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)

        with _fork_rng():
            _set_all_rng_states(*ctx.rng_states)
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs, args = zip(
            *filter(
                lambda x: torch.is_tensor(x[0]) and x[0].requires_grad,
                zip(outputs, args),
            )
        )
        torch.autograd.backward(outputs, args)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else inp
            for inp in detached_inputs
        )
        _unset_checkpointing()
        return (None, None) + grads


def checkpoint(
    function: Callable,
    distribute_saved_activations: bool,
    *args,
):
    """Checkpoint *function* with saved/restored CUDA RNG states.

    Drop-in replacement for torch.utils.checkpoint.checkpoint that also
    handles the tensor-parallel RNG tracker.

    Args:
        function:                     The forward function to checkpoint.
        distribute_saved_activations: Shard first input across TP ranks.
        *args:                        Arguments forwarded to *function*.
    """
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)


# ---------------------------------------------------------------------------
# CheckpointWithoutOutput — zero-copy variant that discards forward outputs
# ---------------------------------------------------------------------------

class CheckpointWithoutOutputFunction(torch.autograd.Function):
    """Helper for CheckpointWithoutOutput: saves context for later recompute."""

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        checkpoint_obj: 'CheckpointWithoutOutput',
        *args,
    ):
        if checkpoint_obj.fp8 and HAVE_TE:
            fp8 = FP8GlobalStateManager.is_fp8_enabled()
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            fwd_ctx = activation_recompute_forward(
                activation_recompute=True, recompute_phase=False
            )
        else:
            ctx.fp8 = False
            ctx.fp8_recipe = None
            fwd_ctx = contextlib.nullcontext()

        with torch.no_grad(), fwd_ctx:
            outputs = run_function(*args)
        ctx.save_for_backward(*detach_variable(args))
        checkpoint_obj.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = ctx.inputs
        outputs = ctx.outputs
        torch.autograd.backward(outputs, args)
        ctx.outputs = None
        ctx.inputs = None
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else inp for inp in inputs
        )
        return (None, None) + grads


class CheckpointWithoutOutput:
    """Checkpoint that discards outputs to save activation memory.

    Usage::

        cwo = CheckpointWithoutOutput()
        out = cwo.checkpoint(fn, *inputs)
        # … use out in next layers …
        out.untyped_storage().resize_(0)  # or call discard_output_and_register_recompute
        cwo.discard_output_and_register_recompute(hook_tensor)

    The output storage is released immediately after forward; it is
    reconstructed via recomputation in backward.
    """

    def __init__(self, fp8: bool = False) -> None:
        self.fp8 = fp8 is not None
        self.run_function: Optional[Callable] = None
        self.rng_states = None
        self.ctx = None
        self.outputs = None

    def checkpoint(self, run_function: Callable, *args):
        """Run *run_function* under checkpoint (no-gradient forward)."""
        self.run_function = run_function
        self.rng_states = _get_all_rng_states()
        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
        return outputs

    def _recompute(self, _):
        """Hook called during backward to recompute forward outputs."""
        if self.ctx is None:
            return

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )

        with _fork_rng():
            _set_all_rng_states(*self.rng_states)

            if self.fp8 and HAVE_TE:
                recompute_ctx = activation_recompute_forward(
                    activation_recompute=True, recompute_phase=True
                )
                fp8_ctx = fp8_autocast(enabled=self.ctx.fp8, fp8_recipe=self.ctx.fp8_recipe)
            else:
                recompute_ctx = contextlib.nullcontext()
                fp8_ctx = contextlib.nullcontext()

            inputs = self.ctx.saved_tensors

            def _detach(t):
                if isinstance(t, torch.Tensor):
                    rg = t.requires_grad
                    t = t.detach()
                    t.requires_grad_(rg)
                return t

            inputs = tuple(_detach(t) for t in inputs)
            with torch.enable_grad(), fp8_ctx, recompute_ctx:
                outputs = self.run_function(*inputs)

        self.run_function = None
        self.rng_states = None

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Zero-copy: point saved output's storage to recomputed data.
        for output, recomputed in zip(self.outputs, outputs):
            try:
                # Best-effort zero-copy via storage pointer swap
                output.set_(recomputed.storage(), recomputed.storage_offset(),
                             recomputed.size(), recomputed.stride())
            except Exception:
                # Fallback: data copy
                output.data.copy_(recomputed.data)

        self.ctx.outputs = outputs
        self.ctx.inputs = inputs
        self.outputs = None
        self.ctx = None

    def discard_output_and_register_recompute(self, hook_tensor: torch.Tensor) -> None:
        """Resize output storages to zero and register backward recompute hook.

        Args:
            hook_tensor: Tensor whose backward trigger fires the recompute.
                         Must be computed before the recomputed outputs are needed.
        """
        if self.outputs is None:
            return
        for output in self.outputs:
            output.untyped_storage().resize_(0)
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)
