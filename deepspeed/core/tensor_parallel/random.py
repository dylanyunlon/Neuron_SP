# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CUDA RNG state tracking and activation checkpointing for tensor parallelism.

Adapted from Megatron-LM megatron/core/tensor_parallel/random.py.
All megatron.core.* imports have been replaced with deepspeed.core.* equivalents.

Key classes/functions:
  * CudaRNGStatesTracker         — tracks named CUDA RNG states for TP/EP/DP
  * get_cuda_rng_tracker         — returns (and lazily initialises) the global tracker
  * model_parallel_cuda_manual_seed — seeds all RNG states after dist init
  * checkpoint                   — gradient-checkpoint wrapper that saves/restores
                                   CUDA RNG states across the TP group
  * CheckpointWithoutOutput      — zero-copy variant that discards outputs

Design notes
------------
**Named RNG states**: Three independent streams are seeded:
  * ``data-parallel-rng``       — same within TP group, differs across DP.
  * ``model-parallel-rng``      — differs within TP group (per tp_rank offset).
  * ``expert-parallel-rng``     — differs across EP and expert-TP ranks.

**Graph-safe RNG**: When ``use_cudagraphable_rng=True``, torch.Generator objects
  are used instead of Tensor-based state so the RNG can be captured in a CUDA
  graph without the state tensor being frozen at capture time.

**Zero-copy storage sharing**: CheckpointWithoutOutput uses a C++ extension to
  point the recomputed output's storage at the already-allocated destination
  without a data copy, matching Megatron's approach.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, List, Optional, TypeVar, Union

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
# Optional: C++ zero-copy storage-sharing extension
# ---------------------------------------------------------------------------
_SHARE_STORAGE_SRC = r"""
#include <torch/extension.h>

void share_storage(at::Tensor dst, at::Tensor src) {
    auto* dst_impl = dst.storage().unsafeGetStorageImpl();

    // Copy src's c10::Storage (increments StorageImpl refcount).
    auto* src_storage_ref = new c10::Storage(src.storage());

    void*       data   = src_storage_ref->data_ptr().get();
    size_t      nbytes = src_storage_ref->nbytes();
    c10::Device device = src_storage_ref->device();

    // Build a DataPtr whose deleter releases our StorageImpl reference.
    c10::DataPtr shared(
        data,
        static_cast<void*>(src_storage_ref),
        [](void* ctx) { delete static_cast<c10::Storage*>(ctx); },
        device);

    dst_impl->set_data_ptr(std::move(shared));
    dst_impl->set_nbytes(nbytes);
}
"""

_share_storage_ext = None


def _get_share_storage():
    """Lazily compile & cache the zero-copy share_storage extension."""
    global _share_storage_ext
    if _share_storage_ext is None:
        try:
            from torch.utils.cpp_extension import load_inline
            _share_storage_ext = load_inline(
                name="share_storage_ext",
                cpp_sources=_SHARE_STORAGE_SRC,
                functions=["share_storage"],
                verbose=False,
            )
            return _share_storage_ext.share_storage
        except Exception:
            return None
    return _share_storage_ext.share_storage if _share_storage_ext else None


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

    Args:
        device:     GPU device (int index, str like "cuda:0", or torch.device).
        clone:      Whether to clone the retrieved state.
        graph_safe: If True, use CUDA-graph-safe Generator API instead of
                    the standard Tensor-based state.

    Returns:
        RNG state tensor (or Generator if graph_safe=True).
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

    Args:
        new_state:  New RNG state (Tensor or Generator).
        device:     GPU device index (-1 = current device).
        graph_safe: Use CUDA-graph-safe Generator API.
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
    """Convert between Tensor-based and graph-safe Generator RNG states.

    Args:
        state:        Current RNG state (Tensor or Generator).
        to_graphable: If True, convert to graph-safe Generator; else to Tensor.
    Returns:
        Converted state.
    """
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

    Thread-safety: Not thread-safe.  Each process uses a single tracker.

    Graph-safe mode: When ``use_cudagraphable_rng=True``, torch.Generator
    objects are used instead of Tensor-based state so the RNG can be captured
    in a CUDA graph.

    Args:
        use_cudagraphable_rng:   Use CUDA-graph-safe Generator API.
        is_inference_rng_tracker: If True, this is a no-op tracker for inference
                                  (``add`` / ``fork`` are effectively no-ops).
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
            ), (
                "Tried using cudagraphs with RNG, but the required PyTorch "
                "Generator graph-safe APIs are not available.  Please upgrade PyTorch."
            )

        self.reset()

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #

    def is_initialized(self) -> bool:
        """Return True once set_states() or add() has been called."""
        return self._is_initialized

    def reset(self) -> None:
        """Clear all tracked states and reset to uninitialised."""
        self._is_initialized = False
        self.states_: dict = {}
        self.seeds_: set = set()
        self._current_state_name: str = "default-rng"

    def get_states(self) -> dict:
        """Return a shallow copy of the state dict."""
        return dict(self.states_)

    def set_states(self, states: dict) -> None:
        """Overwrite the state dict with *states*.

        Args:
            states: Dict mapping tracker name → RNG state.
        """
        self._is_initialized = True
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """Create and track a new named RNG state seeded with *seed*.

        Args:
            name:  Unique tracker name (e.g. 'model-parallel-rng').
            seed:  Seed value.  Must be unique across all tracked states.
        Raises:
            Exception: If *seed* or *name* is already tracked.
        """
        self._is_initialized = True
        if seed in self.seeds_:
            raise Exception(
                f"Seed {seed} already exists in RNG tracker.  "
                f"Every tracked state must have a unique seed."
            )
        self.seeds_.add(seed)
        if name in self.states_:
            raise Exception(
                f"CUDA RNG state '{name}' already exists.  "
                f"Call reset() before re-adding states."
            )

        if self.use_cudagraphable_rng:
            # Graph-safe: use Generator object
            new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
        else:
            # Standard: save/restore to seed without changing current state
            orig = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(seed)
            self.states_[name] = torch.cuda.get_rng_state()
            _set_cuda_rng_state(orig)

    @contextlib.contextmanager
    def fork(self, name: str = _MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork into named RNG state, restoring the previous state on exit.

        Usage::

            with get_cuda_rng_tracker().fork():
                # operations here use the model-parallel RNG
                dropout_output = F.dropout(x, p=0.1)
            # original RNG state is restored here

        Args:
            name: Name of the RNG state to switch to.
        Raises:
            Exception: If *name* is not a tracked state.
        """
        if name not in self.states_:
            raise Exception(
                f"CUDA RNG state '{name}' is not tracked.  "
                f"Call model_parallel_cuda_manual_seed() first or add() manually."
            )

        orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        orig_state_name = self._current_state_name
        if orig_state_name != "default-rng":
            # Save the current non-default state before switching
            self.states_[orig_state_name] = orig_cuda_rng_state

        _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
        self._current_state_name = name
        cpu_rng_state = torch.get_rng_state()

        try:
            yield
        finally:
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning(
                    "CPU RNG state changed within GPU RNG context manager.  "
                    "This may cause non-deterministic behaviour."
                )
            if self._current_state_name != name:
                raise Exception(
                    f"RNG tracker: current state name '{self._current_state_name}' "
                    f"does not match expected '{name}'.  "
                    f"Nested fork() calls on the same name are not supported."
                )
            # Save the forked state so future forks see the advanced state
            self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
            # Restore original state
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

    Must be called before model_parallel_cuda_manual_seed().
    Calling multiple times without force_reset=True is a no-op.

    Args:
        use_te_rng_tracker:    Use TransformerEngine's RNG tracker when available.
                               Falls back to CudaRNGStatesTracker if TE not found.
        inference_rng_tracker: Create a no-op tracker for inference (add/fork no-ops).
        use_cudagraphable_rng: Use CUDA-graph-safe Generator API for the tracker.
        force_reset:           Destroy and recreate the tracker even if already set.
    """
    global _CUDA_RNG_STATE_TRACKER, _CUDA_RNG_STATE_TRACKER_INITIALIZED

    if force_reset:
        _CUDA_RNG_STATE_TRACKER = None
        _CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    base_tracker = None
    tracker_kwargs: dict = {}

    if HAVE_TE and use_te_rng_tracker:
        try:
            from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
            base_tracker = TECudaRNGStatesTracker
            tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
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
            """No-op RNG tracker for inference (no state management needed)."""

            def add(self, name: str, seed: int) -> None:  # type: ignore[override]
                pass

            def set_states(self, states: dict) -> None:  # type: ignore[override]
                pass

            def fork(self, name: str = _MODEL_PARALLEL_RNG_TRACKER_NAME):
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
    """Return the global CUDA RNG tracker, initialising it if needed.

    Args:
        use_te_rng_tracker:    Use TransformerEngine's tracker when available.
        inference_rng_tracker: Use inference (no-op) tracker.
        use_cudagraphable_rng: Use graph-safe RNG APIs.
    Returns:
        The global CudaRNGStatesTracker instance.
    """
    initialize_rng_tracker(use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    return _CUDA_RNG_STATE_TRACKER  # type: ignore[return-value]


def get_all_rng_states() -> dict:
    """Return all named generator states from the current tracker.

    Returns:
        Dict mapping tracker name → RNG state.
    Raises:
        AssertionError: If the tracker has not been initialised.
    """
    assert _CUDA_RNG_STATE_TRACKER_INITIALIZED, (
        "Tried getting all RNG states but the CUDA RNG Tracker has not been "
        "initialised.  Call model_parallel_cuda_manual_seed() first."
    )
    if isinstance(_CUDA_RNG_STATE_TRACKER, CudaRNGStatesTracker):
        return _CUDA_RNG_STATE_TRACKER.states_
    return {}


def is_graph_safe_cuda_rng_tracker(cuda_rng_tracker) -> bool:
    """Return True when *cuda_rng_tracker* uses graph-safe RNG APIs.

    Checks both TransformerEngine tracker (if available) and the
    ``use_cudagraphable_rng`` attribute on the tracker instance.
    """
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

    Must be called after parallel_state is initialised.  After this call,
    torch.cuda.manual_seed() should NOT be called (it would override the
    data-parallel state).

    Seeds three independent RNG streams:
    * **data-parallel** (default): same within a TP group, different across DP.
    * **tensor-model-parallel**: different across the TP group, same across DP.
    * **expert-parallel**: different across EP/ETP ranks.

    The formula for tensor-model-parallel seed is::

        tensor_mp_seed = seed + 2718 + tp_rank

    The formula for expert-parallel seed is::

        expert_seed = seed + 1024 + 100 * ep_rank + etp_rank

    Args:
        seed:                 Base random seed.
        te_rng_tracker:       Use TransformerEngine RNG tracker.
        inference_rng_tracker: Use inference (no-op) tracker.
        use_cudagraphable_rng: Use graph-safe RNG APIs.
        tp_rank:              Override TP rank (auto-detected from parallel_state if None).
        ep_rank:              Override EP rank (auto-detected from parallel_state if None).
        etp_rank:             Override expert-TP rank (auto-detected from parallel_state if None).
        force_reset_rng:      Force tracker reset before seeding.
    """
    # Auto-detect ranks from parallel_state (each may fail if not initialised)
    if tp_rank is None:
        try:
            from deepspeed.core.parallel_state import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
        except (ImportError, AssertionError, Exception):
            tp_rank = 0

    if ep_rank is None:
        try:
            from deepspeed.core.parallel_state import get_expert_model_parallel_rank
            ep_rank = get_expert_model_parallel_rank()
        except (ImportError, AssertionError, Exception):
            ep_rank = 0

    if etp_rank is None:
        try:
            from deepspeed.core.parallel_state import get_expert_tensor_parallel_rank
            etp_rank = get_expert_tensor_parallel_rank()
        except (ImportError, AssertionError, Exception):
            etp_rank = 0

    # 2718 is just for fun; any positive offset works to separate streams
    offset = seed + 2718
    tensor_model_parallel_seed = offset + tp_rank
    # Data parallel gets the original seed (same within TP group)
    data_parallel_seed = seed
    # Expert parallel: combine ep_rank and etp_rank for uniqueness
    expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank

    initialize_rng_tracker(
        te_rng_tracker,
        inference_rng_tracker,
        use_cudagraphable_rng,
        force_reset=force_reset_rng,
    )
    _CUDA_RNG_STATE_TRACKER.reset()

    # Set the default (data-parallel) state
    torch.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    # Set tensor-model-parallel state (unique per TP rank)
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)

    # Set expert-parallel state (unique per EP/ETP rank combination)
    _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)


# ---------------------------------------------------------------------------
# RNG state helpers for checkpointing
# ---------------------------------------------------------------------------

def _get_all_rng_states():
    """Return (cpu_state, cuda_state, tracker_states) tuple.

    Returns:
        Tuple of (cpu_rng_state, cuda_rng_state, tracker_states_dict).
    """
    graph_safe = is_graph_safe_cuda_rng_tracker(get_cuda_rng_tracker())
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = _get_cuda_rng_state(graph_safe=graph_safe)
    cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()
    return cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker


def _set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
    """Restore (cpu_state, cuda_state, tracker_states) tuple.

    Args:
        cpu_rng_state:          CPU RNG state tensor.
        cuda_rng_state:         CUDA RNG state tensor (or Generator).
        cuda_rng_state_tracker: Dict of named tracker states.
    """
    torch.set_rng_state(cpu_rng_state)
    graph_safe = is_graph_safe_cuda_rng_tracker(get_cuda_rng_tracker())
    _set_cuda_rng_state(cuda_rng_state, graph_safe=graph_safe)
    get_cuda_rng_tracker().set_states(cuda_rng_state_tracker)


@contextlib.contextmanager
def _fork_rng():
    """Context manager: fork all RNG states and restore them on exit.

    Used by the checkpoint backward pass to replay the forward with the
    same random state as the original forward pass.
    """
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
    """Return True when currently inside a checkpoint context.

    Can be used by layers (e.g. FlashAttention) that need to know whether
    they are in the first (forward) or second (recompute) pass.
    """
    return IS_CHECKPOINTING


# ---------------------------------------------------------------------------
# CheckpointFunction
# ---------------------------------------------------------------------------

_R = TypeVar('_R')


class CheckpointFunction(torch.autograd.Function):
    """Activation checkpoint that saves/restores CUDA RNG states.

    Adapted from torch.utils.checkpoint with two changes:
    1. ``torch.cuda.set_rng_state`` → ``_set_cuda_rng_state`` (no clone perf issue).
    2. Model-parallel tracker states are also saved/restored so that
       tensor-parallel dropout is deterministic across forward/recompute.

    This allows activation checkpointing to work correctly with tensor parallelism
    where each rank uses a different model-parallel RNG state.
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        distribute_saved_activations: bool,
        *args,
    ):
        """Forward pass: run without grad, save inputs + RNG states.

        Args:
            ctx:                          Autograd context.
            run_function:                 The function to checkpoint.
            distribute_saved_activations: Shard first input across TP ranks.
            *args:                        Arguments forwarded to run_function.
        Returns:
            Output of run_function.
        """
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
        """Backward pass: recompute forward with saved RNG states.

        Restores the exact CPU, CUDA, and model-parallel RNG states from
        the original forward pass before recomputing, ensuring deterministic
        dropout and other stochastic ops.
        """
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
    handles the tensor-parallel RNG tracker, ensuring that model-parallel
    dropout is reproduced exactly during gradient recomputation.

    Args:
        function:                     The forward function to checkpoint.
        distribute_saved_activations: Shard first input across TP ranks to
                                      reduce activation memory per GPU.
        *args:                        Arguments forwarded to *function*.
    Returns:
        Output of function(*args).
    """
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)


# ---------------------------------------------------------------------------
# CheckpointWithoutOutput — zero-copy variant that discards forward outputs
# ---------------------------------------------------------------------------

class CheckpointWithoutOutputFunction(torch.autograd.Function):
    """Helper for CheckpointWithoutOutput: saves context for later recompute.

    Forward pass runs the function and saves the context but does not save
    output activations to the graph.  The CheckpointWithoutOutput object
    holds a reference to allow recomputation during backward.
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        checkpoint_obj: 'CheckpointWithoutOutput',
        *args,
    ):
        """Run forward without tracking outputs.

        Args:
            ctx:            Autograd context.
            run_function:   Function to checkpoint.
            checkpoint_obj: Owning CheckpointWithoutOutput instance.
            *args:          Arguments forwarded to run_function.
        Returns:
            Output of run_function(*args).
        """
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
        """Backward: delegate to saved context's recomputed outputs."""
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

    Unlike the standard checkpoint() which keeps inputs alive, this variant:
    1. Runs forward with no_grad (no activation tensors kept in graph).
    2. Lets the caller explicitly release the output storage.
    3. Recomputes the outputs lazily in backward via a registered hook.

    Zero-copy recomputation uses a C++ extension (``share_storage``) to point
    the original output's UntypedStorage at the recomputed data without copying.

    Usage::

        cwo = CheckpointWithoutOutput()
        out = cwo.checkpoint(fn, *inputs)
        # ... use out in next layers (but don't need to keep it for backward) ...
        cwo.discard_output_and_register_recompute(hook_tensor)
        # out is now a shell; backward will recompute it before it's needed.

    Args:
        fp8: Whether FP8 is enabled (requires TransformerEngine).
    """

    def __init__(self, fp8: bool = False) -> None:
        self.fp8 = fp8 is not None
        self.run_function: Optional[Callable] = None
        self.rng_states = None
        self.ctx = None
        self.outputs = None

    def checkpoint(self, run_function: Callable, *args):
        """Run *run_function* under checkpoint (no-gradient forward).

        Saves RNG states for deterministic recomputation in backward.

        Args:
            run_function: Function to checkpoint.
            *args:        Arguments to pass to run_function.
        Returns:
            Output of run_function(*args).
        """
        self.run_function = run_function
        self.rng_states = _get_all_rng_states()
        outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
        self.outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
        return outputs

    def _recompute(self, _):
        """Hook called during backward to recompute forward outputs.

        This is triggered by a hook on a tensor that is computed just before
        the recomputed outputs are needed in backward.
        """
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

        # Zero-copy: point saved output storage to recomputed data.
        share_storage = _get_share_storage()
        for output, recomputed in zip(self.outputs, outputs):
            try:
                if share_storage is not None:
                    # C++ extension: zero-copy storage pointer swap
                    share_storage(output, recomputed)
                else:
                    # Fallback: use set_() for best-effort zero-copy
                    output.set_(
                        recomputed.storage(),
                        recomputed.storage_offset(),
                        recomputed.size(),
                        recomputed.stride(),
                    )
            except Exception:
                # Last resort: data copy
                output.data.copy_(recomputed.data)

        self.ctx.outputs = tuple(outputs)
        self.ctx.inputs = inputs
        self.outputs = None
        self.ctx = None

    def discard_output_and_register_recompute(self, hook_tensor: torch.Tensor) -> None:
        """Resize output storages to zero and register the backward recompute hook.

        After this call the output tensors become empty shells.  When backward
        reaches *hook_tensor*, the recompute hook fires to rebuild the outputs.

        Args:
            hook_tensor: Tensor whose backward trigger fires the recompute.
                         Must be computed before the recomputed outputs are needed
                         in backward.  Typically the next layer's output.
        """
        if self.outputs is None:
            return
        for output in self.outputs:
            output.untyped_storage().resize_(0)
        if hook_tensor.requires_grad:
            hook_tensor.register_hook(self._recompute)
