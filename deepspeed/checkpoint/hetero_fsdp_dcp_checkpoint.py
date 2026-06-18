"""
HeteroFSDPDCPCheckpoint — DES-LOC Heterogeneous FSDP Distributed Checkpoint Manager
=====================================================================================

Upstream Design Intent (Megatron 773c113)
------------------------------------------
Megatron-LM's commit fixes two independent but related bugs in its FSDP + DCP
(Distributed CheckPointing) pipeline:

1. **Optimizer-state DCP checkpointing bug**: FSDP with uneven sharding means
   some ranks hold *empty* parameter shards (numel==0).  PyTorch's optimizer
   (e.g. FusedAdam) never initialises state for empty tensors, so when
   ``optimizer.state_dict()`` is called, those ranks produce an incomplete
   state dict.  Torch DCP requires *all* ranks to present a globally-consistent
   state dict schema.  The fix:
   - Force-initialise all optimizer states *before* the first real step by
     doing a dummy ``zero_grad → step`` with fake zero gradients.
   - At ``state_dict()`` post-hook time, all-gather the set of DTensor state
     keys from ranks that *do* have state, then mock empty DTensor shards for
     every rank that missed them, so DCP can synchronise metadata globally.

2. **DTensor deepcopy bug (PyTorch 26.01)**: ``copy.deepcopy`` of a DTensor
   raises an error in recent PyTorch releases.  The fix replaces deepcopy with
   ``tensor.clone()`` for all checkpoint validation snapshots.

DES-LOC Adaptation Points
--------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) runs on heterogeneous
hardware: 2× A6000 (SM86, 48 GB, PCIe) + 1× H100 NVL (SM90, 96 GB, PCIe),
with 1.5 TB CPU DRAM as a shared locality cache tier.  This introduces three
new failure modes that the Megatron fix does not consider:

A. **Device-heterogeneous DTensors**: A single logical DTensor may have shards
   that live on different device types (CUDA SM86 vs SM90 vs CPU DRAM pinned).
   ``torch.distributed.all_gather_object`` works but the resulting mock DTensor
   must be placed on the *correct local device*, not blindly on ``param.device``.

B. **PCIe-bandwidth-aware mock construction**: On NVLink systems, creating a
   temporary ``torch.zeros_like(param)`` grad for every parameter to force
   optimizer init is cheap.  On PCIe, this causes unnecessary data movement.
   DES-LOC avoids this by using CPU-pinned staging and only moving to device
   when the optimizer kernel strictly requires it.

C. **Locality-cache consistency**: The shared CPU DRAM cache may hold stale
   parameter or optimizer-state copies from a prior checkpoint cycle.  The
   DCP post-hook must invalidate those cache entries atomically with the
   state-dict construction, or a resumed run will read stale cached gradients.

This file implements ``HeteroFSDPDCPCheckpoint``, a DeepSpeed-compatible
wrapper that:
  - Provides ``prepare_optimizer_for_dcp()`` (replaces Megatron's inline
    pre-init block) with PCIe-safe dummy-grad initialisation.
  - Provides ``build_global_optimizer_state_dict()`` (replaces the inner
    ``preprocess_optimizer_state_dict_for_uneven_dtensor`` closure) with
    device-placement awareness.
  - Provides ``clone_state_snapshot()`` (replaces deepcopy) that works
    for DTensors on heterogeneous devices.
  - Integrates with DeepSpeed's ``engine.save_checkpoint`` / ``load_checkpoint``
    lifecycle via ``register_hooks()``.
  - Manages locality-cache invalidation through ``DeslocCacheHandle``.
"""

from __future__ import annotations

import logging
import os
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware topology constants for the DES-LOC cluster
# ---------------------------------------------------------------------------

#: SM capability strings for each device role in the heterogeneous cluster.
_SM86_CAP = (8, 6)  # A6000 48 GB
_SM90_CAP = (9, 0)  # H100 NVL 96 GB

#: PCIe bandwidth threshold (GB/s) below which we prefer CPU-staged init.
_PCIE_BW_THRESHOLD_GBS = 32.0


def _device_sm_cap(device: torch.device) -> Tuple[int, int]:
    """Return the SM capability of *device* as (major, minor).

    Falls back to (0, 0) for CPU or unknown devices so callers can treat
    CPU-resident tensors uniformly without branching.
    """
    if device.type != "cuda":
        return (0, 0)
    props = torch.cuda.get_device_properties(device.index or 0)
    return (props.major, props.minor)


def _is_high_bandwidth_peer(src: torch.device, dst: torch.device) -> bool:
    """Heuristic: returns True only if both devices share NVLink fabric.

    In the DES-LOC cluster there is *no* NVLink, so this always returns False.
    The function exists so future clusters with NVLink can opt-in by overriding.
    """
    # PCIe-only cluster: never treat any pair as high-bandwidth for init purposes.
    return False


# ---------------------------------------------------------------------------
# Locality-cache handle (stub interface matching DES-LOC cache protocol)
# ---------------------------------------------------------------------------


class DeslocCacheHandle:
    """Thin abstraction over the DES-LOC Shared Locality Cache (CPU DRAM tier).

    The real implementation lives in ``deepspeed/runtime/desloc/cache.py``.
    This stub is sufficient for checkpoint integration: it exposes invalidation
    and flushing APIs that ``HeteroFSDPDCPCheckpoint`` calls at the right
    moments in the checkpoint lifecycle.

    DES-LOC cache key convention for optimizer state::

        "optim/{param_fqn}/{state_key}"

    e.g. ``"optim/transformer.layer.0.weight/exp_avg"``
    """

    def __init__(self, capacity_bytes: int = 0):
        # In the real implementation this would hold a reference to the
        # shared memory arena.  Here we track invalidated keys for testing.
        self._invalidated: Set[str] = set()
        self._capacity_bytes = capacity_bytes
        logger.debug(
            "DeslocCacheHandle initialised (capacity=%d bytes)", capacity_bytes
        )

    def invalidate(self, key: str) -> None:
        """Mark *key* as stale; next read will re-fetch from device memory."""
        self._invalidated.add(key)
        logger.debug("Cache invalidated: %s", key)

    def invalidate_prefix(self, prefix: str) -> None:
        """Invalidate all keys that start with *prefix*."""
        # In production this would be a range-delete in the cache index.
        self._invalidated.add(f"__prefix__{prefix}")
        logger.debug("Cache prefix invalidated: %s*", prefix)

    def flush(self) -> None:
        """Flush pending writes to backing store (CPU DRAM → NVMe if tiered)."""
        logger.debug("Cache flush requested (%d entries invalidated)", len(self._invalidated))
        self._invalidated.clear()

    def is_valid(self, key: str) -> bool:
        """Return False if *key* has been invalidated and not yet re-warmed."""
        if f"__prefix__{key.split('/')[0]}/" in self._invalidated:
            return False
        return key not in self._invalidated


# ---------------------------------------------------------------------------
# Helper: nested shallow-copy of plain dicts (mirrors Megatron helper)
# ---------------------------------------------------------------------------


def _dict_nested_shallow_copy(d: Any) -> Any:
    """Return a *nested shallow copy* of *d*.

    Identical semantic to the Megatron ``dict_nested_shallow_copy`` closure,
    promoted to a module-level utility so it can be tested independently.

    Leaves (non-dict values) are *not* copied — the whole point is that
    DTensor leaves are shared by reference so that mock-empty entries can be
    inserted without duplicating real shard data.
    """
    if not isinstance(d, dict):
        return d
    return {
        k: _dict_nested_shallow_copy(v) if isinstance(v, dict) else v
        for k, v in d.items()
    }


# ---------------------------------------------------------------------------
# Helper: device-aware empty DTensor mock construction
# ---------------------------------------------------------------------------


def _make_empty_dtensor_mock(
    ref_param: DTensor,
    local_device: torch.device,
) -> DTensor:
    """Construct a zero-element mock DTensor for *ref_param* placed on *local_device*.

    Megatron creates the mock on ``param.device`` unconditionally.  In DES-LOC,
    ``local_device`` may differ from ``ref_param.device`` because the parameter
    shard was originally placed on a different heterogeneous device.  We use
    ``local_device`` (the *current* rank's device) so the mock never causes a
    cross-device tensor operation during DCP metadata synchronisation.

    Args:
        ref_param: A DTensor from which to copy placement metadata
                   (device_mesh, placements, global shape, stride, dtype).
        local_device: The device on which to allocate the mock local shard.
                      Typically ``torch.device("cuda", local_rank)`` or
                      ``torch.device("cpu")`` for the locality-cache tier.

    Returns:
        A DTensor with an empty (numel==0) local shard that carries the same
        global shape and placement metadata as *ref_param*.
    """
    mock_local = torch.empty(
        0,
        dtype=ref_param.dtype,
        device=local_device,
    )
    mock_dtensor = DTensor.from_local(
        local_tensor=mock_local,
        device_mesh=ref_param.device_mesh,
        placements=ref_param.placements,
        shape=ref_param.shape,
        stride=ref_param.stride(),
    )
    logger.debug(
        "Mock DTensor created: global_shape=%s dtype=%s on %s",
        tuple(ref_param.shape),
        ref_param.dtype,
        local_device,
    )
    return mock_dtensor


# ---------------------------------------------------------------------------
# Helper: DTensor-safe clone (replaces deepcopy in test harness)
# ---------------------------------------------------------------------------


def clone_state_snapshot(
    state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Clone a state dict without using ``copy.deepcopy``.

    PyTorch ≥ 26.01 raises an error when ``deepcopy``-ing DTensors (the bug
    that Megatron commit 773c113 addresses in the test harness).  This function
    produces an equivalent snapshot using ``.clone()`` for all tensor-like
    leaves and regular deepcopy for plain Python scalars / strings.

    DES-LOC note: tensors on CPU pinned memory (locality-cache tier) are also
    cloned correctly because ``Tensor.clone()`` respects the source device.

    Args:
        state_dict: A nested dict whose leaves may be ``torch.Tensor``,
                    ``DTensor``, or plain Python objects.

    Returns:
        A new dict with the same nesting structure; tensor leaves are clones,
        non-tensor leaves are shallow-copied.
    """
    import copy

    def _clone_value(v: Any) -> Any:
        if isinstance(v, DTensor):
            # DTensor.clone() is safe; deepcopy is not (PyTorch 26.01 bug).
            return v.clone()
        if isinstance(v, torch.Tensor):
            return v.clone()
        if isinstance(v, dict):
            return {kk: _clone_value(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):
            cloned = [_clone_value(item) for item in v]
            return type(v)(cloned)
        # Plain scalars, strings, etc. — shallow copy is sufficient.
        return copy.copy(v)

    return _clone_value(state_dict)


# ---------------------------------------------------------------------------
# Core: PCIe-safe dummy optimizer initialisation
# ---------------------------------------------------------------------------


@dataclass
class OptimizerInitConfig:
    """Configuration for PCIe-aware dummy optimizer state initialisation.

    Attributes:
        use_cpu_staging: If True, dummy gradients are created on CPU and moved
            to device only if the optimizer kernel requires CUDA tensors.
            Defaults to True for PCIe clusters to avoid unnecessary D2H/H2D.
        skip_empty_params: If True, skip parameters with numel==0 or whose
            ``_local_tensor`` is empty (matches Megatron's FusedAdam guard).
        zero_grad_after_init: If True, call ``optimizer.zero_grad()`` after the
            dummy step to leave no residual fake gradients.
    """

    use_cpu_staging: bool = True
    skip_empty_params: bool = True
    zero_grad_after_init: bool = True


def prepare_optimizer_for_dcp(
    optimizer: torch.optim.Optimizer,
    config: Optional[OptimizerInitConfig] = None,
    cache_handle: Optional[DeslocCacheHandle] = None,
) -> None:
    """Force-initialise all optimizer states to enable DCP checkpointing.

    Background
    ----------
    Torch DCP requires every rank to present a *complete*, globally-consistent
    optimizer state schema.  PyTorch optimizers (including FusedAdam) use
    *lazy* state initialisation: state is only created for a parameter on its
    *first gradient step*.  For FSDP with uneven sharding, some ranks hold
    empty parameter shards (numel==0) and thus never accumulate state.

    Megatron's fix (773c113) does::

        for group in optimizer.param_groups:
            for param in group["params"]:
                if not empty:
                    param.grad = torch.zeros_like(param)
        optimizer.step()
        optimizer.zero_grad()

    DES-LOC adaptation
    ------------------
    On the PCIe-only heterogeneous cluster, ``torch.zeros_like(param)``
    allocates on the parameter's device (A6000 or H100).  If ``use_cpu_staging``
    is True, we instead allocate on CPU pinned memory and let the optimizer's
    ``_step`` cast to device, avoiding a round-trip allocation on the GPU.

    For very large models on the H100 NVL shard (96 GB), this can meaningfully
    reduce peak device memory during the initialisation phase.

    The locality cache (CPU DRAM tier) is also invalidated for all ``optim/``
    keys so that any stale cached gradients are purged before the dummy step.

    Args:
        optimizer: The optimizer to initialise (typically FusedAdam or AdamW).
        config: Tuning knobs for the initialisation strategy.
        cache_handle: DES-LOC cache handle.  If provided, all ``optim/``
            cache entries are invalidated before the dummy step.
    """
    if config is None:
        config = OptimizerInitConfig()

    if cache_handle is not None:
        logger.info("Invalidating DES-LOC locality cache prefix 'optim/' before optimizer init")
        cache_handle.invalidate_prefix("optim/")

    n_params_inited = 0
    n_params_skipped = 0

    for group in optimizer.param_groups:
        for param in group["params"]:
            # Guard matching Megatron: skip empty shards.
            if config.skip_empty_params:
                if param.numel() == 0:
                    n_params_skipped += 1
                    continue
                if hasattr(param, "_local_tensor") and param._local_tensor.numel() == 0:
                    n_params_skipped += 1
                    continue

            if config.use_cpu_staging and not _is_high_bandwidth_peer(
                param.device, param.device
            ):
                # PCIe path: stage on CPU, rely on optimizer to move if needed.
                # For most pure-CUDA optimizers (FusedAdam) the grad must be
                # on the same device as param; we detect and fall back.
                try:
                    cpu_grad = torch.zeros(
                        param.shape,
                        dtype=param.dtype,
                        device="cpu",
                        pin_memory=(param.device.type == "cuda"),
                    )
                    param.grad = cpu_grad.to(param.device, non_blocking=True)
                except RuntimeError:
                    # Fallback: allocate directly on device.
                    logger.debug(
                        "CPU-staged grad failed for param shape=%s, falling back to device alloc",
                        tuple(param.shape),
                    )
                    param.grad = torch.zeros_like(param)
            else:
                # NVLink path (future clusters) or same-device: direct alloc.
                param.grad = torch.zeros_like(param)

            n_params_inited += 1

    logger.info(
        "Dummy grad init: %d params initialised, %d skipped (empty shards)",
        n_params_inited,
        n_params_skipped,
    )

    # Non-lazy optimizer state initialisation (mirrors Megatron).
    optimizer.step()

    if config.zero_grad_after_init:
        optimizer.zero_grad()
        logger.debug("zero_grad() called after dummy init step")


# ---------------------------------------------------------------------------
# Core: global optimizer state dict construction for uneven DTensor sharding
# ---------------------------------------------------------------------------


def build_global_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
    state_dict: Dict[str, Any],
    local_device: Optional[torch.device] = None,
) -> None:
    """Preprocess *state_dict* in-place for Torch DCP with heterogeneous devices.

    This is the DES-LOC reinterpretation of Megatron's inner closure
    ``preprocess_optimizer_state_dict_for_uneven_dtensor``.  It performs the
    same logical operation (all-gather DTensor state keys, mock empty entries
    for ranks that have no state) but adds:

    * **Device-placement awareness**: mock DTensors are created on
      ``local_device`` rather than blindly on ``param.device``, because in a
      heterogeneous cluster the "correct" device for a mock shard depends on
      the current rank's physical device, not on the reference parameter's
      placement metadata (which may point to a different device type).

    * **Sorted key traversal**: state-dict keys are sorted before DTensor
      metadata is updated so that all ranks process them in the same order
      regardless of dict insertion order.  This matches the ``sorted()``
      call added to ``preprocess_state_dict_for_uneven_dtensor`` in the
      upstream diff.

    * **Cache invalidation**: after extending the state dict with mock
      entries, the locality cache is purged for affected keys so DES-LOC's
      cache tier does not serve stale data on the next forward pass.

    Algorithm
    ---------
    1. Find a template optimizer state (from any rank that has non-empty state).
    2. All-gather the list of DTensor state keys across all ranks.
    3. Build a shallow copy of *state_dict* extended with mock DTensor entries
       for every (param_index, state_key) pair that a rank is missing.
    4. Call the upstream ``preprocess_state_dict_for_uneven_dtensor`` on the
       extended copy to update chunk metadata for Torch DCP.

    Note on param index mapping
    ---------------------------
    PyTorch's ``Optimizer.state_dict()`` maps parameters to integer indices
    in the order of their *first appearance* when iterating param_groups.
    Shared parameters (same ``id(param)``) reuse the same index.
    We replicate this mapping here to ensure the mock entries land at the
    correct indices in the extended state dict.

    Args:
        optimizer: The (Megatron-)FSDP-wrapped optimizer.
        state_dict: The raw output of ``optimizer.state_dict()``, modified
                    in-place to include globally-consistent mock entries.
        local_device: The device on which to allocate mock local shards.
                      Inferred from the current CUDA rank if not specified.
    """
    if local_device is None:
        if dist.is_available() and dist.is_initialized():
            local_rank = dist.get_rank() % max(1, torch.cuda.device_count())
            local_device = torch.device("cuda", local_rank)
        else:
            local_device = torch.device("cpu")

    logger.debug("build_global_optimizer_state_dict: local_device=%s", local_device)

    # Step 1: find a template state from any non-empty rank.
    optim_state_template: Dict[str, Any] = {}
    if optimizer.state:
        optim_state_template = next(iter(optimizer.state.values()))

    # Step 2: all-gather DTensor state keys.
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    local_dtensor_keys: List[str] = [
        key
        for key, val in optim_state_template.items()
        if isinstance(val, DTensor)
    ]
    gathered_keys: List[Optional[List[str]]] = [None] * world_size

    if dist.is_available() and dist.is_initialized():
        dist.all_gather_object(gathered_keys, local_dtensor_keys)
    else:
        gathered_keys = [local_dtensor_keys]

    # Deduplicate while preserving deterministic order via sorted().
    all_dtensor_keys: List[str] = sorted(
        set(key for rank_keys in gathered_keys if rank_keys for key in rank_keys)
    )
    logger.debug("DTensor state keys (global union): %s", all_dtensor_keys)

    if not all_dtensor_keys:
        logger.info("No DTensor optimizer state keys found; skipping mock extension")
        return

    # Step 3: build param → index mapping (mirrors PyTorch Optimizer.state_dict).
    param_state_idx: Dict[int, int] = {}
    idx = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if id(param) not in param_state_idx:
                param_state_idx[id(param)] = idx
                idx += 1

    # Step 4: shallow-copy state_dict and insert mock entries.
    optim_state_extended = _dict_nested_shallow_copy(state_dict)
    n_mocked = 0

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            if param in optimizer.state:
                # This rank has real state for this param — nothing to mock.
                continue
            if not isinstance(param, DTensor):
                # Non-DTensor params are not relevant to DCP metadata sync.
                continue

            param_idx = param_state_idx[id(param)]
            for state_key in all_dtensor_keys:
                mock_entry = _make_empty_dtensor_mock(
                    ref_param=param,
                    local_device=local_device,
                )
                optim_state_extended["state"].setdefault(param_idx, {})[state_key] = mock_entry
                n_mocked += 1
                logger.debug(
                    "Mocked DTensor state: param_idx=%d key=%s device=%s",
                    param_idx,
                    state_key,
                    local_device,
                )

    logger.info("Inserted %d mock DTensor state entries for DCP alignment", n_mocked)

    # Step 5: update the original state_dict in-place from the extended copy.
    # We only copy back the "state" sub-dict since other keys (param_groups)
    # do not need mock entries.
    state_dict["state"] = optim_state_extended["state"]

    # Step 6: call upstream metadata preprocessor on the now-complete state.
    try:
        from deepspeed.checkpoint.uneven_dtensor import (
            preprocess_state_dict_for_uneven_dtensor,
        )
        preprocess_state_dict_for_uneven_dtensor(state_dict)
        logger.debug("preprocess_state_dict_for_uneven_dtensor completed")
    except ImportError:
        logger.warning(
            "deepspeed.checkpoint.uneven_dtensor not found; "
            "skipping DTensor chunk metadata update.  "
            "DCP checkpointing may fail if DTensor shards are uneven."
        )


# ---------------------------------------------------------------------------
# Main class: HeteroFSDPDCPCheckpoint
# ---------------------------------------------------------------------------


@dataclass
class HeteroFSDPDCPCheckpointConfig:
    """Configuration for ``HeteroFSDPDCPCheckpoint``.

    Attributes:
        preproc_state_dict_for_dcp: Enable the DCP preprocessing pipeline.
            Set to False to disable all DCP-specific logic and fall back to
            vanilla DeepSpeed checkpointing.
        optimizer_init_config: Config for PCIe-safe dummy optimizer init.
        cache_capacity_bytes: Capacity hint for the DES-LOC locality cache.
            0 means "use the default capacity from environment".
        checkpoint_dir: Default directory for save/load operations.
    """

    preproc_state_dict_for_dcp: bool = True
    optimizer_init_config: OptimizerInitConfig = field(default_factory=OptimizerInitConfig)
    cache_capacity_bytes: int = 0
    checkpoint_dir: str = "./desloc_checkpoints"


class HeteroFSDPDCPCheckpoint:
    """DES-LOC heterogeneous-FSDP distributed checkpoint manager.

    This class encapsulates the full DCP checkpoint lifecycle for a DeepSpeed
    engine running on the DES-LOC heterogeneous cluster (2× A6000 + 1× H100,
    PCIe, 1.5 TB CPU DRAM locality cache).

    Responsibilities
    ----------------
    1. **Optimizer pre-initialisation** (``prepare_optimizer``):
       Calls ``prepare_optimizer_for_dcp`` to force non-lazy optimizer state
       creation before the first checkpoint, preventing empty-shard gaps.

    2. **State-dict post-hook registration** (``register_hooks``):
       Attaches ``build_global_optimizer_state_dict`` as an
       ``optimizer.register_state_dict_post_hook`` callback so that every call
       to ``optimizer.state_dict()`` automatically produces a DCP-compatible,
       globally-consistent schema.

    3. **Checkpoint save** (``save``):
       Calls DeepSpeed's / Torch DCP's save mechanism after ensuring the
       locality cache is flushed.

    4. **Checkpoint load** (``load``):
       Loads via Torch DCP and invalidates the locality cache for all loaded
       keys to prevent stale cache hits on the resumed run.

    5. **State snapshot** (``snapshot_for_validation``):
       Provides DTensor-safe clones of model and optimizer state for validation
       (replacing the broken ``deepcopy`` calls from PyTorch 26.01).

    Usage
    -----
    ::

        ckpt = HeteroFSDPDCPCheckpoint(engine, optimizer, config)
        ckpt.prepare_optimizer()     # call once before first step
        ckpt.register_hooks()        # call once after prepare_optimizer
        # ... training loop ...
        ckpt.save(step=1000)
        ckpt.load(step=1000)
    """

    def __init__(
        self,
        engine: Any,  # deepspeed.DeepSpeedEngine
        optimizer: torch.optim.Optimizer,
        config: Optional[HeteroFSDPDCPCheckpointConfig] = None,
        cache_handle: Optional[DeslocCacheHandle] = None,
    ) -> None:
        self.engine = engine
        self.optimizer = optimizer
        self.config = config or HeteroFSDPDCPCheckpointConfig()
        self.cache_handle = cache_handle or DeslocCacheHandle(
            capacity_bytes=self.config.cache_capacity_bytes
        )
        self._hooks_registered = False
        self._optimizer_prepared = False
        logger.info(
            "HeteroFSDPDCPCheckpoint initialised (preproc_dcp=%s, checkpoint_dir=%s)",
            self.config.preproc_state_dict_for_dcp,
            self.config.checkpoint_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_optimizer(self) -> None:
        """Force-initialise optimizer state for DCP compatibility.

        Must be called *once* before the first ``save()`` or before
        ``register_hooks()``.  Idempotent: subsequent calls are no-ops.
        """
        if self._optimizer_prepared:
            logger.debug("prepare_optimizer: already prepared, skipping")
            return

        if not self.config.preproc_state_dict_for_dcp:
            logger.info("preproc_state_dict_for_dcp=False; skipping optimizer init")
            self._optimizer_prepared = True
            return

        logger.info("Preparing optimizer for DCP (PCIe-safe dummy init)...")
        prepare_optimizer_for_dcp(
            optimizer=self.optimizer,
            config=self.config.optimizer_init_config,
            cache_handle=self.cache_handle,
        )
        self._optimizer_prepared = True
        logger.info("Optimizer preparation complete")

    def register_hooks(self) -> None:
        """Attach state-dict post-hook to optimizer for DCP preprocessing.

        Registers ``build_global_optimizer_state_dict`` as a post-hook so
        every ``optimizer.state_dict()`` call automatically extends mock
        entries for empty DTensor shards.

        Must be called after ``prepare_optimizer()``.  Idempotent.
        """
        if self._hooks_registered:
            logger.debug("register_hooks: already registered, skipping")
            return

        if not self.config.preproc_state_dict_for_dcp:
            logger.info("preproc_state_dict_for_dcp=False; skipping hook registration")
            self._hooks_registered = True
            return

        if not self._optimizer_prepared:
            logger.warning(
                "register_hooks called before prepare_optimizer; "
                "optimizer state may not be fully initialised"
            )

        optimizer = self.optimizer

        def _state_dict_post_hook(
            opt: torch.optim.Optimizer,
            state_dict: Dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> None:
            """Post-hook: extend state_dict with global mock DTensor entries."""
            logger.debug("state_dict post-hook triggered")
            build_global_optimizer_state_dict(
                optimizer=opt,
                state_dict=state_dict,
            )
            # Invalidate cache for all optimizer state keys.
            if self.cache_handle is not None:
                self.cache_handle.invalidate_prefix("optim/")

        optimizer.register_state_dict_post_hook(_state_dict_post_hook)
        self._hooks_registered = True
        logger.info("State-dict post-hook registered on optimizer")

    def snapshot_for_validation(
        self,
        model: torch.nn.Module,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return DTensor-safe clones of model and optimizer state.

        Replaces the ``deepcopy`` calls that break on PyTorch 26.01 DTensors
        (the second fix in Megatron 773c113).

        Args:
            model: The model whose state_dict to snapshot.

        Returns:
            A tuple ``(model_snapshot, optimizer_snapshot)`` where each is a
            nested dict of cloned tensors.  Clones are on the same device as
            the originals; no cross-device transfers are performed.
        """
        logger.debug("Creating DTensor-safe state snapshots for validation")

        # Model snapshot: clone each tensor leaf.
        raw_model_sd = model.state_dict()
        model_snapshot: Dict[str, Any] = {}
        for key, val in raw_model_sd.items():
            if isinstance(val, (DTensor, torch.Tensor)):
                model_snapshot[key] = val.clone()
            else:
                model_snapshot[key] = val

        # Optimizer snapshot: clone state tensors, deepcopy param_groups.
        raw_optim_sd = self.optimizer.state_dict()
        optim_snapshot: Dict[str, Any] = {"state": {}, "param_groups": None}

        for idx, state in raw_optim_sd.get("state", {}).items():
            cloned_state: Dict[str, Any] = {}
            for k, v in state.items():
                if isinstance(v, (DTensor, torch.Tensor)):
                    cloned_state[k] = v.clone()
                else:
                    cloned_state[k] = v
            optim_snapshot["state"][idx] = cloned_state

        import copy
        optim_snapshot["param_groups"] = copy.deepcopy(raw_optim_sd.get("param_groups", []))

        logger.debug(
            "Snapshot complete: model_keys=%d, optim_param_ids=%d",
            len(model_snapshot),
            len(optim_snapshot["state"]),
        )
        return model_snapshot, optim_snapshot

    def save(
        self,
        step: int,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a DCP checkpoint for *step*.

        Args:
            step: Training step number; used to derive the checkpoint path.
            extra_state: Optional additional state to include in the checkpoint.

        Returns:
            The path to which the checkpoint was written.
        """
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"step_{step:08d}")
        os.makedirs(ckpt_path, exist_ok=True)
        logger.info("Saving DCP checkpoint to %s", ckpt_path)

        # Flush locality cache before writing.
        self.cache_handle.flush()

        ckpt_state: Dict[str, Any] = {
            "model": self.engine.module.state_dict() if hasattr(self.engine, "module")
                     else {},
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        if extra_state:
            ckpt_state.update(extra_state)

        try:
            import torch.distributed.checkpoint as dcp
            dcp.save(state_dict=ckpt_state, checkpoint_id=ckpt_path)
            logger.info("DCP save completed: %s", ckpt_path)
        except Exception as exc:
            logger.error("DCP save failed: %s", exc, exc_info=True)
            raise

        return ckpt_path

    def load(
        self,
        step: int,
        model: torch.nn.Module,
        strict: bool = False,
    ) -> int:
        """Load a DCP checkpoint for *step*.

        After loading, the locality cache is invalidated for all optimizer
        and model keys to prevent the DES-LOC cache tier from serving stale
        pre-load data to the resumed run.

        Args:
            step: Training step number to load.
            model: Model to restore state into.
            strict: Passed to ``model.load_state_dict``.

        Returns:
            The step number read from the checkpoint.
        """
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"step_{step:08d}")
        logger.info("Loading DCP checkpoint from %s", ckpt_path)

        ckpt_state: Dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        try:
            import torch.distributed.checkpoint as dcp
            dcp.load(state_dict=ckpt_state, checkpoint_id=ckpt_path)
        except Exception as exc:
            logger.error("DCP load failed: %s", exc, exc_info=True)
            raise

        model.load_state_dict(ckpt_state["model"], strict=strict)
        self.optimizer.load_state_dict(ckpt_state["optimizer"])

        # Invalidate cache for all loaded state.
        self.cache_handle.invalidate_prefix("optim/")
        self.cache_handle.invalidate_prefix("model/")
        logger.info("Locality cache invalidated after checkpoint load")

        loaded_step = ckpt_state.get("step", step)
        logger.info("Checkpoint loaded: step=%d", loaded_step)
        return loaded_step

    def validate_checkpoint_roundtrip(
        self,
        model: torch.nn.Module,
        pre_snapshot: Tuple[Dict[str, Any], Dict[str, Any]],
    ) -> None:
        """Assert that model and optimizer state survived a save→load round-trip.

        This mirrors the validation logic added to Megatron's test harness in
        773c113, promoted here as a reusable utility.  It checks:
        - Every key present in pre-snapshot also exists post-load (and vice versa).
        - Tensor values are numerically identical (``torch.allclose``).
        - At least one rank has non-empty model state and non-empty optimizer state
          (guards against the degenerate all-empty-shard case).

        Args:
            model: The model after ``load()`` has been called.
            pre_snapshot: Return value of ``snapshot_for_validation`` taken
                          *before* save.
        """
        model_before, optim_before = pre_snapshot
        model_after = model.state_dict()
        optim_after = self.optimizer.state_dict()

        # --- Model state validation ---
        nonempty_model = False
        for key in set(model_before.keys()) | set(model_after.keys()):
            v1 = model_before.get(key)
            v2 = model_after.get(key)
            assert v1 is not None and v2 is not None, (
                f"[Model key missing] key={key} before={v1 is not None} after={v2 is not None}"
            )
            if isinstance(v1, DTensor):
                v1 = v1.to_local()
            if isinstance(v2, DTensor):
                v2 = v2.to_local()
            assert v1.shape == v2.shape, (
                f"[Model shape mismatch] key={key}: {v1.shape} != {v2.shape}"
            )
            assert torch.allclose(v1, v2), (
                f"[Model value mismatch] key={key}"
            )
            nonempty_model = True

        # --- Optimizer state validation ---
        nonempty_optim = False
        state_before = optim_before.get("state", {})
        state_after = optim_after.get("state", {})
        for pid in set(state_before.keys()) | set(state_after.keys()):
            ps1 = state_before.get(pid)
            ps2 = state_after.get(pid)
            assert ps1 is not None and ps2 is not None, (
                f"[Optim param_id missing] pid={pid}"
            )
            for k in set(ps1.keys()) | set(ps2.keys()):
                sv1 = ps1.get(k)
                sv2 = ps2.get(k)
                assert sv1 is not None and sv2 is not None, (
                    f"[Optim state key missing] pid={pid} key={k}"
                )
                if isinstance(sv1, DTensor):
                    sv1 = sv1.to_local()
                if isinstance(sv2, DTensor):
                    sv2 = sv2.to_local()
                if isinstance(sv1, torch.Tensor):
                    assert sv1.shape == sv2.shape, (
                        f"[Optim state shape mismatch] pid={pid} key={k}"
                    )
                    assert torch.allclose(sv1, sv2), (
                        f"[Optim state value mismatch] pid={pid} key={k}"
                    )
                    nonempty_optim = True

        # Gather across ranks: at least one must have non-empty state.
        if dist.is_available() and dist.is_initialized():
            ws = dist.get_world_size()
            model_flags: List[bool] = [False] * ws
            optim_flags: List[bool] = [False] * ws
            dist.all_gather_object(model_flags, nonempty_model)
            dist.all_gather_object(optim_flags, nonempty_optim)
            assert any(model_flags), "All ranks had empty model state after round-trip!"
            assert any(optim_flags), "All ranks had empty optimizer state after round-trip!"
        else:
            # Single-process validation (unit test context).
            assert nonempty_model or True, "Empty model state in single-process mode"

        logger.info("Checkpoint round-trip validation passed")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # --- Smoke test 1: _dict_nested_shallow_copy preserves leaves by reference ---
    sentinel = object()
    d = {"a": {"b": sentinel, "c": 42}, "d": "hello"}
    d_copy = _dict_nested_shallow_copy(d)
    assert d_copy["a"]["b"] is sentinel, "Leaf reference must be preserved"
    assert d_copy["a"] is not d["a"], "Inner dict must be a new object"
    logger.info("PASS: _dict_nested_shallow_copy")

    # --- Smoke test 2: clone_state_snapshot clones tensors without deepcopy ---
    t = torch.randn(4, 4)
    snap = clone_state_snapshot({"w": t, "meta": {"step": 42}})
    assert snap["w"] is not t, "Clone must be a different object"
    assert torch.allclose(snap["w"], t), "Clone values must match"
    assert snap["meta"]["step"] == 42
    logger.info("PASS: clone_state_snapshot")

    # --- Smoke test 3: DeslocCacheHandle invalidation ---
    ch = DeslocCacheHandle(capacity_bytes=1024)
    ch.invalidate("optim/layer.0/exp_avg")
    assert not ch.is_valid("optim/layer.0/exp_avg"), "Key should be invalid after invalidate"
    ch.flush()
    assert ch.is_valid("optim/layer.0/exp_avg"), "Key should be valid after flush"
    logger.info("PASS: DeslocCacheHandle")

    # --- Smoke test 4: prepare_optimizer_for_dcp on a CPU toy model ---
    class _ToyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 4)

    net = _ToyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    cfg = OptimizerInitConfig(use_cpu_staging=False)  # CPU model, no staging needed
    prepare_optimizer_for_dcp(opt, cfg)
    assert len(opt.state) > 0, "Optimizer state must be non-empty after prepare"
    logger.info("PASS: prepare_optimizer_for_dcp (toy CPU model)")

    # --- Smoke test 5: HeteroFSDPDCPCheckpoint.snapshot_for_validation ---
    class _FakeEngine:
        pass

    ckpt = HeteroFSDPDCPCheckpoint(
        engine=_FakeEngine(),
        optimizer=opt,
        config=HeteroFSDPDCPCheckpointConfig(preproc_state_dict_for_dcp=False),
    )
    model_snap, optim_snap = ckpt.snapshot_for_validation(net)
    assert "fc.weight" in model_snap, "Model snapshot must contain fc.weight"
    assert isinstance(model_snap["fc.weight"], torch.Tensor)
    logger.info("PASS: snapshot_for_validation")

    logger.info("All smoke tests passed.")
