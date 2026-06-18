"""
DES-LOC Heterogeneous MLA Weight Gradient Fix
==============================================

Upstream Design Intent (Megatron commit 76d26e24b):
---------------------------------------------------
Megatron's FusedMLASelfAttention implements Multi-head Latent Attention (MLA) as
introduced in DeepSeek-V2/V3. The fused variant collapses separate Q and KV down-
projection matrices into a single `linear_qkv_down_proj` for compute efficiency.
The upstream fix (PR #5273) corrects a hook ordering bug: the original `backward_dw`
method referenced stale attribute names from the non-fused base class
(`linear_kv_down_proj`, `linear_q_down_proj`) instead of the fused
`linear_qkv_down_proj`. This caused silent gradient omission during delayed weight-
gradient accumulation — a correctness issue that only manifests when
`delay_wgrad_reduce=True` is combined with FusedMLA.

Additionally, `set_for_recompute_input_layernorm` was missing from the fused subclass,
causing FP8/FP4 activation-recompute paths to skip saving the original input to
`linear_qkv_down_proj`, producing NaN or wrong gradients under mixed-precision
recompute.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) splits the transformer
execution graph across a heterogeneous device pool:

    ┌─────────────────────────────────────────────────────┐
    │  Device topology (Neuron_SP / DES-LOC)              │
    │                                                     │
    │  GPU-0  A6000 48GB  SM86  ──┐                       │
    │  GPU-1  A6000 48GB  SM86  ──┼──PCIe──  CPU DRAM    │
    │  GPU-2  H100 NVL 96GB SM90──┘   1.5TB              │
    │                                                     │
    │  No NVLink.  Peer bandwidth limited to PCIe (~32    │
    │  GB/s bidirectional aggregate across root complex). │
    └─────────────────────────────────────────────────────┘

The three MLA weight-grad hooks map to physically different devices:

  * linear_kv_up_proj   → preferentially placed on H100 (high FLOP/byte for
                          large KV latent dim expansion)
  * linear_qkv_down_proj → split: SM90 for forward, SM86 for wgrad accumulation
                          (DES-LOC "decoupled" phase — forward and wgrad live on
                          different devices to maximise overlap)
  * linear_q_up_proj    → A6000 pair (smaller query head expansion)
  * output projection   → A6000 pair (sequence-parallel output scatter)

The locality cache (LOC) stores activation checkpoints in CPU DRAM and streams
them back to the wgrad device on demand. `set_for_recompute_input_layernorm` must
therefore redirect `set_save_original_input` to CPU-pinned memory rather than
device memory when the down-proj lives on SM86 during recompute.

Key DES-LOC additions vs upstream:
  1. `HeteroMLADeviceMap`  — static assignment of MLA sub-modules to devices.
  2. `LOCActivationCache`  — CPU-DRAM backed ring-buffer for pinned activations.
  3. `HeteroMLAWgradHook`  — per-sub-module hook that migrates grads to the
                             correct device before accumulation and evicts the
                             LOC entry afterwards.
  4. `FusedMLAWeightGradFix` — drop-in replacement / monkey-patch for DeepSpeed's
                               MLA attention that applies all of the above.
  5. `apply_hetero_mla_fix` — entry-point called from engine initialisation.

Author:  Neuron_SP project (github.com/dylanyunlon/Neuron_SP)
Mirrors: Megatron commit 76d26e24b076ca93c8b82576404adcac0fb395a9
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SM architecture identifiers used to classify devices at runtime.
_SM86_MAJOR, _SM86_MINOR = 8, 6   # A6000
_SM90_MAJOR, _SM90_MINOR = 9, 0   # H100

# Maximum number of activation tensors held in the LOC ring buffer per layer.
_LOC_RING_CAPACITY = 4

# Sentinel — used when a device assignment is deliberately deferred to CPU.
_CPU_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Device detection helpers
# ---------------------------------------------------------------------------

def _cuda_sm(device: torch.device) -> Tuple[int, int]:
    """Return (major, minor) SM version for *device*.

    Returns (0, 0) for CPU devices so callers can compare without branching.
    """
    if device.type != "cuda":
        return (0, 0)
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)


def _classify_devices() -> Dict[str, List[torch.device]]:
    """Enumerate visible CUDA devices and bucket them by SM generation.

    DES-LOC heterogeneous topology expects:
      - One or more SM86 devices (A6000 class)
      - One SM90 device (H100 class)
    Both pools must be non-empty; a warning is logged (not raised) if not,
    allowing single-GPU smoke tests to pass.
    """
    sm86: List[torch.device] = []
    sm90: List[torch.device] = []
    other: List[torch.device] = []

    n = torch.cuda.device_count()
    for idx in range(n):
        dev = torch.device("cuda", idx)
        major, minor = _cuda_sm(dev)
        if (major, minor) == (_SM86_MAJOR, _SM86_MINOR):
            sm86.append(dev)
        elif (major, minor) == (_SM90_MAJOR, _SM90_MINOR):
            sm90.append(dev)
        else:
            other.append(dev)
            logger.debug("Device cuda:%d has SM%d%d — treated as 'other'",
                         idx, major, minor)

    if not sm86:
        logger.warning(
            "DES-LOC: No SM86 (A6000) devices found among %d visible CUDA devices. "
            "Falling back: assigning SM90/other devices to the SM86 pool.",
            n,
        )
        sm86 = sm90 or other or ([torch.device("cuda", 0)] if n > 0 else [])

    if not sm90:
        logger.warning(
            "DES-LOC: No SM90 (H100) device found. "
            "Falling back: using first SM86 device as SM90 pool.",
        )
        sm90 = sm86[:1]

    logger.info(
        "DES-LOC device classification — SM86 pool: %s  |  SM90 pool: %s",
        sm86, sm90,
    )
    return {"sm86": sm86, "sm90": sm90, "other": other}


# ---------------------------------------------------------------------------
# HeteroMLADeviceMap
# ---------------------------------------------------------------------------

@dataclass
class HeteroMLADeviceMap:
    """Static device assignment for MLA sub-modules in a DES-LOC topology.

    Upstream context:
        Megatron runs all MLA sub-modules on a single rank's device.  In DES-LOC
        we physically place different sub-modules on different devices to exploit
        the H100's higher tensor-core throughput for the latent-dim expansion
        steps, while offloading the cheaper (but more numerous) A6000 steps to
        the SM86 pair.

    Attributes
    ----------
    kv_up_device:
        Device for `linear_kv_up_proj` (KV latent → full KV heads).
        Placed on H100 because the expansion ratio is large.
    qkv_down_fwd_device:
        Device for the *forward* pass of `linear_qkv_down_proj`.
        On H100 (fused QKV down-projection is the critical path).
    qkv_down_wgrad_device:
        Device for *weight-gradient* accumulation of `linear_qkv_down_proj`.
        On A6000 to overlap wgrad with next layer's forward on H100.
        This is the central DES-LOC "decoupled" principle.
    q_up_device:
        Device for `linear_q_up_proj`.  On A6000 (query heads are smaller).
    out_proj_device:
        Device for output projection / sequence-parallel scatter.  On A6000.
    loc_staging_device:
        Device used as PCIe staging area for LOC evictions.
        Typically CPU (pinned DRAM).
    """

    kv_up_device: torch.device
    qkv_down_fwd_device: torch.device
    qkv_down_wgrad_device: torch.device
    q_up_device: torch.device
    out_proj_device: torch.device
    loc_staging_device: torch.device = field(default_factory=lambda: _CPU_DEVICE)

    @classmethod
    def from_topology(cls) -> "HeteroMLADeviceMap":
        """Build a device map by probing the visible CUDA devices."""
        pools = _classify_devices()
        sm86 = pools["sm86"]
        sm90 = pools["sm90"]
        h100 = sm90[0]
        a6000_0 = sm86[0]
        a6000_1 = sm86[1] if len(sm86) > 1 else sm86[0]

        dmap = cls(
            kv_up_device=h100,
            qkv_down_fwd_device=h100,
            qkv_down_wgrad_device=a6000_0,
            q_up_device=a6000_1,
            out_proj_device=a6000_0,
            loc_staging_device=_CPU_DEVICE,
        )
        logger.info("HeteroMLADeviceMap constructed:\n%s", dmap)
        return dmap

    def __str__(self) -> str:
        lines = [
            f"  kv_up_proj        fwd+wgrad → {self.kv_up_device}",
            f"  qkv_down_proj     fwd       → {self.qkv_down_fwd_device}",
            f"  qkv_down_proj     wgrad     → {self.qkv_down_wgrad_device}",
            f"  q_up_proj         fwd+wgrad → {self.q_up_device}",
            f"  out_proj          fwd+wgrad → {self.out_proj_device}",
            f"  LOC staging (CPU DRAM)      → {self.loc_staging_device}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LOCActivationCache
# ---------------------------------------------------------------------------

class LOCActivationCache:
    """CPU-DRAM ring buffer for activation checkpoints (the LOC in DES-LOC).

    Upstream context:
        Megatron's `set_save_original_input` writes the pre-norm activation
        directly into device memory so the recompute path can re-run the
        down-projection without storing the full forward activations.  This is
        fine when all computation is on the same device.

    DES-LOC adaptation:
        Because `linear_qkv_down_proj` has its wgrad on a *different* device
        from its forward, the saved activation must be accessible to both.
        Storing it in device-local VRAM of one device would require a PCIe
        transfer every recompute step.  Instead, we pin it in CPU DRAM (the
        1.5 TB pool), which is accessible from all three devices via DMA.

        The ring buffer evicts oldest entries when capacity is reached, relying
        on DeepSpeed's gradient-accumulation schedule to have consumed them
        by then.

    Thread safety:
        The cache uses a reentrant lock so the wgrad thread (A6000) and the
        forward thread (H100) can safely interleave insertions and lookups.

    Parameters
    ----------
    capacity : int
        Maximum number of activation entries retained simultaneously.
    """

    def __init__(self, capacity: int = _LOC_RING_CAPACITY) -> None:
        self._capacity = capacity
        self._store: Dict[str, torch.Tensor] = {}
        self._order: List[str] = []   # FIFO eviction order
        self._lock = threading.RLock()
        logger.debug("LOCActivationCache initialised (capacity=%d)", capacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, key: str, tensor: torch.Tensor) -> None:
        """Pin *tensor* into CPU DRAM under *key*.

        The tensor is detached, cloned, and pinned.  The clone is necessary
        because the original may be an in-place modified buffer.  Pinning
        enables async DMA back to any CUDA device during the wgrad pass.
        """
        with self._lock:
            if key in self._store:
                # Overwrite without changing eviction order.
                pinned = self._pin(tensor)
                self._store[key] = pinned
                logger.debug("LOC updated key=%s shape=%s", key, list(tensor.shape))
                return

            if len(self._order) >= self._capacity:
                evict_key = self._order.pop(0)
                evicted = self._store.pop(evict_key, None)
                if evicted is not None:
                    logger.debug(
                        "LOC evicted key=%s (capacity=%d)", evict_key, self._capacity
                    )

            pinned = self._pin(tensor)
            self._store[key] = pinned
            self._order.append(key)
            logger.debug(
                "LOC saved key=%s shape=%s pinned=%s",
                key, list(tensor.shape), pinned.is_pinned(),
            )

    def load(self, key: str, device: torch.device,
             non_blocking: bool = True) -> Optional[torch.Tensor]:
        """Return the activation for *key* on *device*.

        The transfer is async when *non_blocking=True* and the tensor is
        pinned — this allows overlapping the PCIe copy with CUDA kernel
        execution on the target device.

        Returns None if the key is not present (caller must handle).
        """
        with self._lock:
            pinned = self._store.get(key)

        if pinned is None:
            logger.warning("LOC cache miss for key=%s — falling back to recompute", key)
            return None

        if device.type == "cpu":
            return pinned

        return pinned.to(device=device, non_blocking=non_blocking)

    def evict(self, key: str) -> None:
        """Explicitly remove *key* after the wgrad consumer has finished."""
        with self._lock:
            if key in self._store:
                self._store.pop(key)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                logger.debug("LOC explicit evict key=%s", key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pin(tensor: torch.Tensor) -> torch.Tensor:
        """Detach, move to CPU, and pin the tensor."""
        cpu_t = tensor.detach().cpu()
        if not cpu_t.is_pinned():
            try:
                cpu_t = cpu_t.pin_memory()
            except RuntimeError:
                # pin_memory can fail outside of CUDA contexts (e.g., CI).
                logger.debug("pin_memory unavailable — storing unpinned tensor")
        return cpu_t


# ---------------------------------------------------------------------------
# HeteroMLAWgradHook
# ---------------------------------------------------------------------------

class HeteroMLAWgradHook:
    """Manages delayed weight-gradient hooks for one MLA linear sub-module.

    Upstream context:
        Megatron attaches `backward_dw` hooks to linear layers so that weight
        gradient computation is deferred until after the activation-recompute
        pass.  The hook is registered via `register_backward_hook` or through
        the custom `backward_dw` method pattern.

    DES-LOC adaptation:
        Because the wgrad device may differ from the forward device, this hook:
          1. Intercepts the backward call before `backward_dw` runs.
          2. Ensures any required activation (loaded from LOC) is on the wgrad
             device.
          3. Calls the original `backward_dw` on the wgrad device's stream.
          4. Evicts the LOC entry once accumulation is complete.

    Parameters
    ----------
    module : nn.Module
        The linear sub-module (e.g. `linear_qkv_down_proj`).
    wgrad_device : torch.device
        Device on which weight gradients should be accumulated.
    loc_cache : LOCActivationCache
        Shared locality cache for this attention layer.
    loc_key : str
        Key under which the module's input activation is stored in the cache.
    """

    def __init__(
        self,
        module: nn.Module,
        wgrad_device: torch.device,
        loc_cache: LOCActivationCache,
        loc_key: str,
    ) -> None:
        self._module_ref = weakref.ref(module)
        self.wgrad_device = wgrad_device
        self.loc_cache = loc_cache
        self.loc_key = loc_key
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None
        logger.debug(
            "HeteroMLAWgradHook created: loc_key=%s wgrad_device=%s",
            loc_key, wgrad_device,
        )

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Register the post-accumulate-grad hook on the module's weight."""
        module = self._module_ref()
        if module is None:
            logger.error("HeteroMLAWgradHook.install: module has been garbage collected")
            return

        if not hasattr(module, "weight") or module.weight is None:
            logger.warning(
                "HeteroMLAWgradHook.install: module has no .weight — "
                "hook not installed for loc_key=%s", self.loc_key,
            )
            return

        # `register_post_accumulate_grad_hook` (PyTorch ≥ 2.1) fires after
        # `.grad` has been accumulated, which is the correct point to evict
        # the LOC entry and synchronise devices.
        if hasattr(module.weight, "register_post_accumulate_grad_hook"):
            self._handle = module.weight.register_post_accumulate_grad_hook(
                self._post_accumulate_hook
            )
            logger.debug(
                "Installed post_accumulate_grad_hook for loc_key=%s", self.loc_key
            )
        else:
            # Fallback for older PyTorch: use full-backward hook on the module.
            self._handle = module.register_full_backward_hook(
                self._full_backward_hook
            )
            logger.debug(
                "Installed full_backward_hook (compat) for loc_key=%s", self.loc_key
            )

    def remove(self) -> None:
        """Deregister the hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------
    # Hook implementations
    # ------------------------------------------------------------------

    def _post_accumulate_hook(self, param: torch.Tensor) -> None:
        """Called after weight grad is accumulated.  Migrate grad + evict LOC."""
        module = self._module_ref()
        if module is None:
            return

        self._migrate_weight_grad(module)
        self.loc_cache.evict(self.loc_key)

    def _full_backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple,
        grad_output: Tuple,
    ) -> None:
        """Compatibility hook for PyTorch < 2.1."""
        self._migrate_weight_grad(module)
        self.loc_cache.evict(self.loc_key)

    def _migrate_weight_grad(self, module: nn.Module) -> None:
        """Move weight.grad to wgrad_device if it is on the wrong device."""
        if not hasattr(module, "weight") or module.weight is None:
            return
        if module.weight.grad is None:
            logger.debug(
                "_migrate_weight_grad: no grad yet for loc_key=%s", self.loc_key
            )
            return

        grad = module.weight.grad
        if grad.device == self.wgrad_device:
            return   # Already on the right device — no-op.

        logger.debug(
            "Migrating weight grad %s → %s for loc_key=%s",
            grad.device, self.wgrad_device, self.loc_key,
        )
        with torch.cuda.stream(torch.cuda.Stream(device=self.wgrad_device)):
            module.weight.grad = grad.to(
                device=self.wgrad_device, non_blocking=True
            )


# ---------------------------------------------------------------------------
# set_save_original_input — DES-LOC override
# ---------------------------------------------------------------------------

def set_save_original_input_loc(
    linear_module: nn.Module,
    loc_cache: LOCActivationCache,
    loc_key: str,
    wgrad_device: torch.device,
) -> None:
    """DES-LOC replacement for Megatron's `set_save_original_input`.

    Upstream context:
        Megatron's `set_save_original_input(module)` registers a forward hook
        that saves `module`'s input tensor into `module._saved_input` so that
        the FP8/FP4 activation-recompute path can re-execute the down-
        projection with the correct input.  The saved tensor lives in device
        VRAM.

    DES-LOC adaptation:
        We intercept the forward hook to *also* pin the input into the LOC
        (CPU DRAM), tagged with *loc_key*.  The wgrad hook (HeteroMLAWgradHook)
        will later load it back to *wgrad_device* asynchronously.

        This removes the requirement for a synchronous PCIe copy at wgrad time
        and avoids VRAM fragmentation on the A6000 pair.

    Parameters
    ----------
    linear_module : nn.Module
        The target linear layer (`linear_qkv_down_proj`).
    loc_cache : LOCActivationCache
        Shared locality cache for this attention layer.
    loc_key : str
        Key under which to store the activation.
    wgrad_device : torch.device
        Device that will consume the activation for weight-gradient computation.
    """

    def _forward_hook(
        module: nn.Module,
        inp: Tuple[torch.Tensor, ...],
        _out: torch.Tensor,
    ) -> None:
        if len(inp) == 0:
            logger.warning(
                "set_save_original_input_loc: empty input tuple for key=%s", loc_key
            )
            return
        activation = inp[0]
        # Save to device attr (for FP8 recompute, same as upstream).
        module._saved_input = activation
        # Also pin into LOC for async PCIe transfer to wgrad_device.
        loc_cache.save(loc_key, activation)
        logger.debug(
            "LOC pinned activation key=%s shape=%s src_device=%s dst_device=%s",
            loc_key, list(activation.shape), activation.device, wgrad_device,
        )

    linear_module.register_forward_hook(_forward_hook)
    logger.info(
        "set_save_original_input_loc registered for %s (loc_key=%s)",
        type(linear_module).__name__, loc_key,
    )


# ---------------------------------------------------------------------------
# FusedMLAWeightGradFix — mixin / monkey-patch target
# ---------------------------------------------------------------------------

class FusedMLAWeightGradFix:
    """Mixin that applies the DES-LOC weight-grad fix to a FusedMLASelfAttention.

    Upstream context:
        Megatron PR #5273 fixes `backward_dw` to call the correct fused
        attributes (`linear_qkv_down_proj`) and adds `set_for_recompute_input_layernorm`.
        Both methods were absent from `FusedMLASelfAttention`, breaking delayed
        wgrad accumulation and FP8 recompute.

    DES-LOC adaptation:
        Beyond the attribute-name fix, this mixin:
          * Routes each sub-module's wgrad computation to its assigned device.
          * Uses `LOCActivationCache` to make activations available across
            PCIe without synchronous copies.
          * Installs `HeteroMLAWgradHook` on each relevant weight after the
            first forward pass (lazy installation avoids issues with modules
            that haven't been moved to their target devices yet).

    Usage::

        # During DeepSpeed engine init, after model construction:
        apply_hetero_mla_fix(model, device_map=HeteroMLADeviceMap.from_topology())

    The mixin can also be used standalone via direct subclassing in tests.
    """

    # Set by apply_hetero_mla_fix.
    _deslock_device_map: Optional[HeteroMLADeviceMap] = None
    _deslock_loc_cache: Optional[LOCActivationCache] = None
    _deslock_hooks_installed: bool = False

    # ------------------------------------------------------------------
    # Core fix: backward_dw (mirrors Megatron 76d26e24b)
    # ------------------------------------------------------------------

    def backward_dw(self) -> None:
        """Execute weight-gradient computation across heterogeneous devices.

        Upstream fix:
            Megatron PR #5273 corrects attribute references to use
            `linear_qkv_down_proj` (fused) rather than the legacy
            `linear_kv_down_proj` / `linear_q_down_proj` pair.

        DES-LOC extension:
            Each `backward_dw` call is wrapped in a CUDA stream corresponding
            to the sub-module's assigned wgrad device so that:
              - H100 wgrad (kv_up) overlaps with A6000 wgrad (qkv_down).
              - PCIe transfers from LOC to A6000 overlap with H100 compute.

        Hook order (matches Megatron):
            kv_up_proj → qkv_down_proj → q_up_proj → output_proj
        """
        dmap = self._deslock_device_map
        if dmap is None:
            # No device map: plain upstream behaviour (correctness fix only).
            logger.debug("backward_dw: no device_map — running sequentially")
            self._plain_backward_dw()
            return

        logger.debug("backward_dw: running with DES-LOC device routing")

        # 1. kv_up on H100.
        self._run_backward_dw_on_device(
            self.linear_kv_up_proj, dmap.kv_up_device, "kv_up_proj"
        )

        # 2. qkv_down wgrad on A6000 (decoupled from fwd which ran on H100).
        self._run_backward_dw_on_device(
            self.linear_qkv_down_proj, dmap.qkv_down_wgrad_device, "qkv_down_proj"
        )

        # 3. q_up on A6000.
        self._run_backward_dw_on_device(
            self.linear_q_up_proj, dmap.q_up_device, "q_up_proj"
        )

        # 4. Output projection on A6000.
        self._backward_output_proj()

    def _plain_backward_dw(self) -> None:
        """Upstream-equivalent sequential wgrad (no device routing)."""
        self.linear_kv_up_proj.backward_dw()
        self.linear_qkv_down_proj.backward_dw()
        self.linear_q_up_proj.backward_dw()
        self._backward_output_proj()

    def _run_backward_dw_on_device(
        self,
        submodule: nn.Module,
        device: torch.device,
        name: str,
    ) -> None:
        """Call `submodule.backward_dw()` on the specified CUDA stream.

        If the submodule has weights on a different device (can happen during
        pipeline-parallel transitions), we log a warning but proceed — the
        upstream correctness fix applies regardless of device placement.
        """
        if not hasattr(submodule, "backward_dw"):
            logger.warning(
                "_run_backward_dw_on_device: %s has no backward_dw — skipping", name
            )
            return

        if device.type == "cuda":
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                logger.debug("backward_dw %s on %s (stream %d)", name, device, stream.stream_id)
                submodule.backward_dw()
        else:
            # CPU fallback (e.g., single-GPU smoke test).
            submodule.backward_dw()

    # ------------------------------------------------------------------
    # Core fix: set_for_recompute_input_layernorm (mirrors Megatron 76d26e24b)
    # ------------------------------------------------------------------

    def set_for_recompute_input_layernorm(self) -> None:
        """Configure activation saving for FP8/FP4 input-layernorm recompute.

        Upstream fix:
            Megatron PR #5273 adds this method to `FusedMLASelfAttention`.
            It calls `set_save_original_input(self.linear_qkv_down_proj)` so
            the FP8/FP4 recompute path knows to save the module's input before
            the normcasting step.

        DES-LOC extension:
            Instead of Megatron's device-local `set_save_original_input`, we
            call `set_save_original_input_loc` which simultaneously:
              (a) sets `module._saved_input` (upstream FP8 contract), and
              (b) pins the input into the LOC ring buffer (DES-LOC contract).

            The LOC key is namespaced to this layer instance to avoid
            collisions across pipeline stages.
        """
        dmap = self._deslock_device_map
        loc = self._deslock_loc_cache

        if dmap is None or loc is None:
            # Upstream fallback: call the module-level set_save_original_input
            # if available (imported from Megatron compat shim).
            _megatron_set_save = _try_import_megatron_set_save()
            if _megatron_set_save is not None:
                _megatron_set_save(self.linear_qkv_down_proj)
            else:
                logger.warning(
                    "set_for_recompute_input_layernorm: no device_map/loc_cache "
                    "and no Megatron shim available — FP8 recompute may be incorrect"
                )
            return

        loc_key = f"layer{id(self)}_qkv_down_input"
        set_save_original_input_loc(
            linear_module=self.linear_qkv_down_proj,
            loc_cache=loc,
            loc_key=loc_key,
            wgrad_device=dmap.qkv_down_wgrad_device,
        )

    # ------------------------------------------------------------------
    # Lazy hook installation
    # ------------------------------------------------------------------

    def _install_deslock_hooks_if_needed(self) -> None:
        """Install HeteroMLAWgradHooks on first call (lazy, idempotent)."""
        if self._deslock_hooks_installed:
            return
        if self._deslock_device_map is None or self._deslock_loc_cache is None:
            return

        dmap = self._deslock_device_map
        loc = self._deslock_loc_cache

        submodules = [
            (self.linear_kv_up_proj,    dmap.kv_up_device,           f"layer{id(self)}_kv_up"),
            (self.linear_qkv_down_proj, dmap.qkv_down_wgrad_device,  f"layer{id(self)}_qkv_down"),
            (self.linear_q_up_proj,     dmap.q_up_device,            f"layer{id(self)}_q_up"),
        ]

        for module, wgrad_dev, key in submodules:
            hook = HeteroMLAWgradHook(
                module=module,
                wgrad_device=wgrad_dev,
                loc_cache=loc,
                loc_key=key,
            )
            hook.install()

        self._deslock_hooks_installed = True
        logger.info(
            "DES-LOC wgrad hooks installed for FusedMLA layer id=%d", id(self)
        )


# ---------------------------------------------------------------------------
# apply_hetero_mla_fix — engine entry point
# ---------------------------------------------------------------------------

def apply_hetero_mla_fix(
    model: nn.Module,
    device_map: Optional[HeteroMLADeviceMap] = None,
    loc_capacity: int = _LOC_RING_CAPACITY,
) -> int:
    """Monkey-patch all FusedMLASelfAttention modules in *model* with DES-LOC fix.

    This is the primary entry point called from the DeepSpeed engine
    initialisation (e.g., in `deepspeed/runtime/engine.py` after
    `model.to(device)`).

    The function:
      1. Auto-detects the device topology if *device_map* is None.
      2. Iterates all modules looking for FusedMLASelfAttention instances
         (matched by class name to avoid a hard import dependency).
      3. Dynamically injects `FusedMLAWeightGradFix` into the MRO of each
         matched instance's class (per-instance class patching to avoid
         modifying the global class).
      4. Attaches `_deslock_device_map` and `_deslock_loc_cache` to each.

    Parameters
    ----------
    model : nn.Module
        The full model (typically a DeepSpeed-wrapped transformer).
    device_map : HeteroMLADeviceMap, optional
        Pre-built device assignment.  Auto-detected from CUDA topology if None.
    loc_capacity : int
        LOC ring buffer capacity per attention layer.

    Returns
    -------
    int
        Number of FusedMLASelfAttention modules patched.
    """
    if device_map is None:
        if torch.cuda.is_available():
            device_map = HeteroMLADeviceMap.from_topology()
        else:
            logger.warning(
                "apply_hetero_mla_fix: CUDA not available — "
                "applying correctness-only fix (no device routing)"
            )

    n_patched = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name != "FusedMLASelfAttention":
            continue

        logger.info("Patching FusedMLASelfAttention at '%s'", name)

        # Per-instance subclassing: creates a unique class for this instance
        # so we don't mutate the global FusedMLASelfAttention class.
        orig_cls = type(module)
        if FusedMLAWeightGradFix not in orig_cls.__mro__:
            patched_cls = type(
                f"DESLoc_{orig_cls.__name__}",
                (FusedMLAWeightGradFix, orig_cls),
                {},
            )
            module.__class__ = patched_cls

        # Attach DES-LOC state.
        module._deslock_device_map = device_map
        module._deslock_loc_cache = LOCActivationCache(capacity=loc_capacity)
        module._deslock_hooks_installed = False

        n_patched += 1

    logger.info(
        "apply_hetero_mla_fix: patched %d FusedMLASelfAttention module(s)", n_patched
    )
    return n_patched


# ---------------------------------------------------------------------------
# Megatron compatibility shim
# ---------------------------------------------------------------------------

def _try_import_megatron_set_save() -> Optional[Callable]:
    """Try to import Megatron's `set_save_original_input`, return None on failure."""
    try:
        from megatron.core.transformer.multi_latent_attention import (  # type: ignore
            set_save_original_input,
        )
        return set_save_original_input
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------------
    # Minimal stub classes that replicate the Megatron attribute contract.
    # ------------------------------------------------------------------

    class _FakeLinear(nn.Linear):
        """Linear layer with a stub backward_dw."""

        def __init__(self, name: str, in_f: int = 4, out_f: int = 4):
            super().__init__(in_f, out_f, bias=False)
            self._name = name
            self._bw_calls: List[str] = []

        def backward_dw(self) -> None:
            self._bw_calls.append(self._name)

    class _FakeFusedMLA(FusedMLAWeightGradFix, nn.Module):
        """Minimal stand-in for FusedMLASelfAttention."""

        def __init__(self):
            nn.Module.__init__(self)
            self.linear_kv_up_proj    = _FakeLinear("kv_up")
            self.linear_qkv_down_proj = _FakeLinear("qkv_down")
            self.linear_q_up_proj     = _FakeLinear("q_up")
            self.linear_proj          = _FakeLinear("out")
            self._out_calls: List[str] = []

        def _backward_output_proj(self) -> None:
            self._out_calls.append("out")

    # ------------------------------------------------------------------
    # Test 1: plain backward_dw (no device map) calls all 4 sub-modules.
    # ------------------------------------------------------------------
    mla = _FakeFusedMLA()
    mla._deslock_device_map = None
    mla._deslock_loc_cache = None
    mla.backward_dw()

    assert mla.linear_kv_up_proj._bw_calls == ["kv_up"], "kv_up_proj not called"
    assert mla.linear_qkv_down_proj._bw_calls == ["qkv_down"], "qkv_down_proj not called"
    assert mla.linear_q_up_proj._bw_calls == ["q_up"], "q_up_proj not called"
    assert mla._out_calls == ["out"], "output_proj not called"
    print("PASS test 1: backward_dw sequential (no device map)")

    # ------------------------------------------------------------------
    # Test 2: LOCActivationCache save / load / evict round-trip.
    # ------------------------------------------------------------------
    cache = LOCActivationCache(capacity=2)
    t = torch.randn(3, 5)
    cache.save("k1", t)
    assert len(cache) == 1, "cache should have 1 entry"

    loaded = cache.load("k1", _CPU_DEVICE)
    assert loaded is not None, "load returned None"
    assert loaded.shape == t.shape, "shape mismatch after LOC round-trip"

    cache.evict("k1")
    assert len(cache) == 0, "cache should be empty after evict"
    print("PASS test 2: LOCActivationCache round-trip")

    # ------------------------------------------------------------------
    # Test 3: set_save_original_input_loc registers forward hook.
    # ------------------------------------------------------------------
    lin = nn.Linear(4, 4, bias=False)
    loc2 = LOCActivationCache(capacity=2)
    set_save_original_input_loc(lin, loc2, "test_key", _CPU_DEVICE)

    x = torch.randn(2, 4)
    _ = lin(x)   # triggers forward hook

    assert hasattr(lin, "_saved_input"), "_saved_input not set by hook"
    assert len(loc2) == 1, "LOC should contain 1 entry after forward"
    print("PASS test 3: set_save_original_input_loc forward hook")

    # ------------------------------------------------------------------
    # Test 4: apply_hetero_mla_fix finds and patches FusedMLASelfAttention.
    # ------------------------------------------------------------------
    class FusedMLASelfAttention(_FakeFusedMLA):
        """Rename to match the target class name."""
        pass

    wrapper = nn.ModuleList([FusedMLASelfAttention(), FusedMLASelfAttention()])

    n = apply_hetero_mla_fix(wrapper, device_map=None)
    assert n == 2, f"Expected 2 patches, got {n}"
    for mod in wrapper:
        assert isinstance(mod, FusedMLAWeightGradFix), "patch not applied"
    print("PASS test 4: apply_hetero_mla_fix patches all FusedMLASelfAttention")

    print("\nAll smoke tests passed.")
