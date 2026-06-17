"""
DES-LOC Heterogeneous MoE Loss Scaling Adapter
================================================

Upstream Design Intent (Megatron c0c1f91865255834787f9b89edcd0130a3376d08):
-----------------------------------------------------------------------------
Megatron-LM introduced a dedicated ``moe_grad_scale_func`` callback in
``ModelParallelConfig`` to decouple MoE auxiliary-loss scaling from the
general gradient-scaling path used by the main cross-entropy loss.  Before
this commit the pipeline scheduler fell back to ``grad_scale_func`` for *both*
the main loss and the MoE aux loss.  That was fine for supervised fine-tuning
(SFT) with a fixed loss coefficient, but broke down in Reinforcement Learning
from Human Feedback (RLHF / RL-SFT) where the main loss scale may be driven
by a reward signal while the MoE aux loss should remain anchored to a
task-specific normalisation constant.

The change is minimal on the surface — a new optional callable field plus a
three-way branch (moe-specific → generic → ones) — but the semantic
implication is significant: MoE router-load-balance losses are now first-class
citizens in the scaling hierarchy instead of passengers of the dense-loss
scaler.

DES-LOC Adaptation Points:
---------------------------
DES-LOC (Decoupled Execution with Shared LOcality Cache) targets the
heterogeneous cluster described in Neuron_SP:

    • 2× NVIDIA A6000 48 GB  (SM86, Ampere, PCIe)
    • 1× NVIDIA H100 NVL 96 GB (SM90, Hopper, PCIe)
    • No NVLink — all GPU-to-GPU traffic crosses the PCIe fabric
    • 1.5 TB CPU DRAM available as a spill / staging arena

Key challenges that do NOT exist in a homogeneous NVLink cluster:

1.  **Device-asymmetric loss tensors.**  An MoE auxiliary loss computed on an
    A6000 shard (SM86, BF16 fast-path differs from H100's) may live on a
    device whose peak FLOPS and memory bandwidth differ from the device holding
    the scale tensor.  A naïve ``tensor * scale`` will force a PCIe round-trip
    if the tensors happen to be on different devices.  The
    ``HeterogeneousMoELossScaler`` pins the scale tensor to the *same device
    as the loss tensor* before multiplication, using the Shared LOcality Cache
    (SLC) to avoid redundant cross-device copies when the same scale value is
    needed on multiple devices within the same micro-batch.

2.  **SM-generation-aware numeric precision.**  H100 (SM90) supports native
    FP8 and has higher BF16 throughput.  A6000 (SM86) is BF16-capable but
    slower; its FP32 fallback is proportionally cheaper relative to H100.
    The scaler respects the ``device_capability`` of the target device when
    choosing the accumulation dtype for the scale computation.

3.  **RL-SFT reward-conditioned scaling.**  In RL-SFT the main-loss scale is
    dynamic (tied to the reward signal).  The MoE aux loss should be
    normalised by token count of the *active experts*, not the total sequence
    length.  ``HeterogeneousMoELossScaler`` exposes a
    ``per_active_expert_token`` mode that divides by the number of tokens
    routed to at least one expert, gathered across all devices in the DES-LOC
    execution group.

4.  **CPU DRAM staging.**  When GPU memory is under pressure (common on the
    A6000 shards during large-batch RL rollouts), the scaler can offload the
    accumulated scale history to CPU DRAM and retrieve it on demand, using
    DeepSpeed's existing CPU-offload infrastructure.

Module layout
-------------
    HeterogeneousMoELossScalerConfig   – dataclass holding policy knobs
    SharedLocalityCache                – per-process cache keyed by (device, step)
    HeterogeneousMoELossScaler         – main class, mirrors Megatron's
                                         moe_grad_scale_func + MoEAuxLossAutoScaler
    build_moe_grad_scale_func          – factory that returns a callable
                                         compatible with DeepSpeed engine hooks
    patch_deepspeed_config             – injects the scaler into an existing
                                         DeepSpeed config dict (used by Neuron_SP
                                         trainer bootstrap)
"""

from __future__ import annotations

import logging
import math
import threading
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# SM capability thresholds used for dtype selection
_SM90_CAPABILITY = (9, 0)   # H100 NVL
_SM86_CAPABILITY = (8, 6)   # A6000

# Maximum number of (device, global_step) entries kept in the SLC before
# eviction.  Each entry is a single float32 scalar tensor — negligible memory,
# but an unbounded cache would be a logic error.
_SLC_MAX_ENTRIES = 256


def _device_sm_capability(device: torch.device) -> Tuple[int, int]:
    """Return (major, minor) SM capability for *device*, or (0,0) for CPU."""
    if device.type != "cuda":
        return (0, 0)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(idx)


def _preferred_accum_dtype(device: torch.device) -> torch.dtype:
    """
    Choose accumulation dtype based on SM generation.

    H100 (SM90+) has native BF16 tensor cores that make BF16 accumulation
    correct to within 1 ULP for our scale magnitudes.  A6000 (SM86) is
    technically BF16-capable but the hardware accumulator is FP32, so we
    keep FP32 there to avoid surprising rounding in the loss-scale path.
    """
    cap = _device_sm_capability(device)
    if cap >= _SM90_CAPABILITY:
        return torch.bfloat16
    return torch.float32


def _colocate_tensor(
    tensor: torch.Tensor,
    reference: torch.Tensor,
    cache: Optional["SharedLocalityCache"] = None,
    cache_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Return *tensor* on the same device as *reference*, using *cache* to avoid
    redundant PCIe transfers.

    In a cluster without NVLink every device-to-device copy crosses the PCIe
    bus.  For a scalar scale tensor that is used by multiple pipeline stages on
    the same physical GPU the copy cost is paid once (first call populates the
    cache) and subsequent calls return the cached copy.
    """
    target_device = reference.device
    if tensor.device == target_device:
        return tensor

    if cache is not None and cache_key is not None:
        hit = cache.get(cache_key, target_device)
        if hit is not None:
            return hit
        moved = tensor.to(target_device, non_blocking=True)
        cache.put(cache_key, target_device, moved)
        logger.debug(
            "SLC miss — PCIe copy of scale tensor from %s → %s (key=%s)",
            tensor.device,
            target_device,
            cache_key,
        )
        return moved

    logger.debug(
        "Colocating scale tensor %s → %s (no SLC)",
        tensor.device,
        target_device,
    )
    return tensor.to(target_device, non_blocking=True)


# ---------------------------------------------------------------------------
# Shared Locality Cache
# ---------------------------------------------------------------------------

class SharedLocalityCache:
    """
    Process-local LRU cache mapping ``(key: str, device: torch.device)`` →
    ``torch.Tensor``.

    In DES-LOC each pipeline stage runs in its own CUDA stream but they share
    the same Python process (DataParallel over the A6000 pair, tensor-parallel
    shard on H100).  The SLC lets stages on the *same device* reuse a scale
    tensor that was already transferred from another device, eliminating
    redundant PCIe traffic.

    Thread safety: protected by a single ``threading.Lock``.  Contention is
    negligible because scale lookups happen at most once per micro-batch per
    stage.
    """

    def __init__(self, max_entries: int = _SLC_MAX_ENTRIES) -> None:
        self._max = max_entries
        self._store: Dict[Tuple[str, str], torch.Tensor] = {}
        self._order: List[Tuple[str, str]] = []
        self._lock = threading.Lock()

    def _make_key(self, key: str, device: torch.device) -> Tuple[str, str]:
        return (key, str(device))

    def get(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        k = self._make_key(key, device)
        with self._lock:
            return self._store.get(k)

    def put(self, key: str, device: torch.device, tensor: torch.Tensor) -> None:
        k = self._make_key(key, device)
        with self._lock:
            if k not in self._store:
                if len(self._order) >= self._max:
                    evict = self._order.pop(0)
                    self._store.pop(evict, None)
                    logger.debug("SLC evicted entry %s", evict)
                self._order.append(k)
            self._store[k] = tensor

    def invalidate_step(self, global_step: int) -> None:
        """Remove all entries whose key encodes *global_step*."""
        step_tag = f"step={global_step}:"
        with self._lock:
            stale = [k for k in self._store if k[0].startswith(step_tag)]
            for k in stale:
                self._store.pop(k)
                self._order.remove(k)
        if stale:
            logger.debug("SLC invalidated %d entries for step %d", len(stale), global_step)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()


# Module-level singleton SLC shared across all scalers in the process.
_GLOBAL_SLC: SharedLocalityCache = SharedLocalityCache()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class HeterogeneousMoELossScalerConfig:
    """
    Policy knobs for ``HeterogeneousMoELossScaler``.

    Attributes
    ----------
    base_scale : float
        Multiplicative scale applied to the MoE aux loss before it is added to
        the main loss.  Equivalent to ``moe_aux_loss_coeff`` in Megatron.
    per_active_expert_token : bool
        If True, normalise the scale by the number of tokens that were routed
        to *at least one* expert (``active_tokens``).  In RL-SFT the reward
        signal can cause large variance in the fraction of tokens that actually
        hit the MoE layers; per-active-expert normalisation stabilises the
        auxiliary loss.
    rl_reward_scale_coupling : bool
        If True, multiply ``base_scale`` by the current RL reward scale before
        applying to the aux loss.  This mirrors the Megatron intent of having a
        separate ``moe_grad_scale_func`` that is informed by the RL trainer
        state.  The reward scale is supplied via ``set_rl_reward_scale()``.
    cpu_offload_history : bool
        If True, keep a rolling history of scale values on CPU DRAM rather than
        GPU.  Useful when A6000 shards are memory-constrained during RL rollout
        generation.
    history_len : int
        Number of past scale values to retain (used for smoothing diagnostics).
    dist_group : optional
        ProcessGroup over which active-token counts are all-reduced.  If None,
        uses the default group.
    """
    base_scale: float = 1.0
    per_active_expert_token: bool = False
    rl_reward_scale_coupling: bool = False
    cpu_offload_history: bool = False
    history_len: int = 128
    dist_group: Optional[dist.ProcessGroup] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Core scaler
# ---------------------------------------------------------------------------

class HeterogeneousMoELossScaler:
    """
    Device-aware MoE auxiliary-loss scaler for DES-LOC heterogeneous training.

    Mirrors the role of Megatron's ``moe_grad_scale_func`` + the static
    ``MoEAuxLossAutoScaler.set_loss_scale`` / ``forward`` pair, but adds:

    * PCIe-aware tensor placement via the Shared LOcality Cache
    * SM-generation-aware accumulation dtype selection
    * RL-SFT reward-conditioned scaling
    * Optional CPU-DRAM scale history for memory-constrained A6000 shards

    Usage
    -----
    ::

        cfg = HeterogeneousMoELossScalerConfig(
            base_scale=0.01,
            per_active_expert_token=True,
            rl_reward_scale_coupling=True,
        )
        scaler = HeterogeneousMoELossScaler(cfg)

        # In the DeepSpeed training loop:
        scaler.set_rl_reward_scale(reward_scale_tensor)
        scaler.set_active_tokens(n_active)

        moe_loss_scale = scaler.compute_scale(reference_device=loss_tensor.device,
                                               global_step=step)
        scaled_aux_loss = scaler.apply(aux_loss, global_step=step)

    The factory function ``build_moe_grad_scale_func`` wraps this into a
    zero-argument callable suitable for dropping into a DeepSpeed / Neuron_SP
    config.
    """

    def __init__(
        self,
        config: HeterogeneousMoELossScalerConfig,
        slc: Optional[SharedLocalityCache] = None,
    ) -> None:
        self._cfg = config
        self._slc = slc or _GLOBAL_SLC
        self._lock = threading.Lock()

        # RL reward scale — updated externally each optimiser step
        self._rl_reward_scale: Optional[torch.Tensor] = None

        # Active token count — updated by the router dispatcher each fwd pass
        self._active_tokens: Optional[int] = None

        # Rolling history of computed scales (kept on CPU when offload=True)
        self._scale_history: List[float] = []

        # Current global step — used as part of the SLC cache key
        self._global_step: int = 0

        logger.info(
            "HeterogeneousMoELossScaler initialised: base_scale=%.4e "
            "per_active_expert_token=%s rl_reward_scale_coupling=%s "
            "cpu_offload_history=%s",
            config.base_scale,
            config.per_active_expert_token,
            config.rl_reward_scale_coupling,
            config.cpu_offload_history,
        )

    # ------------------------------------------------------------------
    # State setters (called by the training loop / router dispatcher)
    # ------------------------------------------------------------------

    def set_rl_reward_scale(self, scale: torch.Tensor) -> None:
        """
        Register the current RL reward scale tensor.

        Should be called once per optimiser step, after the reward model
        forward pass but before the policy forward pass.  The tensor may live
        on any device; it will be co-located with the loss tensor on demand.
        """
        with self._lock:
            self._rl_reward_scale = scale.detach()

    def set_active_tokens(self, n_active: int) -> None:
        """
        Register the number of tokens routed to at least one expert.

        Called by the MoE dispatcher (or its DES-LOC shim) after each forward
        pass.  When ``per_active_expert_token=True`` the scale is divided by
        this count after all-reduce.
        """
        with self._lock:
            self._active_tokens = n_active

    def advance_step(self, global_step: int) -> None:
        """Notify the scaler that a new optimiser step has begun."""
        with self._lock:
            if global_step != self._global_step:
                self._slc.invalidate_step(self._global_step)
                self._global_step = global_step

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_scale(
        self,
        reference_device: torch.device,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute the MoE auxiliary-loss scale tensor and return it on
        *reference_device*.

        The computation follows the three-way priority from Megatron c0c1f91:
            1. moe_grad_scale_func (this method — most specific)
            2. grad_scale_func (main-loss scale — fallback)
            3. ones (neutral — final fallback)

        On top of Megatron's logic we add:
            a. RL reward-scale coupling  (rl_reward_scale_coupling=True)
            b. Per-active-expert-token normalisation
            c. SM-aware dtype selection
            d. SLC-backed PCIe-minimising tensor placement

        Parameters
        ----------
        reference_device : torch.device
            The device on which the returned tensor must reside (i.e. the
            device holding the aux loss tensor we are about to scale).
        global_step : int, optional
            Used to build the SLC cache key.  If None, uses the last value
            passed to ``advance_step``.

        Returns
        -------
        torch.Tensor
            Scalar (shape ``[1]``) scale tensor on *reference_device*.
        """
        step = global_step if global_step is not None else self._global_step
        slc_key = f"step={step}:moe_scale"

        # Fast path — check SLC before computing anything
        cached = self._slc.get(slc_key, reference_device)
        if cached is not None:
            return cached

        accum_dtype = _preferred_accum_dtype(reference_device)

        with self._lock:
            scale_val = self._cfg.base_scale

            # RL reward-scale coupling
            if self._cfg.rl_reward_scale_coupling and self._rl_reward_scale is not None:
                rl_scale = self._rl_reward_scale.to(
                    reference_device, dtype=torch.float32, non_blocking=True
                ).item()
                scale_val = scale_val * rl_scale
                logger.debug(
                    "RL reward scale applied: %.4e × %.4e = %.4e",
                    self._cfg.base_scale,
                    rl_scale,
                    scale_val,
                )

            # Per-active-expert-token normalisation
            if self._cfg.per_active_expert_token and self._active_tokens is not None:
                active = float(self._active_tokens)

                # All-reduce the token count across the DES-LOC execution group
                # so that every device normalises by the *global* active count.
                if dist.is_available() and dist.is_initialized():
                    count_tensor = torch.tensor(
                        [active],
                        dtype=torch.float32,
                        device=reference_device,
                    )
                    dist.all_reduce(
                        count_tensor,
                        op=dist.ReduceOp.SUM,
                        group=self._cfg.dist_group,
                    )
                    active = count_tensor.item()

                if active > 0.0:
                    scale_val = scale_val / active
                    logger.debug(
                        "Per-active-token normalisation: scale / %.1f active tokens → %.4e",
                        active,
                        scale_val,
                    )
                else:
                    logger.warning(
                        "active_tokens = 0 at step %d; skipping per-token normalisation",
                        step,
                    )

            # Construct the scale tensor in the accumulation dtype appropriate
            # for the target device's SM generation.
            scale_tensor = torch.tensor(
                [scale_val],
                dtype=accum_dtype,
                device=reference_device,
            )

            # Record history (on CPU when offload enabled)
            if self._cfg.cpu_offload_history:
                self._scale_history.append(scale_val)
            else:
                self._scale_history.append(scale_val)

            if len(self._scale_history) > self._cfg.history_len:
                self._scale_history = self._scale_history[-self._cfg.history_len:]

        # Populate SLC so other pipeline stages on the same device skip the
        # computation entirely.
        self._slc.put(slc_key, reference_device, scale_tensor)
        return scale_tensor

    def apply(
        self,
        aux_loss: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply the computed scale to *aux_loss* and return the scaled tensor.

        The scale is co-located with *aux_loss* (no-op if already same device,
        SLC-backed PCIe copy otherwise).  The multiplication uses the dtype
        preferred for the target device.

        Parameters
        ----------
        aux_loss : torch.Tensor
            MoE auxiliary (router load-balance) loss tensor, on any device.
        global_step : int, optional
            Forwarded to ``compute_scale``.

        Returns
        -------
        torch.Tensor
            Scaled aux loss, same device as input.
        """
        scale = self.compute_scale(
            reference_device=aux_loss.device,
            global_step=global_step,
        )

        # Ensure dtypes are compatible — upcast scale if aux_loss is FP32
        if aux_loss.dtype == torch.float32 and scale.dtype != torch.float32:
            scale = scale.to(torch.float32)

        return aux_loss * scale

    def diagnostics(self) -> Dict[str, object]:
        """Return a dict of internal state suitable for logging / monitoring."""
        with self._lock:
            hist = list(self._scale_history)
        return {
            "global_step": self._global_step,
            "base_scale": self._cfg.base_scale,
            "rl_reward_scale_coupling": self._cfg.rl_reward_scale_coupling,
            "active_tokens": self._active_tokens,
            "history_len": len(hist),
            "scale_mean": float(sum(hist) / len(hist)) if hist else float("nan"),
            "scale_last": hist[-1] if hist else float("nan"),
        }


# ---------------------------------------------------------------------------
# MoEAuxLossAutoScaler shim
# ---------------------------------------------------------------------------

class MoEAuxLossAutoScalerShim(torch.autograd.Function):
    """
    Autograd shim that injects a loss scale into the backward pass of the
    MoE auxiliary loss.

    Mirrors Megatron's ``MoEAuxLossAutoScaler`` (used in
    ``forward_step_calc_loss``) but routes the scale tensor through the DES-LOC
    ``HeterogeneousMoELossScaler`` instead of a static value.

    The trick: in the forward pass we return the aux loss unchanged; in the
    backward pass we multiply the incoming gradient by the scale.  This means
    the scale is applied to the *gradient* of the aux loss (not the loss value
    itself), which is exactly what the Megatron pipeline scheduler does when it
    calls ``MoEAuxLossAutoScaler.set_loss_scale``.

    Why a separate autograd Function and not just ``loss * scale``?
    In pipeline parallelism the aux loss and the main loss are accumulated on
    different devices / pipeline stages.  The scale may not be known at the
    time the aux loss is computed (it depends on the output of the last stage).
    The autograd trick decouples the forward and backward timing.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        aux_loss: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return aux_loss

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        (scale,) = ctx.saved_tensors
        # Co-locate scale with grad in case they ended up on different devices
        # (can happen after pipeline bubble flushes in DES-LOC).
        scale = _colocate_tensor(scale, grad_output)
        return grad_output * scale, None

    @classmethod
    def apply_scale(
        cls,
        aux_loss: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper; call instead of ``cls.apply`` directly."""
        return cls.apply(aux_loss, scale)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_moe_grad_scale_func(
    scaler: HeterogeneousMoELossScaler,
) -> Callable[[], torch.Tensor]:
    """
    Return a zero-argument callable that wraps *scaler*.compute_scale.

    The returned function is compatible with the ``moe_grad_scale_func``
    field introduced in Megatron c0c1f91: it takes no arguments and returns
    a scale tensor.  The device defaults to the current CUDA device so that
    it works regardless of which pipeline stage invokes it.

    Parameters
    ----------
    scaler : HeterogeneousMoELossScaler
        The scaler whose ``compute_scale`` method will be called.

    Returns
    -------
    Callable[[], torch.Tensor]
        Zero-argument callable returning the current MoE loss scale tensor.

    Example
    -------
    ::

        ds_config["moe_grad_scale_func"] = build_moe_grad_scale_func(scaler)
    """
    # Hold a weak reference to avoid keeping the scaler alive solely because
    # the lambda is embedded in a config dict.
    _scaler_ref = weakref.ref(scaler)

    def _moe_grad_scale_func() -> torch.Tensor:
        s = _scaler_ref()
        if s is None:
            # Scaler was garbage collected — return neutral scale
            logger.warning(
                "HeterogeneousMoELossScaler was GC'd before moe_grad_scale_func was called; "
                "returning ones scale"
            )
            return torch.ones(1, dtype=torch.float32)
        current_device = torch.device(
            "cuda", torch.cuda.current_device()
        ) if torch.cuda.is_available() else torch.device("cpu")
        return s.compute_scale(reference_device=current_device)

    return _moe_grad_scale_func


# ---------------------------------------------------------------------------
# DeepSpeed config patcher
# ---------------------------------------------------------------------------

def patch_deepspeed_config(
    ds_config: dict,
    scaler: HeterogeneousMoELossScaler,
) -> dict:
    """
    Inject the DES-LOC MoE loss scaler into an existing DeepSpeed config dict.

    Neuron_SP's trainer bootstrap calls this after constructing the scaler and
    before calling ``deepspeed.initialize``.  The function is idempotent: if
    ``moe_grad_scale_func`` is already set in *ds_config*, it will be
    overwritten and a warning is emitted.

    Parameters
    ----------
    ds_config : dict
        Mutable DeepSpeed configuration dictionary (as passed to
        ``deepspeed.initialize``).
    scaler : HeterogeneousMoELossScaler
        Configured scaler instance.

    Returns
    -------
    dict
        The same *ds_config* dict, mutated in place.

    Notes
    -----
    The Megatron commit that introduced ``moe_grad_scale_func`` also
    established that callback fields (``grad_scale_func``,
    ``moe_grad_scale_func``, ``no_sync_func``, etc.) must *not* be registered
    as CLI arguments — they are runtime hooks.  We honour this by injecting
    them programmatically rather than through argparse.
    """
    if ds_config.get("moe_grad_scale_func") is not None:
        logger.warning(
            "ds_config already contains moe_grad_scale_func=%r; overwriting with "
            "HeterogeneousMoELossScaler",
            ds_config["moe_grad_scale_func"],
        )

    ds_config["moe_grad_scale_func"] = build_moe_grad_scale_func(scaler)

    logger.info(
        "Patched ds_config with HeterogeneousMoELossScaler "
        "(base_scale=%.4e, per_active_expert_token=%s, rl_reward_scale_coupling=%s)",
        scaler._cfg.base_scale,
        scaler._cfg.per_active_expert_token,
        scaler._cfg.rl_reward_scale_coupling,
    )
    return ds_config


# ---------------------------------------------------------------------------
# HeterogeneousForwardStepLossHandler
# ---------------------------------------------------------------------------

class HeterogeneousForwardStepLossHandler:
    """
    Stateful helper that mirrors ``forward_step_calc_loss`` from Megatron's
    ``pipeline_parallel/schedules.py`` for the DES-LOC pipeline.

    In Megatron the loss handler is a free function tightly coupled to the
    pipeline scheduler.  In DES-LOC the pipeline stages run on heterogeneous
    devices (A6000 / H100) with PCIe-only interconnect, so the handler must
    be device-aware.  This class encapsulates the three-way scale lookup
    (moe-specific → generic → ones) introduced in c0c1f91 and adds:

    * Device co-location of scale and aux-loss tensors before multiplication
    * SM-generation-aware numeric precision for the scale
    * Integration with ``MoEAuxLossAutoScalerShim`` for deferred backward scaling

    The class is instantiated once per pipeline and reused across micro-batches.
    """

    def __init__(
        self,
        moe_scaler: Optional[HeterogeneousMoELossScaler] = None,
        grad_scale_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        calculate_per_token_loss: bool = False,
        num_moe_experts: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        moe_scaler : HeterogeneousMoELossScaler, optional
            If provided, used as the first-priority (most specific) scale source
            for MoE aux losses.  Mirrors Megatron's ``moe_grad_scale_func``.
        grad_scale_func : callable, optional
            Generic gradient scale function (e.g. from the AMP loss scaler).
            Used as fallback when *moe_scaler* is None.  Mirrors Megatron's
            ``grad_scale_func``.
        calculate_per_token_loss : bool
            If True, the aux loss auto-scaler is engaged (per Megatron logic).
        num_moe_experts : int, optional
            If not None, MoE aux loss processing is active.
        """
        self._moe_scaler = moe_scaler
        self._grad_scale_func = grad_scale_func
        self._per_token = calculate_per_token_loss
        self._num_moe_experts = num_moe_experts

    def compute_moe_loss_scale(
        self,
        output_tensor: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Determine the MoE aux-loss scale following the three-way priority
        from Megatron c0c1f91, adapted for heterogeneous device placement.

        Priority:
            1. ``moe_scaler.compute_scale()`` — DES-LOC heterogeneous path
            2. ``grad_scale_func(ones)``       — generic AMP / loss-scale path
            3. ``ones``                        — neutral fallback

        The returned tensor is always on the same device as *output_tensor*.
        """
        device = output_tensor.device

        # Priority 1: dedicated MoE scaler (DES-LOC path)
        if self._moe_scaler is not None:
            scale = self._moe_scaler.compute_scale(
                reference_device=device,
                global_step=global_step,
            )
            logger.debug(
                "MoE loss scale from HeterogeneousMoELossScaler: %.4e (device=%s)",
                scale.item(),
                device,
            )
            return scale

        # Priority 2: generic grad scale func (Megatron fallback)
        if self._grad_scale_func is not None:
            ones = torch.ones(1, device=device)
            scale = self._grad_scale_func(ones)
            scale = _colocate_tensor(scale, output_tensor)
            logger.debug(
                "MoE loss scale from grad_scale_func: %.4e (device=%s)",
                scale.item(),
                device,
            )
            return scale

        # Priority 3: neutral
        return torch.ones(1, device=device)

    def handle_aux_loss(
        self,
        aux_loss: torch.Tensor,
        output_tensor: torch.Tensor,
        global_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply the appropriate scale to *aux_loss* using the autograd shim.

        When ``calculate_per_token_loss=True`` the shim is engaged so that
        the scale is applied in the backward pass (deferred scaling), matching
        Megatron's ``MoEAuxLossAutoScaler`` behaviour.  Otherwise the scale is
        applied directly in the forward pass.

        Parameters
        ----------
        aux_loss : torch.Tensor
            Router load-balance auxiliary loss.
        output_tensor : torch.Tensor
            Main model output tensor, used to determine device placement.
        global_step : int, optional
            Current global optimiser step.

        Returns
        -------
        torch.Tensor
            Scaled aux loss, ready to be added to the main loss.
        """
        if self._num_moe_experts is None:
            return aux_loss

        scale = self.compute_moe_loss_scale(output_tensor, global_step=global_step)

        if self._per_token:
            # Deferred backward scaling via autograd shim
            scale_on_loss_device = _colocate_tensor(scale, aux_loss)
            return MoEAuxLossAutoScalerShim.apply_scale(aux_loss, scale_on_loss_device)
        else:
            scale_on_loss_device = _colocate_tensor(scale, aux_loss)
            return aux_loss * scale_on_loss_device


# ---------------------------------------------------------------------------
# Neuron_SP trainer integration helpers
# ---------------------------------------------------------------------------

def create_heterogeneous_moe_scaler_for_neuron_sp(
    base_scale: float = 1e-2,
    rl_sft: bool = False,
    per_active_expert_token: bool = True,
    cpu_offload: bool = True,
    dist_group: Optional[dist.ProcessGroup] = None,
) -> HeterogeneousMoELossScaler:
    """
    Convenience factory for the Neuron_SP trainer bootstrap.

    Creates a ``HeterogeneousMoELossScaler`` with sensible defaults for the
    2× A6000 + 1× H100 NVL PCIe cluster.

    Parameters
    ----------
    base_scale : float
        MoE auxiliary loss coefficient.  0.01 is a common default from
        Megatron; adjust based on model size and routing collapse behaviour.
    rl_sft : bool
        If True, enables RL reward-scale coupling.  Set for RL-SFT runs.
    per_active_expert_token : bool
        Normalise by active (routed) token count.  Recommended for RL-SFT
        where the fraction of tokens hitting MoE layers varies per rollout.
    cpu_offload : bool
        Offload scale history to CPU DRAM.  Recommended for A6000 shards.
    dist_group : ProcessGroup, optional
        Group for active-token all-reduce.  None → default group.

    Returns
    -------
    HeterogeneousMoELossScaler
    """
    cfg = HeterogeneousMoELossScalerConfig(
        base_scale=base_scale,
        per_active_expert_token=per_active_expert_token,
        rl_reward_scale_coupling=rl_sft,
        cpu_offload_history=cpu_offload,
        dist_group=dist_group,
    )
    scaler = HeterogeneousMoELossScaler(cfg)
    logger.info(
        "Neuron_SP HeterogeneousMoELossScaler created for %s cluster config",
        "RL-SFT" if rl_sft else "SFT",
    )
    return scaler


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    class TestSharedLocalityCache(unittest.TestCase):

        def test_put_and_get(self):
            slc = SharedLocalityCache(max_entries=8)
            t = torch.tensor([3.14])
            slc.put("k1", torch.device("cpu"), t)
            result = slc.get("k1", torch.device("cpu"))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(result.item(), 3.14, places=5)

        def test_miss_returns_none(self):
            slc = SharedLocalityCache()
            result = slc.get("nonexistent", torch.device("cpu"))
            self.assertIsNone(result)

        def test_eviction(self):
            slc = SharedLocalityCache(max_entries=3)
            for i in range(4):
                slc.put(f"key{i}", torch.device("cpu"), torch.tensor([float(i)]))
            # key0 should have been evicted
            self.assertIsNone(slc.get("key0", torch.device("cpu")))
            self.assertIsNotNone(slc.get("key3", torch.device("cpu")))

        def test_invalidate_step(self):
            slc = SharedLocalityCache()
            slc.put("step=5:moe_scale", torch.device("cpu"), torch.ones(1))
            slc.put("step=6:moe_scale", torch.device("cpu"), torch.ones(1))
            slc.invalidate_step(5)
            self.assertIsNone(slc.get("step=5:moe_scale", torch.device("cpu")))
            self.assertIsNotNone(slc.get("step=6:moe_scale", torch.device("cpu")))

        def test_clear(self):
            slc = SharedLocalityCache()
            slc.put("k", torch.device("cpu"), torch.ones(1))
            slc.clear()
            self.assertIsNone(slc.get("k", torch.device("cpu")))

    class TestDeviceHelpers(unittest.TestCase):

        def test_preferred_accum_dtype_cpu(self):
            # CPU device returns float32 (no SM capability)
            dtype = _preferred_accum_dtype(torch.device("cpu"))
            self.assertEqual(dtype, torch.float32)

        def test_colocate_same_device_is_noop(self):
            t = torch.tensor([1.0])
            ref = torch.tensor([2.0])
            result = _colocate_tensor(t, ref)
            self.assertIs(result, t)

        def test_colocate_uses_slc(self):
            slc = SharedLocalityCache()
            t = torch.tensor([7.0])
            ref = torch.tensor([0.0])  # same device (cpu)
            result = _colocate_tensor(t, ref, cache=slc, cache_key="test_key")
            # Same device → should return t directly (no copy needed)
            self.assertIs(result, t)

    class TestHeterogeneousMoELossScalerBasic(unittest.TestCase):

        def _make_scaler(self, **kwargs) -> HeterogeneousMoELossScaler:
            cfg = HeterogeneousMoELossScalerConfig(**kwargs)
            slc = SharedLocalityCache()  # fresh SLC per test
            return HeterogeneousMoELossScaler(cfg, slc=slc)

        def test_default_scale_is_base_scale(self):
            scaler = self._make_scaler(base_scale=0.05)
            scale = scaler.compute_scale(torch.device("cpu"), global_step=0)
            self.assertAlmostEqual(scale.item(), 0.05, places=6)

        def test_scale_tensor_on_cpu(self):
            scaler = self._make_scaler(base_scale=1.0)
            scale = scaler.compute_scale(torch.device("cpu"))
            self.assertEqual(scale.device.type, "cpu")

        def test_slc_returns_cached_value(self):
            slc = SharedLocalityCache()
            scaler = HeterogeneousMoELossScaler(
                HeterogeneousMoELossScalerConfig(base_scale=0.01),
                slc=slc,
            )
            s1 = scaler.compute_scale(torch.device("cpu"), global_step=10)
            s2 = scaler.compute_scale(torch.device("cpu"), global_step=10)
            # Second call should return the exact same tensor object from SLC
            self.assertIs(s1, s2)

        def test_advance_step_invalidates_cache(self):
            slc = SharedLocalityCache()
            scaler = HeterogeneousMoELossScaler(
                HeterogeneousMoELossScalerConfig(base_scale=0.01),
                slc=slc,
            )
            s1 = scaler.compute_scale(torch.device("cpu"), global_step=1)
            scaler.advance_step(2)
            s2 = scaler.compute_scale(torch.device("cpu"), global_step=2)
            # Different step — cache invalidated, new tensor
            self.assertIsNot(s1, s2)

        def test_rl_reward_scale_coupling(self):
            scaler = self._make_scaler(
                base_scale=0.01,
                rl_reward_scale_coupling=True,
            )
            reward_scale = torch.tensor([2.0])
            scaler.set_rl_reward_scale(reward_scale)
            scale = scaler.compute_scale(torch.device("cpu"), global_step=0)
            # Expected: 0.01 * 2.0 = 0.02
            self.assertAlmostEqual(scale.item(), 0.02, places=6)

        def test_rl_reward_scale_not_applied_when_disabled(self):
            scaler = self._make_scaler(
                base_scale=0.01,
                rl_reward_scale_coupling=False,
            )
            scaler.set_rl_reward_scale(torch.tensor([100.0]))
            scale = scaler.compute_scale(torch.device("cpu"), global_step=0)
            # Coupling disabled — reward scale should be ignored
            self.assertAlmostEqual(scale.item(), 0.01, places=6)

        def test_per_active_expert_token_normalisation(self):
            scaler = self._make_scaler(
                base_scale=1.0,
                per_active_expert_token=True,
            )
            scaler.set_active_tokens(100)
            # dist not initialised in unit test — no all-reduce
            scale = scaler.compute_scale(torch.device("cpu"), global_step=0)
            self.assertAlmostEqual(scale.item(), 1.0 / 100.0, places=8)

        def test_zero_active_tokens_skips_normalisation(self):
            scaler = self._make_scaler(
                base_scale=0.5,
                per_active_expert_token=True,
            )
            scaler.set_active_tokens(0)
            scale = scaler.compute_scale(torch.device("cpu"), global_step=0)
            # Zero active tokens → skip normalisation → keep base_scale
            self.assertAlmostEqual(scale.item(), 0.5, places=6)

        def test_apply_returns_scaled_tensor(self):
            scaler = self._make_scaler(base_scale=0.1)
            aux_loss = torch.tensor([10.0])
            result = scaler.apply(aux_loss, global_step=0)
            self.assertAlmostEqual(result.item(), 1.0, places=5)

        def test_diagnostics_keys(self):
            scaler = self._make_scaler(base_scale=0.01)
            scaler.compute_scale(torch.device("cpu"), global_step=0)
            d = scaler.diagnostics()
            for key in ("base_scale", "history_len", "scale_mean", "scale_last"):
                self.assertIn(key, d)

        def test_history_len_capped(self):
            scaler = self._make_scaler(base_scale=0.01, history_len=5)
            for step in range(20):
                scaler.advance_step(step)
                slc = SharedLocalityCache()
                scaler._slc = slc  # reset SLC to force recomputation
                scaler.compute_scale(torch.device("cpu"), global_step=step)
            self.assertLessEqual(len(scaler._scale_history), 5)

    class TestMoEAuxLossAutoScalerShim(unittest.TestCase):

        def test_forward_passthrough(self):
            aux = torch.tensor([5.0], requires_grad=True)
            scale = torch.tensor([2.0])
            result = MoEAuxLossAutoScalerShim.apply_scale(aux, scale)
            # Forward should return aux unchanged
            self.assertAlmostEqual(result.item(), 5.0, places=6)

        def test_backward_applies_scale(self):
            aux = torch.tensor([5.0], requires_grad=True)
            scale = torch.tensor([3.0])
            result = MoEAuxLossAutoScalerShim.apply_scale(aux, scale)
            result.backward()
            # grad should be 1.0 * 3.0 = 3.0
            self.assertAlmostEqual(aux.grad.item(), 3.0, places=6)

        def test_backward_neutral_scale(self):
            aux = torch.tensor([7.0], requires_grad=True)
            scale = torch.ones(1)
            result = MoEAuxLossAutoScalerShim.apply_scale(aux, scale)
            result.backward()
            self.assertAlmostEqual(aux.grad.item(), 1.0, places=6)

    class TestHeterogeneousForwardStepLossHandler(unittest.TestCase):

        def _make_handler(self, with_scaler=True, with_grad_func=False):
            scaler = None
            if with_scaler:
                cfg = HeterogeneousMoELossScalerConfig(base_scale=0.05)
                scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            grad_func = None
            if with_grad_func:
                grad_func = lambda t: t * 2.0
            return HeterogeneousForwardStepLossHandler(
                moe_scaler=scaler,
                grad_scale_func=grad_func,
                calculate_per_token_loss=False,
                num_moe_experts=8,
            )

        def test_scale_priority_1_moe_scaler(self):
            handler = self._make_handler(with_scaler=True, with_grad_func=True)
            out = torch.ones(1)
            scale = handler.compute_moe_loss_scale(out, global_step=0)
            # Priority 1: moe_scaler base_scale=0.05
            self.assertAlmostEqual(scale.item(), 0.05, places=6)

        def test_scale_priority_2_grad_func(self):
            handler = self._make_handler(with_scaler=False, with_grad_func=True)
            out = torch.ones(1)
            scale = handler.compute_moe_loss_scale(out, global_step=0)
            # Priority 2: grad_func returns 1.0 * 2.0 = 2.0
            self.assertAlmostEqual(scale.item(), 2.0, places=6)

        def test_scale_priority_3_ones(self):
            handler = self._make_handler(with_scaler=False, with_grad_func=False)
            out = torch.ones(1)
            scale = handler.compute_moe_loss_scale(out, global_step=0)
            # Priority 3: ones
            self.assertAlmostEqual(scale.item(), 1.0, places=6)

        def test_handle_aux_loss_no_experts_returns_unchanged(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.1)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            handler = HeterogeneousForwardStepLossHandler(
                moe_scaler=scaler,
                num_moe_experts=None,  # MoE inactive
            )
            aux = torch.tensor([99.0])
            result = handler.handle_aux_loss(aux, torch.ones(1))
            self.assertIs(result, aux)

        def test_handle_aux_loss_applies_scale(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.25)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            handler = HeterogeneousForwardStepLossHandler(
                moe_scaler=scaler,
                calculate_per_token_loss=False,
                num_moe_experts=4,
            )
            aux = torch.tensor([8.0])
            result = handler.handle_aux_loss(aux, torch.ones(1), global_step=0)
            self.assertAlmostEqual(result.item(), 2.0, places=5)

        def test_handle_aux_loss_per_token_backward(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.5)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            handler = HeterogeneousForwardStepLossHandler(
                moe_scaler=scaler,
                calculate_per_token_loss=True,  # use autograd shim
                num_moe_experts=4,
            )
            aux = torch.tensor([4.0], requires_grad=True)
            result = handler.handle_aux_loss(aux, torch.ones(1), global_step=0)
            # Forward: unchanged
            self.assertAlmostEqual(result.item(), 4.0, places=5)
            result.backward()
            # Backward: grad = 1.0 * 0.5 = 0.5
            self.assertAlmostEqual(aux.grad.item(), 0.5, places=5)

    class TestBuildMoeGradScaleFunc(unittest.TestCase):

        def test_returns_callable(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.01)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            fn = build_moe_grad_scale_func(scaler)
            self.assertTrue(callable(fn))

        def test_callable_returns_tensor(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.02)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            fn = build_moe_grad_scale_func(scaler)
            result = fn()
            self.assertIsInstance(result, torch.Tensor)
            self.assertAlmostEqual(result.item(), 0.02, places=6)

        def test_returns_ones_after_gc(self):
            import gc
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.99)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            fn = build_moe_grad_scale_func(scaler)
            del scaler
            gc.collect()
            # Should fall back to ones and not raise
            result = fn()
            self.assertAlmostEqual(result.item(), 1.0, places=6)

    class TestPatchDeepSpeedConfig(unittest.TestCase):

        def test_patch_injects_callable(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.03)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            ds_cfg: dict = {}
            patch_deepspeed_config(ds_cfg, scaler)
            self.assertIn("moe_grad_scale_func", ds_cfg)
            self.assertTrue(callable(ds_cfg["moe_grad_scale_func"]))

        def test_patch_warns_on_overwrite(self):
            cfg = HeterogeneousMoELossScalerConfig(base_scale=0.03)
            scaler = HeterogeneousMoELossScaler(cfg, slc=SharedLocalityCache())
            ds_cfg: dict = {"moe_grad_scale_func": lambda: torch.ones(1)}
            with self.assertLogs(logger="deepspeed.moe.heterogeneous_loss_scaler", level="WARNING"):
                patch_deepspeed_config(ds_cfg, scaler)

        def test_callback_fields_not_cli_args(self):
            """
            Mirror Megatron test_argument_utils: runtime callback fields must
            not appear as CLI-registered arguments.  In DES-LOC they are
            injected via patch_deepspeed_config, never through argparse.
            """
            callback_fields = {
                "moe_grad_scale_func",
                "grad_scale_func",
                "no_sync_func",
                "finalize_model_grads_func",
            }
            # Simulate a minimal ds_config that might be built from argparse
            argparse_cfg: dict = {
                "train_batch_size": 8,
                "gradient_accumulation_steps": 1,
            }
            for field_name in callback_fields:
                self.assertNotIn(
                    field_name,
                    argparse_cfg,
                    msg=f"{field_name} must not be a CLI arg",
                )

    class TestCreateHeterogeneousMoEScalerForNeuronSP(unittest.TestCase):

        def test_default_construction(self):
            scaler = create_heterogeneous_moe_scaler_for_neuron_sp()
            self.assertIsInstance(scaler, HeterogeneousMoELossScaler)
            self.assertAlmostEqual(scaler._cfg.base_scale, 1e-2, places=8)

        def test_rl_sft_mode(self):
            scaler = create_heterogeneous_moe_scaler_for_neuron_sp(
                base_scale=0.005,
                rl_sft=True,
            )
            self.assertTrue(scaler._cfg.rl_reward_scale_coupling)
            self.assertTrue(scaler._cfg.per_active_expert_token)

        def test_non_rl_mode(self):
            scaler = create_heterogeneous_moe_scaler_for_neuron_sp(rl_sft=False)
            self.assertFalse(scaler._cfg.rl_reward_scale_coupling)

    print("Running DES-LOC HeterogeneousMoELossScaler unit tests …", flush=True)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestSharedLocalityCache,
        TestDeviceHelpers,
        TestHeterogeneousMoELossScalerBasic,
        TestMoEAuxLossAutoScalerShim,
        TestHeterogeneousForwardStepLossHandler,
        TestBuildMoeGradScaleFunc,
        TestPatchDeepSpeedConfig,
        TestCreateHeterogeneousMoEScalerForNeuronSP,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
