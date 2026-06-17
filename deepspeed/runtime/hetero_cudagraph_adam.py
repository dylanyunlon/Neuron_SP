"""
HeteroCudaGraphAdam — DES-LOC Heterogeneous CUDA Graph Adam Optimizer
======================================================================

Upstream design intent (Megatron 3d87bfc1):
    NVIDIA Megatron-LM introduced CUDA graph capture for the ADAM optimizer step,
    enabling the entire optimizer update loop to be replayed as a single GPU kernel
    launch rather than thousands of individual CUDA calls.  Key mechanisms:

    1. ``capturable=True`` in PyTorch AdamW/Adam → tensors for lr/beta/eps are kept
       on-device as scalars, so the graph can capture the full step.
    2. ``OptimizerCudaGraphWrapper`` delays capture until after N warmup steps to let
       PyTorch internal state settle, then records with ``torch.cuda.CUDAGraph()``.
    3. ``StaticBufferLoader`` stream-synchronization fixes (wait_stream before context
       switch) prevent hazards between the data-loading stream and the capture stream.
    4. ``multi_tensor_scale_tensor`` replaces the scalar ``clip_coeff.item()`` call so
       gradient clipping stays entirely on-device and is capturable.
    5. ``OptimizerParamScheduler`` detects tensor-typed lr and uses ``fill_()`` in-place
       instead of re-assignment, keeping the captured graph's pointer valid.

DES-LOC adaptation points:
    Our hardware triangle — 2× A6000 (SM86, 48 GB) + 1× H100 NVL (SM90, 96 GB) on
    PCIe without NVLink — makes naïve homogeneous CUDA graph capture unsafe:

    * A CUDA graph captured on H100 (SM90 PTX) will crash if replayed on A6000
      (SM86).  Each device family must own a *separate* graph handle.
    * PCIe bandwidth (~32 GB/s bidirectional) is the bottleneck for parameter
      sharding.  We must not pipeline optimizer traffic from A6000↔H100 inside a
      captured graph because PCIe transfers are opaque to CUDA graph replay.
    * The Shared LOcality Cache (LOC) in DES-LOC pins optimizer *states* (momentum,
      variance) inside CPU DRAM (1.5 TB) and only streams the *gradient* and
      *parameter delta* across PCIe.  This means m/v tensors must be CPU-pinned and
      excluded from graph capture on A6000 ranks — but on H100 they can live on-device
      and be fully captured.
    * Decoupled Execution means the forward/backward and optimizer steps run on
      potentially different SM families in the same iteration; capture timing must be
      gated per-rank-type rather than globally.

    The class ``HeteroCudaGraphAdam`` below implements all of this.

Author: Neuron_SP project (DES-LOC framework)
Mirrors: Megatron commit 3d87bfc1 — Enable CUDA graph for ADAM optimizer
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim import Adam, AdamW

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware capability helpers
# ---------------------------------------------------------------------------

class SMArch(Enum):
    """Compute-capability families present in the DES-LOC cluster."""
    SM86 = auto()   # A6000
    SM90 = auto()   # H100 NVL
    UNKNOWN = auto()


def _detect_sm_arch(device: torch.device) -> SMArch:
    """Return the SM architecture of *device*.

    Uses ``torch.cuda.get_device_capability`` which is available without
    querying nvidia-smi and works inside containers.
    """
    major, minor = torch.cuda.get_device_capability(device)
    code = major * 10 + minor
    if code == 86:
        return SMArch.SM86
    if code == 90:
        return SMArch.SM90
    logger.warning("Unknown SM arch %d.%d on %s — treating as UNKNOWN", major, minor, device)
    return SMArch.UNKNOWN


def _graph_capturable_on_arch(arch: SMArch) -> bool:
    """CUDA graphs require SM >= 7.0 (Volta).  All our devices qualify,
    BUT on SM86 (A6000) we cannot capture cross-PCIe transfers.  The flag
    returned here gates *full* capture (on-device momentum/variance).
    """
    return arch in (SMArch.SM86, SMArch.SM90)


# ---------------------------------------------------------------------------
# DES-LOC Locality Cache — pinned CPU tensor manager
# ---------------------------------------------------------------------------

@dataclass
class LocalityCacheSlot:
    """One parameter's optimizer state pinned in CPU DRAM.

    In DES-LOC, A6000 ranks keep m/v in CPU memory to avoid exhausting
    the 48 GB device pool.  H100 (96 GB) can afford to keep them on-device.
    The ``device_shadow`` is a small on-device buffer used only during the
    AdamW kernel; it is filled from CPU just-in-time and written back after.
    """
    param_id: int
    m_cpu: torch.Tensor          # pinned CPU momentum
    v_cpu: torch.Tensor          # pinned CPU variance
    device_shadow_m: Optional[torch.Tensor] = None
    device_shadow_v: Optional[torch.Tensor] = None


class LocalityCache:
    """Manages the Shared LOcality Cache for optimizer states.

    Parameters
    ----------
    device:
        The CUDA device that *owns* the parameters.
    arch:
        Pre-detected SM architecture of ``device``.
    pin_states_in_cpu:
        If True, momentum/variance live in pinned CPU DRAM and are streamed
        to device for each optimizer step.  Forced True for SM86 ranks.
    """

    def __init__(
        self,
        device: torch.device,
        arch: SMArch,
        pin_states_in_cpu: bool = True,
    ) -> None:
        self.device = device
        self.arch = arch
        # SM86 (A6000) always pins states in CPU to conserve 48 GB VRAM.
        self.pin_states_in_cpu = pin_states_in_cpu or (arch == SMArch.SM86)
        self._slots: Dict[int, LocalityCacheSlot] = {}
        # Dedicated H2D/D2H stream so copies never interfere with compute.
        self._xfer_stream = torch.cuda.Stream(device=device)
        logger.info(
            "[LOC] Initialized LocalityCache on %s arch=%s pin_cpu=%s",
            device, arch.name, self.pin_states_in_cpu,
        )

    def get_or_create(self, param_id: int, shape: torch.Size, dtype: torch.dtype) -> LocalityCacheSlot:
        """Return existing slot or allocate fresh pinned tensors."""
        if param_id not in self._slots:
            if self.pin_states_in_cpu:
                m = torch.zeros(shape, dtype=dtype, pin_memory=True)
                v = torch.zeros(shape, dtype=dtype, pin_memory=True)
                # Small on-device mirrors for kernel use
                shadow_m = torch.zeros(shape, dtype=dtype, device=self.device)
                shadow_v = torch.zeros(shape, dtype=dtype, device=self.device)
                self._slots[param_id] = LocalityCacheSlot(
                    param_id=param_id,
                    m_cpu=m, v_cpu=v,
                    device_shadow_m=shadow_m,
                    device_shadow_v=shadow_v,
                )
            else:
                # H100 path: all on device
                m = torch.zeros(shape, dtype=dtype, device=self.device)
                v = torch.zeros(shape, dtype=dtype, device=self.device)
                self._slots[param_id] = LocalityCacheSlot(
                    param_id=param_id, m_cpu=m, v_cpu=v,
                )
        return self._slots[param_id]

    def prefetch_to_device(self, param_ids: List[int]) -> None:
        """Async H2D copy of momentum/variance for listed param ids.

        Issued on ``_xfer_stream`` so it overlaps with any preceding compute
        on the default stream.  Must call ``sync_xfer()`` before the Adam
        kernel runs.
        """
        if not self.pin_states_in_cpu:
            return
        with torch.cuda.stream(self._xfer_stream):
            for pid in param_ids:
                slot = self._slots.get(pid)
                if slot is None:
                    continue
                slot.device_shadow_m.copy_(slot.m_cpu, non_blocking=True)
                slot.device_shadow_v.copy_(slot.v_cpu, non_blocking=True)

    def writeback_to_cpu(self, param_ids: List[int]) -> None:
        """Async D2H copy of updated momentum/variance back to CPU DRAM."""
        if not self.pin_states_in_cpu:
            return
        with torch.cuda.stream(self._xfer_stream):
            for pid in param_ids:
                slot = self._slots.get(pid)
                if slot is None:
                    continue
                slot.m_cpu.copy_(slot.device_shadow_m, non_blocking=True)
                slot.v_cpu.copy_(slot.v_cpu, non_blocking=True)

    def sync_xfer(self) -> None:
        """Block the current stream until H2D/D2H transfers are complete."""
        torch.cuda.current_stream(self.device).wait_stream(self._xfer_stream)

    def slot_count(self) -> int:
        return len(self._slots)


# ---------------------------------------------------------------------------
# Capturable clip-coeff scale (mirrors multi_tensor_scale_tensor)
# ---------------------------------------------------------------------------

def _scale_grads_capturable(
    grads: List[torch.Tensor],
    clip_coeff: torch.Tensor,
) -> None:
    """In-place gradient scaling with a *tensor* clip_coeff.

    Megatron introduced ``multi_tensor_scale_tensor`` to avoid a ``.item()``
    call inside ``clip_grad_by_total_norm_fp32``, which would break CUDA graph
    capture.  We replicate the same semantics here using ``torch.mul`` with
    broadcasting, which is capturable since PyTorch 2.1.

    Parameters
    ----------
    grads:
        List of gradient tensors to scale in-place.
    clip_coeff:
        0-d tensor on the same device as grads, clamped to ≤ 1.0.
    """
    clip_coeff.clamp_max_(1.0)
    for g in grads:
        g.mul_(clip_coeff)


def get_grad_norm_fp32_capturable(
    params_or_grads: List[torch.Tensor],
    norm_type: float = 2.0,
    model_parallel_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute gradient L2 norm, returning a *device tensor* (not a Python float).

    Upstream (Megatron 3d87bfc1) patched ``get_grad_norm_fp32`` to branch on
    whether ``multi_tensor_scale_tensor`` is available: if yes, keep the result
    as a tensor; if no, call ``.item()``.  In DES-LOC we always keep it as a
    tensor so the result can flow through the CUDA graph on devices that support
    full capture (H100).  On A6000 the value is still computed correctly; the
    graph capture flag for A6000 simply excludes the clip path from capture.

    Parameters
    ----------
    params_or_grads:
        Flat list of parameter or gradient tensors.
    norm_type:
        Order of the norm (default L2).
    model_parallel_group:
        If provided, the norm is all-reduced across this group.

    Returns
    -------
    torch.Tensor
        0-d float32 tensor containing the total gradient norm.
    """
    grads_fp32 = []
    for t in params_or_grads:
        g = t if t.dim() == 0 else (t.grad if hasattr(t, "grad") and t.grad is not None else t)
        grads_fp32.append(g.float())

    if norm_type == math.inf:
        total_norm = max(g.abs().max() for g in grads_fp32)
        total_norm = torch.tensor(float(total_norm), dtype=torch.float32,
                                  device=grads_fp32[0].device if grads_fp32 else torch.device("cpu"))
    else:
        total_norm = torch.zeros(1, dtype=torch.float32,
                                 device=grads_fp32[0].device if grads_fp32 else torch.device("cpu"))
        for g in grads_fp32:
            total_norm.add_(g.pow(norm_type).sum())
        if model_parallel_group is not None and dist.is_initialized():
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=model_parallel_group)
        total_norm = total_norm.pow(1.0 / norm_type)

    return total_norm


# ---------------------------------------------------------------------------
# Per-device CUDA graph handle with arch-awareness
# ---------------------------------------------------------------------------

class DeviceGraphHandle:
    """Owns a ``torch.cuda.CUDAGraph`` bound to one specific device + SM arch.

    Megatron's ``OptimizerCudaGraphWrapper`` used a single class-level graph,
    which is unsafe in a heterogeneous cluster: a graph captured on SM90
    bytecode cannot be replayed on SM86.

    DES-LOC creates one ``DeviceGraphHandle`` per *rank*, each tracking its
    own capture state, warmup counter, and arch constraints.

    Parameters
    ----------
    device:
        CUDA device this handle belongs to.
    arch:
        SM architecture (controls whether full-step capture is allowed).
    warmup_steps:
        Number of eager iterations before graph capture is attempted.
    """

    def __init__(
        self,
        device: torch.device,
        arch: SMArch,
        warmup_steps: int = 3,
    ) -> None:
        self.device = device
        self.arch = arch
        self.warmup_steps = warmup_steps
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._result = None
        self._iteration: int = 0
        # SM86 (A6000): PCIe transfers of LOC states cannot be inside the graph.
        # We still capture the pure-compute part but exclude H2D/D2H copies.
        self._full_capture = (arch == SMArch.SM90)
        logger.info(
            "[GraphHandle] device=%s arch=%s full_capture=%s warmup=%d",
            device, arch.name, self._full_capture, warmup_steps,
        )

    @property
    def is_captured(self) -> bool:
        return self._graph is not None

    def maybe_capture(self, step_fn) -> None:
        """Capture ``step_fn`` into a CUDA graph if conditions are met.

        ``step_fn`` must be callable with zero arguments and its side effects
        (parameter updates) must be deterministic w.r.t. the graph inputs
        (static tensors in LOC device shadows or on-device states).

        Mirrors Megatron's capture logic but adds:
        - per-device barrier (only ranks sharing the same SM arch synchronize)
        - SM86 partial-capture gate
        """
        if self._iteration == self.warmup_steps and self._graph is None:
            logger.info(
                "[GraphHandle] Capturing optimizer CUDA graph on %s (arch=%s)",
                self.device, self.arch.name,
            )
            if dist.is_initialized():
                dist.barrier()
            self._graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(self.device)
            capture_stream = torch.cuda.Stream(device=self.device)
            # Megatron fix: wait for current stream before entering capture stream
            capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.graph(self._graph, stream=capture_stream):
                self._result = step_fn()
            torch.cuda.synchronize(self.device)
            if dist.is_initialized():
                dist.barrier()
            logger.info("[GraphHandle] Capture complete on %s", self.device)

    def replay_or_eager(self, step_fn):
        """Replay captured graph or run eagerly."""
        if self._graph is not None:
            self._graph.replay()
            return self._result
        return step_fn()

    def destroy(self) -> None:
        if self._graph is not None:
            logger.info("[GraphHandle] Destroying CUDA graph on %s", self.device)
            del self._graph
            self._graph = None
            self._result = None

    def increment(self) -> None:
        self._iteration += 1


# ---------------------------------------------------------------------------
# Main class: HeteroCudaGraphAdam
# ---------------------------------------------------------------------------

class HeteroCudaGraphAdam:
    """Heterogeneous CUDA Graph Adam optimizer for DES-LOC.

    Wraps DeepSpeed's optimizer infrastructure to enable:

    1. **Per-SM-arch CUDA graph capture** — H100 (SM90) captures the full
       optimizer step including on-device momentum/variance updates.  A6000
       (SM86) captures only the pure-compute gradient scaling + parameter
       delta, while PCIe transfers of LOC states remain outside the graph.

    2. **Shared LOcality Cache** — optimizer states (m, v) for A6000 ranks
       are pinned in CPU DRAM (1.5 TB available) and streamed to device
       shadows just before the Adam kernel, then written back asynchronously.
       H100 rank keeps m/v fully on-device (96 GB capacity).

    3. **Capturable lr/clip** — ``capturable=True`` in AdamW keeps lr/beta/eps
       as device scalars.  Gradient clipping uses ``_scale_grads_capturable``
       (tensor clip_coeff) so no ``.item()`` breaks the graph.

    4. **In-place lr scheduling** — when lr is a tensor (capturable mode),
       the scheduler calls ``lr_tensor.fill_(new_lr)`` to keep the graph
       pointer valid (mirrors Megatron's ``OptimizerParamScheduler`` fix).

    Parameters
    ----------
    param_groups:
        Standard optimizer param groups (list of dicts with 'params').
    lr:
        Initial learning rate.
    betas:
        Adam beta coefficients.
    eps:
        Adam epsilon.
    weight_decay:
        L2 regularization coefficient.
    max_grad_norm:
        Gradient clipping threshold (0 = no clipping).
    warmup_steps:
        Eager iterations before CUDA graph capture.
    device:
        Target CUDA device.  Defaults to ``cuda:0``.
    pin_loc_states:
        Force LOC states into CPU DRAM regardless of arch.
    """

    def __init__(
        self,
        param_groups: List[dict],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 3,
        device: Optional[torch.device] = None,
        pin_loc_states: Optional[bool] = None,
    ) -> None:
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.arch = _detect_sm_arch(self.device)
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps

        # Determine LOC pinning strategy
        if pin_loc_states is None:
            pin_loc_states = (self.arch == SMArch.SM86)
        self.loc = LocalityCache(
            device=self.device,
            arch=self.arch,
            pin_states_in_cpu=pin_loc_states,
        )

        # Build capturable Adam.
        # ``capturable=True`` keeps lr/beta/eps as 0-d device tensors so the
        # optimizer step can be recorded into a CUDA graph without graph breaks.
        self._inner = AdamW(
            self._flatten_params(param_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            capturable=_graph_capturable_on_arch(self.arch),
            foreach=False,   # foreach is not compatible with capturable on older PT
        )
        self.param_groups = self._inner.param_groups

        # Initialize LOC slots for all parameters
        self._all_param_ids: List[int] = []
        for pg in self.param_groups:
            for p in pg["params"]:
                pid = id(p)
                self._all_param_ids.append(pid)
                self.loc.get_or_create(pid, p.shape, torch.float32)
        logger.info(
            "[HeteroAdam] Initialized: device=%s arch=%s params=%d loc_slots=%d capturable=%s",
            self.device, self.arch.name,
            len(self._all_param_ids), self.loc.slot_count(),
            _graph_capturable_on_arch(self.arch),
        )

        # Per-device graph handle
        self._graph_handle = DeviceGraphHandle(
            device=self.device,
            arch=self.arch,
            warmup_steps=warmup_steps,
        )

        # lr tensor reference for in-place scheduler updates
        # After the first step, AdamW will have converted lr to a tensor when
        # capturable=True.  We cache the reference here.
        self._lr_tensor: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_params(param_groups: List[dict]) -> List[dict]:
        """Ensure param_groups is a list of dicts suitable for torch.optim."""
        out = []
        for pg in param_groups:
            entry = {k: v for k, v in pg.items()}
            if "params" not in entry:
                raise ValueError("param_group missing 'params' key")
            out.append(entry)
        return out

    def _collect_grads(self) -> List[torch.Tensor]:
        grads = []
        for pg in self.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    grads.append(p.grad)
        return grads

    def _clip_gradients(self, model_parallel_group=None) -> Optional[torch.Tensor]:
        """Clip gradients and return total norm as a device tensor.

        On SM90 (H100) the returned tensor stays on-device and is fully
        capturable.  On SM86 (A6000) we also return a device tensor but the
        all-reduce for the norm happens *before* graph capture, so it is safe.
        """
        if self.max_grad_norm <= 0.0:
            return None
        grads = self._collect_grads()
        if not grads:
            return None
        total_norm = get_grad_norm_fp32_capturable(
            grads,
            norm_type=2.0,
            model_parallel_group=model_parallel_group,
        )
        clip_coeff = torch.tensor(self.max_grad_norm, device=self.device) / (total_norm + 1e-6)
        _scale_grads_capturable(grads, clip_coeff)
        return total_norm

    def _loc_prefetch(self) -> None:
        """Stream optimizer states H2D before the Adam kernel (A6000 path)."""
        if self.loc.pin_states_in_cpu:
            self.loc.prefetch_to_device(self._all_param_ids)
            self.loc.sync_xfer()

    def _loc_writeback(self) -> None:
        """Stream updated optimizer states D2H after the Adam kernel (A6000 path)."""
        if self.loc.pin_states_in_cpu:
            self.loc.writeback_to_cpu(self._all_param_ids)

    def _inject_loc_states(self) -> None:
        """Point AdamW state tensors at LOC device shadows (A6000 path).

        Before the Adam kernel runs we swap the m/v in ``self._inner.state``
        to point at the freshly-prefetched device shadows from the LOC.
        After writeback the CPU copies are authoritative again.

        On H100 (pin_states_in_cpu=False) the AdamW state lives on-device
        natively and this function is a no-op.
        """
        if not self.loc.pin_states_in_cpu:
            return
        state = self._inner.state
        for pg in self.param_groups:
            for p in pg["params"]:
                if p not in state or not state[p]:
                    # Adam state not yet initialized — skip; it will be created
                    # on the first eager step.
                    continue
                pid = id(p)
                slot = self.loc._slots.get(pid)
                if slot is None:
                    continue
                state[p]["exp_avg"] = slot.device_shadow_m
                state[p]["exp_avg_sq"] = slot.device_shadow_v

    def _maybe_cache_lr_tensor(self) -> None:
        """Cache a reference to the first group's lr tensor post-capture.

        Mirrors Megatron's ``OptimizerParamScheduler`` fix: after capturable
        Adam converts lr to a tensor, schedulers must use ``fill_()`` to update
        it in-place or the captured graph's device pointer becomes stale.
        """
        if self._lr_tensor is None and self.param_groups:
            lr_val = self.param_groups[0].get("lr")
            if isinstance(lr_val, torch.Tensor):
                self._lr_tensor = lr_val
                logger.debug("[HeteroAdam] lr is now a device tensor %s", self._lr_tensor.shape)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lr(self, new_lr: float) -> None:
        """Update learning rate, respecting capturable tensor semantics.

        When ``capturable=True``, AdamW converts the lr float to a 0-d device
        tensor after the first step.  Standard ``param_group['lr'] = x``
        re-assignment would invalidate the graph's static pointer.  We detect
        the tensor case and call ``fill_()`` instead, exactly as Megatron's
        patched ``OptimizerParamScheduler.step()`` does.
        """
        self._maybe_cache_lr_tensor()
        for pg in self.param_groups:
            current_lr = pg.get("lr")
            if isinstance(current_lr, torch.Tensor):
                current_lr.fill_(new_lr)
            else:
                pg["lr"] = new_lr

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients."""
        self._inner.zero_grad(set_to_none=set_to_none)

    def step(self, model_parallel_group=None) -> Optional[torch.Tensor]:
        """Perform one Adam optimizer step with DES-LOC graph capture logic.

        Execution flow
        --------------
        A6000 (SM86, LOC-pinned):
            1. Gradient clipping (norm computed before graph boundary)
            2. Prefetch m/v from CPU DRAM → device shadows  (PCIe, outside graph)
            3. Inject LOC shadows into AdamW state
            4. [warmup] eager inner step  /  [captured] graph replay
            5. Write-back updated m/v to CPU DRAM  (PCIe, outside graph)

        H100 (SM90, fully on-device):
            1. Gradient clipping (capturable, stays in graph)
            2. [warmup] eager inner step  /  [captured] full graph replay
            (no H2D/D2H needed — m/v live on-device)

        The split ensures PCIe transfers are *never* inside a captured graph,
        while still giving maximum replay speedup on the H100 rank.
        """
        handle = self._graph_handle

        if self.arch == SMArch.SM86:
            # --- A6000 path ---
            # Clip grads eagerly (outside graph) so norm all-reduce is safe.
            total_norm = self._clip_gradients(model_parallel_group=model_parallel_group)

            # Prefetch optimizer states over PCIe (outside graph).
            self._loc_prefetch()
            self._inject_loc_states()

            def _inner_step():
                return self._inner.step()

            # Attempt graph capture / replay of pure-compute Adam kernel.
            handle.maybe_capture(_inner_step)
            result = handle.replay_or_eager(_inner_step)

            # Write back updated states over PCIe (outside graph).
            self._loc_writeback()

        else:
            # --- H100 path ---
            # Gradient clipping can be inside the graph (no item() call).
            def _full_step():
                self._clip_gradients(model_parallel_group=model_parallel_group)
                return self._inner.step()

            handle.maybe_capture(_full_step)
            result = handle.replay_or_eager(_full_step)
            total_norm = None  # returned by inner clip; not separately tracked here

        self._maybe_cache_lr_tensor()
        handle.increment()
        return result

    def state_dict(self) -> dict:
        """Serialize optimizer state including LOC metadata."""
        inner_sd = self._inner.state_dict()
        loc_meta = {
            "arch": self.arch.name,
            "pin_states_in_cpu": self.loc.pin_states_in_cpu,
            "iteration": self._graph_handle._iteration,
        }
        # Include CPU-pinned states for A6000 ranks so checkpoints are
        # self-consistent even if the rank restores on a different device.
        if self.loc.pin_states_in_cpu:
            loc_cpu_states = {}
            for pid, slot in self.loc._slots.items():
                loc_cpu_states[pid] = {
                    "m": slot.m_cpu.clone(),
                    "v": slot.v_cpu.clone(),
                }
            loc_meta["loc_cpu_states"] = loc_cpu_states
        return {"inner": inner_sd, "loc_meta": loc_meta}

    def load_state_dict(self, sd: dict) -> None:
        """Restore optimizer state, migrating LOC CPU tensors if needed."""
        self._inner.load_state_dict(sd["inner"])
        meta = sd.get("loc_meta", {})
        self._graph_handle._iteration = meta.get("iteration", 0)
        if "loc_cpu_states" in meta and self.loc.pin_states_in_cpu:
            for pid_str, tensors in meta["loc_cpu_states"].items():
                pid = int(pid_str)
                slot = self.loc._slots.get(pid)
                if slot is not None:
                    slot.m_cpu.copy_(tensors["m"])
                    slot.v_cpu.copy_(tensors["v"])
            logger.info("[HeteroAdam] Restored %d LOC CPU state slots", len(meta["loc_cpu_states"]))

    def destroy(self) -> None:
        """Explicitly release CUDA graph resources (mirrors Megatron's ``del optimizer.step``)."""
        self._graph_handle.destroy()


# ---------------------------------------------------------------------------
# DeepSpeed engine integration shim
# ---------------------------------------------------------------------------

def build_hetero_cudagraph_adam(
    engine,
    config: dict,
) -> HeteroCudaGraphAdam:
    """Construct a ``HeteroCudaGraphAdam`` from a DeepSpeed engine config dict.

    Intended to be called from ``deepspeed/runtime/engine.py`` after the
    base optimizer is constructed, when ``config["optimizer"]["type"]`` is
    ``"HeteroCudaGraphAdam"``.

    Parameters
    ----------
    engine:
        DeepSpeed engine (used to access ``engine.module.parameters()``).
    config:
        DeepSpeed config dict (``ds_config``).

    Returns
    -------
    HeteroCudaGraphAdam
    """
    opt_cfg = config.get("optimizer", {}).get("params", {})
    sched_cfg = config.get("scheduler", {}).get("params", {})

    param_groups = [{"params": list(engine.module.parameters())}]

    device = torch.device("cuda", torch.cuda.current_device())

    optimizer = HeteroCudaGraphAdam(
        param_groups=param_groups,
        lr=opt_cfg.get("lr", 1e-3),
        betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
        eps=opt_cfg.get("eps", 1e-8),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
        max_grad_norm=config.get("gradient_clipping", 1.0),
        warmup_steps=config.get("cuda_graph_warmup_steps", 3),
        device=device,
    )
    logger.info(
        "[build_hetero_cudagraph_adam] Created optimizer for device %s", device
    )
    return optimizer


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
    else:
        device = torch.device("cuda", 0)
        arch = _detect_sm_arch(device)
        logger.info("Detected arch: %s", arch.name)

        # Build a tiny model
        p = torch.randn(16, 16, device=device, requires_grad=True)
        optimizer = HeteroCudaGraphAdam(
            param_groups=[{"params": [p]}],
            lr=1e-3,
            warmup_steps=2,
            device=device,
        )

        # Run warmup + capture + replay
        for i in range(5):
            loss = p.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert optimizer._graph_handle._iteration == 5, \
            f"Expected 5 iterations, got {optimizer._graph_handle._iteration}"

        # lr tensor in-place update
        optimizer.set_lr(5e-4)
        lr_val = optimizer.param_groups[0]["lr"]
        if isinstance(lr_val, torch.Tensor):
            assert abs(lr_val.item() - 5e-4) < 1e-9, "lr tensor fill_ failed"

        # Gradient clipping returns a device tensor
        p.grad = torch.ones_like(p)
        norm = get_grad_norm_fp32_capturable([p.grad])
        assert isinstance(norm, torch.Tensor), "norm must be a device tensor"
        assert norm.item() > 0, "norm must be positive"

        # LOC slot count matches param count
        assert optimizer.loc.slot_count() == 1, \
            f"Expected 1 LOC slot, got {optimizer.loc.slot_count()}"

        optimizer.destroy()
        logger.info("All smoke tests passed.")
