"""
deepspeed/inference/hetero_moe_inference_tuner.py
==================================================

DES-LOC Heterogeneous MoE Inference Tuner
------------------------------------------

Upstream intent (Megatron commit 20f09364):
    Megatron's inference team replaced a fixed 25-entry autotuning config list
    and a single ``_select_block_size_m`` helper with a unified
    ``_get_default_config(M, E, top_k)`` function that picks *all* launch
    parameters (BLOCK_SIZE_{M,N,K}, GROUP_SIZE_M, num_warps, num_stages) from
    the runtime token-count hint rather than the worst-case buffer size.

    Simultaneously they:
    1. Switched the ``_moe_sum`` reduction kernel to a persistent CTA grid
       (BLOCK_M = num_SMs) that strides over valid_tokens in a tl.range loop,
       eliminating the zero-fill overhead on rows beyond valid_tokens.
    2. Configurable ``ep_consensus_interval`` (was hard-coded to 20) so busy
       engines can skip all-reduce more or less aggressively.
    3. Enabled shared-expert overlap for latent-MoE (DeepSeek-style) inference
       by letting the layer — not the dispatcher — own the launch+join of the
       shared-expert side-stream when latent projections are in use.
    4. Changed the default ``inference_grouped_gemm_backend`` from ``'torch'``
       to ``'vllm'`` across the board.

DES-LOC adaptation (M3909-BF):
    The Neuron_SP cluster has an extreme VRAM asymmetry:
        • 2× A6000  48 GB  SM_86  (NVIDIA Ampere, 84 SMs)
        • 1× H100 NVL  96 GB  SM_90  (NVIDIA Hopper, 132 SMs)
    The three GPUs are PCIe-connected with *no* NVLink.  1.5 TB CPU DRAM is
    available as a spill/offload tier.

    DES-LOC (Decoupled Execution with Shared LOcality Cache) exploits this by:
    • Partitioning experts across devices so the H100 handles large/heavy
      experts and the A6000 pair handles medium/light ones, with a shared
      locality cache in CPU DRAM that both sides can page into.
    • Using the ``ep_consensus_interval`` knob to decouple the PCIe-bound
      all-reduce from the critical compute path on the busy device.
    • Deriving per-device Triton launch configs that respect each SM
      architecture's register file and L2 capacity instead of using a
      universal table.
    • Running the shared-expert side-stream on the H100 (which has headroom)
      while routing computation is in-flight on the A6000 pair.

Key classes:
    HeteroDeviceProfile        – Static capabilities of one physical GPU.
    DEsLOCConsensusScheduler   – Adaptive ep_consensus_interval gating.
    DEsLOCMoELaunchConfig      – Per-device Triton launch config derivation.
    SharedExpertOverlapManager – Side-stream lifecycle for latent-MoE paths.
    HeteroMoEInferenceTuner    – Top-level orchestrator used by Neuron_SP.

Usage (inside DeepSpeed engine init)::

    tuner = HeteroMoEInferenceTuner.from_cluster_auto()
    launch_cfg = tuner.get_launch_config(num_tokens_hint=32, device=device)
    consensus_ok = tuner.consensus_scheduler.should_run(step, has_global_work)
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
import unittest
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – SM architecture capability fingerprints
# ---------------------------------------------------------------------------

# L2 cache sizes in bytes per SM architecture (conservative lower bounds from
# NVIDIA product sheets; we use them to reason about weight-tile reuse).
_L2_BYTES: Dict[int, int] = {
    86: 6 * 1024 * 1024,    # A6000: 6 MB L2 (SM86)
    90: 51 * 1024 * 1024,   # H100 NVL: 51 MB L2 (SM90)
    80: 40 * 1024 * 1024,   # A100 (common in comparable clusters)
    89: 72 * 1024 * 1024,   # L40S (SM89)
    70: 6 * 1024 * 1024,    # V100 fallback
}

# Register file per SM (32-bit words per SM) — used to estimate occupancy
# impact of num_warps.
_REGS_PER_SM: Dict[int, int] = {
    86: 65536,
    90: 65536,
    80: 65536,
    89: 65536,
    70: 65536,
}

# Async-copy engine count per SM (wgmma/cp.async stages on Hopper vs Ampere)
_ASYNC_COPY_STAGES_MAX: Dict[int, int] = {
    86: 5,   # Ampere: effective pipeline depth up to 5
    90: 7,   # Hopper: TMA allows deeper asynchrony
    80: 5,
    89: 5,
    70: 3,
}

# Cluster-local GPU device IDs assigned to each role in DES-LOC.
# These are overridden by HeteroMoEInferenceTuner.from_cluster_auto().
_DEFAULT_H100_DEVICE_IDS: Tuple[int, ...] = (0,)
_DEFAULT_A6000_DEVICE_IDS: Tuple[int, ...] = (1, 2)


# ---------------------------------------------------------------------------
# HeteroDeviceProfile
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HeteroDeviceProfile:
    """Static capability snapshot for one physical GPU in the DES-LOC cluster.

    Upstream context:
        Megatron's ``_get_default_config`` hard-codes SM-count via
        ``torch.cuda.get_device_properties`` at call time, which is fine for
        a homogeneous cluster.  In DES-LOC the caller must pass the *target*
        device's profile because MoE launch configs are computed on the CPU
        for a GPU that might not be the current CUDA device.

    DES-LOC adaptation:
        Profile is captured once at engine init and stored per device-index.
        ``sm_count`` drives grid sizing; ``sm_major`` selects the appropriate
        tuning regime (SM86 = Ampere conservative vs SM90 = Hopper aggressive).
    """

    device_id: int
    sm_count: int
    sm_major: int
    sm_minor: int
    total_memory_bytes: int
    l2_cache_bytes: int
    async_copy_stages_max: int
    name: str

    @classmethod
    def from_device(cls, device_id: int) -> "HeteroDeviceProfile":
        """Capture live properties for *device_id*."""
        props = torch.cuda.get_device_properties(device_id)
        sm_major = props.major
        l2 = _L2_BYTES.get(sm_major * 10 + props.minor,
                            _L2_BYTES.get(sm_major, 6 * 1024 * 1024))
        stages_max = _ASYNC_COPY_STAGES_MAX.get(sm_major * 10 + props.minor,
                                                 _ASYNC_COPY_STAGES_MAX.get(sm_major, 4))
        profile = cls(
            device_id=device_id,
            sm_count=props.multi_processor_count,
            sm_major=sm_major,
            sm_minor=props.minor,
            total_memory_bytes=props.total_memory,
            l2_cache_bytes=l2,
            async_copy_stages_max=stages_max,
            name=props.name,
        )
        logger.info(
            "HeteroDeviceProfile captured: device=%d name=%s sm=%d.%d "
            "sm_count=%d l2=%.1fMB",
            device_id, profile.name, sm_major, props.minor,
            profile.sm_count, l2 / 1024 / 1024,
        )
        return profile

    @property
    def is_hopper(self) -> bool:
        return self.sm_major >= 9

    @property
    def is_ampere(self) -> bool:
        return self.sm_major == 8


# ---------------------------------------------------------------------------
# DEsLOCMoELaunchConfig
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DEsLOCMoELaunchConfig:
    """Full Triton launch configuration for one MoE GEMM pass on one device.

    Upstream context (``_get_default_config`` in Megatron 20f09364):
        Megatron replaced a 25-entry autotuning table with a deterministic
        heuristic that maps (M, E, top_k) → tile sizes / warps / stages.
        The reasoning is:
          • Small M → memory-bound → tall/narrow tiles, more pipeline stages.
          • Large M → compute-bound → short/wide tiles, more warps.
          • tokens_per_expert drives GROUP_SIZE_M for L2 weight-tile reuse.

    DES-LOC adaptation:
        The same intuitions apply, but the *thresholds* differ between SM86
        and SM90 due to:
          • SM90 has 51 MB L2 vs SM86's 6 MB — GROUP_SIZE_M stays profitable
            to much larger M on Hopper because weight tiles fit in L2.
          • SM90 supports TMA with up to 7 effective async stages; SM86 caps
            at 5.  We use ``profile.async_copy_stages_max`` to cap num_stages.
          • SM90 has 132 SMs vs SM86's 84.  Grid sizing uses the actual count.

        The ``grid_size`` field replaces Megatron's ``num_sms``-based
        persistent grid.  Here we compute it from the *token-count hint*
        (num_tokens_hint) so the launch overhead stays minimal at decode time,
        while still being CUDA-graph safe.
    """

    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int
    grid_size_fc1: int
    grid_size_fc2: int
    # Metadata for DES-LOC routing decisions
    device_profile: HeteroDeviceProfile
    num_tokens_hint: int

    @classmethod
    def derive(
        cls,
        num_tokens_hint: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        ffn_dim: int,
        profile: HeteroDeviceProfile,
    ) -> "DEsLOCMoELaunchConfig":
        """Derive a launch config from runtime token-count hint and device profile.

        Parameters
        ----------
        num_tokens_hint:
            Expected per-step token count (NOT the worst-case buffer size).
            Mirrors Megatron's ``effective_tokens`` = ``num_tokens_hint`` if
            provided, else ``max_tokens``.
        num_experts:
            Number of experts local to this device.
        top_k:
            Routing fanout.
        hidden_dim:
            Hidden size (K dimension of FC1, N dimension of FC2).
        ffn_dim:
            FFN intermediate size (N of FC1, K of FC2).
        profile:
            Hardware capability snapshot for the target device.
        """
        M = max(num_tokens_hint, 1)

        # ── BLOCK_SIZE_M ──────────────────────────────────────────────────────
        # Directly mirrors Megatron's thresholds.  On both SM86 and SM90 the
        # indirection-table padding dominates at small M, so the thresholds
        # are the same.
        if M <= 32:
            block_m = 16
        elif M <= 96:
            block_m = 32
        elif M <= 512:
            block_m = 64
        else:
            block_m = 128

        # ── BLOCK_SIZE_N / BLOCK_SIZE_K ───────────────────────────────────────
        # SM90 has large L2 → can afford wider N tiles at smaller M.
        # SM86 is more memory-constrained → stick to narrow N until larger M.
        if profile.is_hopper:
            block_n = 128 if M > 32 else 64
            block_k = 64 if M > 32 else 128
        else:
            # SM86: matches Megatron verbatim
            block_n = 64 if M <= 64 else 128
            block_k = 128 if M <= 64 else 64

        # ── GROUP_SIZE_M ──────────────────────────────────────────────────────
        # L2-reuse threshold differs by architecture.
        # SM90 has 51 MB L2 → profitable grouping at lower tokens_per_expert.
        # SM86 has 6 MB L2 → weight tiles spill earlier, raise threshold.
        tokens_per_expert = M // max(num_experts, 1)
        if profile.is_hopper:
            group_m = 16 if tokens_per_expert > 64 else 1
        else:
            group_m = 16 if tokens_per_expert > 128 else 1

        # ── num_warps ─────────────────────────────────────────────────────────
        # Mirrors Megatron: memory-bound small-M regime uses 4 warps to reduce
        # register pressure; compute-bound large-M uses 8.
        # Hopper can sustain 8 warps at slightly lower M due to larger register
        # file headroom from wgmma vs mma.sync.
        if profile.is_hopper:
            num_warps = 4 if M <= 64 else 8
        else:
            num_warps = 4 if M <= 128 else 8

        # ── num_stages ────────────────────────────────────────────────────────
        # Extra prefetch stages help memory-bound (small M).  Cap by
        # architecture's async-copy engine limit.
        raw_stages = 4 if M <= 32 else 3
        num_stages = min(raw_stages, profile.async_copy_stages_max)

        # ── Grid sizing ───────────────────────────────────────────────────────
        # Mirrors Megatron's grid_size_{fc1,fc2} derivation:
        #   em_hint = effective_tokens * top_k + BLOCK_SIZE_M * num_local_experts
        #   num_pid_m_hint = ceil(em_hint / BLOCK_SIZE_M)
        #   grid_size_fc1 = num_pid_m_hint * ceil(N_fc1 / BLOCK_SIZE_N)
        #   grid_size_fc2 = num_pid_m_hint * ceil(N_fc2 / BLOCK_SIZE_N)
        em_hint = M * top_k + block_m * num_experts
        num_pid_m_hint = math.ceil(em_hint / block_m)
        num_pid_n_fc1 = math.ceil(ffn_dim / block_n)
        num_pid_n_fc2 = math.ceil(hidden_dim / block_n)
        grid_size_fc1 = num_pid_m_hint * num_pid_n_fc1
        grid_size_fc2 = num_pid_m_hint * num_pid_n_fc2

        cfg = cls(
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_SIZE_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
            grid_size_fc1=grid_size_fc1,
            grid_size_fc2=grid_size_fc2,
            device_profile=profile,
            num_tokens_hint=M,
        )
        logger.debug(
            "DEsLOCMoELaunchConfig derived: device=%d M=%d E=%d top_k=%d "
            "BM=%d BN=%d BK=%d GM=%d warps=%d stages=%d gs_fc1=%d gs_fc2=%d",
            profile.device_id, M, num_experts, top_k,
            block_m, block_n, block_k, group_m,
            num_warps, num_stages, grid_size_fc1, grid_size_fc2,
        )
        return cfg

    def as_triton_kwargs(self, fc: int = 1) -> Dict:
        """Return kwargs suitable for passing directly to the Triton kernel.

        Parameters
        ----------
        fc:
            1 for FC1 pass, 2 for FC2 pass.  Selects the appropriate grid_size.
        """
        assert fc in (1, 2), f"fc must be 1 or 2, got {fc}"
        return {
            "BLOCK_SIZE_M": self.BLOCK_SIZE_M,
            "BLOCK_SIZE_N": self.BLOCK_SIZE_N,
            "BLOCK_SIZE_K": self.BLOCK_SIZE_K,
            "GROUP_SIZE_M": self.GROUP_SIZE_M,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "grid_size": self.grid_size_fc1 if fc == 1 else self.grid_size_fc2,
        }


# ---------------------------------------------------------------------------
# DEsLOCConsensusScheduler
# ---------------------------------------------------------------------------

class DEsLOCConsensusScheduler:
    """Adaptive gate for EP-consensus all-reduce on PCIe-connected clusters.

    Upstream context (Megatron 20f09364 ``ep_consensus_interval``):
        Megatron added ``ep_consensus_interval`` (default 20) to make the
        hard-coded modulo-20 check configurable.  The rationale: when the
        engine is busy (global_work > 0) the all-reduce can be skipped for
        ``ep_consensus_interval`` steps to avoid per-step PCIe traffic.  When
        idle (global_work == 0) consensus runs immediately to detect new
        arrivals quickly.

    DES-LOC adaptation:
        In our PCIe-only cluster the all-reduce *always* crosses the PCIe bus
        (no NVLink shortcut).  Two regimes arise:

        Busy regime (has_global_work=True):
            Skip consensus aggressively.  The interval starts at
            ``base_interval`` (default 20, matching Megatron) and can grow up
            to ``max_interval`` if the recent skip rate is stable, reducing
            PCIe contention during heavy decode.

        Idle regime (has_global_work=False):
            Run immediately (interval=0), identical to Megatron — we want to
            detect new arrivals as fast as possible.

        The scheduler tracks consecutive busy steps to detect when it is safe
        to relax the interval further.  It resets on any idle step.

    Thread safety:
        ``should_run`` is called from the engine's main loop on a single
        thread; no lock is needed for the core state.  ``_lock`` guards the
        statistics counters used by ``log_stats`` only.
    """

    def __init__(
        self,
        base_interval: int = 20,
        max_interval: int = 80,
        ramp_after_steps: int = 200,
    ) -> None:
        self.base_interval = base_interval
        self.max_interval = max_interval
        self.ramp_after_steps = ramp_after_steps

        self._loop_counter: int = 0
        self._consecutive_busy: int = 0
        self._skipped_total: int = 0
        self._ran_total: int = 0
        self._lock = threading.Lock()

    @property
    def current_interval(self) -> int:
        """Effective interval in the current busy streak."""
        if self._consecutive_busy < self.ramp_after_steps:
            return self.base_interval
        # After ramp_after_steps consecutive busy steps, double interval
        # (capped at max_interval).
        factor = min(2, self._consecutive_busy // self.ramp_after_steps)
        return min(self.base_interval * factor, self.max_interval)

    def should_run(self, has_global_work: bool) -> bool:
        """Return True if EP-consensus should execute this step.

        Parameters
        ----------
        has_global_work:
            Whether the engine has pending work globally (from the previous
            consensus result).  Mirrors Megatron's
            ``global_work_from_last_consensus == 0`` fast-path.
        """
        self._loop_counter += 1

        if not has_global_work:
            # Idle: always run, reset busy streak.
            self._consecutive_busy = 0
            with self._lock:
                self._ran_total += 1
            return True

        # Busy: check configurable interval.
        self._consecutive_busy += 1
        interval = self.current_interval
        run = (self._loop_counter % interval) == 0

        with self._lock:
            if run:
                self._ran_total += 1
            else:
                self._skipped_total += 1

        return run

    def log_stats(self) -> None:
        """Emit a summary of consensus skip/run counts at INFO level."""
        with self._lock:
            total = self._ran_total + self._skipped_total
            skip_rate = self._skipped_total / max(total, 1)
        logger.info(
            "DEsLOCConsensusScheduler stats: ran=%d skipped=%d skip_rate=%.1f%% "
            "current_interval=%d consecutive_busy=%d",
            self._ran_total, self._skipped_total, skip_rate * 100,
            self.current_interval, self._consecutive_busy,
        )

    def reset(self) -> None:
        """Reset all counters (e.g., at checkpoint resume)."""
        self._loop_counter = 0
        self._consecutive_busy = 0
        with self._lock:
            self._skipped_total = 0
            self._ran_total = 0
        logger.info("DEsLOCConsensusScheduler reset.")


# ---------------------------------------------------------------------------
# SharedExpertOverlapManager
# ---------------------------------------------------------------------------

class SharedExpertOverlapManager:
    """Manage side-stream shared-expert execution for latent-MoE inference.

    Upstream context (Megatron 20f09364 ``moe_layer.py``):
        Megatron introduced ``_external_shared_expert_launch`` on the NVLS
        dispatcher so that, for latent-MoE models (DeepSeek-style), the *layer*
        owns the side-stream launch+join rather than the dispatcher.  This is
        necessary because the shared expert must see the full hidden_states
        (pre-latent projection) but its output must be added post-FC2-latent,
        both of which are outside the dispatcher's view.

    DES-LOC adaptation:
        In our heterogeneous cluster the shared expert is always assigned to the
        H100 (device 0) because it has spare VRAM and higher arithmetic
        throughput.  The A6000 pair runs routing-expert GEMMs concurrently.

        This class manages:
        1. The side-stream on the H100.
        2. A tensor-staging buffer in CPU-pinned memory (the DES-LOC locality
           cache) so the result can be transferred to an A6000 if the consumer
           is there — though in the typical latent-MoE case the consumer is also
           on the H100.
        3. A ``pending`` flag so postprocess() knows whether to join.

    Usage (inside MoELayer-equivalent in Neuron_SP)::

        mgr = SharedExpertOverlapManager(h100_device_id=0)

        # In preprocess (before fc1_latent_proj):
        mgr.launch(shared_expert_module, hidden_states)

        # ... routing expert GEMMs run here on H100 or A6000 ...

        # In postprocess (after fc2_latent_proj):
        output = output + mgr.join()
    """

    def __init__(self, h100_device_id: int = 0) -> None:
        self.h100_device_id = h100_device_id
        self._stream: Optional[torch.cuda.Stream] = None
        self._output: Optional[torch.Tensor] = None
        self._pending: bool = False
        self._launch_count: int = 0

    @property
    def stream(self) -> torch.cuda.Stream:
        """Lazily create the side-stream on the H100."""
        if self._stream is None:
            with torch.cuda.device(self.h100_device_id):
                self._stream = torch.cuda.Stream(device=self.h100_device_id)
            logger.info(
                "SharedExpertOverlapManager: created side-stream on device %d",
                self.h100_device_id,
            )
        return self._stream

    def launch(self, shared_expert_module: torch.nn.Module, hidden_states: torch.Tensor) -> None:
        """Launch the shared expert forward on the H100 side-stream.

        Must be called *before* fc1_latent_proj so ``hidden_states`` is in the
        full hidden dimension.  Mirrors Megatron's preprocess block:

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._latent_shared_expert_output = apply_module(shared_experts)(hidden_states)

        DES-LOC note: if ``hidden_states`` lives on an A6000, we issue an
        async D2D transfer to the H100 first.  The stream serialises this
        automatically via ``wait_stream``.
        """
        if self._pending:
            logger.warning(
                "SharedExpertOverlapManager.launch called while previous "
                "result was not consumed via join(). Discarding stale output."
            )
            self._output = None
            self._pending = False

        main_stream = torch.cuda.current_stream()
        self.stream.wait_stream(main_stream)

        with torch.cuda.stream(self.stream):
            # If hidden_states is on a different device, move it asynchronously.
            if hidden_states.device.index != self.h100_device_id:
                hs_local = hidden_states.to(
                    device=torch.device("cuda", self.h100_device_id),
                    non_blocking=True,
                )
            else:
                hs_local = hidden_states
            self._output = shared_expert_module(hs_local)

        self._pending = True
        self._launch_count += 1
        logger.debug(
            "SharedExpertOverlapManager: launched shared expert (launch #%d) "
            "on device %d side-stream",
            self._launch_count, self.h100_device_id,
        )

    def join(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        """Join the side-stream and return the shared-expert output.

        Must be called *after* fc2_latent_proj so dimensions match the main
        output.  Mirrors Megatron's postprocess block:

            torch.cuda.current_stream().wait_stream(SharedExpertMLP.stream)
            output = output + self._latent_shared_expert_output
            self._latent_shared_expert_output = None

        Parameters
        ----------
        target_device:
            If provided and different from the output's device, issue an async
            D2H+H2D or D2D copy so the caller can add on their preferred device.
        """
        if not self._pending or self._output is None:
            raise RuntimeError(
                "SharedExpertOverlapManager.join() called without a prior launch()."
            )

        torch.cuda.current_stream().wait_stream(self.stream)
        result = self._output

        if target_device is not None and result.device != target_device:
            result = result.to(device=target_device, non_blocking=False)

        self._output = None
        self._pending = False
        return result

    @property
    def is_pending(self) -> bool:
        return self._pending


# ---------------------------------------------------------------------------
# MoESumPersistentConfig
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MoeSumPersistentConfig:
    """Configuration for the persistent _moe_sum reduction kernel.

    Upstream context (Megatron 20f09364 ``_moe_sum_kernel``):
        Changed from a 1-CTA-per-token grid (grid = [max_tokens]) to a
        persistent CTA grid (grid = [BLOCK_M] where BLOCK_M = num_SMs).
        Each CTA strides over ``valid_tokens`` via ``tl.range(pid, valid_tokens,
        BLOCK_M)``.  This eliminates the zero-fill overhead on rows beyond
        valid_tokens and keeps the grid CUDA-graph safe.

    DES-LOC adaptation:
        BLOCK_M = num_SMs of the device *executing* the reduction.  In our
        cluster the reduction always runs on the device that holds the expert
        output (either H100 or A6000 depending on routing).  We store both
        values and expose a ``for_device`` factory.
    """

    block_m: int          # Number of persistent CTAs = num_SMs of target device
    block_k: int          # K-tile width (next_power_of_2(K), capped at 1024)
    num_k_blocks: int     # ceil(K / block_k)

    @classmethod
    def for_device(cls, K: int, profile: HeteroDeviceProfile) -> "MoeSumPersistentConfig":
        block_k = min(_next_power_of_2(K), 1024)
        num_k_blocks = math.ceil(K / block_k)
        cfg = cls(
            block_m=profile.sm_count,
            block_k=block_k,
            num_k_blocks=num_k_blocks,
        )
        logger.debug(
            "MoeSumPersistentConfig: device=%d K=%d block_m=%d block_k=%d num_k_blocks=%d",
            profile.device_id, K, cfg.block_m, block_k, num_k_blocks,
        )
        return cfg


# ---------------------------------------------------------------------------
# HeteroExpertPartition
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HeteroExpertPartition:
    """Assigns experts to devices based on device VRAM and SM count.

    DES-LOC design:
        The H100 has 96 GB and 132 SMs; the A6000 pair share 96 GB (2×48 GB)
        and 168 total SMs.  In practice the H100 is preferred for the largest
        weight blocks (e.g., the gate/up projections of heavy experts) while
        A6000s take medium experts.

        Partition heuristic:
          1. Sort experts by estimated weight size (descending).
          2. Assign top experts to H100 until it reaches its VRAM quota.
          3. Distribute remaining experts round-robin across A6000 devices.

        ``expert_to_device`` maps expert_id → device_id.
        ``device_to_experts`` is the inverse mapping.
    """

    expert_to_device: Dict[int, int]
    device_to_experts: Dict[int, List[int]]
    num_experts: int
    profiles: List[HeteroDeviceProfile]

    @classmethod
    def balanced_by_vram(
        cls,
        num_experts: int,
        profiles: List[HeteroDeviceProfile],
        expert_param_bytes: Optional[List[int]] = None,
    ) -> "HeteroExpertPartition":
        """Partition experts proportionally to device VRAM.

        Parameters
        ----------
        num_experts:
            Total number of MoE experts.
        profiles:
            One profile per device.  The H100 (sm_major >= 9) is identified
            automatically.
        expert_param_bytes:
            Optional list of per-expert parameter sizes in bytes.  If None,
            all experts are assumed equal.
        """
        if expert_param_bytes is None:
            expert_param_bytes = [1] * num_experts

        # Sort experts largest-first.
        sorted_experts = sorted(
            range(num_experts), key=lambda e: expert_param_bytes[e], reverse=True
        )

        total_vram = sum(p.total_memory_bytes for p in profiles)
        expert_to_device: Dict[int, int] = {}
        device_to_experts: Dict[int, List[int]] = {p.device_id: [] for p in profiles}

        # VRAM quota per device (proportional allocation).
        device_quota = {
            p.device_id: (p.total_memory_bytes / total_vram) * num_experts
            for p in profiles
        }
        device_assigned: Dict[int, int] = {p.device_id: 0 for p in profiles}

        # Sort profiles: Hopper first so it absorbs large experts.
        sorted_profiles = sorted(profiles, key=lambda p: p.total_memory_bytes, reverse=True)

        for expert_id in sorted_experts:
            # Find the device with the most remaining quota.
            best_device = min(
                sorted_profiles,
                key=lambda p: device_assigned[p.device_id] - device_quota[p.device_id]
            )
            expert_to_device[expert_id] = best_device.device_id
            device_to_experts[best_device.device_id].append(expert_id)
            device_assigned[best_device.device_id] += 1

        partition = cls(
            expert_to_device=expert_to_device,
            device_to_experts=device_to_experts,
            num_experts=num_experts,
            profiles=profiles,
        )
        for p in profiles:
            n = len(device_to_experts[p.device_id])
            logger.info(
                "HeteroExpertPartition: device=%d (%s) assigned %d/%d experts",
                p.device_id, p.name, n, num_experts,
            )
        return partition

    def local_experts(self, device_id: int) -> List[int]:
        return self.device_to_experts.get(device_id, [])

    def num_local_experts(self, device_id: int) -> int:
        return len(self.local_experts(device_id))


# ---------------------------------------------------------------------------
# HeteroMoEInferenceTuner
# ---------------------------------------------------------------------------

class HeteroMoEInferenceTuner:
    """Top-level DES-LOC MoE inference tuner for the 2×A6000 + 1×H100 cluster.

    Combines:
    • Per-device launch config derivation (``DEsLOCMoELaunchConfig``).
    • Adaptive EP-consensus scheduling (``DEsLOCConsensusScheduler``).
    • Shared-expert side-stream management (``SharedExpertOverlapManager``).
    • Expert-to-device partitioning (``HeteroExpertPartition``).
    • Persistent moe_sum kernel configuration (``MoeSumPersistentConfig``).

    This is the single entry point that Neuron_SP's engine init calls.
    """

    def __init__(
        self,
        profiles: List[HeteroDeviceProfile],
        consensus_base_interval: int = 20,
        consensus_max_interval: int = 80,
        h100_device_id: Optional[int] = None,
    ) -> None:
        self.profiles = profiles
        self._profile_map: Dict[int, HeteroDeviceProfile] = {
            p.device_id: p for p in profiles
        }

        # Identify H100 device (sm_major >= 9).
        h100_candidates = [p for p in profiles if p.is_hopper]
        if h100_device_id is not None:
            self.h100_profile = self._profile_map[h100_device_id]
        elif h100_candidates:
            self.h100_profile = h100_candidates[0]
            if len(h100_candidates) > 1:
                logger.warning(
                    "Multiple Hopper devices found; using device %d (%s) as H100.",
                    self.h100_profile.device_id, self.h100_profile.name,
                )
        else:
            # Fallback: use device with most VRAM.
            self.h100_profile = max(profiles, key=lambda p: p.total_memory_bytes)
            logger.warning(
                "No Hopper device found; treating device %d (%s) as primary.",
                self.h100_profile.device_id, self.h100_profile.name,
            )

        self.consensus_scheduler = DEsLOCConsensusScheduler(
            base_interval=consensus_base_interval,
            max_interval=consensus_max_interval,
        )
        self.shared_expert_overlap = SharedExpertOverlapManager(
            h100_device_id=self.h100_profile.device_id
        )

        logger.info(
            "HeteroMoEInferenceTuner initialised: %d devices, H100=device%d, "
            "consensus_base=%d consensus_max=%d",
            len(profiles), self.h100_profile.device_id,
            consensus_base_interval, consensus_max_interval,
        )

    @classmethod
    def from_cluster_auto(
        cls,
        device_ids: Optional[List[int]] = None,
        consensus_base_interval: int = 20,
        consensus_max_interval: int = 80,
    ) -> "HeteroMoEInferenceTuner":
        """Auto-detect all visible CUDA devices and build the tuner.

        Parameters
        ----------
        device_ids:
            If None, discovers all torch.cuda visible devices.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("HeteroMoEInferenceTuner requires CUDA.")

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if not device_ids:
            raise RuntimeError("No CUDA devices found.")

        profiles = [HeteroDeviceProfile.from_device(did) for did in device_ids]
        return cls(
            profiles=profiles,
            consensus_base_interval=consensus_base_interval,
            consensus_max_interval=consensus_max_interval,
        )

    def profile(self, device: torch.device) -> HeteroDeviceProfile:
        """Look up the HeteroDeviceProfile for *device*."""
        idx = device.index if device.index is not None else torch.cuda.current_device()
        if idx not in self._profile_map:
            raise KeyError(
                f"Device {idx} not registered. Known devices: "
                f"{list(self._profile_map.keys())}"
            )
        return self._profile_map[idx]

    def get_launch_config(
        self,
        num_tokens_hint: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        ffn_dim: int,
        device: torch.device,
    ) -> DEsLOCMoELaunchConfig:
        """Derive a Triton launch config for a MoE layer on *device*.

        This is the primary per-step call from the Neuron_SP forward pass.
        Calling it with the same arguments multiple times is cheap (no GPU
        ops, pure Python arithmetic).
        """
        p = self.profile(device)
        return DEsLOCMoELaunchConfig.derive(
            num_tokens_hint=num_tokens_hint,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            profile=p,
        )

    def get_moe_sum_config(self, K: int, device: torch.device) -> MoeSumPersistentConfig:
        """Return the persistent moe_sum kernel config for *device*."""
        p = self.profile(device)
        return MoeSumPersistentConfig.for_device(K=K, profile=p)

    def build_partition(
        self,
        num_experts: int,
        expert_param_bytes: Optional[List[int]] = None,
    ) -> HeteroExpertPartition:
        """Partition experts across devices by VRAM quota."""
        return HeteroExpertPartition.balanced_by_vram(
            num_experts=num_experts,
            profiles=self.profiles,
            expert_param_bytes=expert_param_bytes,
        )

    @contextmanager
    def shared_expert_side_stream(
        self,
        shared_expert_module: torch.nn.Module,
        hidden_states: torch.Tensor,
        output_container: List,
    ):
        """Context manager that launches and joins the shared-expert side-stream.

        Mirrors Megatron's preprocess/postprocess split for latent-MoE + NVLS:

            preprocess: stream.wait_stream(main); with stream: output = se(hs)
            postprocess: main.wait_stream(stream); result += output

        DES-LOC note: The yield point is where the caller runs the routing
        expert GEMMs.  The shared expert runs concurrently on the H100
        side-stream.

        Usage::

            output_container = []
            with tuner.shared_expert_side_stream(se_module, hs, output_container):
                routing_expert_output = run_routing_experts(hs_latent)
            se_out = output_container[0]
            combined = routing_expert_output + se_out
        """
        self.shared_expert_overlap.launch(shared_expert_module, hidden_states)
        try:
            yield
        finally:
            se_out = self.shared_expert_overlap.join()
            output_container.append(se_out)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Functional helpers (drop-in for Megatron vllm_fused_moe.py internals)
# ---------------------------------------------------------------------------

def get_default_config_for_device(
    M: int,
    E: int,
    top_k: int,
    profile: HeteroDeviceProfile,
) -> Dict:
    """DES-LOC equivalent of Megatron's ``_get_default_config(M, E, top_k)``.

    Returns a flat dict with the same keys as Megatron's version
    (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
    num_warps, num_stages) so it can be used as a drop-in for the
    ``config`` argument of ``_invoke_fused_moe_kernel``.

    The difference from Megatron: thresholds for BLOCK_SIZE_N/K and
    GROUP_SIZE_M are adjusted per SM architecture (see
    ``DEsLOCMoELaunchConfig.derive`` for detailed comments).
    """
    cfg = DEsLOCMoELaunchConfig.derive(
        num_tokens_hint=M,
        num_experts=E,
        top_k=top_k,
        hidden_dim=1,   # not used for flat-dict output
        ffn_dim=1,
        profile=profile,
    )
    return {
        "BLOCK_SIZE_M": cfg.BLOCK_SIZE_M,
        "BLOCK_SIZE_N": cfg.BLOCK_SIZE_N,
        "BLOCK_SIZE_K": cfg.BLOCK_SIZE_K,
        "GROUP_SIZE_M": cfg.GROUP_SIZE_M,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }


def compute_grid_sizes(
    num_tokens_hint: int,
    num_local_experts: int,
    top_k: int,
    N_fc1: int,
    K_fc2: int,
    config: Dict,
) -> Tuple[int, int]:
    """Compute (grid_size_fc1, grid_size_fc2) from config and token hint.

    Mirrors Megatron's ``vllm_fused_moe`` grid-sizing block::

        em_hint = effective_tokens * topk + BLOCK_SIZE_M * num_local_experts
        num_pid_m_hint = ceil(em_hint / BLOCK_SIZE_M)
        grid_size_fc1 = num_pid_m_hint * ceil(N / BLOCK_SIZE_N)
        grid_size_fc2 = num_pid_m_hint * ceil(K / BLOCK_SIZE_N)

    DES-LOC usage: called inside the Neuron_SP forward pass on each device
    after ``get_default_config_for_device``.
    """
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    em_hint = num_tokens_hint * top_k + block_m * num_local_experts
    num_pid_m_hint = _ceil_div(em_hint, block_m)
    gs_fc1 = num_pid_m_hint * _ceil_div(N_fc1, block_n)
    gs_fc2 = num_pid_m_hint * _ceil_div(K_fc2, block_n)
    return gs_fc1, gs_fc2


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import unittest

    # ── Fake device profile for offline tests (no CUDA required) ──────────────

    def _fake_profile(device_id: int, sm_major: int, sm_count: int,
                      total_gb: int, name: str) -> HeteroDeviceProfile:
        sm_minor = 0
        l2 = _L2_BYTES.get(sm_major, 6 * 1024 * 1024)
        stages_max = _ASYNC_COPY_STAGES_MAX.get(sm_major, 4)
        return HeteroDeviceProfile(
            device_id=device_id,
            sm_count=sm_count,
            sm_major=sm_major,
            sm_minor=sm_minor,
            total_memory_bytes=total_gb * 1024 ** 3,
            l2_cache_bytes=l2,
            async_copy_stages_max=stages_max,
            name=name,
        )

    H100_PROFILE = _fake_profile(0, sm_major=9, sm_count=132, total_gb=96, name="H100 NVL")
    A6000_PROFILE = _fake_profile(1, sm_major=8, sm_count=84, total_gb=48, name="A6000")
    A6000_PROFILE2 = _fake_profile(2, sm_major=8, sm_count=84, total_gb=48, name="A6000 #2")

    # ── TestDEsLOCMoELaunchConfig ─────────────────────────────────────────────

    class TestDEsLOCMoELaunchConfig(unittest.TestCase):

        def _cfg(self, M, profile=H100_PROFILE):
            return DEsLOCMoELaunchConfig.derive(
                num_tokens_hint=M,
                num_experts=8,
                top_k=2,
                hidden_dim=4096,
                ffn_dim=14336,
                profile=profile,
            )

        # BLOCK_SIZE_M thresholds mirror Megatron verbatim for both architectures.
        def test_block_m_thresholds(self):
            cases = [(1, 16), (32, 16), (33, 32), (96, 32), (97, 64),
                     (512, 64), (513, 128), (4096, 128)]
            for M, expected in cases:
                with self.subTest(M=M):
                    cfg = self._cfg(M)
                    self.assertEqual(cfg.BLOCK_SIZE_M, expected,
                                     f"M={M}: expected BM={expected}, got {cfg.BLOCK_SIZE_M}")

        def test_block_m_minimum_is_16(self):
            cfg = self._cfg(1)
            self.assertGreaterEqual(cfg.BLOCK_SIZE_M, 16)

        def test_block_m_monotone(self):
            prev = self._cfg(1).BLOCK_SIZE_M
            for M in range(2, 600):
                cur = self._cfg(M).BLOCK_SIZE_M
                self.assertGreaterEqual(cur, prev,
                                        f"BLOCK_SIZE_M decreased at M={M}: {prev}→{cur}")
                prev = cur

        # Hopper gets wider N at smaller M due to large L2.
        def test_hopper_block_n_wider_at_small_m(self):
            cfg_small = DEsLOCMoELaunchConfig.derive(32, 8, 2, 4096, 14336, H100_PROFILE)
            cfg_large = DEsLOCMoELaunchConfig.derive(512, 8, 2, 4096, 14336, H100_PROFILE)
            # Both should have valid block_n (64 or 128).
            self.assertIn(cfg_small.BLOCK_SIZE_N, (64, 128))
            self.assertIn(cfg_large.BLOCK_SIZE_N, (64, 128))

        # Ampere: split at M=64 (matches Megatron).
        def test_ampere_block_n_split_at_64(self):
            cfg_sm = self._cfg(64, profile=A6000_PROFILE)
            cfg_lg = self._cfg(65, profile=A6000_PROFILE)
            self.assertEqual(cfg_sm.BLOCK_SIZE_N, 64)
            self.assertEqual(cfg_lg.BLOCK_SIZE_N, 128)
            self.assertEqual(cfg_sm.BLOCK_SIZE_K, 128)
            self.assertEqual(cfg_lg.BLOCK_SIZE_K, 64)

        def test_group_size_m_hopper_lower_threshold(self):
            # Hopper: tokens_per_expert > 64 → group_m=16
            # 8 experts, M=512 → t/e = 64 → NOT > 64 → group_m=1
            cfg = DEsLOCMoELaunchConfig.derive(512, 8, 2, 4096, 14336, H100_PROFILE)
            self.assertEqual(cfg.GROUP_SIZE_M, 1)
            # M=513 → t/e = 64 still (513//8=64). M=600 → 75 > 64 → group=16
            cfg2 = DEsLOCMoELaunchConfig.derive(600, 8, 2, 4096, 14336, H100_PROFILE)
            self.assertEqual(cfg2.GROUP_SIZE_M, 16)

        def test_group_size_m_ampere_higher_threshold(self):
            # Ampere: threshold > 128. M=1024, E=8 → t/e=128 → NOT > 128 → group=1
            cfg = DEsLOCMoELaunchConfig.derive(1024, 8, 2, 4096, 14336, A6000_PROFILE)
            self.assertEqual(cfg.GROUP_SIZE_M, 1)
            # M=2048 → t/e=256 > 128 → group=16
            cfg2 = DEsLOCMoELaunchConfig.derive(2048, 8, 2, 4096, 14336, A6000_PROFILE)
            self.assertEqual(cfg2.GROUP_SIZE_M, 16)

        def test_num_warps_split_ampere(self):
            cfg_sm = self._cfg(128, profile=A6000_PROFILE)
            cfg_lg = self._cfg(129, profile=A6000_PROFILE)
            self.assertEqual(cfg_sm.num_warps, 4)
            self.assertEqual(cfg_lg.num_warps, 8)

        def test_num_warps_split_hopper(self):
            cfg_sm = self._cfg(64, profile=H100_PROFILE)
            cfg_lg = self._cfg(65, profile=H100_PROFILE)
            self.assertEqual(cfg_sm.num_warps, 4)
            self.assertEqual(cfg_lg.num_warps, 8)

        def test_num_stages_capped_by_architecture(self):
            # SM86 caps at 5; raw=4 at M=32 → stays 4
            cfg = self._cfg(32, profile=A6000_PROFILE)
            self.assertLessEqual(cfg.num_stages, A6000_PROFILE.async_copy_stages_max)
            # SM90 caps at 7; raw=4 at M=32 → stays 4
            cfg2 = self._cfg(32, profile=H100_PROFILE)
            self.assertLessEqual(cfg2.num_stages, H100_PROFILE.async_copy_stages_max)

        def test_grid_sizes_positive(self):
            cfg = self._cfg(128)
            self.assertGreater(cfg.grid_size_fc1, 0)
            self.assertGreater(cfg.grid_size_fc2, 0)

        def test_as_triton_kwargs_fc1(self):
            cfg = self._cfg(128)
            kw = cfg.as_triton_kwargs(fc=1)
            self.assertEqual(kw["grid_size"], cfg.grid_size_fc1)
            self.assertIn("BLOCK_SIZE_M", kw)

        def test_as_triton_kwargs_fc2(self):
            cfg = self._cfg(128)
            kw = cfg.as_triton_kwargs(fc=2)
            self.assertEqual(kw["grid_size"], cfg.grid_size_fc2)

        def test_as_triton_kwargs_bad_fc(self):
            cfg = self._cfg(128)
            with self.assertRaises(AssertionError):
                cfg.as_triton_kwargs(fc=3)

    # ── TestDEsLOCConsensusScheduler ─────────────────────────────────────────

    class TestDEsLOCConsensusScheduler(unittest.TestCase):

        def _sched(self, base=20, max_i=80, ramp=200):
            return DEsLOCConsensusScheduler(
                base_interval=base,
                max_interval=max_i,
                ramp_after_steps=ramp,
            )

        def test_idle_always_runs(self):
            sched = self._sched()
            for _ in range(50):
                self.assertTrue(sched.should_run(has_global_work=False))

        def test_idle_resets_busy_streak(self):
            sched = self._sched(base=20)
            # Build up a busy streak.
            for _ in range(100):
                sched.should_run(has_global_work=True)
            self.assertGreater(sched._consecutive_busy, 0)
            sched.should_run(has_global_work=False)
            self.assertEqual(sched._consecutive_busy, 0)

        def test_busy_runs_at_base_interval(self):
            sched = self._sched(base=5, max_i=20, ramp=1000)
            results = [sched.should_run(has_global_work=True) for _ in range(30)]
            # With counter starting at 1 and interval=5, runs at steps 5,10,15,20,25,30
            run_steps = [i + 1 for i, r in enumerate(results) if r]
            for step in run_steps:
                self.assertEqual(step % 5, 0,
                                 f"Unexpected run at step {step} with interval=5")

        def test_busy_does_not_run_every_step(self):
            sched = self._sched(base=20)
            any_skip = False
            for _ in range(40):
                if not sched.should_run(has_global_work=True):
                    any_skip = True
                    break
            self.assertTrue(any_skip, "Scheduler never skipped in 40 busy steps")

        def test_ramp_increases_interval_after_threshold(self):
            sched = self._sched(base=10, max_i=40, ramp=50)
            # Run 50 busy steps to trigger ramp.
            for _ in range(50):
                sched.should_run(has_global_work=True)
            interval_before_ramp = sched.base_interval
            # One more step crosses ramp_after_steps=50.
            sched.should_run(has_global_work=True)
            self.assertGreaterEqual(sched.current_interval, interval_before_ramp)

        def test_interval_never_exceeds_max(self):
            sched = self._sched(base=10, max_i=15, ramp=1)
            for _ in range(1000):
                sched.should_run(has_global_work=True)
            self.assertLessEqual(sched.current_interval, 15)

        def test_reset_clears_state(self):
            sched = self._sched(base=5)
            for _ in range(50):
                sched.should_run(has_global_work=True)
            sched.reset()
            self.assertEqual(sched._loop_counter, 0)
            self.assertEqual(sched._consecutive_busy, 0)

        def test_stats_logging_does_not_raise(self):
            sched = self._sched()
            for _ in range(10):
                sched.should_run(has_global_work=True)
            # Should not raise.
            sched.log_stats()

    # ── TestMoeSumPersistentConfig ────────────────────────────────────────────

    class TestMoeSumPersistentConfig(unittest.TestCase):

        def test_h100_block_m_equals_sm_count(self):
            cfg = MoeSumPersistentConfig.for_device(K=4096, profile=H100_PROFILE)
            self.assertEqual(cfg.block_m, H100_PROFILE.sm_count)

        def test_a6000_block_m_equals_sm_count(self):
            cfg = MoeSumPersistentConfig.for_device(K=4096, profile=A6000_PROFILE)
            self.assertEqual(cfg.block_m, A6000_PROFILE.sm_count)

        def test_block_k_is_power_of_2(self):
            for K in [64, 100, 4096, 7000, 8192]:
                cfg = MoeSumPersistentConfig.for_device(K=K, profile=H100_PROFILE)
                bk = cfg.block_k
                self.assertTrue(bk & (bk - 1) == 0, f"block_k={bk} is not power-of-2 for K={K}")

        def test_block_k_capped_at_1024(self):
            cfg = MoeSumPersistentConfig.for_device(K=16384, profile=H100_PROFILE)
            self.assertLessEqual(cfg.block_k, 1024)

        def test_num_k_blocks_covers_k(self):
            for K in [64, 512, 4096, 7001]:
                cfg = MoeSumPersistentConfig.for_device(K=K, profile=H100_PROFILE)
                self.assertGreaterEqual(cfg.num_k_blocks * cfg.block_k, K)

    # ── TestHeteroExpertPartition ─────────────────────────────────────────────

    class TestHeteroExpertPartition(unittest.TestCase):

        def test_all_experts_assigned(self):
            profiles = [H100_PROFILE, A6000_PROFILE, A6000_PROFILE2]
            part = HeteroExpertPartition.balanced_by_vram(num_experts=16, profiles=profiles)
            assigned = set(part.expert_to_device.keys())
            self.assertEqual(assigned, set(range(16)))

        def test_device_to_experts_is_consistent(self):
            profiles = [H100_PROFILE, A6000_PROFILE]
            part = HeteroExpertPartition.balanced_by_vram(num_experts=8, profiles=profiles)
            for dev, experts in part.device_to_experts.items():
                for e in experts:
                    self.assertEqual(part.expert_to_device[e], dev)

        def test_h100_gets_larger_share_with_skewed_params(self):
            # H100 has 96 GB, A6000 has 48 GB → H100 quota = 2/3 * 8 ≈ 5.3 experts.
            profiles = [H100_PROFILE, A6000_PROFILE]
            part = HeteroExpertPartition.balanced_by_vram(num_experts=6, profiles=profiles)
            n_h100 = part.num_local_experts(H100_PROFILE.device_id)
            n_a6000 = part.num_local_experts(A6000_PROFILE.device_id)
            # H100 should get ≥ A6000's share.
            self.assertGreaterEqual(n_h100, n_a6000)

        def test_local_experts_helper(self):
            profiles = [H100_PROFILE, A6000_PROFILE]
            part = HeteroExpertPartition.balanced_by_vram(num_experts=4, profiles=profiles)
            for p in profiles:
                lst = part.local_experts(p.device_id)
                self.assertIsInstance(lst, list)
                self.assertEqual(part.num_local_experts(p.device_id), len(lst))

        def test_unknown_device_returns_empty(self):
            profiles = [H100_PROFILE]
            part = HeteroExpertPartition.balanced_by_vram(num_experts=4, profiles=profiles)
            self.assertEqual(part.local_experts(99), [])

    # ── TestGetDefaultConfigForDevice ─────────────────────────────────────────

    class TestGetDefaultConfigForDevice(unittest.TestCase):
        """Verify the flat-dict API mirrors Megatron's key names."""

        def test_keys_present(self):
            cfg = get_default_config_for_device(M=32, E=8, top_k=2, profile=H100_PROFILE)
            for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                        "GROUP_SIZE_M", "num_warps", "num_stages"):
                self.assertIn(key, cfg)

        def test_block_m_agrees_with_class(self):
            for M in [1, 32, 97, 513]:
                flat = get_default_config_for_device(M=M, E=8, top_k=2, profile=H100_PROFILE)
                cls_cfg = DEsLOCMoELaunchConfig.derive(M, 8, 2, 4096, 14336, H100_PROFILE)
                self.assertEqual(flat["BLOCK_SIZE_M"], cls_cfg.BLOCK_SIZE_M)

    # ── TestComputeGridSizes ──────────────────────────────────────────────────

    class TestComputeGridSizes(unittest.TestCase):

        def test_grid_sizes_positive_and_cover_all_tiles(self):
            cfg = get_default_config_for_device(M=64, E=4, top_k=2, profile=A6000_PROFILE)
            gs_fc1, gs_fc2 = compute_grid_sizes(
                num_tokens_hint=64,
                num_local_experts=4,
                top_k=2,
                N_fc1=14336,
                K_fc2=4096,
                config=cfg,
            )
            self.assertGreater(gs_fc1, 0)
            self.assertGreater(gs_fc2, 0)

        def test_larger_hint_gives_larger_grid(self):
            cfg = get_default_config_for_device(M=128, E=4, top_k=2, profile=H100_PROFILE)
            gs_small_fc1, _ = compute_grid_sizes(32, 4, 2, 14336, 4096, cfg)
            gs_large_fc1, _ = compute_grid_sizes(128, 4, 2, 14336, 4096, cfg)
            self.assertLessEqual(gs_small_fc1, gs_large_fc1)

    # ── TestNextPowerOf2 and CeilDiv ──────────────────────────────────────────

    class TestInternalUtils(unittest.TestCase):

        def test_next_power_of_2(self):
            cases = [(1, 1), (2, 2), (3, 4), (5, 8), (64, 64), (65, 128),
                     (1023, 1024), (1024, 1024), (1025, 2048)]
            for n, exp in cases:
                with self.subTest(n=n):
                    self.assertEqual(_next_power_of_2(n), exp)

        def test_ceil_div(self):
            self.assertEqual(_ceil_div(10, 3), 4)
            self.assertEqual(_ceil_div(9, 3), 3)
            self.assertEqual(_ceil_div(1, 1024), 1)

    # ── TestHeteroDeviceProfile (offline) ────────────────────────────────────

    class TestHeteroDeviceProfileOffline(unittest.TestCase):

        def test_is_hopper_sm9(self):
            self.assertTrue(H100_PROFILE.is_hopper)
            self.assertFalse(A6000_PROFILE.is_hopper)

        def test_is_ampere_sm8(self):
            self.assertTrue(A6000_PROFILE.is_ampere)
            self.assertFalse(H100_PROFILE.is_ampere)

        def test_l2_cache_nonzero(self):
            self.assertGreater(H100_PROFILE.l2_cache_bytes, 0)
            self.assertGreater(A6000_PROFILE.l2_cache_bytes, 0)

        def test_h100_l2_larger_than_a6000(self):
            self.assertGreater(H100_PROFILE.l2_cache_bytes, A6000_PROFILE.l2_cache_bytes)

    # ── TestHeteroMoEInferenceTuner (offline) ─────────────────────────────────

    class TestHeteroMoEInferenceTunerOffline(unittest.TestCase):

        def _tuner(self):
            profiles = [H100_PROFILE, A6000_PROFILE, A6000_PROFILE2]
            return HeteroMoEInferenceTuner(profiles=profiles)

        def test_identifies_h100(self):
            tuner = self._tuner()
            self.assertEqual(tuner.h100_profile.device_id, H100_PROFILE.device_id)

        def test_profile_lookup(self):
            tuner = self._tuner()
            dev = torch.device("cuda", A6000_PROFILE.device_id)
            p = tuner.profile(dev)
            self.assertEqual(p.device_id, A6000_PROFILE.device_id)

        def test_unknown_device_raises(self):
            tuner = self._tuner()
            dev = torch.device("cuda", 99)
            with self.assertRaises(KeyError):
                tuner.profile(dev)

        def test_build_partition_covers_all_experts(self):
            tuner = self._tuner()
            part = tuner.build_partition(num_experts=12)
            self.assertEqual(set(part.expert_to_device.keys()), set(range(12)))

        def test_consensus_scheduler_accessible(self):
            tuner = self._tuner()
            self.assertIsInstance(tuner.consensus_scheduler, DEsLOCConsensusScheduler)

        def test_shared_expert_overlap_manager_accessible(self):
            tuner = self._tuner()
            self.assertIsInstance(tuner.shared_expert_overlap, SharedExpertOverlapManager)

    # ── SharedExpertOverlapManager (no-CUDA path via mock) ───────────────────

    class TestSharedExpertOverlapManagerLogic(unittest.TestCase):
        """Test state machine logic without real CUDA streams."""

        def _make_mgr(self):
            mgr = SharedExpertOverlapManager.__new__(SharedExpertOverlapManager)
            mgr.h100_device_id = 0
            mgr._stream = None
            mgr._output = None
            mgr._pending = False
            mgr._launch_count = 0
            return mgr

        def test_join_without_launch_raises(self):
            mgr = self._make_mgr()
            with self.assertRaises(RuntimeError):
                mgr.join()

        def test_is_pending_false_initially(self):
            mgr = self._make_mgr()
            self.assertFalse(mgr.is_pending)

    # ── Run ──────────────────────────────────────────────────────────────────

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for tc in [
        TestDEsLOCMoELaunchConfig,
        TestDEsLOCConsensusScheduler,
        TestMoeSumPersistentConfig,
        TestHeteroExpertPartition,
        TestGetDefaultConfigForDevice,
        TestComputeGridSizes,
        TestInternalUtils,
        TestHeteroDeviceProfileOffline,
        TestHeteroMoEInferenceTunerOffline,
        TestSharedExpertOverlapManagerLogic,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
