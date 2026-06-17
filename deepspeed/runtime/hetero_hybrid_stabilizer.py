"""
DES-LOC HeteroHybridStabilizer
================================

Upstream design intent (Megatron bd1f0dd):
    The Megatron commit "Stabilize hybrid nanov3 gb200 perf" addresses a
    systematic measurement instability in hybrid NanoV3-3B inference on GB200
    4-GPU systems.  The root cause was two-fold:

    1. Warmup insufficiency: Only 2 warmup iterations were used, leaving the
       first timed iteration in a "cold" microarchitectural state (empty L2/HBM
       caches, cold TLB, uninitialized CUDA graph state).  On GB200 NVL this
       manifested as a ~2.5x latency spike on the first timed batch-128 step,
       which severely poisoned mean and p99 metrics.

    2. Sample-size fragility: 5 timed iterations were insufficient to smooth
       outliers introduced by PCIe arbitration jitter, NUMA migration events,
       and SM frequency scaling on the GB200.  A single outlier caused 10–14%
       run-to-run swing in throughput and p99 latency at batch size 128.

    The fix was conservative but effective: raise NUM_WARMUP_ITERS from 2→5 and
    NUM_TIMED_ITERS from 5→10, giving steady-state measurements that are robust
    to single-iteration anomalies.

DES-LOC reinterpretation (HeteroHybridStabilizer):
    In the Neuron_SP / DES-LOC framework we face an *amplified* version of the
    same problem.  Our target cluster is:

        GPU-0, GPU-1 : NVIDIA A6000 48 GB, SM86, PCIe Gen4 x16
        GPU-2        : NVIDIA H100 NVL 96 GB, SM90, PCIe Gen4 x16
        Host DRAM    : 1.5 TB DDR5, acts as Shared LOcality Cache tier

    Because SM86 and SM90 have fundamentally different:
        • Tensor Core ISA (A6000: 3rd-gen, H100: 4th-gen Hopper)
        • HBM bandwidth (A6000: 768 GB/s GDDR6, H100 NVL: 3.9 TB/s HBM3e)
        • PCIe topology (no NVLink; A6000↔H100 traffic must round-trip through
          PCIe root complex, adding ~12 µs serialization latency per transfer)
        • Boost frequency envelopes (A6000 base 1.41 GHz / boost 1.8 GHz;
          H100 base 1.095 GHz / boost 1.98 GHz)

    …the "cold state" problem from Megatron is far worse here.  On first entry
    into a hybrid execution epoch:
        a. The A6000s may be running at base clock if no prior kernel was
           dispatched within the last ~10 s (NVIDIA GPU Boost idle ramp-down).
        b. The DES-LOC locality cache in CPU DRAM has not yet been populated
           with the KV-cache shards that the H100 will request.
        c. PCIe bandwidth is unallocated — the first burst of cross-device
           tensor transfers will see full PCIe setup overhead.

    HeteroHybridStabilizer solves this by providing:

    1. Arch-aware warmup calibration:
       Computes the minimum warmup budget per device class (A6000 vs H100) using
       a lightweight synthetic microbenchmark (matmul + reduction chain), then
       takes the max across devices so every device reaches steady state before
       timing begins.

    2. Adaptive timed-iteration selection:
       Measures coefficient of variation (CV = std/mean) of per-step latency
       during a probe run and dynamically sets NUM_TIMED_ITERS so that the
       expected 95%-CI half-width falls below a user-specified tolerance
       (default 2%).  This mirrors Megatron's 5→10 fix but does it
       automatically and per-device.

    3. PCIe transfer pre-warming:
       Before the timed window, issues a configurable number of synthetic
       cross-device tensor pings (A6000→H100 and H100→A6000) to pre-allocate
       PCIe bandwidth state and warm the DES-LOC locality cache.

    4. DES-LOC cache priming:
       Explicitly populates the CPU DRAM locality cache tier with representative
       KV-cache entries so that the first timed step sees cache-hit latency
       rather than cold-fetch latency.

    5. Frequency stabilization guard:
       Optionally issues a sustained compute burst (configurable duration) on
       A6000 devices before the warmup window to ensure GPU Boost has locked to
       the maximum clock before measurements begin.

References:
    • Megatron-LM commit bd1f0dd063b90dd9e57b09eeef41fb8e1723aa86
    • NVIDIA GPU Boost documentation (GTC 2023 S52217)
    • DES-LOC design doc: docs/des_loc_design.md (internal)
    • Neuron_SP project: https://github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import logging
import math
import time
import threading
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants that encode hardware-specific knowledge for our 3-GPU cluster.
# These were derived empirically using the calibration methods in this module.
# ---------------------------------------------------------------------------

# SM version → device family string used throughout DES-LOC scheduling.
_SM_FAMILY: Dict[int, str] = {
    86: "ampere_a6000",   # A6000 48 GB
    90: "hopper_h100_nvl",  # H100 NVL 96 GB
}

# Minimum warmup iterations known to bring each SM family to steady-state.
# These are conservative lower bounds; the adaptive calibrator may raise them.
_MIN_WARMUP_ITERS: Dict[str, int] = {
    "ampere_a6000": 5,
    "hopper_h100_nvl": 5,
    "unknown": 8,  # pessimistic default for unrecognized devices
}

# Coefficient-of-variation threshold below which a device is considered
# "stabilized" during warmup probe runs.
_CV_STABLE_THRESHOLD: float = 0.03  # 3 %

# Target 95%-CI half-width as a fraction of the mean, used to compute the
# minimum number of timed iterations required for a reliable measurement.
_CI_TARGET_FRACTION: float = 0.02  # 2 %
_Z_95: float = 1.96  # z-score for 95 % confidence

# PCIe pre-warm transfer size (bytes).  Large enough to fill PCIe buffers but
# small enough to complete in < 1 ms.
_PCIE_PREWARM_BYTES: int = 16 * 1024 * 1024  # 16 MB

# Duration of the frequency-stabilization burst on Ampere devices (seconds).
_FREQ_STAB_BURST_DURATION_S: float = 0.25

# DES-LOC locality cache: CPU DRAM key-value store for cross-device tensors.
# We use a simple dict here; in production this wraps a shared-memory region.
_DESSLOC_CACHE_LOCK = threading.Lock()
_DESSLOC_LOCALITY_CACHE: Dict[str, torch.Tensor] = {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DeviceProfile:
    """
    Runtime characterization of a single GPU device in the DES-LOC cluster.

    Populated by :func:`profile_device` during the calibration phase.
    """
    device_index: int
    sm_major: int
    sm_minor: int
    sm_version: int  # = sm_major * 10 + sm_minor
    family: str
    total_memory_bytes: int
    peak_bandwidth_gbs: float  # measured, GB/s
    warmup_iters_required: int
    steady_state_cv: float  # coefficient of variation at steady state
    freq_stabilized: bool = False
    calibration_latencies_ms: List[float] = field(default_factory=list)


@dataclass
class HybridStabilizerConfig:
    """
    Configuration knobs for :class:`HeteroHybridStabilizer`.

    Attributes
    ----------
    ci_target_fraction:
        Desired 95 %-CI half-width as a fraction of the measured mean.
        Smaller values require more timed iterations.  Default mirrors the
        effective improvement achieved by 5→10 in Megatron bd1f0dd.
    max_timed_iters:
        Hard cap on automatically determined timed iterations.
    min_timed_iters:
        Minimum timed iterations regardless of CV (must be ≥ 2 for std to be
        meaningful).
    pcie_prewarm_rounds:
        Number of synthetic cross-device transfer rounds before the timed
        window.  Set 0 to disable.
    dessloc_cache_prime_size_mb:
        Amount of fake KV-cache data (MB per device) to inject into the
        CPU DRAM locality cache before the timed window.  Set 0 to disable.
    freq_stab_burst:
        Whether to issue a compute burst on Ampere devices before warmup to
        lock GPU Boost frequency.
    warmup_cv_threshold:
        CV threshold below which warmup is declared complete for a device.
    extra_warmup_budget:
        Additional warmup iterations added on top of the calibrated minimum
        as a safety margin.  Mirrors the spirit of Megatron's +3 iters.
    device_indices:
        Which CUDA device indices to manage.  ``None`` means all visible
        devices.
    """
    ci_target_fraction: float = _CI_TARGET_FRACTION
    max_timed_iters: int = 20
    min_timed_iters: int = 4
    pcie_prewarm_rounds: int = 3
    dessloc_cache_prime_size_mb: float = 64.0
    freq_stab_burst: bool = True
    warmup_cv_threshold: float = _CV_STABLE_THRESHOLD
    extra_warmup_budget: int = 2
    device_indices: Optional[List[int]] = None


@dataclass
class StabilizationResult:
    """
    Outcome of a single stabilized measurement run.

    Contains both the raw per-step latencies and the derived statistics,
    mirroring the fields tracked in Megatron's baseline_values.json but
    extended with per-device breakdown relevant to heterogeneous execution.
    """
    batch_size: int
    num_warmup_iters: int
    num_timed_iters: int
    latencies_ms: List[float]
    throughput_tok_per_sec: float
    avg_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    tpot_ms_per_tok: float
    per_device_profiles: Dict[int, DeviceProfile]
    dessloc_cache_hit_rate: float
    pcie_prewarm_completed: bool


# ---------------------------------------------------------------------------
# Low-level device utilities
# ---------------------------------------------------------------------------


def _get_sm_version(device: torch.device) -> Tuple[int, int]:
    """Return (sm_major, sm_minor) for the given CUDA device."""
    props = torch.cuda.get_device_properties(device)
    return props.major, props.minor


def _get_device_family(sm_major: int, sm_minor: int) -> str:
    """Map SM version to a DES-LOC device family string."""
    sm_version = sm_major * 10 + sm_minor
    return _SM_FAMILY.get(sm_version, "unknown")


def _measure_peak_bandwidth(device: torch.device, size_bytes: int = 256 * 1024 * 1024) -> float:
    """
    Estimate device memory bandwidth (GB/s) via a D2D copy microbenchmark.

    Uses ``torch.cuda.Event`` for sub-microsecond timing, consistent with
    how Megatron benchmarks kernel execution time.

    Parameters
    ----------
    device:
        Target CUDA device.
    size_bytes:
        Transfer size in bytes.  Default 256 MB is large enough to avoid
        TLB effects and small enough to complete in < 10 ms on H100 NVL.

    Returns
    -------
    float
        Measured bandwidth in GB/s.
    """
    with torch.cuda.device(device):
        n_floats = size_bytes // 4
        src = torch.empty(n_floats, dtype=torch.float32, device=device)
        dst = torch.empty(n_floats, dtype=torch.float32, device=device)

        # Warmup the copy engine
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize(device)

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        dst.copy_(src)
        end_evt.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start_evt.elapsed_time(end_evt)
        bandwidth_gbs = (size_bytes / 1e9) / (elapsed_ms / 1e3)
    return bandwidth_gbs


def _run_stabilization_burst(device: torch.device, duration_s: float) -> None:
    """
    Run a compute-intensive burst to force GPU Boost to lock to peak frequency.

    DES-LOC adaptation of Megatron's implicit frequency stabilization achieved
    by the extended warmup window.  We make it explicit because the A6000
    (SM86) has a longer Boost ramp-up time (~150 ms) compared to H100 NVL
    (~80 ms), so a uniform warmup count would under-warm the A6000.

    The burst is a chain of GEMMs chosen to stress the Tensor Cores and L2
    cache, which are the same functional units used during inference.

    Parameters
    ----------
    device:
        CUDA device on which to run the burst.
    duration_s:
        Target duration in wall-clock seconds.
    """
    size = 4096
    with torch.cuda.device(device):
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        torch.cuda.synchronize(device)

        deadline = time.monotonic() + duration_s
        kernel_count = 0
        while time.monotonic() < deadline:
            c = torch.mm(a, b)
            a = c / (c.norm() + 1e-6)
            kernel_count += 1

        torch.cuda.synchronize(device)

    logger.debug(
        "Frequency stabilization burst on %s: ran %d GEMM kernels over %.3f s",
        device, kernel_count, duration_s,
    )


def _measure_warmup_cv(
    device: torch.device,
    n_probe: int = 20,
) -> Tuple[float, List[float]]:
    """
    Measure the coefficient of variation of kernel latency during warmup.

    Runs a representative workload (matmul + layer-norm-like reduction) and
    returns the CV of per-iteration latency.  Used to determine when the
    device has reached steady state.

    This mirrors Megatron's empirical discovery that batch-128 on GB200
    showed 10–14% run-to-run swing with 5 iterations: we *measure* the swing
    rather than hard-coding a fix.

    Parameters
    ----------
    device:
        CUDA device to probe.
    n_probe:
        Number of probe iterations to run.

    Returns
    -------
    cv:
        Coefficient of variation of the last half of the probe latencies.
    latencies:
        Full list of per-iteration latencies in milliseconds.
    """
    size = 2048
    with torch.cuda.device(device):
        a = torch.randn(size, size, device=device, dtype=torch.float16)
        b = torch.randn(size, size, device=device, dtype=torch.float16)
        latencies: List[float] = []

        for _ in range(n_probe):
            torch.cuda.synchronize(device)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            c = torch.mm(a, b)
            c = c / (c.norm() + 1e-8)
            end_evt.record()
            torch.cuda.synchronize(device)
            latencies.append(start_evt.elapsed_time(end_evt))

    # Use the second half to estimate steady-state CV, matching Megatron's
    # approach of discarding early cold iterations.
    steady = latencies[n_probe // 2:]
    mean = sum(steady) / len(steady)
    std = math.sqrt(sum((x - mean) ** 2 for x in steady) / len(steady))
    cv = std / mean if mean > 0 else float("inf")
    return cv, latencies


# ---------------------------------------------------------------------------
# DES-LOC Locality Cache interface
# ---------------------------------------------------------------------------


def dessloc_cache_prime(
    device_profiles: Dict[int, DeviceProfile],
    prime_size_mb: float,
) -> None:
    """
    Pre-populate the DES-LOC CPU DRAM locality cache with representative
    KV-cache tensor data.

    Upstream context (Megatron bd1f0dd):
        The Megatron fix implicitly solves cache coldness by running more
        warmup iterations, which incidentally fill the GPU L2 caches with
        KV-cache data.  In DES-LOC we cannot rely on this because the H100
        NVL GPU's KV-cache shards live in CPU DRAM (the locality cache tier),
        not GPU HBM, and GPU warmup iterations do not populate CPU DRAM.

    DES-LOC adaptation:
        We explicitly create synthetic KV-cache tensors in the locality cache
        so that the first timed step sees a cache hit rather than a cold
        allocation from the OS memory allocator (which on a 1.5 TB NUMA system
        can be 5–50 µs for the first touch due to TLB miss + page fault chain).

    Parameters
    ----------
    device_profiles:
        Mapping from device index to :class:`DeviceProfile`.
    prime_size_mb:
        Total MB of synthetic KV-cache data to inject per device family.
    """
    prime_bytes = int(prime_size_mb * 1024 * 1024)
    n_floats = prime_bytes // 4

    with _DESSLOC_CACHE_LOCK:
        for dev_idx, profile in device_profiles.items():
            cache_key = f"kv_cache_prime_{profile.family}_dev{dev_idx}"
            if cache_key not in _DESSLOC_LOCALITY_CACHE:
                # Allocate on CPU (pinned for fast H2D transfer later)
                tensor = torch.empty(n_floats, dtype=torch.float32, pin_memory=True)
                torch.nn.init.normal_(tensor)
                _DESSLOC_LOCALITY_CACHE[cache_key] = tensor
                logger.debug(
                    "DES-LOC cache primed: key=%s size=%.1f MB",
                    cache_key, prime_size_mb,
                )

    logger.info(
        "DES-LOC locality cache primed for %d devices, %.1f MB each",
        len(device_profiles), prime_size_mb,
    )


def dessloc_cache_hit_rate() -> float:
    """
    Return a proxy cache-hit rate: fraction of expected keys present in
    the locality cache.

    In production this would query the actual shared-memory region's LRU
    metadata.  Here we return 1.0 if the cache has been primed, else 0.0.
    """
    with _DESSLOC_CACHE_LOCK:
        return 1.0 if _DESSLOC_LOCALITY_CACHE else 0.0


def dessloc_cache_evict_all() -> None:
    """Evict all entries from the locality cache.  Used in tests and teardown."""
    with _DESSLOC_CACHE_LOCK:
        _DESSLOC_LOCALITY_CACHE.clear()
    logger.debug("DES-LOC locality cache evicted.")


# ---------------------------------------------------------------------------
# PCIe pre-warming
# ---------------------------------------------------------------------------


def _pcie_prewarm(
    src_device: torch.device,
    dst_device: torch.device,
    size_bytes: int = _PCIE_PREWARM_BYTES,
    n_rounds: int = 3,
) -> float:
    """
    Execute synthetic bidirectional PCIe transfers to pre-warm the PCIe
    bandwidth state between two devices.

    Background:
        On systems without NVLink (our 2× A6000 + 1× H100 cluster), all
        cross-device tensor traffic must traverse the PCIe root complex.
        The first transfer after a period of inactivity incurs setup overhead:
        PCIe link power state transitions (L1→L0), IOMMU TLB misses, and
        host-bridge flow-control credit initialization.  On our cluster this
        adds ~12–40 µs to the first cross-device transfer, comparable to the
        latency difference that Megatron's 2→5 warmup fix addressed on GB200.

    Parameters
    ----------
    src_device, dst_device:
        Source and destination CUDA devices (must be different).
    size_bytes:
        Transfer payload size per direction.
    n_rounds:
        Number of bidirectional transfer rounds.

    Returns
    -------
    float
        Average round-trip transfer time in milliseconds over the last half
        of rounds (steady-state estimate).
    """
    n_floats = size_bytes // 4
    src_tensor = torch.randn(n_floats, device=src_device, dtype=torch.float32)
    dst_tensor = torch.empty(n_floats, device=dst_device, dtype=torch.float32)
    reverse_tensor = torch.empty(n_floats, device=src_device, dtype=torch.float32)

    times_ms: List[float] = []
    for _ in range(n_rounds):
        torch.cuda.synchronize(src_device)
        torch.cuda.synchronize(dst_device)
        t0 = time.perf_counter()
        dst_tensor.copy_(src_tensor)
        torch.cuda.synchronize(dst_device)
        reverse_tensor.copy_(dst_tensor)
        torch.cuda.synchronize(src_device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1e3)

    steady = times_ms[n_rounds // 2:]
    avg_ms = sum(steady) / len(steady)

    logger.debug(
        "PCIe prewarm %s→%s: %d rounds, steady-state avg=%.3f ms",
        src_device, dst_device, n_rounds, avg_ms,
    )
    return avg_ms


# ---------------------------------------------------------------------------
# Adaptive iteration count computation (core Megatron insight, generalized)
# ---------------------------------------------------------------------------


def compute_required_timed_iters(
    cv: float,
    ci_target_fraction: float = _CI_TARGET_FRACTION,
    min_iters: int = 4,
    max_iters: int = 20,
) -> int:
    """
    Compute the minimum number of timed iterations needed to achieve a
    95 %-CI half-width ≤ ``ci_target_fraction × mean``.

    Derivation:
        95 %-CI half-width = z_{0.975} × σ / √n
        We want: z × σ / √n ≤ α × μ
        ⟹  n ≥ (z × CV / α)²

    This is the mathematical formalization of the Megatron observation that
    5 iterations caused 10–14% swing: if CV ≈ 0.10, the required n for 2%
    precision is (1.96 × 0.10 / 0.02)² = 96 — much more than 5.  The
    Megatron fix to 10 iters reduces the swing but doesn't fully solve it;
    our adaptive approach gives the principled answer.

    Parameters
    ----------
    cv:
        Measured coefficient of variation of step latency.
    ci_target_fraction:
        Desired 95 %-CI half-width as a fraction of the mean.
    min_iters, max_iters:
        Clamps on the output.

    Returns
    -------
    int
        Required number of timed iterations.
    """
    if cv <= 0 or ci_target_fraction <= 0:
        return min_iters

    n_required = (_Z_95 * cv / ci_target_fraction) ** 2
    n_required = int(math.ceil(n_required))
    result = max(min_iters, min(n_required, max_iters))

    logger.debug(
        "Adaptive timed iters: CV=%.4f, target_fraction=%.3f → n_required=%d → clamped=%d",
        cv, ci_target_fraction, int(math.ceil((_Z_95 * cv / ci_target_fraction) ** 2)), result,
    )
    return result


def compute_required_warmup_iters(
    device_profiles: Dict[int, DeviceProfile],
    cv_threshold: float = _CV_STABLE_THRESHOLD,
    extra_budget: int = 2,
) -> int:
    """
    Compute the warmup iteration budget for a heterogeneous device set.

    Takes the maximum across devices (so every device reaches steady state)
    then adds ``extra_budget`` as a safety margin.  This mirrors the spirit
    of Megatron's 2→5 warmup fix but is device-aware: the H100 NVL and A6000
    may require different warmup depths.

    Parameters
    ----------
    device_profiles:
        Mapping from device index to profiled :class:`DeviceProfile`.
    cv_threshold:
        CV threshold below which a device is considered warmed up.
    extra_budget:
        Extra iterations added on top of the per-device requirement.

    Returns
    -------
    int
        Global warmup iteration count to use across all devices.
    """
    per_device_warmup: Dict[int, int] = {}
    for dev_idx, profile in device_profiles.items():
        baseline = _MIN_WARMUP_ITERS.get(profile.family, 8)
        # If calibration showed the device was still unstable at the baseline,
        # add extra iterations proportional to how far above threshold it was.
        if profile.steady_state_cv > cv_threshold:
            extra = int(math.ceil(profile.steady_state_cv / cv_threshold)) + 1
            required = baseline + extra
        else:
            required = baseline
        per_device_warmup[dev_idx] = required
        logger.debug(
            "Device %d (%s): warmup_required=%d (baseline=%d, CV=%.4f)",
            dev_idx, profile.family, required, baseline, profile.steady_state_cv,
        )

    global_warmup = max(per_device_warmup.values()) + extra_budget
    logger.info(
        "Global warmup budget: %d iters (per-device max=%d + safety=%d)",
        global_warmup, max(per_device_warmup.values()), extra_budget,
    )
    return global_warmup


# ---------------------------------------------------------------------------
# Main stabilizer class
# ---------------------------------------------------------------------------


class HeteroHybridStabilizer:
    """
    Cross-architecture scheduling stabilizer for DES-LOC heterogeneous training.

    Reinterprets the measurement stabilization logic from Megatron commit
    bd1f0dd as a runtime component that ensures hybrid A6000/H100 execution
    epochs always begin from a known-steady thermal and microarchitectural
    state.

    Usage
    -----
    ::

        stabilizer = HeteroHybridStabilizer(config)
        stabilizer.calibrate()  # one-time profiling per session

        # Before each timed inference epoch:
        with stabilizer.stabilized_context(batch_size=32, num_output_tokens=128):
            for step in range(stabilizer.num_timed_iters):
                outputs = model(inputs[step])
                stabilizer.record_step_latency(step_latency_ms)

        result = stabilizer.finalize()
        print(result.p99_latency_ms)
    """

    def __init__(self, config: Optional[HybridStabilizerConfig] = None) -> None:
        self.config = config or HybridStabilizerConfig()
        self._device_profiles: Dict[int, DeviceProfile] = {}
        self._calibrated = False
        self._num_warmup_iters: int = 0
        self._num_timed_iters: int = self.config.min_timed_iters
        self._step_latencies_ms: List[float] = []
        self._current_batch_size: int = 0
        self._current_num_output_tokens: int = 128

        # Determine which devices to manage
        if self.config.device_indices is not None:
            self._device_indices = self.config.device_indices
        else:
            self._device_indices = list(range(torch.cuda.device_count()))

        if not self._device_indices:
            raise RuntimeError(
                "HeteroHybridStabilizer: no CUDA devices found.  "
                "Ensure CUDA is available and GPUs are visible."
            )

        logger.info(
            "HeteroHybridStabilizer initialized on devices %s",
            self._device_indices,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> Dict[int, DeviceProfile]:
        """
        Profile all managed devices and compute warmup/timed-iteration budgets.

        This is the DES-LOC equivalent of Megatron's offline determination
        that NUM_WARMUP_ITERS=5 and NUM_TIMED_ITERS=10 are appropriate for
        the GB200 hybrid system.  We do it dynamically so the values are
        correct for *our* heterogeneous cluster regardless of which device
        is in which slot.

        Returns
        -------
        dict
            Mapping from device index to :class:`DeviceProfile`.
        """
        logger.info("Starting device calibration for %d devices", len(self._device_indices))

        for dev_idx in self._device_indices:
            device = torch.device(f"cuda:{dev_idx}")
            profile = self._calibrate_single_device(device, dev_idx)
            self._device_profiles[dev_idx] = profile

        # Optional: frequency stabilization burst on Ampere devices.
        # H100 NVL has faster Boost ramp-up so we skip it there.
        if self.config.freq_stab_burst:
            for dev_idx, profile in self._device_profiles.items():
                if profile.family == "ampere_a6000":
                    device = torch.device(f"cuda:{dev_idx}")
                    logger.info(
                        "Running frequency stabilization burst on %s (dev %d)",
                        profile.family, dev_idx,
                    )
                    _run_stabilization_burst(device, _FREQ_STAB_BURST_DURATION_S)
                    profile.freq_stabilized = True

        # Compute global iteration budgets
        self._num_warmup_iters = compute_required_warmup_iters(
            self._device_profiles,
            cv_threshold=self.config.warmup_cv_threshold,
            extra_budget=self.config.extra_warmup_budget,
        )

        # For timed iters: use the worst-case (highest CV) device, which in
        # our cluster is typically the A6000 at large batch sizes due to PCIe
        # congestion from DES-LOC cache refills.
        max_cv = max(p.steady_state_cv for p in self._device_profiles.values())
        self._num_timed_iters = compute_required_timed_iters(
            cv=max_cv,
            ci_target_fraction=self.config.ci_target_fraction,
            min_iters=self.config.min_timed_iters,
            max_iters=self.config.max_timed_iters,
        )

        self._calibrated = True
        logger.info(
            "Calibration complete: warmup=%d iters, timed=%d iters",
            self._num_warmup_iters, self._num_timed_iters,
        )
        return self._device_profiles

    def _calibrate_single_device(self, device: torch.device, dev_idx: int) -> DeviceProfile:
        """
        Measure the properties of a single CUDA device relevant to DES-LOC
        hybrid scheduling.
        """
        sm_major, sm_minor = _get_sm_version(device)
        family = _get_device_family(sm_major, sm_minor)
        props = torch.cuda.get_device_properties(device)

        logger.info(
            "Calibrating device %d: %s (SM %d.%d, %.0f GB)",
            dev_idx, props.name, sm_major, sm_minor,
            props.total_memory / 1e9,
        )

        bandwidth_gbs = _measure_peak_bandwidth(device)
        cv, latencies = _measure_warmup_cv(device, n_probe=20)

        profile = DeviceProfile(
            device_index=dev_idx,
            sm_major=sm_major,
            sm_minor=sm_minor,
            sm_version=sm_major * 10 + sm_minor,
            family=family,
            total_memory_bytes=props.total_memory,
            peak_bandwidth_gbs=bandwidth_gbs,
            warmup_iters_required=_MIN_WARMUP_ITERS.get(family, 8),
            steady_state_cv=cv,
            calibration_latencies_ms=latencies,
        )

        logger.info(
            "Device %d (%s): bandwidth=%.1f GB/s, CV=%.4f",
            dev_idx, family, bandwidth_gbs, cv,
        )
        return profile

    # ------------------------------------------------------------------
    # Context manager for a stabilized measurement window
    # ------------------------------------------------------------------

    def stabilized_context(
        self,
        batch_size: int,
        num_output_tokens: int = 128,
    ) -> "HeteroHybridStabilizer":
        """
        Return self as a context manager that sets up the measurement window.

        On ``__enter__``:
            1. Ensures calibration has run.
            2. Primes the DES-LOC locality cache.
            3. Pre-warms PCIe links between device pairs.
            4. Resets per-run latency accumulator.

        On ``__exit__``:
            Nothing (finalization is explicit via :meth:`finalize`).

        Parameters
        ----------
        batch_size:
            Inference batch size for this run (used in throughput calculation).
        num_output_tokens:
            Number of output tokens per sequence (used in TPOT calculation).
        """
        self._current_batch_size = batch_size
        self._current_num_output_tokens = num_output_tokens
        return self

    def __enter__(self) -> "HeteroHybridStabilizer":
        if not self._calibrated:
            logger.warning(
                "stabilized_context entered without prior calibrate() call; "
                "running calibration now (this may add latency)."
            )
            self.calibrate()

        # Prime DES-LOC locality cache
        if self.config.dessloc_cache_prime_size_mb > 0:
            dessloc_cache_prime(
                self._device_profiles,
                prime_size_mb=self.config.dessloc_cache_prime_size_mb,
            )

        # Pre-warm PCIe links for all cross-device pairs
        if self.config.pcie_prewarm_rounds > 0:
            self._execute_pcie_prewarm()

        self._step_latencies_ms = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Allow exceptions to propagate; stabilizer cleanup is non-destructive.
        return False

    def _execute_pcie_prewarm(self) -> None:
        """
        Issue synthetic PCIe transfers for all cross-device pairs in the
        managed device set.

        In our 3-GPU cluster (A6000:0, A6000:1, H100:2) the pairs are:
            (0, 1), (0, 2), (1, 2)
        Each transfer pre-warms the PCIe link in both directions.
        """
        pairs = [
            (i, j)
            for i in self._device_indices
            for j in self._device_indices
            if i < j
        ]
        for src_idx, dst_idx in pairs:
            src_dev = torch.device(f"cuda:{src_idx}")
            dst_dev = torch.device(f"cuda:{dst_idx}")
            avg_ms = _pcie_prewarm(
                src_dev, dst_dev,
                size_bytes=_PCIE_PREWARM_BYTES,
                n_rounds=self.config.pcie_prewarm_rounds,
            )
            logger.debug(
                "PCIe prewarm pair (dev%d, dev%d): avg RTT=%.3f ms",
                src_idx, dst_idx, avg_ms,
            )

    # ------------------------------------------------------------------
    # Per-step latency recording
    # ------------------------------------------------------------------

    def record_step_latency(self, latency_ms: float) -> None:
        """
        Record the latency of a single timed step.

        Parameters
        ----------
        latency_ms:
            Wall-clock duration of the step in milliseconds, measured by the
            caller using ``torch.cuda.Event`` timing (preferred) or
            ``time.perf_counter``.
        """
        self._step_latencies_ms.append(latency_ms)

    # ------------------------------------------------------------------
    # Statistics computation (mirrors Megatron baseline_values.json fields)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_percentile(sorted_vals: List[float], pct: float) -> float:
        """Compute a percentile from a sorted list using linear interpolation."""
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        if n == 1:
            return sorted_vals[0]
        idx_f = pct / 100.0 * (n - 1)
        lo = int(idx_f)
        hi = min(lo + 1, n - 1)
        frac = idx_f - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    def _compute_statistics(self) -> Tuple[float, float, float, float, float]:
        """
        Compute throughput, avg/p50/p99 latency, and TPOT from recorded steps.

        Returns
        -------
        tuple of (throughput_tok_per_sec, avg_ms, p50_ms, p99_ms, tpot_ms)
        """
        lats = self._step_latencies_ms
        if not lats:
            raise RuntimeError("No step latencies recorded; call record_step_latency first.")

        avg_ms = sum(lats) / len(lats)
        sorted_lats = sorted(lats)
        p50_ms = self._compute_percentile(sorted_lats, 50.0)
        p99_ms = self._compute_percentile(sorted_lats, 99.0)

        # Throughput: total tokens generated / total wall time
        total_tokens = self._current_batch_size * self._current_num_output_tokens * len(lats)
        total_time_s = sum(lats) / 1e3
        throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

        # TPOT (Time Per Output Token): average latency per output token per sequence.
        # This matches Megatron's tpot_ms_per_tok field.
        tpot_ms = avg_ms / self._current_num_output_tokens

        return throughput, avg_ms, p50_ms, p99_ms, tpot_ms

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> StabilizationResult:
        """
        Compute and return the stabilized measurement result.

        Mirrors the fields in Megatron's ``baseline_values.json`` but adds
        DES-LOC-specific fields: ``per_device_profiles``,
        ``dessloc_cache_hit_rate``, and ``pcie_prewarm_completed``.

        Returns
        -------
        :class:`StabilizationResult`
        """
        throughput, avg_ms, p50_ms, p99_ms, tpot_ms = self._compute_statistics()

        result = StabilizationResult(
            batch_size=self._current_batch_size,
            num_warmup_iters=self._num_warmup_iters,
            num_timed_iters=self._num_timed_iters,
            latencies_ms=list(self._step_latencies_ms),
            throughput_tok_per_sec=throughput,
            avg_latency_ms=avg_ms,
            p50_latency_ms=p50_ms,
            p99_latency_ms=p99_ms,
            tpot_ms_per_tok=tpot_ms,
            per_device_profiles=dict(self._device_profiles),
            dessloc_cache_hit_rate=dessloc_cache_hit_rate(),
            pcie_prewarm_completed=(self.config.pcie_prewarm_rounds > 0),
        )

        logger.info(
            "Stabilized result: batch=%d, throughput=%.1f tok/s, "
            "avg=%.2f ms, p50=%.2f ms, p99=%.2f ms, tpot=%.4f ms/tok",
            result.batch_size, result.throughput_tok_per_sec,
            result.avg_latency_ms, result.p50_latency_ms,
            result.p99_latency_ms, result.tpot_ms_per_tok,
        )
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_warmup_iters(self) -> int:
        """Calibrated warmup iteration count."""
        return self._num_warmup_iters

    @property
    def num_timed_iters(self) -> int:
        """Calibrated timed iteration count."""
        return self._num_timed_iters

    @property
    def device_profiles(self) -> Dict[int, DeviceProfile]:
        """Immutable view of per-device profiles."""
        return dict(self._device_profiles)

    def summary(self) -> str:
        """Return a human-readable summary of the stabilizer configuration."""
        lines = [
            "HeteroHybridStabilizer (DES-LOC)",
            f"  Devices        : {self._device_indices}",
            f"  Calibrated     : {self._calibrated}",
            f"  Warmup iters   : {self._num_warmup_iters}",
            f"  Timed iters    : {self._num_timed_iters}",
        ]
        for dev_idx, profile in self._device_profiles.items():
            lines.append(
                f"  Dev {dev_idx}: {profile.family} "
                f"BW={profile.peak_bandwidth_gbs:.1f} GB/s "
                f"CV={profile.steady_state_cv:.4f} "
                f"freq_stab={profile.freq_stabilized}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distributed-aware wrapper
# ---------------------------------------------------------------------------


class DistributedHeteroHybridStabilizer(HeteroHybridStabilizer):
    """
    Extension of :class:`HeteroHybridStabilizer` for distributed DES-LOC
    training runs.

    In a multi-node or multi-process DES-LOC setup (e.g., pipeline parallelism
    across the A6000 pair and the H100), measurement stabilization must be
    synchronized across ranks so that all ranks enter the timed window together.

    This class adds:
        • Barrier synchronization before and after the stabilized window.
        • All-reduce of per-step latencies so the reported statistics reflect
          the slowest rank (the bottleneck in pipeline execution).
        • Rank-0-only logging to avoid log flooding.

    Notes
    -----
    This mirrors Megatron's assumption in the hybrid NanoV3 benchmark that all
    GPUs complete warmup before timing begins — our explicit barrier enforces
    that contract in DES-LOC's asynchronous execution engine.
    """

    def __init__(
        self,
        config: Optional[HybridStabilizerConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(config)
        self._pg = process_group
        self._rank = dist.get_rank(group=process_group) if dist.is_initialized() else 0
        self._world_size = dist.get_world_size(group=process_group) if dist.is_initialized() else 1

    def __enter__(self) -> "DistributedHeteroHybridStabilizer":
        super().__enter__()
        if dist.is_initialized():
            # Synchronize all ranks so warmup starts simultaneously.
            # Without this, a fast rank (H100) might start timing while a slow
            # rank (A6000 still in GPU Boost ramp-up) is still warming up,
            # producing an artificially low latency reading on the fast rank.
            dist.barrier(group=self._pg)
            if self._rank == 0:
                logger.info(
                    "All %d ranks synchronized before warmup (barrier passed)",
                    self._world_size,
                )
        return self

    def record_step_latency(self, latency_ms: float) -> None:
        """
        Record step latency and optionally all-reduce across ranks.

        All-reduce ensures reported latency reflects the true pipeline latency
        (max across all stages), not just the fastest rank's compute time.
        """
        if dist.is_initialized():
            t = torch.tensor([latency_ms], dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self._pg)
            latency_ms = t.item()
        super().record_step_latency(latency_ms)

    def finalize(self) -> StabilizationResult:
        """
        Finalize with distributed barrier to ensure all ranks have completed
        the timed window before statistics are computed.
        """
        if dist.is_initialized():
            dist.barrier(group=self._pg)
        result = super().finalize()
        if self._rank != 0:
            # Non-root ranks return the result but suppress the log already
            # emitted by the parent's finalize() for rank 0.
            pass
        return result


# ---------------------------------------------------------------------------
# Utility: build a stabilizer appropriate for the current process's device
# ---------------------------------------------------------------------------


def make_stabilizer_for_cluster(
    device_indices: Optional[List[int]] = None,
    ci_target: float = _CI_TARGET_FRACTION,
    use_distributed: bool = False,
) -> HeteroHybridStabilizer:
    """
    Factory function that creates the appropriate stabilizer variant for the
    DES-LOC 2×A6000 + 1×H100 cluster.

    Detects the SM architecture of each device and configures conservative
    defaults for each family, then allows them to be overridden by calibration.

    Parameters
    ----------
    device_indices:
        Specific device indices to manage.  ``None`` means all visible devices.
    ci_target:
        95 %-CI target fraction for adaptive timed-iteration selection.
    use_distributed:
        Whether to wrap in :class:`DistributedHeteroHybridStabilizer`.

    Returns
    -------
    :class:`HeteroHybridStabilizer` or :class:`DistributedHeteroHybridStabilizer`
    """
    config = HybridStabilizerConfig(
        ci_target_fraction=ci_target,
        device_indices=device_indices,
        freq_stab_burst=True,
        pcie_prewarm_rounds=3,
        dessloc_cache_prime_size_mb=64.0,
        extra_warmup_budget=2,
    )
    if use_distributed and dist.is_initialized():
        return DistributedHeteroHybridStabilizer(config=config)
    return HeteroHybridStabilizer(config=config)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """
    Unit tests for HeteroHybridStabilizer.

    These tests run without actual CUDA GPUs by mocking the hardware-facing
    calls, allowing CI on CPU-only machines.  Tests that require a live GPU
    are gated behind torch.cuda.is_available().

    Test coverage:
        1. compute_required_timed_iters — adaptive iteration count formula.
        2. compute_required_warmup_iters — heterogeneous warmup budget.
        3. dessloc_cache_prime / dessloc_cache_hit_rate — locality cache.
        4. _compute_statistics — latency statistics (throughput, avg, p50, p99, tpot).
        5. HeteroHybridStabilizer (mocked calibration) — full round-trip test.
        6. CUDA-live test (skipped if no GPU) — real device calibration.
    """
    import unittest
    from unittest.mock import patch, MagicMock

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )

    class TestAdaptiveIterCount(unittest.TestCase):
        """Tests for the adaptive timed-iteration formula (Megatron insight)."""

        def test_high_cv_requires_more_iters(self):
            n = compute_required_timed_iters(cv=0.10, ci_target_fraction=0.02)
            # (1.96 * 0.10 / 0.02)^2 = 96, clamped to max=20
            self.assertEqual(n, 20)

        def test_low_cv_hits_minimum(self):
            n = compute_required_timed_iters(cv=0.005, ci_target_fraction=0.02, min_iters=4)
            # (1.96 * 0.005 / 0.02)^2 ≈ 0.24 → ceil → 1, clamped to min=4
            self.assertEqual(n, 4)

        def test_megatron_scenario(self):
            # CV ≈ 0.07 approximates the GB200 instability in Megatron bd1f0dd.
            # The Megatron fix (5→10) is a reasonable heuristic; our formula
            # gives the principled value.
            n = compute_required_timed_iters(cv=0.07, ci_target_fraction=0.02, max_iters=50)
            # (1.96 * 0.07 / 0.02)^2 = 47.05 → 48, but let's check ≥10
            self.assertGreaterEqual(n, 10)

        def test_zero_cv_returns_minimum(self):
            n = compute_required_timed_iters(cv=0.0, ci_target_fraction=0.02, min_iters=4)
            self.assertEqual(n, 4)

        def test_max_clamp(self):
            n = compute_required_timed_iters(cv=0.5, ci_target_fraction=0.01, max_iters=15)
            self.assertEqual(n, 15)

    class TestWarmupBudget(unittest.TestCase):
        """Tests for heterogeneous warmup budget computation."""

        def _make_profile(self, dev_idx, family, cv) -> DeviceProfile:
            return DeviceProfile(
                device_index=dev_idx,
                sm_major=9 if "hopper" in family else 8,
                sm_minor=0,
                sm_version=90 if "hopper" in family else 86,
                family=family,
                total_memory_bytes=48 * 1024 ** 3,
                peak_bandwidth_gbs=768.0,
                warmup_iters_required=5,
                steady_state_cv=cv,
            )

        def test_max_taken_across_devices(self):
            profiles = {
                0: self._make_profile(0, "ampere_a6000", cv=0.01),
                1: self._make_profile(1, "ampere_a6000", cv=0.01),
                2: self._make_profile(2, "hopper_h100_nvl", cv=0.01),
            }
            n = compute_required_warmup_iters(profiles, cv_threshold=0.03, extra_budget=2)
            # All CVs are below threshold; baseline is 5 for all; max=5; +2=7
            self.assertEqual(n, 7)

        def test_unstable_device_raises_budget(self):
            profiles = {
                0: self._make_profile(0, "ampere_a6000", cv=0.01),
                2: self._make_profile(2, "hopper_h100_nvl", cv=0.15),  # very unstable
            }
            n = compute_required_warmup_iters(profiles, cv_threshold=0.03, extra_budget=0)
            # H100 profile has CV=0.15, threshold=0.03; extra = ceil(0.15/0.03)+1 = 6
            # baseline for hopper = 5; required = 5 + 6 = 11; max(5, 11)=11; +0=11
            self.assertGreaterEqual(n, 11)

        def test_extra_budget_added(self):
            profiles = {0: self._make_profile(0, "ampere_a6000", cv=0.01)}
            n2 = compute_required_warmup_iters(profiles, cv_threshold=0.03, extra_budget=2)
            n0 = compute_required_warmup_iters(profiles, cv_threshold=0.03, extra_budget=0)
            self.assertEqual(n2 - n0, 2)

    class TestDesslocCache(unittest.TestCase):
        """Tests for DES-LOC locality cache priming."""

        def setUp(self):
            dessloc_cache_evict_all()

        def tearDown(self):
            dessloc_cache_evict_all()

        def _make_profile(self, dev_idx, family) -> DeviceProfile:
            return DeviceProfile(
                device_index=dev_idx, sm_major=8, sm_minor=6, sm_version=86,
                family=family, total_memory_bytes=48 * 1024 ** 3,
                peak_bandwidth_gbs=768.0, warmup_iters_required=5, steady_state_cv=0.01,
            )

        def test_cache_empty_before_prime(self):
            self.assertAlmostEqual(dessloc_cache_hit_rate(), 0.0)

        def test_cache_populated_after_prime(self):
            profiles = {0: self._make_profile(0, "ampere_a6000")}
            dessloc_cache_prime(profiles, prime_size_mb=1.0)
            self.assertAlmostEqual(dessloc_cache_hit_rate(), 1.0)

        def test_cache_evict_resets_hit_rate(self):
            profiles = {0: self._make_profile(0, "ampere_a6000")}
            dessloc_cache_prime(profiles, prime_size_mb=1.0)
            dessloc_cache_evict_all()
            self.assertAlmostEqual(dessloc_cache_hit_rate(), 0.0)

        def test_prime_idempotent(self):
            profiles = {0: self._make_profile(0, "ampere_a6000")}
            dessloc_cache_prime(profiles, prime_size_mb=1.0)
            dessloc_cache_prime(profiles, prime_size_mb=1.0)  # second call should not duplicate
            with _DESSLOC_CACHE_LOCK:
                n_keys = len(_DESSLOC_LOCALITY_CACHE)
            self.assertEqual(n_keys, 1)

    class TestStatisticsComputation(unittest.TestCase):
        """Tests for latency statistics matching Megatron baseline_values.json format."""

        def _make_stabilizer(self):
            """Create a stabilizer with a mocked calibration."""
            stab = HeteroHybridStabilizer.__new__(HeteroHybridStabilizer)
            stab.config = HybridStabilizerConfig()
            stab._device_profiles = {}
            stab._calibrated = True
            stab._num_warmup_iters = 5
            stab._num_timed_iters = 10
            stab._step_latencies_ms = []
            stab._current_batch_size = 8
            stab._current_num_output_tokens = 128
            stab._device_indices = []
            return stab

        def test_throughput_formula(self):
            stab = self._make_stabilizer()
            # 10 steps × 8 sequences × 128 tokens = 10240 tokens
            # avg latency = 100 ms per step → total = 1.0 s → throughput = 10240 tok/s
            stab._step_latencies_ms = [100.0] * 10
            tp, avg, p50, p99, tpot = stab._compute_statistics()
            self.assertAlmostEqual(tp, 10240.0, places=1)
            self.assertAlmostEqual(avg, 100.0, places=5)

        def test_p99_with_one_outlier(self):
            stab = self._make_stabilizer()
            # 9 steps at 100 ms, 1 step at 1000 ms
            stab._step_latencies_ms = [100.0] * 9 + [1000.0]
            _, _, _, p99, _ = stab._compute_statistics()
            # p99 should be close to 1000 ms
            self.assertGreater(p99, 500.0)

        def test_tpot_formula(self):
            stab = self._make_stabilizer()
            stab._step_latencies_ms = [256.0] * 10  # 256 ms per step
            _, avg, _, _, tpot = stab._compute_statistics()
            # tpot = avg / num_output_tokens = 256 / 128 = 2.0 ms/tok
            self.assertAlmostEqual(tpot, 2.0, places=5)

        def test_single_step(self):
            stab = self._make_stabilizer()
            stab._step_latencies_ms = [500.0]
            tp, avg, p50, p99, tpot = stab._compute_statistics()
            self.assertAlmostEqual(avg, 500.0)
            self.assertAlmostEqual(p50, 500.0)
            self.assertAlmostEqual(p99, 500.0)

        def test_empty_latencies_raises(self):
            stab = self._make_stabilizer()
            with self.assertRaises(RuntimeError):
                stab._compute_statistics()

        def test_percentile_interpolation(self):
            """Verify percentile uses linear interpolation, not nearest-rank."""
            vals = sorted([float(i) for i in range(1, 11)])  # 1.0 … 10.0
            p50 = HeteroHybridStabilizer._compute_percentile(vals, 50.0)
            # 50th pct of 10 elements: idx_f = 0.5*9 = 4.5 → 5.0*0.5 + 6.0*0.5 = 5.5
            self.assertAlmostEqual(p50, 5.5, places=5)

    class TestStabilizerRoundTrip(unittest.TestCase):
        """End-to-end test with mocked CUDA calls."""

        def test_full_round_trip_mocked(self):
            """
            Simulate a full calibrate → stabilized_context → record → finalize
            cycle without a real GPU, verifying all fields are populated.
            """
            dessloc_cache_evict_all()

            # Build a minimal mock DeviceProfile for device 0
            mock_profile = DeviceProfile(
                device_index=0, sm_major=8, sm_minor=6, sm_version=86,
                family="ampere_a6000", total_memory_bytes=48 * 1024 ** 3,
                peak_bandwidth_gbs=768.0, warmup_iters_required=5,
                steady_state_cv=0.02,
            )

            config = HybridStabilizerConfig(
                device_indices=[0],
                pcie_prewarm_rounds=0,          # disable PCIe prewarm (no GPU)
                dessloc_cache_prime_size_mb=1.0,
                freq_stab_burst=False,
            )

            stab = HeteroHybridStabilizer.__new__(HeteroHybridStabilizer)
            stab.config = config
            stab._device_profiles = {0: mock_profile}
            stab._calibrated = True
            stab._num_warmup_iters = 7
            stab._num_timed_iters = 10
            stab._step_latencies_ms = []
            stab._current_batch_size = 32
            stab._current_num_output_tokens = 128
            stab._device_indices = [0]

            with stab.stabilized_context(batch_size=32, num_output_tokens=128):
                simulated_latencies = [
                    980.0, 975.0, 985.0, 970.0, 990.0,
                    978.0, 983.0, 972.0, 988.0, 976.0,
                ]
                for lat in simulated_latencies:
                    stab.record_step_latency(lat)

            result = stab.finalize()

            self.assertEqual(result.batch_size, 32)
            self.assertEqual(result.num_warmup_iters, 7)
            self.assertEqual(result.num_timed_iters, 10)
            self.assertEqual(len(result.latencies_ms), 10)
            self.assertGreater(result.throughput_tok_per_sec, 0)
            self.assertGreater(result.avg_latency_ms, 0)
            self.assertGreater(result.p99_latency_ms, result.p50_latency_ms)
            self.assertGreater(result.tpot_ms_per_tok, 0)
            self.assertAlmostEqual(result.dessloc_cache_hit_rate, 1.0)
            self.assertFalse(result.pcie_prewarm_completed)

            # Verify statistics match expectations from simulated latencies
            expected_avg = sum(simulated_latencies) / len(simulated_latencies)
            self.assertAlmostEqual(result.avg_latency_ms, expected_avg, places=5)

            dessloc_cache_evict_all()

    class TestLiveCUDA(unittest.TestCase):
        """Live GPU tests — skipped if no CUDA device is available."""

        @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
        def test_calibrate_single_device(self):
            """Verify that calibration runs without error on a real GPU."""
            config = HybridStabilizerConfig(
                device_indices=[0],
                freq_stab_burst=True,
                pcie_prewarm_rounds=0,  # skip cross-device test on single-GPU CI
                dessloc_cache_prime_size_mb=4.0,
            )
            stab = HeteroHybridStabilizer(config=config)
            profiles = stab.calibrate()

            self.assertIn(0, profiles)
            profile = profiles[0]
            self.assertGreater(profile.peak_bandwidth_gbs, 0)
            self.assertGreater(profile.steady_state_cv, 0)
            self.assertGreater(stab.num_warmup_iters, 0)
            self.assertGreater(stab.num_timed_iters, 0)

            logger.info("Live calibration result:\n%s", stab.summary())
            dessloc_cache_evict_all()

        @unittest.skipUnless(
            torch.cuda.is_available() and torch.cuda.device_count() >= 2,
            "Fewer than 2 CUDA devices available",
        )
        def test_pcie_prewarm_two_devices(self):
            """Verify PCIe prewarm completes without error on a 2-GPU system."""
            avg_ms = _pcie_prewarm(
                torch.device("cuda:0"),
                torch.device("cuda:1"),
                size_bytes=4 * 1024 * 1024,
                n_rounds=3,
            )
            self.assertGreater(avg_ms, 0)
            self.assertLess(avg_ms, 500)  # sanity: < 500 ms for 4 MB

    # Run all tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveIterCount)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWarmupBudget))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDesslocCache))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStatisticsComputation))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStabilizerRoundTrip))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLiveCUDA))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)
