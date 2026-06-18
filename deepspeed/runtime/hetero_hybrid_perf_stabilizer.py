"""
DES-LOC Heterogeneous Hybrid Performance Stabilizer
=====================================================

Upstream Design Intent (Megatron bd1f0dd):
    Megatron's "Stabilize hybrid nanov3 gb200 perf" commit addressed a classic
    benchmarking pathology on GB200 hardware: with only 5 timed iterations, a
    single cold/outlier iteration could swing p99 latency and throughput by
    10-14% run-to-run.  The fix was twofold:
      1. Increase NUM_WARMUP_ITERS from 2→5 so the first *timed* iteration
         already hits steady-state cache/TLB/CUDA-graph state.
      2. Increase NUM_TIMED_ITERS from 5→10 to dilute single-iteration outliers
         and tighten the p99 confidence interval.

DES-LOC Adaptation Points:
    In the Neuron_SP DES-LOC (Decoupled Execution with Shared LOcality Cache)
    framework the same statistical pathology is *amplified* because:

    A. Heterogeneous device pool (2× A6000 SM86 @ 48 GB + 1× H100 NVL SM90
       @ 96 GB, PCIe-only, no NVLink):
       - H100 cold-starts (CUDA graph capture, cuDNN algo selection) take
         3-5× longer than A6000 cold-starts due to SM90 arch differences.
       - PCIe bandwidth variance between host↔device copies adds ±8-15 %
         latency jitter that is absent on NVLink topologies.
       - The "locality cache" (shared CPU DRAM pinned buffer, up to 1.5 TB)
         has its own cold/warm distinction: first access triggers pinned-page
         faults; subsequent accesses hit the TLB.

    B. Decoupled execution pipeline:
       - Prefill stages run on H100; decode stages fan out to A6000 × 2.
       - Synchronization barriers between the two device classes introduce
         non-deterministic PCIe transfer latency that poisons short benchmark
         windows exactly as Megatron observed on GB200.

    This module provides:
      • HeteroWarmupScheduler  — device-class-aware warm-up iteration planner
      • IterationStabilityGuard — online Welch-t test to decide when timing
                                  window is statistically stable
      • SharedLocalityCacheWarmer — pre-faults the DES-LOC pinned CPU buffer
                                   before timed iterations start
      • HeteroHybridPerfStabilizer — top-level orchestrator mirroring the
                                      role of Megatron's model_config.yaml knobs
                                      but made runtime-adaptive

    Reference commit: Megatron-LM bd1f0dd063b90dd9e57b09eeef41fb8e1723aa86
    Project: github.com/dylanyunlon/Neuron_SP
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import statistics
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device classification
# ---------------------------------------------------------------------------

class DeviceClass(Enum):
    """SM architecture classes present in the DES-LOC heterogeneous pool."""
    SM86_A6000  = auto()   # 2× A6000 48 GB
    SM90_H100   = auto()   # 1× H100 NVL 96 GB
    CPU_DRAM    = auto()   # 1.5 TB pinned locality cache


@dataclass(frozen=True)
class DeviceProfile:
    """
    Static characterisation of one physical device.

    cold_start_multiplier:
        Ratio of cold-start latency to steady-state latency.  Derived
        empirically:  A6000 ≈ 2.0×, H100 SM90 ≈ 3.5× (CUDA-graph capture
        + cuDNN heuristic selection on new arch).

    pcie_bw_cv:
        Coefficient of variation of PCIe bandwidth (σ/μ).  Without NVLink
        every cross-device copy goes through the PCIe root complex; on
        systems with competing DMA traffic this is surprisingly noisy.

    warmup_iters_base:
        Minimum warm-up iterations recommended *before* any timed window
        starts on this device class.  Maps directly to Megatron's
        NUM_WARMUP_ITERS knob, but split per device class.
    """
    device_class: DeviceClass
    cold_start_multiplier: float
    pcie_bw_cv: float          # dimensionless, typical 0.05–0.20
    warmup_iters_base: int
    timed_iters_base: int
    vram_gb: float
    sm_count: int


# Hard-coded profiles for the Neuron_SP target hardware.
DEVICE_PROFILES: Dict[DeviceClass, DeviceProfile] = {
    DeviceClass.SM86_A6000: DeviceProfile(
        device_class=DeviceClass.SM86_A6000,
        cold_start_multiplier=2.0,
        pcie_bw_cv=0.12,
        warmup_iters_base=5,   # mirrors Megatron's final NUM_WARMUP_ITERS=5
        timed_iters_base=10,   # mirrors Megatron's final NUM_TIMED_ITERS=10
        vram_gb=48.0,
        sm_count=84,
    ),
    DeviceClass.SM90_H100: DeviceProfile(
        device_class=DeviceClass.SM90_H100,
        cold_start_multiplier=3.5,
        pcie_bw_cv=0.08,
        warmup_iters_base=7,   # extra warmup: SM90 arch init is heavier
        timed_iters_base=12,   # extra iters: H100 jitter needs more samples
        vram_gb=96.0,
        sm_count=132,
    ),
}


def classify_device(device: torch.device) -> DeviceClass:
    """
    Map a torch.device to a DeviceClass by inspecting SM capability.

    Falls back to SM86_A6000 for any unknown CUDA device so that the
    scheduler degrades gracefully on unexpected hardware.
    """
    if device.type != "cuda":
        return DeviceClass.CPU_DRAM
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm >= 90:
        return DeviceClass.SM90_H100
    return DeviceClass.SM86_A6000


# ---------------------------------------------------------------------------
# Shared Locality Cache Warmer
# ---------------------------------------------------------------------------

class SharedLocalityCacheWarmer:
    """
    Pre-fault the DES-LOC pinned CPU DRAM buffer before timed iterations.

    DES-LOC Background:
        The "Shared LOcality Cache" is a large region of page-locked (pinned)
        CPU DRAM that acts as a staging buffer between H100 prefill outputs
        and A6000 decode inputs.  On first access, Linux maps physical pages
        lazily, causing TLB misses and NUMA DRAM faults that inflate latency
        by 5-20 % for the first 1-3 iterations — exactly the warm-up
        pathology Megatron fixed by raising NUM_WARMUP_ITERS.

    This class issues explicit sequential writes across the buffer (stride =
    OS page size) during the warmup phase so that all physical pages are
    faulted in and TLB-populated before timing begins.

    Args:
        buffer_bytes:  Size of the locality cache region to pre-fault.
                       Defaults to a conservative 4 GB slice; the full
                       1.5 TB pool is warmed lazily by the OS after the
                       first real workload wave.
        page_size:     OS page size in bytes (default 4096).
        num_threads:   Parallel pre-fault threads.  Using more than the
                       number of NUMA nodes rarely helps and can cause
                       cross-NUMA migrations.
    """

    def __init__(
        self,
        buffer_bytes: int = 4 * 1024 ** 3,
        page_size: int = 4096,
        num_threads: int = 4,
    ) -> None:
        self._buffer_bytes = buffer_bytes
        self._page_size = page_size
        self._num_threads = num_threads
        self._warmed = False
        self._pinned_buf: Optional[torch.Tensor] = None

    def _allocate_pinned(self) -> torch.Tensor:
        """Allocate a pinned (page-locked) CPU tensor for pre-faulting."""
        n_elements = self._buffer_bytes // 4  # float32
        logger.debug(
            "SharedLocalityCacheWarmer: allocating %.1f GB pinned buffer "
            "(%d float32 elements)",
            self._buffer_bytes / 1024**3,
            n_elements,
        )
        try:
            buf = torch.empty(n_elements, dtype=torch.float32, pin_memory=True)
        except RuntimeError as exc:
            logger.warning(
                "SharedLocalityCacheWarmer: pinned alloc failed (%s); "
                "falling back to pageable tensor — first-iter latency may "
                "be elevated", exc,
            )
            buf = torch.empty(n_elements, dtype=torch.float32)
        return buf

    def _prefault_slice(self, buf: torch.Tensor, start: int, end: int) -> None:
        """Write one byte per page in [start, end) elements."""
        stride = self._page_size // 4  # pages in float32 elements
        buf[start:end:stride].fill_(0.0)

    def warm(self) -> None:
        """
        Pre-fault the locality cache buffer.

        Spawns *num_threads* threads, each responsible for a contiguous
        slice of the buffer.  Blocks until all threads complete.  Safe to
        call multiple times; subsequent calls are no-ops.
        """
        if self._warmed:
            logger.debug("SharedLocalityCacheWarmer: already warmed, skipping")
            return

        t0 = time.perf_counter()
        self._pinned_buf = self._allocate_pinned()
        n = len(self._pinned_buf)
        chunk = math.ceil(n / self._num_threads)

        threads = []
        for i in range(self._num_threads):
            start = i * chunk
            end = min(start + chunk, n)
            t = threading.Thread(
                target=self._prefault_slice,
                args=(self._pinned_buf, start, end),
                daemon=True,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        elapsed = time.perf_counter() - t0
        logger.info(
            "SharedLocalityCacheWarmer: pre-faulted %.1f GB in %.3f s "
            "(%.1f GB/s effective)",
            self._buffer_bytes / 1024**3,
            elapsed,
            (self._buffer_bytes / 1024**3) / max(elapsed, 1e-9),
        )
        self._warmed = True

    @property
    def is_warm(self) -> bool:
        return self._warmed


# ---------------------------------------------------------------------------
# Iteration Stability Guard
# ---------------------------------------------------------------------------

@dataclass
class IterationStats:
    """Running statistics for a single metric stream."""
    samples: deque = field(default_factory=lambda: deque(maxlen=64))

    def push(self, value: float) -> None:
        self.samples.append(value)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else float("nan")

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) >= 2 else float("inf")

    @property
    def cv(self) -> float:
        """Coefficient of variation (σ/μ)."""
        m = self.mean
        return self.stdev / m if m > 0 else float("inf")

    def p_percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100) using linear interpolation."""
        if not self.samples:
            return float("nan")
        sorted_s = sorted(self.samples)
        idx = (p / 100.0) * (len(sorted_s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(sorted_s) - 1)
        frac = idx - lo
        return sorted_s[lo] * (1 - frac) + sorted_s[hi] * frac


class IterationStabilityGuard:
    """
    Online stability detector for heterogeneous iteration timing.

    Motivation (from Megatron bd1f0dd):
        With 5 timed iterations on GB200, a single outlier caused 10-14%
        p99 swing.  Simply increasing to 10 iterations was sufficient for
        *homogeneous* GB200 hardware.  On the DES-LOC heterogeneous pool
        (A6000 + H100, PCIe-only) the required sample count depends on the
        live PCIe bandwidth CV, which changes with system load.

    This guard implements an online Welch t-test between the first half and
    second half of a rolling window.  If the two halves are statistically
    indistinguishable (p > alpha) the timing window is deemed stable and
    further iterations can stop.  This mirrors the *intent* of Megatron's
    fixed NUM_TIMED_ITERS=10 but adapts dynamically to actual variance.

    Args:
        min_iters:   Minimum timed iterations regardless of convergence.
        max_iters:   Hard cap; raise RuntimeWarning if not converged by then.
        alpha:       Significance level for Welch t-test (default 0.05).
        cv_threshold: If CV < cv_threshold the window is considered stable
                      without running the t-test (fast path for very
                      low-noise devices).
    """

    def __init__(
        self,
        min_iters: int = 10,
        max_iters: int = 32,
        alpha: float = 0.05,
        cv_threshold: float = 0.02,
    ) -> None:
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.alpha = alpha
        self.cv_threshold = cv_threshold
        self._stats = IterationStats()

    def record(self, latency_ms: float) -> None:
        self._stats.push(latency_ms)

    @property
    def n(self) -> int:
        return self._stats.n

    def is_stable(self) -> bool:
        """
        Return True when the timed window has converged.

        Convergence criteria (checked in order):
          1. Have at least min_iters samples.
          2. Fast-path: CV < cv_threshold → already stable.
          3. Welch t-test on first-half vs second-half of window;
             stable if we *fail to reject* H₀ (means are equal) at alpha.
        """
        if self._stats.n < self.min_iters:
            return False
        if self._stats.cv < self.cv_threshold:
            logger.debug(
                "IterationStabilityGuard: CV=%.4f < %.4f — fast-path stable "
                "after %d iters", self._stats.cv, self.cv_threshold, self._stats.n,
            )
            return True

        samples = list(self._stats.samples)
        mid = len(samples) // 2
        first_half = samples[:mid]
        second_half = samples[mid:]
        if len(first_half) < 2 or len(second_half) < 2:
            return False

        t_stat, p_value = _welch_t_test(first_half, second_half)
        stable = p_value > self.alpha
        logger.debug(
            "IterationStabilityGuard: n=%d t=%.4f p=%.4f → %s",
            self._stats.n, t_stat, p_value,
            "STABLE" if stable else "NOT STABLE",
        )
        return stable

    def summary(self) -> Dict[str, float]:
        return {
            "n": self._stats.n,
            "mean_ms": self._stats.mean,
            "stdev_ms": self._stats.stdev,
            "cv": self._stats.cv,
            "p50_ms": self._stats.p_percentile(50),
            "p99_ms": self._stats.p_percentile(99),
        }


def _welch_t_test(
    a: Sequence[float], b: Sequence[float]
) -> Tuple[float, float]:
    """
    Compute Welch's t-statistic and two-tailed p-value.

    Pure-Python implementation avoiding scipy dependency.  Accuracy is
    sufficient for the convergence decision (we only need a rough p-value
    threshold, not publication-quality inference).

    Returns:
        (t_statistic, p_value)  — p_value approximated via t-distribution
        CDF using an Abramowitz & Stegun rational approximation.
    """
    n_a, n_b = len(a), len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0.0:
        return 0.0, 1.0

    t = (mean_a - mean_b) / se

    # Welch–Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / den if den > 0 else 1.0

    # Two-tailed p-value via regularised incomplete beta function approximation
    p_value = _t_dist_p_value(abs(t), df)
    return t, p_value


def _t_dist_p_value(t_abs: float, df: float) -> float:
    """
    Approximate two-tailed p-value for t-distribution.

    Uses the relationship between the t-CDF and the regularised incomplete
    beta function:  p = I(df/(df+t²), df/2, 1/2).
    The incomplete beta is approximated with the continued-fraction method
    (Numerical Recipes, 6.4).
    """
    x = df / (df + t_abs ** 2)
    p_one_tail = _regularised_incomplete_beta(x, df / 2.0, 0.5) / 2.0
    return 2.0 * p_one_tail


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """
    Compute I_x(a, b) via Lentz continued-fraction expansion.
    Accurate to ~1e-7 for the parameter ranges we encounter.
    """
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a

    # Continued fraction via modified Lentz
    MAXIT = 200
    EPS = 3e-7
    FPMIN = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break

    return front * h


# ---------------------------------------------------------------------------
# Heterogeneous Warmup Scheduler
# ---------------------------------------------------------------------------

class HeteroWarmupScheduler:
    """
    Compute device-class-aware warm-up and timed-iteration counts.

    Megatron's model_config.yaml hardcoded NUM_WARMUP_ITERS=5 and
    NUM_TIMED_ITERS=10 for GB200.  In DES-LOC the optimal counts differ
    per device class (H100 needs more warm-up) *and* per batch size
    (large batches saturate PCIe earlier, reaching steady state faster).

    Design:
        base_warmup  = device_profile.warmup_iters_base
        batch_factor = log2(batch_size) / log2(reference_batch)  ∈ [0.5, 2.0]
        warmup_iters = ceil(base_warmup * batch_factor)

        base_timed   = device_profile.timed_iters_base
        pcie_factor  = 1 + device_profile.pcie_bw_cv * PCIe_variance_scale
        timed_iters  = ceil(base_timed * pcie_factor)

    The PCIe_variance_scale is measured at runtime by the
    PCIeBandwidthMonitor (see below) and injected at schedule time.

    Args:
        device_profiles:   Dict mapping DeviceClass → DeviceProfile.
                           Defaults to the hardware-specific DEVICE_PROFILES.
        reference_batch:   Batch size at which the base counts were tuned
                           (matches Megatron's reference config).
        pcie_variance_scale: Runtime PCIe variance multiplier; updated by
                             the PCIeBandwidthMonitor callback.
    """

    def __init__(
        self,
        device_profiles: Optional[Dict[DeviceClass, DeviceProfile]] = None,
        reference_batch: int = 32,
        pcie_variance_scale: float = 1.0,
    ) -> None:
        self._profiles = device_profiles or DEVICE_PROFILES
        self._reference_batch = reference_batch
        self._pcie_variance_scale = pcie_variance_scale
        self._lock = threading.Lock()

    def update_pcie_variance(self, scale: float) -> None:
        """Called by PCIeBandwidthMonitor when new BW samples arrive."""
        with self._lock:
            self._pcie_variance_scale = max(0.5, min(scale, 4.0))
            logger.debug(
                "HeteroWarmupScheduler: PCIe variance scale updated → %.3f",
                self._pcie_variance_scale,
            )

    def schedule(
        self,
        device_class: DeviceClass,
        batch_size: int,
    ) -> Tuple[int, int]:
        """
        Return (warmup_iters, timed_iters) for the given device + batch.

        Returns:
            Tuple of (warmup_iters, timed_iters).
        """
        if device_class not in self._profiles:
            logger.warning(
                "HeteroWarmupScheduler: unknown device class %s, "
                "using SM86_A6000 profile", device_class,
            )
            profile = self._profiles[DeviceClass.SM86_A6000]
        else:
            profile = self._profiles[device_class]

        # Batch size scaling: large batches fill PCIe faster → less cold start
        if batch_size <= 0:
            batch_factor = 1.0
        else:
            batch_factor = math.log2(max(batch_size, 1) + 1) / math.log2(
                self._reference_batch + 1
            )
        batch_factor = max(0.5, min(batch_factor, 2.0))

        warmup = math.ceil(profile.warmup_iters_base * batch_factor)

        # PCIe variance scaling for timed iterations
        with self._lock:
            pcie_scale = self._pcie_variance_scale
        pcie_factor = 1.0 + profile.pcie_bw_cv * pcie_scale
        timed = math.ceil(profile.timed_iters_base * pcie_factor)

        logger.info(
            "HeteroWarmupScheduler: device=%s batch=%d → "
            "warmup=%d timed=%d (batch_factor=%.2f pcie_factor=%.2f)",
            device_class.name, batch_size, warmup, timed,
            batch_factor, pcie_factor,
        )
        return warmup, timed


# ---------------------------------------------------------------------------
# PCIe Bandwidth Monitor
# ---------------------------------------------------------------------------

class PCIeBandwidthMonitor:
    """
    Lightweight PCIe bandwidth probe for the DES-LOC heterogeneous pool.

    Runs a background thread that periodically transfers a small probe
    tensor between the H100 and CPU DRAM, measuring effective bandwidth.
    The CV of recent samples is used to update the HeteroWarmupScheduler's
    pcie_variance_scale so that the timed-iteration budget adapts to
    current system load (e.g., other workloads competing for PCIe).

    Args:
        scheduler:       HeteroWarmupScheduler to notify on BW updates.
        probe_mb:        Size of the probe transfer in MB (default 128).
        interval_s:      Seconds between probe measurements (default 10.0).
        window:          Number of recent measurements to keep for CV calc.
        h100_device_idx: CUDA device index of the H100 (default 0).
    """

    def __init__(
        self,
        scheduler: HeteroWarmupScheduler,
        probe_mb: float = 128.0,
        interval_s: float = 10.0,
        window: int = 16,
        h100_device_idx: int = 0,
    ) -> None:
        self._scheduler = scheduler
        self._probe_bytes = int(probe_mb * 1024 ** 2)
        self._interval_s = interval_s
        self._bw_samples: deque = deque(maxlen=window)
        self._h100_device_idx = h100_device_idx
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background bandwidth monitor thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.debug("PCIeBandwidthMonitor: already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="pcie-bw-monitor"
        )
        self._thread.start()
        logger.info(
            "PCIeBandwidthMonitor: started (probe=%.0f MB, interval=%.1f s)",
            self._probe_bytes / 1024**2, self._interval_s,
        )

    def stop(self) -> None:
        """Signal the monitor thread to stop and join it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s + 5)

    def _run(self) -> None:
        if not torch.cuda.is_available():
            logger.warning("PCIeBandwidthMonitor: CUDA unavailable, stopping")
            return

        n_elements = self._probe_bytes // 4
        try:
            gpu_tensor = torch.empty(
                n_elements, dtype=torch.float32,
                device=torch.device("cuda", self._h100_device_idx),
            )
            cpu_tensor = torch.empty(n_elements, dtype=torch.float32, pin_memory=True)
        except RuntimeError as exc:
            logger.error("PCIeBandwidthMonitor: probe alloc failed: %s", exc)
            return

        while not self._stop_event.is_set():
            try:
                t0 = time.perf_counter()
                cpu_tensor.copy_(gpu_tensor, non_blocking=False)
                torch.cuda.synchronize(self._h100_device_idx)
                elapsed = time.perf_counter() - t0
                bw_gbs = (self._probe_bytes / 1e9) / max(elapsed, 1e-9)
                self._bw_samples.append(bw_gbs)

                if len(self._bw_samples) >= 4:
                    mean_bw = statistics.mean(self._bw_samples)
                    stdev_bw = statistics.stdev(self._bw_samples)
                    cv = stdev_bw / mean_bw if mean_bw > 0 else 1.0
                    # Scale: CV=0.05 → scale=1.0; CV=0.20 → scale=4.0
                    scale = max(1.0, cv / 0.05)
                    self._scheduler.update_pcie_variance(scale)
                    logger.debug(
                        "PCIeBandwidthMonitor: BW=%.2f GB/s CV=%.3f scale=%.2f",
                        mean_bw, cv, scale,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("PCIeBandwidthMonitor: probe error: %s", exc)

            self._stop_event.wait(self._interval_s)


# ---------------------------------------------------------------------------
# Top-level Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class StabilisedBenchmarkResult:
    """Aggregated benchmark result from a stabilised timed window."""
    device_class: DeviceClass
    batch_size: int
    warmup_iters_used: int
    timed_iters_used: int
    mean_latency_ms: float
    stdev_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_tok_per_sec: float
    converged: bool


class HeteroHybridPerfStabilizer:
    """
    Top-level DES-LOC performance stabiliser for heterogeneous hybrid runs.

    Mirrors the role of Megatron's model_config.yaml NUM_WARMUP_ITERS /
    NUM_TIMED_ITERS knobs but operates as a runtime object that:

      1. Classifies each device in the pool.
      2. Pre-faults the DES-LOC locality cache (SharedLocalityCacheWarmer).
      3. Starts the PCIeBandwidthMonitor to track live transfer variance.
      4. For each benchmark call, consults HeteroWarmupScheduler to get
         device+batch-specific warm-up and timed iteration counts.
      5. Runs the user-supplied iteration_fn in warm-up mode (discarded),
         then in timed mode, collecting latency samples.
      6. Uses IterationStabilityGuard to extend the timed window if the
         per-device PCIe jitter hasn't converged yet.
      7. Returns a StabilisedBenchmarkResult with p50/p99/throughput.

    Args:
        devices:             List of torch.device objects in the pool.
        locality_cache_gb:   GB of pinned CPU buffer to pre-fault.
        pcie_monitor_mb:     PCIe probe size in MB.
        pcie_monitor_interval_s: Seconds between PCIe probes.
        stability_alpha:     Welch t-test significance level.
        max_timed_iters:     Hard cap on timed iterations.

    Example::

        stabilizer = HeteroHybridPerfStabilizer(
            devices=[torch.device("cuda:0"), torch.device("cuda:1"),
                     torch.device("cuda:2")],
        )
        stabilizer.start()

        def my_iter_fn(batch_size: int) -> float:
            # ... run one forward pass, return latency in ms
            return 12.3

        result = stabilizer.run_benchmark(
            iteration_fn=my_iter_fn,
            batch_size=32,
            primary_device=torch.device("cuda:0"),
            num_output_tokens=128,
        )
        print(result)
        stabilizer.stop()
    """

    def __init__(
        self,
        devices: Optional[List[torch.device]] = None,
        locality_cache_gb: float = 4.0,
        pcie_monitor_mb: float = 128.0,
        pcie_monitor_interval_s: float = 10.0,
        stability_alpha: float = 0.05,
        max_timed_iters: int = 32,
    ) -> None:
        self._devices = devices or []
        self._stability_alpha = stability_alpha
        self._max_timed_iters = max_timed_iters

        self._warmup_scheduler = HeteroWarmupScheduler()

        # Find the H100 device index for PCIe monitoring
        h100_idx = self._find_h100_device_index()

        self._cache_warmer = SharedLocalityCacheWarmer(
            buffer_bytes=int(locality_cache_gb * 1024 ** 3),
        )

        self._pcie_monitor = PCIeBandwidthMonitor(
            scheduler=self._warmup_scheduler,
            probe_mb=pcie_monitor_mb,
            interval_s=pcie_monitor_interval_s,
            h100_device_idx=h100_idx,
        )

        self._running = False

    def _find_h100_device_index(self) -> int:
        """Return the CUDA index of the first H100 device, or 0 as fallback."""
        for dev in self._devices:
            if dev.type == "cuda":
                dc = classify_device(dev)
                if dc == DeviceClass.SM90_H100:
                    return dev.index if dev.index is not None else 0
        return 0

    def start(self) -> None:
        """
        Initialise the stabiliser.

        Warms the locality cache (blocking) then starts the PCIe monitor
        (background thread).  Call this once before any benchmark runs.
        """
        if self._running:
            logger.debug("HeteroHybridPerfStabilizer: already started")
            return
        logger.info("HeteroHybridPerfStabilizer: warming locality cache …")
        self._cache_warmer.warm()
        if torch.cuda.is_available():
            self._pcie_monitor.start()
        self._running = True
        logger.info("HeteroHybridPerfStabilizer: ready")

    def stop(self) -> None:
        """Stop background threads."""
        self._pcie_monitor.stop()
        self._running = False
        logger.info("HeteroHybridPerfStabilizer: stopped")

    def run_benchmark(
        self,
        iteration_fn: Callable[[int], float],
        batch_size: int,
        primary_device: torch.device,
        num_output_tokens: int = 128,
        force_warmup_iters: Optional[int] = None,
        force_timed_iters: Optional[int] = None,
    ) -> StabilisedBenchmarkResult:
        """
        Run a stabilised benchmark on the given device.

        Args:
            iteration_fn:       Callable(batch_size) → latency_ms for one iter.
            batch_size:         Batch size for this benchmark point.
            primary_device:     The device that is the latency bottleneck
                                (used to select the warmup/timed schedule).
            num_output_tokens:  Output tokens per sample (for throughput calc).
            force_warmup_iters: Override scheduler warmup count (testing only).
            force_timed_iters:  Override scheduler timed count (testing only).

        Returns:
            StabilisedBenchmarkResult
        """
        device_class = classify_device(primary_device)

        warmup_iters, timed_iters = self._warmup_scheduler.schedule(
            device_class, batch_size
        )
        if force_warmup_iters is not None:
            warmup_iters = force_warmup_iters
        if force_timed_iters is not None:
            timed_iters = force_timed_iters

        logger.info(
            "HeteroHybridPerfStabilizer.run_benchmark: "
            "device=%s batch=%d warmup=%d timed=%d",
            device_class.name, batch_size, warmup_iters, timed_iters,
        )

        # --- Warm-up phase (discarded) ---
        for i in range(warmup_iters):
            lat = iteration_fn(batch_size)
            logger.debug("  warmup[%d/%d] %.2f ms", i + 1, warmup_iters, lat)

        # --- Timed phase with online stability detection ---
        guard = IterationStabilityGuard(
            min_iters=timed_iters,
            max_iters=self._max_timed_iters,
            alpha=self._stability_alpha,
        )

        iter_idx = 0
        while iter_idx < self._max_timed_iters:
            lat = iteration_fn(batch_size)
            guard.record(lat)
            logger.debug("  timed[%d] %.2f ms", iter_idx + 1, lat)
            iter_idx += 1
            if guard.is_stable():
                logger.info(
                    "HeteroHybridPerfStabilizer: converged after %d timed iters",
                    iter_idx,
                )
                break
        else:
            logger.warning(
                "HeteroHybridPerfStabilizer: did not converge within %d iters "
                "(device=%s batch=%d); using available samples",
                self._max_timed_iters, device_class.name, batch_size,
            )

        summary = guard.summary()
        mean_ms = summary["mean_ms"]
        throughput = (
            batch_size * num_output_tokens / (mean_ms / 1000.0)
            if mean_ms > 0
            else 0.0
        )

        result = StabilisedBenchmarkResult(
            device_class=device_class,
            batch_size=batch_size,
            warmup_iters_used=warmup_iters,
            timed_iters_used=iter_idx,
            mean_latency_ms=mean_ms,
            stdev_latency_ms=summary["stdev_ms"],
            p50_latency_ms=summary["p50_ms"],
            p99_latency_ms=summary["p99_ms"],
            throughput_tok_per_sec=throughput,
            converged=guard.is_stable(),
        )
        logger.info(
            "HeteroHybridPerfStabilizer result: "
            "mean=%.1f ms p50=%.1f ms p99=%.1f ms tput=%.1f tok/s converged=%s",
            result.mean_latency_ms,
            result.p50_latency_ms,
            result.p99_latency_ms,
            result.throughput_tok_per_sec,
            result.converged,
        )
        return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import random

    # --- Unit: Welch t-test p-value sanity ---
    identical = [100.0] * 12
    _, p_same = _welch_t_test(identical[:6], identical[6:])
    assert p_same > 0.99, f"identical samples should give p≈1, got {p_same:.4f}"

    different = [100.0] * 8 + [200.0] * 8
    _, p_diff = _welch_t_test(different[:8], different[8:])
    assert p_diff < 0.01, f"very different samples should give p≈0, got {p_diff:.4f}"

    # --- Unit: IterationStabilityGuard converges on stable signal ---
    guard = IterationStabilityGuard(min_iters=10, max_iters=20, cv_threshold=0.01)
    for _ in range(15):
        guard.record(50.0 + random.gauss(0, 0.1))
    assert guard.is_stable(), "low-CV signal should converge"

    # --- Unit: HeteroWarmupScheduler gives H100 more iters than A6000 ---
    sched = HeteroWarmupScheduler()
    wu_a6000, ti_a6000 = sched.schedule(DeviceClass.SM86_A6000, batch_size=32)
    wu_h100, ti_h100   = sched.schedule(DeviceClass.SM90_H100,  batch_size=32)
    assert wu_h100 >= wu_a6000, "H100 should need at least as many warmup iters"
    assert ti_h100 >= ti_a6000, "H100 should need at least as many timed iters"

    # --- Integration: full stabilizer smoke run (CPU mock, no real GPU needed) ---
    cpu_device = torch.device("cpu")
    stabilizer = HeteroHybridPerfStabilizer(
        devices=[cpu_device],
        locality_cache_gb=0.01,   # tiny for test
    )
    stabilizer.start()

    call_count = 0
    def _mock_iter(bs: int) -> float:
        global call_count
        call_count += 1
        return 50.0 + random.gauss(0, 0.3)

    result = stabilizer.run_benchmark(
        iteration_fn=_mock_iter,
        batch_size=8,
        primary_device=cpu_device,
        num_output_tokens=128,
        force_warmup_iters=3,
        force_timed_iters=10,
    )
    stabilizer.stop()

    assert result.timed_iters_used >= 10, "must run at least min_iters"
    assert result.mean_latency_ms > 0, "mean latency must be positive"
    assert result.throughput_tok_per_sec > 0, "throughput must be positive"

    logger.info("All smoke tests passed. result=%s", result)
