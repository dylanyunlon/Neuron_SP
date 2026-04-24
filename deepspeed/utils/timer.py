# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# DES-LOC (Desynced Low Communication) phase timing integration
# Ref: Section 4.1 — training time = compute + param_comm + mom_comm + idle

import time
import math
from deepspeed.utils.logging import print_dist
from deepspeed.accelerator import get_accelerator

FORWARD_MICRO_TIMER = 'fwd_microstep'
FORWARD_GLOBAL_TIMER = 'fwd'
BACKWARD_MICRO_TIMER = 'bwd_microstep'
BACKWARD_GLOBAL_TIMER = 'bwd'
BACKWARD_INNER_MICRO_TIMER = 'bwd_inner_microstep'
BACKWARD_INNER_GLOBAL_TIMER = 'bwd_inner'
BACKWARD_REDUCE_MICRO_TIMER = 'bwd_allreduce_microstep'
BACKWARD_REDUCE_GLOBAL_TIMER = 'bwd_allreduce'
STEP_MICRO_TIMER = 'step_microstep'
STEP_GLOBAL_TIMER = 'step'
TIME_EPSILON = 1e-6

# DES-LOC phase identifiers for MFU decomposition
# Ref: Nick Joseph — "model efficiency with paper and pen, then implement, then profile"
# training_time = compute + param_comm + mom_comm + optimizer + data_load + idle
DESLOC_PHASE_COMPUTE = 'desloc_compute'        # forward + backward FLOPS
DESLOC_PHASE_PARAM_COMM = 'desloc_param_comm'  # Kx-gated gradient allreduce
DESLOC_PHASE_MOM_COMM = 'desloc_mom_comm'      # Ku/Kv-gated momentum averaging
DESLOC_PHASE_OPTIMIZER = 'desloc_optimizer'    # local optimizer step (Adam/ADOPT)
DESLOC_PHASE_DATA_LOAD = 'desloc_data_load'    # data loading + preprocessing
DESLOC_PHASE_IDLE = 'desloc_idle'              # straggler waiting / synchronization

# Map standard DeepSpeed timer names to DES-LOC phases
_TIMER_TO_PHASE = {
    FORWARD_MICRO_TIMER: DESLOC_PHASE_COMPUTE,
    FORWARD_GLOBAL_TIMER: DESLOC_PHASE_COMPUTE,
    BACKWARD_MICRO_TIMER: DESLOC_PHASE_COMPUTE,
    BACKWARD_GLOBAL_TIMER: DESLOC_PHASE_COMPUTE,
    BACKWARD_INNER_MICRO_TIMER: DESLOC_PHASE_COMPUTE,
    BACKWARD_INNER_GLOBAL_TIMER: DESLOC_PHASE_COMPUTE,
    BACKWARD_REDUCE_MICRO_TIMER: DESLOC_PHASE_PARAM_COMM,
    BACKWARD_REDUCE_GLOBAL_TIMER: DESLOC_PHASE_PARAM_COMM,
    STEP_MICRO_TIMER: DESLOC_PHASE_OPTIMIZER,
    STEP_GLOBAL_TIMER: DESLOC_PHASE_OPTIMIZER,
}

ALL_DESLOC_PHASES = [
    DESLOC_PHASE_COMPUTE, DESLOC_PHASE_PARAM_COMM, DESLOC_PHASE_MOM_COMM,
    DESLOC_PHASE_OPTIMIZER, DESLOC_PHASE_DATA_LOAD, DESLOC_PHASE_IDLE,
]

try:
    import psutil

    PSUTILS_INSTALLED = True
except ImportError:
    PSUTILS_INSTALLED = False
    pass


class CudaEventTimer(object):
    """Timer using CUDA events for accurate GPU timing.

    DES-LOC note: CUDA event timing captures actual GPU execution time,
    which is critical for separating compute-bound vs comm-bound phases.
    Ref: Megatron-LM timers.py — same pattern for GPU-accurate timing.
    """

    def __init__(self, start_event: get_accelerator().Event, end_event: get_accelerator().Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        """Get elapsed time in milliseconds between start and end events."""
        get_accelerator().current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class SynchronizedWallClockTimer:
    """Group of named timers with DES-LOC phase tracking.

    Rewritten from Megatron-based original to add per-phase cumulative
    timing. Each timer's elapsed time is automatically classified into
    a DES-LOC phase (compute, param_comm, mom_comm, optimizer, idle)
    based on its name, enabling MFU decomposition.

    Ref: Section 4.1 — DES-LOC improves MFU by reducing comm phases.
    """

    class Timer:
        """Individual named timer with DES-LOC phase association."""

        def __init__(self, name, desloc_phase=None):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.use_host_timer = get_accelerator().use_host_timers()
            self.start_event = None
            self.elapsed_records = None
            self.start_time = 0.0
            self.end_time = 0.0
            # DES-LOC: associate this timer with a training phase
            self.desloc_phase = desloc_phase or _TIMER_TO_PHASE.get(name)

        def start(self):
            """Start the timer."""
            assert not self.started_, f"{self.name_} timer has already been started"
            if self.use_host_timer:
                self.start_time = time.time()
            else:
                event_class = get_accelerator().Event
                self.start_event = event_class(enable_timing=True)
                self.start_event.record()
            self.started_ = True

        def stop(self, reset=False, record=False):
            """Stop the timer."""
            assert self.started_, "timer is not started"
            if self.use_host_timer:
                self.end_time = time.time()
                self.event_timers.append(self.end_time - self.start_time)
            else:
                event_class = get_accelerator().Event
                end_event = event_class(enable_timing=True)
                end_event.record()
                self.event_timers.append(CudaEventTimer(self.start_event, end_event))
                self.start_event = None
            self.started_ = False

        def _get_elapsed_msec(self):
            """Get total elapsed milliseconds across all recorded intervals."""
            if self.use_host_timer:
                self.elapsed_records = [et * 1000.0 for et in self.event_timers]
            else:
                self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer state."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time in milliseconds."""
            started_ = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self._get_elapsed_msec()
            if reset:
                self.reset()
            if started_:
                self.start()
            return elapsed_

        def mean(self):
            """Get trimmed mean of elapsed records."""
            self.elapsed(reset=False)
            return trim_mean(self.elapsed_records, 0.1)

    def __init__(self):
        self.timers = {}
        # DES-LOC: per-phase cumulative timing for MFU decomposition
        # Ref: MFU = compute_flops / (elapsed_time * peak_flops)
        # DES-LOC improves MFU by reducing param_comm and mom_comm phases
        self._desloc_phases = {phase: 0.0 for phase in ALL_DESLOC_PHASES}
        self._desloc_step_count = 0
        self._desloc_phase_history = []  # per-step phase breakdown

    def get_timers(self):
        """Return the dict of all timers."""
        return self.timers

    def __call__(self, name):
        """Get or create a named timer."""
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        """Return formatted string of GPU memory usage."""
        alloc = "mem_allocated: {:.4f} GB".format(
            get_accelerator().memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(
            get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(
            get_accelerator().memory_cached() / (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(
            get_accelerator().max_memory_cached() / (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        """Log a group of timers, with optional DES-LOC phase breakdown.

        DES-LOC extension: after logging standard timer values, also
        accumulates each timer's elapsed time into its DES-LOC phase
        bucket for MFU decomposition.
        """
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            if name in self.timers:
                elapsed_time = self.timers[name].elapsed(reset=reset) / normalizer
                string += " | {}: {:.2f}".format(name, elapsed_time)
                # DES-LOC: accumulate into phase bucket
                phase = _TIMER_TO_PHASE.get(name)
                if phase and phase in self._desloc_phases:
                    self._desloc_phases[phase] += elapsed_time

        print_dist(string, ranks=ranks or [0])

    def get_mean(self, names, normalizer=1.0, reset=True):
        """Get the mean of a group of timers."""
        assert normalizer > 0.0
        means = {}
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].mean() * 1000.0 / normalizer)
                means[name] = elapsed_time
        return means

    # DES-LOC phase tracking methods

    def accumulate_desloc_phase(self, phase, elapsed_ms):
        """Accumulate elapsed time into a DES-LOC phase bucket.

        Called by engine.py at each training step to record how much
        time was spent in each phase.

        Args:
            phase: One of DESLOC_PHASE_* constants.
            elapsed_ms: Time in milliseconds.

        Ref: MFU = compute_time / total_time. DES-LOC reduces
        param_comm and mom_comm phases, increasing MFU.
        """
        if phase in self._desloc_phases:
            self._desloc_phases[phase] += elapsed_ms

    def get_desloc_breakdown(self):
        """Return DES-LOC phase breakdown as percentages.

        Returns dict mapping phase names to percentage of total time.
        Ref: Section 5.4 — training speedup over DDP.
        """
        total = sum(self._desloc_phases.values())
        if total <= 0:
            return {p: 0.0 for p in self._desloc_phases}
        return {p: round(100.0 * v / total, 4) for p, v in self._desloc_phases.items()}

    def get_desloc_mfu_estimate(self, peak_tflops=0, model_flops_per_step=0):
        """Estimate Model FLOPS Utilization from DES-LOC timing data.

        MFU = model_flops / (step_time * peak_flops)
        This is the key metric for evaluating DES-LOC's benefit.

        Args:
            peak_tflops: Hardware peak TFLOPS (e.g. 312 for H100 BF16).
            model_flops_per_step: FLOPs per training step (6*N*S*B).

        Returns:
            dict with mfu, achieved_tflops, step_time, compute/comm/idle pct.

        Ref: Nick Joseph — "six or seven bottleneck constraints determine MFU."
        """
        total = sum(self._desloc_phases.values())
        if total <= 0 or peak_tflops <= 0 or model_flops_per_step <= 0:
            return {'mfu': 0.0, 'step_time_ms': 0.0}
        steps = max(1, self._desloc_step_count)
        step_time_s = total / 1000.0 / steps
        achieved_tflops = model_flops_per_step / step_time_s / 1e12
        mfu = achieved_tflops / peak_tflops
        compute_ms = self._desloc_phases[DESLOC_PHASE_COMPUTE]
        comm_ms = (self._desloc_phases[DESLOC_PHASE_PARAM_COMM] +
                   self._desloc_phases[DESLOC_PHASE_MOM_COMM])
        idle_ms = self._desloc_phases[DESLOC_PHASE_IDLE]
        return {
            'mfu': round(mfu, 6),
            'achieved_tflops': round(achieved_tflops, 4),
            'step_time_ms': round(total / steps, 4),
            'compute_pct': round(100.0 * compute_ms / total, 2),
            'comm_pct': round(100.0 * comm_ms / total, 2),
            'idle_pct': round(100.0 * idle_ms / total, 2),
            'optimizer_pct': round(100.0 * self._desloc_phases[DESLOC_PHASE_OPTIMIZER] / total, 2),
            'total_steps': steps,
        }

    def record_desloc_step(self):
        """Record one DES-LOC training step completed.

        Call this at end of each step to maintain accurate per-step averages.
        Also snapshots per-step phase breakdown for time-series analysis.
        """
        self._desloc_step_count += 1
        # Snapshot current breakdown for history
        if self._desloc_step_count <= 1000:  # cap history to avoid memory leak
            self._desloc_phase_history.append(
                {p: v for p, v in self._desloc_phases.items()})

    def reset_desloc_phases(self):
        """Reset all DES-LOC phase accumulators."""
        for k in self._desloc_phases:
            self._desloc_phases[k] = 0.0
        self._desloc_step_count = 0
        self._desloc_phase_history.clear()

    def log_desloc_phases(self, ranks=None):
        """Log DES-LOC phase breakdown to console.

        Ref: NKI-FA draw_plot.py format — structured metric output.
        """
        breakdown = self.get_desloc_breakdown()
        total = sum(self._desloc_phases.values())
        if total > 0:
            parts = []
            for phase in ALL_DESLOC_PHASES:
                pct = breakdown.get(phase, 0)
                if pct > 0.01:
                    short = phase.replace('desloc_', '')
                    parts.append(f'{short}: {pct:.1f}%')
            msg = 'DES-LOC phases: ' + ' | '.join(parts)
            print_dist(msg, ranks=ranks or [0])

    def get_desloc_comm_fraction(self):
        """Return fraction of time spent in communication.

        This is the key metric for evaluating whether Kx should be increased.
        If comm_fraction is high, a larger Kx would help.
        If comm_fraction is near zero, Kx=1 is fine (compute-bound).
        """
        total = sum(self._desloc_phases.values())
        if total <= 0:
            return 0.0
        comm = (self._desloc_phases[DESLOC_PHASE_PARAM_COMM] +
                self._desloc_phases[DESLOC_PHASE_MOM_COMM])
        return round(comm / total, 6)

    def state_dict(self):
        """Serialize DES-LOC timing state for checkpointing."""
        return {
            'phases': dict(self._desloc_phases),
            'step_count': self._desloc_step_count,
        }

    def load_state_dict(self, sd):
        """Restore DES-LOC timing state from checkpoint."""
        if 'phases' in sd:
            for k, v in sd['phases'].items():
                if k in self._desloc_phases:
                    self._desloc_phases[k] = v
        self._desloc_step_count = sd.get('step_count', 0)


class NoopTimer:
    """No-op timer that silently ignores all timing calls.

    DES-LOC extension: also implements DES-LOC phase methods as no-ops
    so that code paths don't need to check timer type before calling them.
    """

    class Timer:

        def start(self):
            ...

        def reset(self):
            ...

        def stop(self, **kwargs):
            ...

        def elapsed(self, **kwargs):
            return 0

        def mean(self):
            return 0

    def __init__(self):
        self.timer = self.Timer()

    def __call__(self, name):
        return self.timer

    def get_timers(self):
        return {}

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        ...

    def get_mean(self, names, normalizer=1.0, reset=True):
        ...

    # DES-LOC no-op implementations
    def accumulate_desloc_phase(self, phase, elapsed_ms):
        pass

    def get_desloc_breakdown(self):
        return {}

    def get_desloc_mfu_estimate(self, peak_tflops=0, model_flops_per_step=0):
        return {'mfu': 0.0}

    def record_desloc_step(self):
        pass

    def reset_desloc_phases(self):
        pass

    def log_desloc_phases(self, ranks=None):
        pass

    def get_desloc_comm_fraction(self):
        return 0.0


class ThroughputTimer:
    """Training throughput timer with DES-LOC annotations.

    Rewritten to add DES-LOC step-level logging in NKI-FA format
    and to track whether each step was a sync boundary.

    Ref: NKI-FA draw_plot.py — structured 'metric: value' log lines.
    """

    def __init__(self, config, batch_size, start_step=2, steps_per_output=None,
                 monitor_memory=False, logging_fn=None):
        from deepspeed.utils import logger
        self.config = config
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = 1 if batch_size is None else batch_size
        self.start_step = start_step
        self.epoch_count = 0
        self.micro_step_count = 0
        self.global_step_count = 0
        self.total_elapsed_time = 0
        self.step_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            self.logging = logger.info
        self.initialized = False

        if self.monitor_memory and not PSUTILS_INSTALLED:
            raise ImportError("Unable to import 'psutils', please install package")

        # DES-LOC: per-step timing history for NKI-FA style log output
        self._desloc_step_times = []   # list of per-step durations
        self._desloc_is_sync = []      # whether each step was a Kx boundary
        self._desloc_mfu_history = []  # per-step MFU estimates

    def update_epoch_count(self):
        """Increment epoch counter and reset micro step counter."""
        self.epoch_count += 1
        self.micro_step_count = 0

    def _init_timer(self):
        """Mark timer as initialized."""
        self.initialized = True

    def start(self):
        """Start timing a training step."""
        if not self.config.enabled:
            return
        self._init_timer()
        self.started = True
        if self.global_step_count >= self.start_step:
            if self.config.synchronized:
                get_accelerator().synchronize()
            self.start_time = time.time()

    def _is_report_boundary(self):
        """Check if current step should trigger a throughput report."""
        if self.steps_per_output is None:
            return False
        return self.global_step_count % self.steps_per_output == 0

    def stop(self, global_step=False, report_speed=True, desloc_is_sync=None):
        """Stop timing and optionally report throughput.

        DES-LOC extension: accepts desloc_is_sync flag to record whether
        this step was a Kx sync boundary. This enables the per-step
        analysis of sync vs non-sync step timing differences.

        Args:
            global_step: Whether this completes a global step.
            report_speed: Whether to log throughput at report boundaries.
            desloc_is_sync: True if this step did param allreduce.
        """
        if not self.config.enabled or not self.started:
            return
        self.started = False
        self.micro_step_count += 1
        if global_step:
            self.global_step_count += 1

        if self.start_time > 0:
            if self.config.synchronized:
                get_accelerator().synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            self.step_elapsed_time += duration

            # DES-LOC: record per-step timing data
            self._desloc_step_times.append(duration * 1000.0)  # convert to ms
            if desloc_is_sync is not None:
                self._desloc_is_sync.append(desloc_is_sync)

            if global_step:
                if report_speed and self._is_report_boundary():
                    self.logging(
                        "epoch={}/micro_step={}/global_step={}, "
                        "RunningAvgSamplesPerSec={}, CurrSamplesPerSec={}, "
                        "MemAllocated={}GB, MaxMemAllocated={}GB".format(
                            self.epoch_count,
                            self.micro_step_count,
                            self.global_step_count,
                            self.avg_samples_per_sec(),
                            self.batch_size / (self.step_elapsed_time + TIME_EPSILON),
                            round(get_accelerator().memory_allocated() / 1024**3, 2),
                            round(get_accelerator().max_memory_allocated() / 1024**3, 2),
                        ))
                    if self.monitor_memory:
                        virt_mem = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        self.logging(
                            "epoch={}/micro_step={}/global_step={}, "
                            "vm %: {}, swap %: {}".format(
                                self.epoch_count,
                                self.micro_step_count,
                                self.global_step_count,
                                virt_mem.percent,
                                swap.percent,
                            ))
                self.step_elapsed_time = 0

    def avg_samples_per_sec(self):
        """Calculate running average training throughput."""
        if self.global_step_count > 0:
            total_step_offset = self.global_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            return self.batch_size / avg_time_per_step
        return float("-inf")

    @property
    def desloc_step_time_ms(self):
        """Return last step duration in ms for DES-LOC logging.

        Ref: NKI-FA draw_plot.py format — 'step_time_ms: value'.
        """
        return self.step_elapsed_time * 1000.0

    def get_desloc_sync_timing_analysis(self):
        """Compare step times for sync vs non-sync steps.

        DES-LOC insight: non-sync steps should be faster because they
        skip allreduce. The ratio reveals communication overhead.

        Returns dict with sync/non-sync step time averages.
        Ref: Section 5.4 — training speedup from skipping allreduce.
        """
        if not self._desloc_step_times or not self._desloc_is_sync:
            return None
        n = min(len(self._desloc_step_times), len(self._desloc_is_sync))
        sync_times = [self._desloc_step_times[i] for i in range(n)
                      if self._desloc_is_sync[i]]
        nosync_times = [self._desloc_step_times[i] for i in range(n)
                        if not self._desloc_is_sync[i]]
        if not sync_times or not nosync_times:
            return None
        avg_sync = sum(sync_times) / len(sync_times)
        avg_nosync = sum(nosync_times) / len(nosync_times)
        return {
            'avg_sync_step_ms': round(avg_sync, 4),
            'avg_nosync_step_ms': round(avg_nosync, 4),
            'comm_overhead_ms': round(avg_sync - avg_nosync, 4),
            'comm_overhead_pct': round(100 * (avg_sync - avg_nosync) / avg_sync, 2) if avg_sync > 0 else 0,
            'n_sync_steps': len(sync_times),
            'n_nosync_steps': len(nosync_times),
        }


def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Rewritten to use pure Python (no numpy dependency).
    Ref: M167 rule — zero numpy.random/numpy imports in DES-LOC code.

    Args:
        data (list): List of numbers.
        trim_percent (float): Fraction of data to trim from each end.

    Returns:
        float: Trimmed mean, or 0 for empty input.
    """
    assert 0.0 <= trim_percent <= 1.0
    n = len(data)
    if n == 0:
        return 0
    data.sort()
    k = int(round(n * trim_percent))
    trimmed = data[k:n - k]
    if not trimmed:
        return 0
    return sum(trimmed) / len(trimmed)


# =========================================================================

# =========================================================================
# DES-LOC Roofline: compute vs comm bound estimation (Section 4.1)
# =========================================================================

def desloc_roofline_bound(flops_per_step, bytes_per_allreduce,
                          peak_tflops, bandwidth_gbps, Kx=32):
    """Determine if training is compute-bound or comm-bound under DES-LOC.
    Ref: Section 4.1 + Appendix G.1 wall-clock model.
    Returns (bound_type, compute_time_ms, comm_time_ms)."""
    compute_ms = flops_per_step / (peak_tflops * 1e9)  # TFLOPS -> GFLOPS/ms
    comm_ms = bytes_per_allreduce / (bandwidth_gbps * 1e6 / 8.0) / Kx
    bound = 'compute' if compute_ms > comm_ms else 'comm'
    return bound, compute_ms, comm_ms


def desloc_mfu(actual_tflops, peak_tflops):
    """Model FLOPs Utilization. Ref: PaLM paper definition."""
    if peak_tflops <= 0:
        return 0.0
    return actual_tflops / peak_tflops

# Hardware roofline model for MFU computation
class DeslocRooflineModel:
    """Simple roofline: peak_tflops, mem_bw (GB/s), interconnect_bw (GB/s)."""
    def __init__(self, peak_tflops, mem_bw_gbs, interconnect_bw_gbs):
        self.peak_tflops = peak_tflops
        self.mem_bw_gbs = mem_bw_gbs
        self.interconnect_bw_gbs = interconnect_bw_gbs
    def mfu(self, achieved_tflops):
        return achieved_tflops / self.peak_tflops if self.peak_tflops > 0 else 0.0

# Hardware presets for common GPU configurations
DESLOC_HW_PRESETS = {
    'H100_NVL': DeslocRooflineModel(312, 3350, 600),
    'H100_PCIe': DeslocRooflineModel(312, 3350, 32),
    'A100_80GB': DeslocRooflineModel(312, 2039, 600),
    'A6000': DeslocRooflineModel(77.4, 768, 32),
    'Trainium2': DeslocRooflineModel(380, 3200, 100),
}


# =============================================================================
# M232 (Claude-15): Training Time vs Comm Time Separator for Figure 2
# Ref: Section 5.3 — throughput speedup = f(compute_time, comm_time)
# Ref: Nick Joseph — "model with pen-and-paper, verify with profiler"
# Ref: NKI-FA draw_plot.py — Time (ms) bar chart
# =============================================================================




# M284: Phase timer convenience
def desloc_time_step(fn, phase_timer=None, phase="compute"):
    import time
    t0 = time.monotonic(); result = fn(); elapsed = time.monotonic()-t0
    if phase_timer: phase_timer.start(phase); phase_timer.stop()
    return result, elapsed*1000

def desloc_format_breakdown(bd):
    total = sum(bd.values())
    if total <= 0: return "no data"
    return " | ".join(f"{p}={bd.get(p,0):.1f}ms ({bd.get(p,0)/total*100:.1f}%)"
                      for p in ["compute","comm","idle"])

# M299 — Claude-19: PhaseTimer + Roofline + MFU
import time as _t299


# M314: MFU + roofline + progress
import time as _tm; import math as _tmath

class DeslocSTimer:
    def __init__(self): self._ss = 0; self._cp = None; self._ps = 0; self._pt = {}; self._st = []; self._w = 100
    def start_step(self): self._ss = _tm.perf_counter_ns()
    def end_step(self):
        e = _tm.perf_counter_ns() - self._ss; self._st.append(e)
        if len(self._st) > self._w: self._st.pop(0)
        return e / 1e6
    def start_ph(self, n): self._cp = n; self._ps = _tm.perf_counter_ns()
    def end_ph(self):
        if self._cp is None: return
        self._pt[self._cp] = self._pt.get(self._cp, 0) + _tm.perf_counter_ns() - self._ps; self._cp = None
    def avg_ms(self): return sum(self._st) / max(1, len(self._st)) / 1e6 if self._st else 0
    def breakdown(self):
        t = sum(self._pt.values())
        return {p: {'ms': round(n / 1e6, 4), '%': round(100 * n / max(1, t), 2)} for p, n in self._pt.items()} if t > 0 else {}
    def reset(self): self._pt.clear(); self._st.clear()

def desloc_mfu(ach, peak): return round(ach / max(.01, peak), 4) if peak > 0 else 0

def desloc_roof(flops, bytesacc, peak_tf, peak_bw):
    if bytesacc <= 0 or peak_bw <= 0: return ('compute', 1.0, 0)
    ai = flops / bytesacc; ridge = (peak_tf * 1e12) / (peak_bw * 1e9)
    if ai >= ridge: return ('compute', round(flops / max(1, peak_tf * 1e12), 4), round(ridge, 2))
    a = ai * peak_bw * 1e9; return ('memory', round(a / max(1, peak_tf * 1e12), 4), round(ridge, 2))

def desloc_h_mfu(profiles, times):
    pd = {}; ta = 0; tp = 0
    for p in profiles:
        idx = p.get('idx', 0); pk = p.get('tf', 50); tp += pk
        if idx in times:
            m = times[idx].get('mfu', 0); a = pk * m; ta += a
            pd[idx] = {'gpu': p.get('name', '?'), 'mfu': round(m, 4), 'ach': round(a, 2)}
    return {'pd': pd, 'cmfu': round(ta / max(.01, tp), 4), 'ach': round(ta, 2), 'peak': round(tp, 2)}

class DeslocProgress:
    def __init__(self): self._l = []; self._m = []; self._t = []; self._c = []
    def record(self, step, loss=None, mfu=None, tps=None, cb=None):
        if loss is not None: self._l.append((step, loss))
        if mfu is not None: self._m.append((step, mfu))
        if tps is not None: self._t.append((step, tps))
        if cb is not None: self._c.append((step, cb))
    def summary(self):
        def _a(l): return sum(x[1] for x in l) / max(1, len(l)) if l else 0
        return {'loss': round(_a(self._l), 6), 'mfu': round(_a(self._m), 4), 'tps': round(_a(self._t), 1), 'n': max(len(self._l), len(self._m))}
# --- End M314 ---


# =====================================================================
# M338 — Claude-30: FlashAttention-Style CUDA Event Timing
# Source: flash-attention/flash_attn/utils/benchmark.py:8 (benchmark_forward)
#         flash-attention/flash_attn/utils/benchmark.py:30 (benchmark_backward)
#         flash-attention/flash_attn/utils/benchmark.py:258 (benchmark_memory)
# Adapted: CUDA event-based timing for DES-LOC step profiling with
#          warmup iterations and statistical aggregation
# =====================================================================

class DeslocCudaEventTimer:
    """High-precision CUDA event timer for DES-LOC step profiling.

    Adapted from FlashAttention benchmark_forward() (flash_attn/utils/benchmark.py:8)
    which uses torch.utils.benchmark.Timer with CUDA synchronization.
    This implementation uses raw CUDA events for sub-microsecond precision
    without the torch.utils.benchmark overhead, suitable for per-step
    measurement during training (not just offline benchmarking).

    The key insight from FA benchmark.py is: always warmup before timing,
    always synchronize, and collect enough samples for statistical validity.
    """

    def __init__(self, warmup_iters=10, device='cuda'):
        self._warmup = warmup_iters
        self._device = device
        self._iter_count = 0
        self._phase_times = {}
        self._step_times = []
        self._current_phase = None
        self._phase_start_event = None
        self._phase_end_event = None
        self._step_start_ns = 0
        self._cuda_available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._cuda_available = True
        except ImportError:
            pass

    def _create_events(self):
        """Create CUDA events for timing (FA pattern: enable_timing=True)."""
        if self._cuda_available:
            import torch
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            return start, end
        return None, None

    def start_step(self):
        """Begin timing a training step."""
        self._iter_count += 1
        self._step_start_ns = _tm.perf_counter_ns()
        if self._cuda_available:
            import torch
            torch.cuda.synchronize()

    def end_step(self):
        """End timing a training step and record duration."""
        if self._cuda_available:
            import torch
            torch.cuda.synchronize()
        dur_ns = _tm.perf_counter_ns() - self._step_start_ns
        if self._iter_count > self._warmup:
            self._step_times.append(dur_ns / 1e6)  # store in ms
        return dur_ns / 1e6

    def start_phase(self, phase_name):
        """Start timing a sub-phase (forward, backward, comm, etc.).

        Adapted from FA benchmark_forward/benchmark_backward pattern:
        each phase gets its own timing with CUDA sync boundaries.
        """
        self._current_phase = phase_name
        if self._cuda_available:
            import torch
            self._phase_start_event, self._phase_end_event = self._create_events()
            torch.cuda.synchronize()
            if self._phase_start_event:
                self._phase_start_event.record()
        else:
            self._phase_start_event = _tm.perf_counter_ns()

    def end_phase(self):
        """End timing current sub-phase."""
        if self._current_phase is None:
            return 0.0
        phase = self._current_phase
        if self._cuda_available and self._phase_end_event is not None:
            self._phase_end_event.record()
            import torch
            torch.cuda.synchronize()
            dur_ms = self._phase_start_event.elapsed_time(self._phase_end_event)
        else:
            dur_ms = (_tm.perf_counter_ns() - (self._phase_start_event or 0)) / 1e6

        if self._iter_count > self._warmup:
            if phase not in self._phase_times:
                self._phase_times[phase] = []
            self._phase_times[phase].append(dur_ms)
        self._current_phase = None
        return dur_ms

    def get_phase_stats(self, phase_name=None):
        """Get statistical summary for a phase or all phases.

        Returns mean, std, min, max, p50, p99 — matching FA benchmark
        output format for NeurIPS-quality reporting.
        """
        def _stats(times):
            if not times:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'p50': 0, 'p99': 0, 'n': 0}
            n = len(times)
            s = sorted(times)
            mean = sum(s) / n
            var = sum((x - mean) ** 2 for x in s) / max(1, n - 1)
            return {
                'mean': round(mean, 4),
                'std': round(var ** 0.5, 4),
                'min': round(s[0], 4),
                'max': round(s[-1], 4),
                'p50': round(s[n // 2], 4),
                'p99': round(s[min(n - 1, int(n * 0.99))], 4),
                'n': n,
            }
        if phase_name:
            return _stats(self._phase_times.get(phase_name, []))
        return {p: _stats(t) for p, t in self._phase_times.items()}

    def get_step_stats(self):
        """Get step-level timing statistics."""
        return self.get_phase_stats.__func__(None, {'step': self._step_times}).get('step', {})

    def is_warmed_up(self):
        return self._iter_count > self._warmup


# =====================================================================
# M338 — Claude-30: Benchmark Memory Tracker
# Source: flash-attention/flash_attn/utils/benchmark.py:258 (benchmark_memory)
# Adapted: Peak memory tracking per training step for DES-LOC
# =====================================================================

class DeslocMemoryTracker:
    """Peak memory tracker adapted from FlashAttention benchmark_memory().

    FA pattern: torch.cuda.reset_peak_memory_stats() before operation,
    torch.cuda.max_memory_allocated() after. Extended for per-step
    tracking with allocation delta reporting.
    """

    def __init__(self):
        self._peak_samples = []
        self._alloc_samples = []
        self._cuda_available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._cuda_available = True
        except ImportError:
            pass

    def snapshot_before(self):
        """Take memory snapshot before an operation (FA benchmark_memory pattern)."""
        if not self._cuda_available:
            return 0
        import torch
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()

    def snapshot_after(self, before_alloc=0):
        """Take memory snapshot after operation and compute delta."""
        if not self._cuda_available:
            return {'peak_mb': 0, 'delta_mb': 0, 'alloc_mb': 0}
        import torch
        peak = torch.cuda.max_memory_allocated()
        current = torch.cuda.memory_allocated()
        delta = current - before_alloc
        result = {
            'peak_mb': round(peak / 1e6, 2),
            'delta_mb': round(delta / 1e6, 2),
            'alloc_mb': round(current / 1e6, 2),
        }
        self._peak_samples.append(peak / 1e6)
        self._alloc_samples.append(current / 1e6)
        return result

    def get_memory_stats(self):
        if not self._peak_samples:
            return {'avg_peak_mb': 0, 'max_peak_mb': 0, 'n': 0}
        return {
            'avg_peak_mb': round(sum(self._peak_samples) / len(self._peak_samples), 2),
            'max_peak_mb': round(max(self._peak_samples), 2),
            'avg_alloc_mb': round(sum(self._alloc_samples) / len(self._alloc_samples), 2),
            'n': len(self._peak_samples),
        }


# =====================================================================
# M338 — Claude-30: Per-Step MFU Calculator with Roofline
# Source: TransformerEngine/benchmarks/attention/benchmark_dot_product_attention.py:44
#         Megatron-LM/megatron/core/transformer/cuda_graphs.py (is_graph_capturing)
# Adapted: Step-level MFU with automatic GPU detection and roofline analysis
# =====================================================================

class DeslocStepMFUCalculator:
    """Per-step MFU calculator with roofline model.

    Adapted from TransformerEngine benchmark_dot_product_attention.py
    TFLOPS measurement pattern. Computes achieved TFLOPS per step
    and compares against theoretical peak for the detected GPU.

    Also integrates Megatron cuda_graphs.py is_graph_capturing() check
    to skip MFU measurement during CUDA graph capture phases.
    """

    # GPU peak TFLOPS database (extended from M336 constants)
    _GPU_SPECS = {
        'A100': {'bf16': 312.0, 'fp32': 19.5, 'hbm_bw_gbps': 2039},
        'H100': {'bf16': 989.5, 'fp32': 67.0, 'hbm_bw_gbps': 3352},
        'A6000': {'bf16': 38.7, 'fp32': 38.7, 'hbm_bw_gbps': 768},
        'L40': {'bf16': 181.0, 'fp32': 90.5, 'hbm_bw_gbps': 864},
        'RTX_4090': {'bf16': 165.2, 'fp32': 82.6, 'hbm_bw_gbps': 1008},
        'V100': {'fp16': 125.0, 'fp32': 15.7, 'hbm_bw_gbps': 900},
        'H20': {'bf16': 148.0, 'fp16': 148.0, 'fp32': 39.5, 'fp8': 296.0, 'hbm_bw_gbps': 4000},
    }

    def __init__(self, n_params, gpu_name=None, dtype='bf16', seq_len=2048, batch_size=1):
        self.n_params = n_params
        self.dtype = dtype
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.gpu_name = gpu_name or self._detect_gpu()
        specs = self._GPU_SPECS.get(self.gpu_name, {'bf16': 312.0, 'hbm_bw_gbps': 2039})
        self.peak_tflops = specs.get(dtype, specs.get('bf16', 312.0))
        self.hbm_bw_gbps = specs.get('hbm_bw_gbps', 2039)
        self._mfu_history = []
        self._tflops_history = []

    def _detect_gpu(self):
        """Auto-detect GPU name (Megatron cuda_graphs.py pattern)."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                for key in self._GPU_SPECS:
                    if key.replace('_', ' ').lower() in name.lower():
                        return key
                if 'H100' in name or 'H800' in name:
                    return 'H100'
                if 'A100' in name:
                    return 'A100'
                if 'A6000' in name or 'RTX A6000' in name:
                    return 'A6000'
        except Exception:
            pass
        return 'A100'  # default fallback

    def compute_step_mfu(self, step_time_ms, tokens_per_step=None):
        """Compute MFU for a single training step.

        FLOPS formula (6ND rule from Kaplan et al.):
        FLOPS_per_step = 6 * N_params * tokens_per_step (forward + backward)

        MFU = achieved_TFLOPS / peak_TFLOPS

        Args:
            step_time_ms: Wall-clock time for one step in milliseconds
            tokens_per_step: Number of tokens processed. If None, uses
                batch_size * seq_len.

        Returns:
            Dict with mfu, achieved_tflops, peak_tflops, tokens
        """
        if tokens_per_step is None:
            tokens_per_step = self.batch_size * self.seq_len
        if step_time_ms <= 0:
            return {'mfu': 0.0, 'achieved_tflops': 0.0, 'peak_tflops': self.peak_tflops}

        step_time_s = step_time_ms / 1000.0
        flops = 6 * self.n_params * tokens_per_step
        achieved_tflops = (flops / step_time_s) / 1e12

        mfu = achieved_tflops / self.peak_tflops if self.peak_tflops > 0 else 0.0

        self._mfu_history.append(mfu)
        self._tflops_history.append(achieved_tflops)

        return {
            'mfu': round(mfu, 4),
            'achieved_tflops': round(achieved_tflops, 2),
            'peak_tflops': self.peak_tflops,
            'tokens': tokens_per_step,
            'gpu': self.gpu_name,
        }

    def roofline_analysis(self, step_time_ms, bytes_accessed=None, tokens_per_step=None):
        """Full roofline analysis combining compute and memory bounds.

        Extends desloc_roof() from M314 with per-step tracking and
        automatic bytes estimation from model size.

        Args:
            step_time_ms: Step wall-clock time
            bytes_accessed: Total bytes accessed from HBM. If None,
                estimates from 2 * n_params * dtype_bytes (read weights + write grads)
            tokens_per_step: Tokens per step

        Returns:
            Dict with bottleneck type, MFU, arithmetic intensity, ridge point
        """
        if tokens_per_step is None:
            tokens_per_step = self.batch_size * self.seq_len
        if bytes_accessed is None:
            dtype_bytes = 2 if self.dtype in ('bf16', 'fp16') else 4
            bytes_accessed = 2 * self.n_params * dtype_bytes * 3

        step_time_s = step_time_ms / 1000.0 if step_time_ms > 0 else 1e-9
        flops = 6 * self.n_params * tokens_per_step
        achieved_tflops = (flops / step_time_s) / 1e12
        achieved_bw = (bytes_accessed / step_time_s) / 1e9  # GB/s

        if bytes_accessed > 0:
            arithmetic_intensity = flops / bytes_accessed  # FLOP/byte
        else:
            arithmetic_intensity = float('inf')

        ridge_point = (self.peak_tflops * 1e12) / (self.hbm_bw_gbps * 1e9)
        if arithmetic_intensity >= ridge_point:
            bottleneck = 'compute'
            utilization = achieved_tflops / self.peak_tflops if self.peak_tflops > 0 else 0
        else:
            bottleneck = 'memory'
            utilization = achieved_bw / self.hbm_bw_gbps if self.hbm_bw_gbps > 0 else 0

        return {
            'bottleneck': bottleneck,
            'utilization': round(utilization, 4),
            'mfu': round(achieved_tflops / self.peak_tflops, 4) if self.peak_tflops > 0 else 0,
            'arithmetic_intensity': round(arithmetic_intensity, 2),
            'ridge_point': round(ridge_point, 2),
            'achieved_tflops': round(achieved_tflops, 2),
            'achieved_bw_gbps': round(achieved_bw, 2),
        }

    def get_mfu_summary(self):
        """Get aggregate MFU statistics across all recorded steps."""
        if not self._mfu_history:
            return {'mean_mfu': 0, 'std_mfu': 0, 'n': 0}
        n = len(self._mfu_history)
        mean = sum(self._mfu_history) / n
        var = sum((x - mean) ** 2 for x in self._mfu_history) / max(1, n - 1)
        return {
            'mean_mfu': round(mean, 4),
            'std_mfu': round(var ** 0.5, 4),
            'min_mfu': round(min(self._mfu_history), 4),
            'max_mfu': round(max(self._mfu_history), 4),
            'mean_tflops': round(sum(self._tflops_history) / n, 2),
            'gpu': self.gpu_name,
            'peak_tflops': self.peak_tflops,
            'n': n,
        }


# =====================================================================
# M338 — Claude-30: DES-LOC Comm/Compute Phase Profiler
# Source: Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
#         (overlap_grad_reduce hooks in __init__)
# Adapted: Track comm vs compute time per step to measure DES-LOC
#          communication reduction impact on actual training throughput
# =====================================================================

class DeslocPhaseProfiler:
    """Profiler tracking comm vs compute phases per training step.

    Adapted from Megatron DDP overlap_grad_reduce hook pattern where
    backward computation and gradient reduction overlap via CUDA streams.
    This profiler measures the actual overlap achieved by DES-LOC gating.

    Phases tracked: forward, backward, comm (AllReduce/ReduceScatter),
    optimizer, data_loading, idle.
    """

    PHASES = ('forward', 'backward', 'comm', 'optimizer', 'data_loading', 'idle')

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._step_data = []
        self._current_step = {}
        self._current_phase = None
        self._phase_start_ns = 0

    def begin_step(self, step_num):
        """Begin a new step for profiling."""
        if not self.enabled:
            return
        self._current_step = {'step': step_num}
        for p in self.PHASES:
            self._current_step[p + '_ms'] = 0.0

    def begin_phase(self, phase):
        """Begin timing a phase within current step."""
        if not self.enabled or phase not in self.PHASES:
            return
        self._current_phase = phase
        self._phase_start_ns = _tm.perf_counter_ns()

    def end_phase(self):
        """End current phase timing."""
        if not self.enabled or self._current_phase is None:
            return
        dur_ms = (_tm.perf_counter_ns() - self._phase_start_ns) / 1e6
        key = self._current_phase + '_ms'
        self._current_step[key] = self._current_step.get(key, 0) + dur_ms
        self._current_phase = None

    def end_step(self):
        """Finalize step profiling data."""
        if not self.enabled:
            return
        total = sum(self._current_step.get(p + '_ms', 0) for p in self.PHASES)
        self._current_step['total_ms'] = round(total, 4)
        if total > 0:
            self._current_step['comm_fraction'] = round(
                self._current_step.get('comm_ms', 0) / total, 4
            )
            self._current_step['compute_fraction'] = round(
                (self._current_step.get('forward_ms', 0) +
                 self._current_step.get('backward_ms', 0)) / total, 4
            )
        self._step_data.append(self._current_step)

    def get_phase_breakdown(self):
        """Get average phase breakdown across all recorded steps."""
        if not self._step_data:
            return {}
        n = len(self._step_data)
        result = {}
        for p in self.PHASES:
            key = p + '_ms'
            vals = [s.get(key, 0) for s in self._step_data]
            result[p] = {
                'mean_ms': round(sum(vals) / n, 4),
                'total_ms': round(sum(vals), 2),
                'fraction': round(sum(vals) / max(1, sum(
                    sum(s.get(pp + '_ms', 0) for pp in self.PHASES)
                    for s in self._step_data
                )), 4),
            }
        return result

    def comm_reduction_impact(self):
        """Measure how comm reduction from DES-LOC affects step time.

        Computes what fraction of step time is communication, and
        estimates speedup if comm were reduced by DES-LOC factor.
        """
        if not self._step_data:
            return {'avg_comm_frac': 0, 'avg_step_ms': 0}
        n = len(self._step_data)
        comm_fracs = [s.get('comm_fraction', 0) for s in self._step_data]
        step_times = [s.get('total_ms', 0) for s in self._step_data]
        return {
            'avg_comm_frac': round(sum(comm_fracs) / n, 4),
            'avg_step_ms': round(sum(step_times) / n, 4),
            'n_steps': n,
        }

    def to_nkifa_log(self):
        """Export phase data in NKI-FA format for draw_plot.py."""
        lines = []
        for s in self._step_data:
            step = s.get('step', 0)
            lines.append(f"### phase_profile, step={step} ###")
            for p in self.PHASES:
                lines.append(f"{p}_ms: {s.get(p + '_ms', 0):.4f}")
            lines.append(f"total_ms: {s.get('total_ms', 0):.4f}")
            lines.append(f"comm_fraction: {s.get('comm_fraction', 0):.4f}")
            lines.append("")
        return '\n'.join(lines)
# --- End M338 ---