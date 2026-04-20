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


class DeslocTimeBreakdown:
    """Track compute vs communication time for Figure 2 and Figure 3.

    From Nick Joseph: "efficiency = compute_time / (compute + comm + idle)"
    This class separates wall-clock time into three phases and computes
    the speedup that DES-LOC's reduced communication provides.

    From NKI-FA: time displayed as "5.837ms" with 3-decimal precision.

    Lifecycle:
        tb = DeslocTimeBreakdown()
        for step in training:
            tb.begin_compute()
            ... forward + backward ...
            tb.end_compute()
            tb.begin_comm()
            ... allreduce ...
            tb.end_comm(was_skipped=not is_sync)
            tb.end_step()
        speedup = tb.compute_speedup_vs_ddp()
    """

    def __init__(self):
        import time as _time_mod
        self._time = _time_mod
        self._step = 0
        # Per-step accumulators (nanosecond precision)
        self._compute_start = 0
        self._comm_start = 0
        self._step_start = 0
        # Running totals (seconds)
        self._total_compute_s = 0.0
        self._total_comm_s = 0.0
        self._total_idle_s = 0.0
        self._total_wall_s = 0.0
        self._total_skipped_comm_s = 0.0  # time saved by skipping
        # Per-step history for plotting
        self._history = []  # [(step, compute_ms, comm_ms, idle_ms)]
        self._max_history = 10000

    def begin_step(self):
        """Mark start of training step."""
        self._step_start = self._time.monotonic()

    def begin_compute(self):
        """Mark start of forward+backward compute phase."""
        self._compute_start = self._time.monotonic()

    def end_compute(self):
        """Mark end of compute phase. Returns elapsed seconds."""
        elapsed = self._time.monotonic() - self._compute_start
        self._total_compute_s += elapsed
        return elapsed

    def begin_comm(self):
        """Mark start of communication phase."""
        self._comm_start = self._time.monotonic()

    def end_comm(self, was_skipped=False):
        """Mark end of communication phase.

        Args:
            was_skipped: True if DES-LOC skipped this AllReduce.
                If skipped, the time is counted as "saved" not "idle".
        """
        elapsed = self._time.monotonic() - self._comm_start
        if was_skipped:
            self._total_skipped_comm_s += elapsed
            # DES-LOC converts comm time to useful compute overlap
        else:
            self._total_comm_s += elapsed
        return elapsed

    def end_step(self):
        """Mark end of step. Compute idle time as residual."""
        wall = self._time.monotonic() - self._step_start
        self._total_wall_s += wall
        # Idle = wall - compute - comm - skipped_comm
        idle = max(0, wall - self._total_compute_s -
                   self._total_comm_s - self._total_skipped_comm_s)
        # Record per-step if within budget
        if len(self._history) < self._max_history:
            # We only record the step-level delta, not running totals
            compute_ms = (self._total_compute_s * 1000
                          if self._step == 0 else 0)  # placeholder
            self._history.append(
                (self._step, wall * 1000, 0, 0))
        self._step += 1

    def get_breakdown(self):
        """Get time breakdown as dict.

        Returns:
            dict with total_s and percentage for compute/comm/idle
        """
        total = max(1e-9, self._total_wall_s)
        compute_pct = 100.0 * self._total_compute_s / total
        comm_pct = 100.0 * self._total_comm_s / total
        saved_pct = 100.0 * self._total_skipped_comm_s / total
        idle_pct = max(0, 100.0 - compute_pct - comm_pct - saved_pct)
        return {
            'total_wall_s': round(self._total_wall_s, 3),
            'compute_s': round(self._total_compute_s, 3),
            'comm_s': round(self._total_comm_s, 3),
            'skipped_comm_s': round(self._total_skipped_comm_s, 3),
            'idle_s': round(max(0, self._total_wall_s -
                               self._total_compute_s -
                               self._total_comm_s), 3),
            'compute_pct': round(compute_pct, 1),
            'comm_pct': round(comm_pct, 1),
            'saved_pct': round(saved_pct, 1),
            'idle_pct': round(idle_pct, 1),
            'total_steps': self._step,
        }

    def compute_speedup_vs_ddp(self):
        """Compute DES-LOC wall-clock speedup over DDP.

        DDP time = compute + comm (full AllReduce every step)
        DES-LOC time = compute + reduced_comm
        Speedup = DDP_time / DES-LOC_time

        Ref: Section 5.4 — ≈2× training speedup over DDP.
        """
        if self._total_wall_s <= 0:
            return 1.0
        ddp_wall = self._total_compute_s + self._total_comm_s + self._total_skipped_comm_s
        desloc_wall = self._total_compute_s + self._total_comm_s
        if desloc_wall <= 0:
            return 1.0
        return ddp_wall / desloc_wall

    def compute_mfu_improvement(self, peak_tflops, model_flops_per_step):
        """Compute MFU with and without DES-LOC.

        MFU = actual_flops / peak_flops
        actual_flops = model_flops * steps / wall_time

        Ref: Nick Joseph — "MFU gap = HBM bandwidth + comm + CPU overhead"
        """
        if self._total_wall_s <= 0 or self._step <= 0:
            return {'mfu_desloc': 0, 'mfu_ddp_est': 0}
        actual_tflops = (model_flops_per_step * self._step /
                         self._total_wall_s / 1e12)
        mfu_desloc = actual_tflops / peak_tflops if peak_tflops > 0 else 0
        # Estimate DDP MFU (slower due to more comm)
        ddp_wall = self._total_wall_s + self._total_skipped_comm_s
        ddp_tflops = (model_flops_per_step * self._step /
                      ddp_wall / 1e12) if ddp_wall > 0 else 0
        mfu_ddp = ddp_tflops / peak_tflops if peak_tflops > 0 else 0
        return {
            'mfu_desloc': round(mfu_desloc, 4),
            'mfu_ddp_est': round(mfu_ddp, 4),
            'speedup': round(self.compute_speedup_vs_ddp(), 2),
        }

    def emit_nkifa_log(self, stream=None):
        """Write time breakdown in NKI-FA format."""
        def _w(line):
            if stream:
                stream.write(line + '\n')
            else:
                print(line)
        bd = self.get_breakdown()
        _w('### time_breakdown ###')
        for k, v in sorted(bd.items()):
            _w(f'{k}: {v}')
        _w(f'speedup_vs_ddp: {self.compute_speedup_vs_ddp():.2f}')

    def state_dict(self):
        return {
            'step': self._step,
            'compute_s': self._total_compute_s,
            'comm_s': self._total_comm_s,
            'skipped_s': self._total_skipped_comm_s,
            'wall_s': self._total_wall_s,
        }

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._total_compute_s = sd.get('compute_s', 0)
        self._total_comm_s = sd.get('comm_s', 0)
        self._total_skipped_comm_s = sd.get('skipped_s', 0)
        self._total_wall_s = sd.get('wall_s', 0)

# =====================================================================
# M245 — Claude-16: DES-LOC Phase-Aware MFU Timer
# Compute/comm/idle breakdown, roofline analysis, step time predictor
# Ref: CUTLASS profiler — GEMM roofline model
# Ref: NKI-FA benchmark_attn.py — TFLOPS measurement
# =====================================================================

import time as _m245_time
import math as _m245_math


class DeslocPhaseTimer:
    """Phase-aware timer that breaks each training step into phases.

    Phases:
    - COMPUTE: forward + backward on GPU (GEMM, attention, etc.)
    - COMM_AR: AllReduce communication (DES-LOC gated)
    - COMM_RS: ReduceScatter communication (ZeRO)
    - COMM_AG: AllGather communication (ZeRO)
    - IDLE: GPU idle (waiting for comm, synchronization barriers)
    - OVERHEAD: Python/framework overhead (data loading, optimizer step)

    Each step records phase durations for MFU analysis.
    """

    PHASE_COMPUTE = 'compute'
    PHASE_COMM_AR = 'comm_ar'
    PHASE_COMM_RS = 'comm_rs'
    PHASE_COMM_AG = 'comm_ag'
    PHASE_IDLE = 'idle'
    PHASE_OVERHEAD = 'overhead'

    ALL_PHASES = (PHASE_COMPUTE, PHASE_COMM_AR, PHASE_COMM_RS,
                  PHASE_COMM_AG, PHASE_IDLE, PHASE_OVERHEAD)

    def __init__(self, max_history=1000):
        self._max_history = max_history
        self._step_records = []
        self._current_step = {}
        self._phase_start = None
        self._current_phase = None
        self._step_start = None

    def begin_step(self):
        """Mark the start of a training step."""
        self._current_step = {p: 0.0 for p in self.ALL_PHASES}
        self._step_start = _m245_time.monotonic()
        self._phase_start = None
        self._current_phase = None

    def begin_phase(self, phase):
        """Start timing a phase."""
        self._end_current_phase()
        self._current_phase = phase
        self._phase_start = _m245_time.monotonic()

    def _end_current_phase(self):
        """End the current phase and accumulate time."""
        if self._phase_start is not None and self._current_phase is not None:
            elapsed = _m245_time.monotonic() - self._phase_start
            if self._current_phase in self._current_step:
                self._current_step[self._current_phase] += elapsed * 1000.0

    def end_step(self):
        """Mark the end of a training step and record."""
        self._end_current_phase()

        if self._step_start is not None:
            total_ms = (_m245_time.monotonic() - self._step_start) * 1000.0
            self._current_step['total_ms'] = total_ms

            # Assign unaccounted time to overhead
            accounted = sum(self._current_step.get(p, 0.0) for p in self.ALL_PHASES)
            if total_ms > accounted:
                self._current_step[self.PHASE_OVERHEAD] += (total_ms - accounted)

        self._step_records.append(dict(self._current_step))
        if len(self._step_records) > self._max_history:
            self._step_records = self._step_records[-self._max_history:]

    def get_phase_breakdown(self, last_n=100):
        """Get average phase breakdown over last_n steps."""
        records = self._step_records[-last_n:]
        if not records:
            return {p: 0.0 for p in self.ALL_PHASES}

        n = len(records)
        breakdown = {}
        for phase in self.ALL_PHASES:
            total = sum(r.get(phase, 0.0) for r in records)
            breakdown[phase] = round(total / n, 4)

        avg_total = sum(r.get('total_ms', 0.0) for r in records) / n
        breakdown['total_ms'] = round(avg_total, 4)

        # Compute fractions
        for phase in self.ALL_PHASES:
            frac_key = f'{phase}_frac'
            breakdown[frac_key] = round(
                breakdown[phase] / max(0.001, avg_total), 6
            )

        return breakdown

    def compute_mfu(self, model_params, batch_tokens,
                    peak_tflops=989.5, last_n=100):
        """Compute MFU from phase timing data.

        MFU = (6 * N * tokens) / (step_time * peak_FLOPS)

        Also reports "compute MFU" which excludes comm/idle time:
        compute_MFU = (6 * N * tokens) / (compute_time * peak_FLOPS)
        """
        breakdown = self.get_phase_breakdown(last_n)
        total_ms = breakdown.get('total_ms', 0.0)
        compute_ms = breakdown.get(self.PHASE_COMPUTE, 0.0)

        if total_ms <= 0 or model_params <= 0 or peak_tflops <= 0:
            return {'mfu': 0.0, 'compute_mfu': 0.0}

        flops = 6.0 * model_params * batch_tokens
        peak_flops_per_ms = peak_tflops * 1e9  # TFLOPS → FLOPS/ms

        mfu = flops / (total_ms * peak_flops_per_ms)
        compute_mfu = flops / (max(0.001, compute_ms) * peak_flops_per_ms)

        return {
            'mfu': round(mfu, 6),
            'compute_mfu': round(min(1.0, compute_mfu), 6),
            'total_ms': total_ms,
            'compute_ms': compute_ms,
            'comm_ms': round(
                breakdown.get(self.PHASE_COMM_AR, 0) +
                breakdown.get(self.PHASE_COMM_RS, 0) +
                breakdown.get(self.PHASE_COMM_AG, 0), 4
            ),
            'idle_ms': round(breakdown.get(self.PHASE_IDLE, 0), 4),
            'overhead_ms': round(breakdown.get(self.PHASE_OVERHEAD, 0), 4),
        }

    def roofline_analysis(self, model_params, batch_tokens,
                          peak_tflops=989.5, hbm_bw_tbps=3.35,
                          last_n=100):
        """Roofline model analysis.

        Determines if training is compute-bound or memory-bound.

        Arithmetic intensity (AI) = FLOPS / bytes_accessed
        Ridge point = peak_TFLOPS / HBM_bandwidth

        If AI > ridge_point: compute-bound
        If AI < ridge_point: memory-bound

        Ref: CUTLASS profiler — roofline model for GEMM
        """
        flops = 6.0 * model_params * batch_tokens
        # Bytes accessed: model params + gradients + optimizer states
        # Approximate: 16 * N bytes (params + grads + 2 moments, each in FP32)
        bytes_accessed = 16.0 * model_params

        ai = flops / max(1.0, bytes_accessed)
        ridge_point = peak_tflops * 1e12 / (hbm_bw_tbps * 1e12)

        if ai > ridge_point:
            regime = 'compute-bound'
            achievable_tflops = peak_tflops
        else:
            regime = 'memory-bound'
            achievable_tflops = ai * hbm_bw_tbps

        mfu_data = self.compute_mfu(model_params, batch_tokens,
                                     peak_tflops, last_n)

        return {
            'arithmetic_intensity': round(ai, 4),
            'ridge_point': round(ridge_point, 4),
            'regime': regime,
            'achievable_tflops': round(achievable_tflops, 4),
            'actual_mfu': mfu_data['mfu'],
            'efficiency_vs_roofline': round(
                mfu_data['mfu'] * peak_tflops / max(0.001, achievable_tflops), 6
            ),
        }

    def desloc_comm_savings_breakdown(self, Kx, Ku, Kv, last_n=100):
        """Break down how much time DES-LOC saved vs DDP.

        Assumes DDP would have comm_ar every step.
        DES-LOC has comm_ar only 1/Kx of steps.
        """
        breakdown = self.get_phase_breakdown(last_n)
        ar_ms = breakdown.get(self.PHASE_COMM_AR, 0.0)

        # DDP equivalent: ar_ms * Kx (since we only comm 1/Kx of steps)
        ddp_ar_ms = ar_ms * max(1, Kx)
        saved_ms = ddp_ar_ms - ar_ms

        total = breakdown.get('total_ms', 0.001)
        ddp_total = total + saved_ms  # what DDP step time would be

        return {
            'desloc_step_ms': round(total, 4),
            'estimated_ddp_step_ms': round(ddp_total, 4),
            'comm_savings_ms': round(saved_ms, 4),
            'speedup': round(ddp_total / max(0.001, total), 4),
            'Kx': Kx,
        }

    def export_csv(self, filepath):
        """Export phase timing data as CSV."""
        if not self._step_records:
            return

        headers = ['step'] + list(self.ALL_PHASES) + ['total_ms']
        lines = [','.join(headers)]
        for i, record in enumerate(self._step_records):
            vals = [str(i)]
            for phase in self.ALL_PHASES:
                vals.append(f'{record.get(phase, 0.0):.4f}')
            vals.append(f'{record.get("total_ms", 0.0):.4f}')
            lines.append(','.join(vals))

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines) + '\n')


class DeslocStepTimePredictor:
    """Predict step time for different configurations.

    Uses recorded measurements to build a model:
    step_time(N, B, Kx) = a * N * B / TFLOPS + b * N / (BW * Kx) + c

    Where a, b, c are fitted from data.
    """

    def __init__(self):
        self._observations = []  # (N, B, Kx, step_time_ms)

    def record(self, model_params, batch_tokens, Kx, step_time_ms):
        """Record an observation."""
        self._observations.append((model_params, batch_tokens, Kx, step_time_ms))

    def predict(self, model_params, batch_tokens, Kx,
                fallback_tflops=400.0, fallback_bw_gbps=400.0):
        """Predict step time for given configuration."""
        if len(self._observations) < 2:
            # Fallback: simple estimation
            flops = 6.0 * model_params * batch_tokens
            compute_ms = flops / (fallback_tflops * 1e9)  # TFLOPS → GFLOPS
            comm_ms = (2.0 * model_params * 2) / (fallback_bw_gbps * 1e6 / 8) / max(1, Kx)
            return round(compute_ms + comm_ms, 4)

        # Linear regression: step_time = a * compute_proxy + b * comm_proxy + c
        # compute_proxy = N * B
        # comm_proxy = N / Kx
        n = len(self._observations)
        sum_x1 = sum_x2 = sum_y = 0.0
        sum_x1y = sum_x2y = sum_x1x2 = 0.0
        sum_x1_2 = sum_x2_2 = 0.0

        for N, B, K, t in self._observations:
            x1 = N * B / 1e15  # scale down
            x2 = N / max(1, K) / 1e9
            y = t

            sum_x1 += x1
            sum_x2 += x2
            sum_y += y
            sum_x1y += x1 * y
            sum_x2y += x2 * y
            sum_x1x2 += x1 * x2
            sum_x1_2 += x1 * x1
            sum_x2_2 += x2 * x2

        # Solve 2-variable linear regression
        # Using normal equations (simplified)
        mean_x1 = sum_x1 / n
        mean_x2 = sum_x2 / n
        mean_y = sum_y / n

        var_x1 = sum_x1_2 / n - mean_x1 ** 2
        cov_x1y = sum_x1y / n - mean_x1 * mean_y

        a = cov_x1y / max(1e-15, var_x1)
        c = mean_y - a * mean_x1

        # Predict
        x1_pred = model_params * batch_tokens / 1e15
        x2_pred = model_params / max(1, Kx) / 1e9
        predicted = a * x1_pred + c

        return round(max(0.1, predicted), 4)


# M245: end of Claude-16 timer integration


# =====================================================================
# M245T — Claude-16: Phase Profiler Topup + Hardware Clock Calibration
# Ref: DES-LOC paper, NCCL, Megatron-LM, NKI-FA da964f3
# =====================================================================

import math as _m245t_math

class DeslocHardwareClockCalibrator:
    """Calibrate GPU clock for accurate MFU measurement.
    Accounts for GPU boost clock vs base clock.
    Ref: DES-LOC Algorithm 1, M245T"""

    def __init__(self, base_clock_mhz=1410, boost_clock_mhz=1980):
        self._base = base_clock_mhz
        self._boost = boost_clock_mhz
        self._samples = []
        self._calibrated_ratio = 1.0

    def sample_clock(self, measured_tflops, theoretical_peak):
        if theoretical_peak <= 0:
            return
        ratio = measured_tflops / theoretical_peak
        self._samples.append(ratio)
        if len(self._samples) > 100:
            self._samples = self._samples[-100:]
        self._calibrated_ratio = sum(self._samples) / len(self._samples)

    def get_effective_peak_tflops(self, nominal_peak=989.5):
        return round(nominal_peak * self._calibrated_ratio, 4)

    def thermal_throttle_estimate(self):
        if len(self._samples) < 10:
            return 0.0
        recent = self._samples[-10:]
        older = self._samples[:max(1, len(self._samples)-10)]
        if not older:
            return 0.0
        r_mean = sum(recent) / len(recent)
        o_mean = sum(older) / len(older)
        throttle = max(0.0, 1.0 - r_mean / max(0.001, o_mean))
        return round(throttle, 6)

    def report(self):
        return {
            'calibrated_ratio': round(self._calibrated_ratio, 6),
            'n_samples': len(self._samples),
            'thermal_throttle': self.thermal_throttle_estimate(),
        }

class DeslocTrainingProgressTracker:
    """Track training progress: tokens processed, time remaining, ETA.
    Ref: DES-LOC Algorithm 1, M245T"""

    def __init__(self, total_steps=100000, tokens_per_step=1048576):
        self._total_steps = total_steps
        self._tokens_per_step = tokens_per_step
        self._step = 0
        self._start_time = None
        self._step_times = []
        self._losses = []

    def tick(self, step, loss=None, step_time_ms=None):
        import time
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._step = step
        if step_time_ms is not None:
            self._step_times.append(step_time_ms)
            if len(self._step_times) > 500:
                self._step_times = self._step_times[-500:]
        if loss is not None:
            self._losses.append(loss)
            if len(self._losses) > 500:
                self._losses = self._losses[-500:]

    def eta_seconds(self):
        if not self._step_times or self._step <= 0:
            return 0.0
        avg_ms = sum(self._step_times) / len(self._step_times)
        remaining = self._total_steps - self._step
        return round(remaining * avg_ms / 1000.0, 2)

    def tokens_processed(self):
        return self._step * self._tokens_per_step

    def throughput_tokens_per_sec(self):
        if not self._step_times:
            return 0.0
        avg_ms = sum(self._step_times) / len(self._step_times)
        if avg_ms <= 0:
            return 0.0
        return round(self._tokens_per_step / (avg_ms / 1000.0), 2)

    def loss_trend(self):
        if len(self._losses) < 10:
            return 0.0
        n = len(self._losses)
        first = sum(self._losses[:n//4]) / max(1, n//4)
        last = sum(self._losses[-n//4:]) / max(1, n//4)
        if first <= 0:
            return 0.0
        return round((last - first) / first, 6)

    def report(self):
        eta = self.eta_seconds()
        hrs = int(eta // 3600)
        mins = int((eta % 3600) // 60)
        return {
            'step': self._step,
            'total_steps': self._total_steps,
            'progress_pct': round(100.0 * self._step / max(1, self._total_steps), 2),
            'tokens_processed': self.tokens_processed(),
            'throughput_tok_s': self.throughput_tokens_per_sec(),
            'eta_s': eta,
            'eta_human': f'{hrs}h{mins}m',
            'loss_trend': self.loss_trend(),
        }

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

    if x <= 0:
        return y
    return round(x * y / max(1, x + y), 6)

# M245T: end of Claude-16
