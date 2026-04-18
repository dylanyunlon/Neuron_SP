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
# DES-LOC Roofline Model
# Ref: Nick Joseph — "you can use paper and pen to calculate theoretical MFU.
# The reasons for efficiency gap are usually: HBM bandwidth bottleneck,
# CPU transfer bottleneck, etc. About six or seven constraints."
# =========================================================================

class DeslocRooflineModel:
    """Analytical performance model for DES-LOC training.

    Models three bottleneck dimensions:
    1. Compute bound: FLOPS / peak_FLOPS
    2. Memory bound: bytes_accessed / HBM_bandwidth
    3. Communication bound: comm_bytes / network_bandwidth

    DES-LOC specifically targets #3 by reducing comm_bytes via
    independent sync periods (Kx, Ku, Kv).

    Ref: Section 4.1 — Ring-AllReduce: 2*(N-1)/N * data_size bytes.
    """

    def __init__(self, peak_tflops, hbm_bw_gbps, net_bw_gbps):
        """Initialize with hardware specifications.

        Args:
            peak_tflops: Peak TFLOPS (e.g. 312 for H100 BF16 Tensor Core).
            hbm_bw_gbps: HBM bandwidth in GB/s (e.g. 3350 for H100).
            net_bw_gbps: Network bandwidth in GB/s.
                NVLink 4.0: ~600, PCIe Gen4: ~32, EFA v2: ~12.5.
        """
        self.peak_tflops = peak_tflops
        self.hbm_bw = hbm_bw_gbps
        self.net_bw = net_bw_gbps

    def predict_step_time(self, model_flops, param_bytes, num_workers,
                          Kx=1, Ku=3, Kv=6):
        """Predict training step time under DES-LOC.

        Ref: Section 4.1 — DES-LOC comm cost per step:
          comm_rate = 1/Kx + 1/Ku + 1/Kv (allreduces per step)
          allreduce_time = 2*(N-1)/N * param_bytes / net_bw

        Returns dict with predicted times, MFU, and bottleneck diagnosis.
        """
        compute_s = model_flops / (self.peak_tflops * 1e12) if self.peak_tflops > 0 else 0
        if num_workers <= 1:
            comm_s = 0.0
            ddp_comm_s = 0.0
        else:
            ring = 2.0 * (num_workers - 1) / num_workers
            single_ar_s = ring * param_bytes / (self.net_bw * 1e9) if self.net_bw > 0 else 0
            desloc_rate = 1.0/max(1,Kx) + 1.0/max(1,Ku) + 1.0/max(1,Kv)
            comm_s = single_ar_s * desloc_rate
            ddp_comm_s = single_ar_s * 3.0

        total_s = compute_s + comm_s
        ddp_total_s = compute_s + ddp_comm_s
        mfu = model_flops / (total_s * self.peak_tflops * 1e12) if total_s > 0 and self.peak_tflops > 0 else 0
        ddp_mfu = model_flops / (ddp_total_s * self.peak_tflops * 1e12) if ddp_total_s > 0 and self.peak_tflops > 0 else 0
        bottleneck = 'comm_bound' if comm_s > compute_s else 'compute_bound'

        return {
            'compute_ms': round(compute_s * 1000, 4),
            'comm_ms': round(comm_s * 1000, 4),
            'total_ms': round(total_s * 1000, 4),
            'ddp_total_ms': round(ddp_total_s * 1000, 4),
            'speedup_vs_ddp': round(ddp_total_s / total_s, 4) if total_s > 0 else 1.0,
            'mfu': round(mfu, 6),
            'ddp_mfu': round(ddp_mfu, 6),
            'bottleneck': bottleneck,
        }

    def sweep_kx(self, model_flops, param_bytes, num_workers,
                 kx_values=None):
        """Sweep Kx to find optimal operating point.

        Ref: Section 5.4 — "setting Kx for sufficient throughput."
        """
        if kx_values is None:
            kx_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        results = []
        for kx in kx_values:
            ku = max(1, kx * 3)
            kv = max(1, kx * 6)
            pred = self.predict_step_time(model_flops, param_bytes, num_workers,
                                          Kx=kx, Ku=ku, Kv=kv)
            pred.update({'Kx': kx, 'Ku': ku, 'Kv': kv})
            results.append(pred)
        return results

    def find_optimal_kx(self, model_flops, param_bytes, num_workers):
        """Find smallest Kx where training becomes compute-bound.

        Returns the prediction dict for the optimal Kx, or the
        largest tested Kx if always comm-bound.
        """
        sweep = self.sweep_kx(model_flops, param_bytes, num_workers)
        for pred in sweep:
            if pred['bottleneck'] == 'compute_bound':
                return pred
        return sweep[-1] if sweep else None


# Hardware presets for common GPU configurations
DESLOC_HW_PRESETS = {
    'H100_NVL': DeslocRooflineModel(312, 3350, 600),
    'H100_PCIe': DeslocRooflineModel(312, 3350, 32),
    'A100_80GB': DeslocRooflineModel(312, 2039, 600),
    'A6000': DeslocRooflineModel(77.4, 768, 32),
    'Trainium2': DeslocRooflineModel(380, 3200, 100),
}
