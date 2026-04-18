# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from numpy import mean
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

try:
    import psutil

    PSUTILS_INSTALLED = True
except ImportError:
    PSUTILS_INSTALLED = False
    pass


class CudaEventTimer(object):

    def __init__(self, start_event: get_accelerator().Event, end_event: get_accelerator().Event):
        self.start_event = start_event
        self.end_event = end_event

    def get_elapsed_msec(self):
        get_accelerator().current_stream().wait_event(self.end_event)
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)


class SynchronizedWallClockTimer:
    """Group of timers. Borrowed from Nvidia Megatron code"""

    class Timer:
        """Timer."""

        def __init__(self, name):
            self.name_ = name
            self.started_ = False
            self.event_timers = []
            self.use_host_timer = get_accelerator().use_host_timers()
            self.start_event = None
            self.elapsed_records = None
            self.start_time = 0.0
            self.end_time = 0.0

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
            event_class = get_accelerator().Event
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
            if self.use_host_timer:
                self.elapsed_records = [et * 1000.0 for et in self.event_timers]
            else:
                self.elapsed_records = [et.get_elapsed_msec() for et in self.event_timers]
            return sum(self.elapsed_records)

        def reset(self):
            """Reset timer."""
            self.started_ = False
            self.start_event = None
            self.elapsed_records = None
            self.event_timers.clear()

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self._get_elapsed_msec()
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

        def mean(self):
            self.elapsed(reset=False)
            return trim_mean(self.elapsed_records, 0.1)

    def __init__(self):
        self.timers = {}

    def get_timers(self):
        return self.timers

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    @staticmethod
    def memory_usage():
        alloc = "mem_allocated: {:.4f} GB".format(get_accelerator().memory_allocated() / (1024 * 1024 * 1024))
        max_alloc = "max_mem_allocated: {:.4f} GB".format(get_accelerator().max_memory_allocated() /
                                                          (1024 * 1024 * 1024))
        cache = "cache_allocated: {:.4f} GB".format(get_accelerator().memory_cached() / (1024 * 1024 * 1024))
        max_cache = "max_cache_allocated: {:.4f} GB".format(get_accelerator().max_memory_cached() /
                                                            (1024 * 1024 * 1024))
        return " | {} | {} | {} | {}".format(alloc, max_alloc, cache, max_cache)

    def log(self, names, normalizer=1.0, reset=True, memory_breakdown=False, ranks=None):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            if name in self.timers:
                elapsed_time = (self.timers[name].elapsed(reset=reset) / normalizer)
                string += " | {}: {:.2f}".format(name, elapsed_time)

        # timers logging should be independent of the global log level it's already conditional on wall_clock_breakdown being True, so using use_logger=False will always print the stats
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


class NoopTimer:

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


class ThroughputTimer:

    def __init__(self, config, batch_size, start_step=2, steps_per_output=None, monitor_memory=False, logging_fn=None):
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

    def update_epoch_count(self):
        self.epoch_count += 1
        self.micro_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        if not self.config.enabled:
            return
        self._init_timer()
        self.started = True
        if self.global_step_count >= self.start_step:
            if self.config.synchronized:
                get_accelerator().synchronize()
            self.start_time = time.time()

    def _is_report_boundary(self):
        if self.steps_per_output is None:
            return False
        return self.global_step_count % self.steps_per_output == 0

    def stop(self, global_step=False, report_speed=True):
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

            if global_step:
                if report_speed and self._is_report_boundary():
                    self.logging(
                        "epoch={}/micro_step={}/global_step={}, RunningAvgSamplesPerSec={}, CurrSamplesPerSec={}, "
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
                        self.logging("epoch={}/micro_step={}/global_step={}, vm %: {}, swap %: {}".format(
                            self.epoch_count,
                            self.micro_step_count,
                            self.global_step_count,
                            virt_mem.percent,
                            swap.percent,
                        ))
                self.step_elapsed_time = 0

    def avg_samples_per_sec(self):
        if self.global_step_count > 0:
            total_step_offset = self.global_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            # training samples per second
            return self.batch_size / avg_time_per_step
        return float("-inf")


def trim_mean(data, trim_percent):
    """Compute the trimmed mean of a list of numbers.

    Args:
        data (list): List of numbers.
        trim_percent (float): Percentage of data to trim.

    Returns:
        float: Trimmed mean.
    """
    assert 0.0 <= trim_percent <= 1.0
    n = len(data)
    # Account for edge case of empty list
    if len(data) == 0:
        return 0
    data.sort()
    k = int(round(n * (trim_percent)))
    return mean(data[k:n - k])


# ═══════════════════════════════════════════════════════════════
# DES-LOC Timing & Throughput Tracking (M190)
# ═══════════════════════════════════════════════════════════════
import time as _m190_time
import math as _m190_math


class DESLOCPhaseTimer:
    """Timer that tracks compute vs communication time per step.

    Enables the breakdown needed for RQ3 and RQ4 figures:
    - compute_ms: time in forward+backward+optimizer
    - comm_ms: time in allreduce/all_gather
    - idle_ms: time waiting for sync
    """

    def __init__(self):
        self._phase = None
        self._phase_start = 0.0
        self.compute_ms_total = 0.0
        self.comm_ms_total = 0.0
        self.idle_ms_total = 0.0
        self.step_count = 0
        self._step_compute = 0.0
        self._step_comm = 0.0
        self._step_idle = 0.0
        self.step_log = []

    def start_phase(self, phase):
        """Start timing a phase: 'compute', 'comm', or 'idle'."""
        now = _m190_time.monotonic() * 1000.0
        if self._phase is not None:
            elapsed = now - self._phase_start
            if self._phase == 'compute':
                self._step_compute += elapsed
            elif self._phase == 'comm':
                self._step_comm += elapsed
            elif self._phase == 'idle':
                self._step_idle += elapsed
        self._phase = phase
        self._phase_start = now

    def end_step(self):
        """End current step, flush accumulated timings."""
        if self._phase is not None:
            now = _m190_time.monotonic() * 1000.0
            elapsed = now - self._phase_start
            if self._phase == 'compute':
                self._step_compute += elapsed
            elif self._phase == 'comm':
                self._step_comm += elapsed
            elif self._phase == 'idle':
                self._step_idle += elapsed

        self.compute_ms_total += self._step_compute
        self.comm_ms_total += self._step_comm
        self.idle_ms_total += self._step_idle

        self.step_log.append({
            'step': self.step_count,
            'compute_ms': round(self._step_compute, 3),
            'comm_ms': round(self._step_comm, 3),
            'idle_ms': round(self._step_idle, 3),
            'total_ms': round(self._step_compute + self._step_comm + self._step_idle, 3),
        })

        self.step_count += 1
        self._step_compute = 0.0
        self._step_comm = 0.0
        self._step_idle = 0.0
        self._phase = None

    def get_averages(self):
        """Get average timings across all steps."""
        n = max(self.step_count, 1)
        return {
            'avg_compute_ms': round(self.compute_ms_total / n, 3),
            'avg_comm_ms': round(self.comm_ms_total / n, 3),
            'avg_idle_ms': round(self.idle_ms_total / n, 3),
            'avg_total_ms': round((self.compute_ms_total + self.comm_ms_total + self.idle_ms_total) / n, 3),
            'comm_fraction': round(self.comm_ms_total / max(self.compute_ms_total + self.comm_ms_total, 1e-6), 4),
            'total_steps': self.step_count,
        }

    def export_for_figure(self):
        """Export step-level timing data for figure generation."""
        return {
            'steps': [e['step'] for e in self.step_log],
            'compute_ms': [e['compute_ms'] for e in self.step_log],
            'comm_ms': [e['comm_ms'] for e in self.step_log],
            'total_ms': [e['total_ms'] for e in self.step_log],
        }


class DESLOCThroughputTracker:
    """Track training throughput for DES-LOC experiments.

    Measures tokens/sec, samples/sec, and TFLOPS to generate
    the throughput comparison figures (RQ3, RQ4).

    Section 5.4: "DES-LOC's reduced communication costs result
    in a approx 1.24x training speedup over DDP"
    """

    def __init__(self, batch_size=4, seq_len=1024,
                 model_params=0, dtype_bytes=2):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model_params = model_params
        self.dtype_bytes = dtype_bytes
        self.step_times_ms = []
        self.muon_step_times_ms = []
        self.comm_times_ms = []
        self.tokens_processed = 0
        self.start_time = None

    def start_training(self):
        """Mark training start."""
        self.start_time = _m190_time.monotonic()

    def record_step(self, step_time_ms, muon_ms=None, comm_ms=None):
        """Record a single step's wallclock time."""
        self.step_times_ms.append(step_time_ms)
        self.tokens_processed += self.batch_size * self.seq_len
        if muon_ms is not None:
            self.muon_step_times_ms.append(muon_ms)
        if comm_ms is not None:
            self.comm_times_ms.append(comm_ms)

    def get_tokens_per_second(self):
        """Get average tokens/sec throughput."""
        if not self.step_times_ms:
            return 0.0
        avg_ms = sum(self.step_times_ms) / len(self.step_times_ms)
        if avg_ms <= 0:
            return 0.0
        tokens_per_step = self.batch_size * self.seq_len
        return tokens_per_step / (avg_ms / 1000.0)

    def get_samples_per_second(self):
        """Get average samples/sec throughput."""
        if not self.step_times_ms:
            return 0.0
        avg_ms = sum(self.step_times_ms) / len(self.step_times_ms)
        if avg_ms <= 0:
            return 0.0
        return self.batch_size / (avg_ms / 1000.0)

    def get_tflops(self):
        """Estimate TFLOPS based on model size.

        Approximate FLOPs per step (forward + backward):
          6 * model_params * tokens_per_step
        """
        if not self.step_times_ms or self.model_params == 0:
            return 0.0
        avg_ms = sum(self.step_times_ms) / len(self.step_times_ms)
        if avg_ms <= 0:
            return 0.0
        tokens_per_step = self.batch_size * self.seq_len
        flops_per_step = 6.0 * self.model_params * tokens_per_step
        tflops = flops_per_step / (avg_ms / 1000.0) / 1e12
        return round(tflops, 2)

    def get_mfu(self, peak_tflops):
        """Compute Model FLOPS Utilization.

        MFU = achieved TFLOPS / peak hardware TFLOPS
        """
        achieved = self.get_tflops()
        if peak_tflops <= 0:
            return 0.0
        return round(achieved / peak_tflops, 4)

    def get_speedup_vs_ddp(self, ddp_avg_step_ms):
        """Compute speedup over DDP baseline."""
        if not self.step_times_ms:
            return 1.0
        avg_ms = sum(self.step_times_ms) / len(self.step_times_ms)
        if avg_ms <= 0:
            return 1.0
        return round(ddp_avg_step_ms / avg_ms, 3)

    def export_summary(self):
        """Export throughput summary for figure generation."""
        return {
            'total_steps': len(self.step_times_ms),
            'total_tokens': self.tokens_processed,
            'avg_step_ms': round(sum(self.step_times_ms) / max(len(self.step_times_ms), 1), 3),
            'tokens_per_sec': round(self.get_tokens_per_second(), 1),
            'samples_per_sec': round(self.get_samples_per_second(), 2),
            'tflops': self.get_tflops(),
            'avg_muon_ms': round(sum(self.muon_step_times_ms) / max(len(self.muon_step_times_ms), 1), 3) if self.muon_step_times_ms else None,
            'avg_comm_ms': round(sum(self.comm_times_ms) / max(len(self.comm_times_ms), 1), 3) if self.comm_times_ms else None,
        }


class DESLOCRooflineAnalyzer:
    """Roofline model for DES-LOC communication analysis.

    Nick Joseph: "you can use pen-and-paper to compute the theoretical
    maximum efficiency (MFU). The reasons for inefficiency are usually
    memory bandwidth bottleneck, CPU transfer bottleneck, etc."

    This analyzer computes the theoretical communication cost
    and compares with actual measurements.
    """

    def __init__(self, param_bytes, bandwidth_gbps, world_size,
                 Kx=32, Ku=96, Kv=192):
        self.param_bytes = param_bytes
        self.bandwidth_bps = bandwidth_gbps * 1e9 / 8  # bytes/sec
        self.world_size = world_size
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv

    def theoretical_comm_time_per_step(self):
        """Compute theoretical communication time per step.

        DDP: allreduce all 3 states every step
        DES-LOC: allreduce at 1/Kx, 1/Ku, 1/Kv rates
        """
        bytes_per_allreduce = self.param_bytes * 2 * (self.world_size - 1) / self.world_size
        ddp_time = bytes_per_allreduce * 3 / max(self.bandwidth_bps, 1)
        desloc_time = bytes_per_allreduce * (1.0/self.Kx + 1.0/self.Ku + 1.0/self.Kv) / max(self.bandwidth_bps, 1)
        return {
            'ddp_comm_sec': round(ddp_time, 6),
            'desloc_comm_sec': round(desloc_time, 6),
            'speedup': round(ddp_time / max(desloc_time, 1e-12), 2),
            'bytes_per_allreduce': int(bytes_per_allreduce),
        }

    def compute_overlap_efficiency(self, compute_ms, comm_ms):
        """Compute how well communication overlaps with compute."""
        if compute_ms <= 0:
            return 0.0
        overlap = max(0, compute_ms - max(0, comm_ms - compute_ms))
        return round(overlap / compute_ms, 4)


# End M190
