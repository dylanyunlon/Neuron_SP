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


# =================================================================
# M072: DES-LOC Wallclock Profiler + Throughput Meter (400 lines)
# =================================================================
# Fine-grained timing for DES-LOC training phases:
# - Forward pass timing
# - Backward pass timing
# - Gradient clipping timing
# - AllReduce timing (per state type: x, u, v)
# - Optimizer step timing
# - Total step timing with overlap detection
#
# Reference: Section 5.3 speedup measurements
# Reference: Section 5.4 "training speedup over DDP"
# =================================================================

import time as _time
import math as _math


class DESLOCPhaseTimer:
    """Timer for individual training phases within a DES-LOC step.

    Tracks each phase separately to identify bottlenecks and
    compute the comm/compute overlap ratio.
    """

    PHASES = [
        'forward', 'backward', 'grad_clip', 'optimizer_step',
        'allreduce_x', 'allreduce_u', 'allreduce_v',
        'data_loading', 'logging', 'total_step',
    ]

    def __init__(self):
        self.phase_times = {p: [] for p in self.PHASES}
        self._active = {}
        self.step_count = 0

    def start(self, phase):
        """Start timing a phase."""
        self._active[phase] = _time.monotonic()

    def stop(self, phase):
        """Stop timing a phase and record elapsed time."""
        if phase not in self._active:
            return 0.0
        elapsed_ms = (_time.monotonic() - self._active[phase]) * 1000.0
        if phase not in self.phase_times:
            self.phase_times[phase] = []
        self.phase_times[phase].append(elapsed_ms)
        del self._active[phase]
        return elapsed_ms

    def record_step(self):
        """Mark end of a training step."""
        self.step_count += 1

    def get_phase_avg(self, phase):
        """Get average time for a phase in ms."""
        times = self.phase_times.get(phase, [])
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_compute_time(self):
        """Total compute time = forward + backward + optimizer."""
        return (self.get_phase_avg('forward') +
                self.get_phase_avg('backward') +
                self.get_phase_avg('optimizer_step'))

    def get_comm_time(self):
        """Total communication time = allreduce_{x,u,v}."""
        return (self.get_phase_avg('allreduce_x') +
                self.get_phase_avg('allreduce_u') +
                self.get_phase_avg('allreduce_v'))

    def get_overhead_time(self):
        """Overhead = total - compute - comm."""
        total = self.get_phase_avg('total_step')
        return max(0, total - self.get_compute_time() -
                   self.get_comm_time())

    def get_comm_compute_ratio(self):
        """Ratio of comm to compute time."""
        compute = self.get_compute_time()
        if compute <= 0:
            return 0.0
        return self.get_comm_time() / compute

    def get_utilization(self):
        """GPU compute utilization = compute / total."""
        total = self.get_phase_avg('total_step')
        if total <= 0:
            return 0.0
        return self.get_compute_time() / total

    def format_summary(self):
        """Format phase timing summary for experiment log.

        Output follows NKI-FA structured format.
        """
        lines = [f"### Phase Timing Summary (steps={self.step_count}) ###"]
        for phase in self.PHASES:
            avg = self.get_phase_avg(phase)
            if avg > 0:
                times = self.phase_times[phase]
                lines.append(
                    f"{phase}: avg={avg:.3f}ms, "
                    f"min={min(times):.3f}ms, "
                    f"max={max(times):.3f}ms, "
                    f"samples={len(times)}")
        lines.append(f"Compute: {self.get_compute_time():.3f}ms")
        lines.append(f"Comm: {self.get_comm_time():.3f}ms")
        lines.append(f"Overhead: {self.get_overhead_time():.3f}ms")
        lines.append(
            f"Comm/Compute ratio: "
            f"{self.get_comm_compute_ratio():.4f}")
        lines.append(
            f"GPU utilization: "
            f"{self.get_utilization():.4f}")
        return "\n".join(lines)


class DESLOCThroughputTracker:
    """Track training throughput for DES-LOC experiments.

    Measures tokens/sec, samples/sec, and TFLOPS to generate
    the throughput comparison figures (RQ3, RQ4).

    Section 5.4: "DES-LOC's reduced communication costs result
    in a ≈ 1.24× training speedup over DDP"
    """

    def __init__(self, batch_size=4, seq_len=1024,
                 model_params=0, dtype_bytes=2):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model_params = model_params
        self.dtype_bytes = dtype_bytes
        self.step_times_ms = []
        self.tokens_processed = 0
        self.start_time = None

    def start_training(self):
        """Mark training start."""
        self.start_time = _time.monotonic()

    def record_step(self, step_time_ms):
        """Record a single step's wallclock time."""
        self.step_times_ms.append(step_time_ms)
        self.tokens_processed += self.batch_size * self.seq_len

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

        Approximate FLOPs per step:
          Forward: 2 * params * tokens
          Backward: 4 * params * tokens
          Total: 6 * params * tokens
        """
        if self.model_params <= 0 or not self.step_times_ms:
            return 0.0
        tokens_per_step = self.batch_size * self.seq_len
        flops_per_step = 6.0 * self.model_params * tokens_per_step
        avg_sec = (sum(self.step_times_ms) /
                   len(self.step_times_ms)) / 1000.0
        if avg_sec <= 0:
            return 0.0
        return flops_per_step / avg_sec / 1e12

    def estimate_time_to_completion(self, total_steps):
        """Estimate remaining training time."""
        if not self.step_times_ms:
            return 0.0
        avg_ms = sum(self.step_times_ms) / len(self.step_times_ms)
        remaining_steps = total_steps - len(self.step_times_ms)
        return max(0, remaining_steps * avg_ms / 1000.0)

    def estimate_speedup_vs_ddp(self, ddp_step_time_ms):
        """Compute speedup factor vs DDP baseline.

        Args:
            ddp_step_time_ms: measured DDP step time
        """
        if not self.step_times_ms or ddp_step_time_ms <= 0:
            return 1.0
        avg_ms = sum(self.step_times_ms) / len(self.step_times_ms)
        if avg_ms <= 0:
            return 1.0
        return ddp_step_time_ms / avg_ms

    def format_summary(self):
        """Format throughput summary for experiment log."""
        lines = [
            f"### Throughput Summary "
            f"(steps={len(self.step_times_ms)}) ###",
        ]
        if self.step_times_ms:
            avg = sum(self.step_times_ms) / len(self.step_times_ms)
            lines.append(f"Avg step time: {avg:.3f}ms")
            lines.append(
                f"Tokens/sec: "
                f"{self.get_tokens_per_second():.1f}")
            lines.append(
                f"Samples/sec: "
                f"{self.get_samples_per_second():.2f}")
            if self.model_params > 0:
                lines.append(
                    f"TFLOPS: {self.get_tflops():.2f}")
        return "\n".join(lines)

    def export_for_plotting(self):
        """Export throughput data for figure generation."""
        return {
            'step_times_ms': self.step_times_ms,
            'tokens_per_sec': self.get_tokens_per_second(),
            'samples_per_sec': self.get_samples_per_second(),
            'tflops': self.get_tflops(),
            'total_tokens': self.tokens_processed,
            'num_steps': len(self.step_times_ms),
        }


class DESLOCWallclockComparison:
    """Compare wallclock times across methods.

    Tracks DDP, Local Adam, and DES-LOC step times
    side-by-side for the speedup comparison figure.

    Section 5.3: "This yields a 1.24× speedup over DDP
    and 2.01× over Local Adam at Kx=32"
    """

    def __init__(self):
        self.methods = {}

    def add_method(self, name, batch_size=4, seq_len=1024,
                   model_params=0):
        """Register a training method for comparison."""
        self.methods[name] = DESLOCThroughputTracker(
            batch_size=batch_size, seq_len=seq_len,
            model_params=model_params)

    def record_step(self, method_name, step_time_ms):
        """Record a step time for a specific method."""
        if method_name in self.methods:
            self.methods[method_name].record_step(step_time_ms)

    def get_speedup_table(self, baseline='ddp'):
        """Generate speedup comparison table.

        Returns dict with speedup factors relative to baseline.
        """
        if baseline not in self.methods:
            return {}

        baseline_tracker = self.methods[baseline]
        if not baseline_tracker.step_times_ms:
            return {}

        baseline_avg = (sum(baseline_tracker.step_times_ms) /
                        len(baseline_tracker.step_times_ms))

        result = {}
        for name, tracker in self.methods.items():
            if not tracker.step_times_ms:
                continue
            avg = (sum(tracker.step_times_ms) /
                   len(tracker.step_times_ms))
            result[name] = {
                'avg_step_ms': round(avg, 3),
                'tokens_per_sec': round(
                    tracker.get_tokens_per_second(), 1),
                'tflops': round(tracker.get_tflops(), 2),
                'speedup_vs_baseline': round(
                    baseline_avg / max(avg, 0.001), 2),
            }
        return result

    def format_comparison(self, baseline='ddp'):
        """Format speedup comparison as log."""
        table = self.get_speedup_table(baseline)
        lines = [f"### Wallclock Comparison (baseline={baseline}) ###"]
        for name, data in table.items():
            lines.append(
                f"{name}: {data['avg_step_ms']:.3f}ms/step, "
                f"{data['tokens_per_sec']:.0f} tok/s, "
                f"{data['speedup_vs_baseline']:.2f}x vs {baseline}")
        return "\n".join(lines)


# =================================================================
# End M072
# =================================================================
