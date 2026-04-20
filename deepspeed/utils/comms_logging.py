# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# DES-LOC (Desynced Low Communication) integration — Algorithm 1 sync tier tracking
# Ref: Section 4.1 — independent sync periods for params, first momentum, second momentum

import math
import re as _re
import json as _json
from deepspeed.utils import log_dist

# DES-LOC sync tier identifiers
# Ref: Algorithm 1 line 7 — Kx for parameters, Ku for first momentum, Kv for second
DESLOC_TIER_PARAM = 0       # parameter gradient allreduce (period Kx)
DESLOC_TIER_MOMENTUM = 1    # first momentum state averaging (period Ku)
DESLOC_TIER_VARIANCE = 2    # second momentum state averaging (period Kv)
DESLOC_TIER_LOCAL = -1      # local-only op, no communication needed
DESLOC_TIER_NAMES = {0: 'param', 1: 'momentum', 2: 'variance', -1: 'local'}


def get_caller_func(frame=3, desloc_tier=None):
    """Get name of the calling function for logging.

    DES-LOC extension: optionally annotate the caller name with the
    sync tier so that log output can be parsed by tier.

    Args:
        frame: Stack frame offset to inspect.
        desloc_tier: If provided, appends [tier=N] to the function name.
            Ref: Algorithm 1 — each comm op belongs to exactly one tier.
    """
    import sys
    name = sys._getframe(frame).f_code.co_name
    if desloc_tier is not None:
        tier_label = DESLOC_TIER_NAMES.get(desloc_tier, str(desloc_tier))
        name = f'{name}[tier={tier_label}]'
    return name


def print_rank_0(message, desloc_step=None, desloc_tier=None):
    """Print a message only from rank 0.

    DES-LOC extension: prefix message with step counter and tier tag
    so log parsers (DeslocLogParser) can correlate comm events with
    training steps and sync tiers.

    Args:
        message: The message string to print.
        desloc_step: Current DES-LOC local step counter.
        desloc_tier: Sync tier for this message.

    Ref: NKI-FA draw_plot.py — structured log format for parsing.
    """
    import deepspeed.comm as dist
    if dist.get_rank() == 0:
        prefix_parts = []
        if desloc_step is not None:
            prefix_parts.append(f'desloc_step={desloc_step}')
        if desloc_tier is not None:
            tier_name = DESLOC_TIER_NAMES.get(desloc_tier, str(desloc_tier))
            prefix_parts.append(f'tier={tier_name}')
        if prefix_parts:
            prefix = '[' + ', '.join(prefix_parts) + '] '
            message = prefix + message
        print(message)


def convert_size(size_bytes, desloc_tier=None):
    """Convert byte count to human-readable string with optional tier label.

    DES-LOC extension: when desloc_tier is provided, appends the tier
    name in brackets so that bandwidth reports distinguish between
    parameter sync, momentum sync, and variance sync traffic.

    Args:
        size_bytes: Number of bytes to format.
        desloc_tier: Optional sync tier identifier.

    Returns:
        Formatted string like "1.23 GB [param]".

    Ref: Section 5.3 — per-tier comm volume comparison.
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    result = "%s %s" % (s, size_name[i])
    if desloc_tier is not None and desloc_tier in DESLOC_TIER_NAMES:
        result += f' [{DESLOC_TIER_NAMES[desloc_tier]}]'
    return result


def calc_bw_log(comm_op, size, duration, desloc_tier=None):
    """Calculate algorithmic and bus bandwidth for a communication op.

    DES-LOC extension: accepts desloc_tier for per-tier bandwidth tracking.
    The tier is not used in the calculation itself (bandwidth formula depends
    only on the collective type and world size) but is threaded through to
    allow the caller to tag the resulting measurement.

    Ref: NCCL-tests performance doc — ring allreduce: busbw = size * 2(N-1)/N.
    Ref: Section 4.1 — Ring-AllReduce is the primary collective for DES-LOC.
    """
    import deepspeed.comm as dist

    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all_single":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op in ("all_gather", "all_gather_into_tensor",
                     "reduce_scatter", "reduce_scatter_tensor"):
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op in ("all_reduce", "all_reduce_coalesced", "inference_all_reduce"):
        # DES-LOC: allreduce is the primary collective for param/momentum sync
        # Ref: Ring-AllReduce sends 2*(N-1)/N * data_size bytes total
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op in ("send", "recv", "isend", "irecv", "broadcast",
                     "reduce", "gather", "scatter", "barrier"):
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0(f"unknown comm_op: {comm_op}")
        return 0, 0

    # Convert bytes/sec to Gbps
    tput = tput * 8 / 1e6
    busbw = busbw * 8 / 1e6

    return tput, busbw


def desloc_classify_op(op_name):
    """Auto-classify a comm operation into a DES-LOC sync tier.

    Heuristic based on operation name patterns:
    - 'gradient', 'allreduce', 'reduce_scatter' → tier 0 (param)
    - 'momentum', 'first_moment', 'exp_avg' → tier 1 (momentum)
    - 'variance', 'second_moment', 'exp_avg_sq' → tier 2 (variance)
    - everything else → tier -1 (local)

    Ref: Algorithm 1 — each optimizer state has independent sync period.
    """
    op = op_name.lower() if isinstance(op_name, str) else ''
    if any(k in op for k in ('momentum', 'first_moment', 'exp_avg')):
        if 'sq' not in op:
            return DESLOC_TIER_MOMENTUM
    if any(k in op for k in ('variance', 'second_moment', 'exp_avg_sq')):
        return DESLOC_TIER_VARIANCE
    if any(k in op for k in ('allreduce', 'reduce_scatter', 'gradient', 'param')):
        return DESLOC_TIER_PARAM
    return DESLOC_TIER_LOCAL


def desloc_should_sync(step, period):
    """Check if a sync should happen at this step for a given period.

    Ref: Algorithm 1 line 10 — deterministic sync: step % K == 0.
    When K=1, every step syncs (DDP baseline behavior preserved).

    Args:
        step: Current training step.
        period: Sync period (Kx, Ku, or Kv).

    Returns:
        True if sync should happen, False if this step should be skipped.
    """
    if period <= 1:
        return True
    return (step % period) == 0


def desloc_half_life(beta):
    """Half-life of EMA with decay factor beta.

    Ref: Section 2, Eq.(1) — tau_0.5(beta) = -1/ln(beta).
    This is the number of steps for the EMA to lose half its "memory"
    of a past value. Determines how often each state needs sync.

    Examples:
        beta=0.9   → tau ≈ 6.58 steps
        beta=0.95  → tau ≈ 13.51 steps
        beta=0.999 → tau ≈ 692.80 steps

    The ratio tau(beta2)/tau(beta1) = ln(beta1)/ln(beta2) justifies
    setting Kv > Ku, since the second momentum changes more slowly.
    """
    if beta <= 0 or beta >= 1:
        return float('inf')
    return -1.0 / math.log(beta)


def desloc_recommend_periods(beta1, beta2, Kx):
    """Recommend Ku and Kv based on half-life ratios.

    Ref: Section 5.3 — default heuristic: Ku=3*Kx, Kv=6*Kx.
    Refined formula: Kv/Kx ~ tau(beta2)/tau(beta1) = ln(beta1)/ln(beta2).
    Capped at 10x to avoid extreme staleness.

    Args:
        beta1: First momentum decay (e.g. 0.9).
        beta2: Second momentum decay (e.g. 0.999).
        Kx: Parameter sync period.

    Returns:
        dict with recommended Ku, Kv, and diagnostic info.
    """
    tau1 = desloc_half_life(beta1)
    tau2 = desloc_half_life(beta2)
    ratio = tau2 / tau1 if tau1 > 0 else 1.0
    Ku = max(1, int(round(Kx * 3)))
    Kv = max(Ku, int(round(Kx * min(ratio, 10))))
    return {
        'Ku': Ku, 'Kv': Kv, 'half_life_ratio': round(ratio, 4),
        'tau_beta1': round(tau1, 4), 'tau_beta2': round(tau2, 4),
    }


def desloc_psi_factor(Kx, Ku, beta1):
    """Compute the psi factor from Theorem 1's convergence bound.

    psi = 4(1-px)/px^2 * (1-beta1)(1-pu) / (6(1-(1-pu)*beta1))
    where px=1/Kx, pu=1/Ku.

    Smaller psi → smaller higher-order convergence term.
    Kx=1 → px=1 → numerator=0 → psi=0 → recovers standard SGD bound.

    Ref: Section 3, Theorem 1 — bound = O(1/sqrt(T)) + O(psi/T).
    """
    px = 1.0 / max(1, Kx)
    pu = 1.0 / max(1, Ku)
    num = 4.0 * (1.0 - px) * (1.0 - beta1) * (1.0 - pu)
    den = (px * px) * 6.0 * (1.0 - (1.0 - pu) * beta1)
    if abs(den) < 1e-15:
        return float('inf')
    return num / den


def desloc_comm_reduction(Kx, Ku, Kv):
    """Compute communication reduction ratios vs DDP and Local Adam.

    Ref: Section 5.3 — per-step allreduce frequency:
      DDP:        3/1 = 3.0 (all three states every step)
      Local Adam: 3/Kx      (all three at same period Kx)
      DES-LOC:    1/Kx + 1/Ku + 1/Kv (independent periods)

    Returns dict with rates and reduction ratios.
    """
    if Kx <= 0 or Ku <= 0 or Kv <= 0:
        return {'error': 'sync periods must be positive'}
    desloc_rate = 1.0/Kx + 1.0/Ku + 1.0/Kv
    ddp_rate = 3.0
    local_adam_rate = 3.0 / Kx
    return {
        'desloc_per_step': round(desloc_rate, 6),
        'ddp_per_step': ddp_rate,
        'local_adam_per_step': round(local_adam_rate, 6),
        'reduction_vs_ddp': round(1.0 - desloc_rate / ddp_rate, 6),
        'reduction_vs_local_adam': round(1.0 - desloc_rate / local_adam_rate, 6),
        'Kx': Kx, 'Ku': Ku, 'Kv': Kv,
    }


class CommsLogger:
    """Communication logger with DES-LOC per-tier tracking.

    This is a full rewrite of the original CommsLogger that adds
    independent tracking for each DES-LOC sync tier. Every recorded
    comm op is classified into param/momentum/variance/local tiers,
    and byte counts, latencies, and bandwidths are accumulated per tier.

    Ref: Algorithm 1 — Kx, Ku, Kv gating; each op belongs to one tier.
    """

    def __init__(self, desloc_enabled=False):
        from deepspeed.comm.constants import (
            COMMS_LOGGER_VERBOSE_DEFAULT, COMMS_LOGGER_DEBUG_DEFAULT,
            COMMS_LOGGER_PROF_OPS_DEFAULT, COMMS_LOGGER_PROF_ALL_DEFAULT,
            COMMS_LOGGER_ENABLED_DEFAULT,
        )
        self.comms_dict = {}
        self.verbose = COMMS_LOGGER_VERBOSE_DEFAULT
        self.debug = COMMS_LOGGER_DEBUG_DEFAULT
        self.prof_ops = COMMS_LOGGER_PROF_OPS_DEFAULT
        self.prof_all = COMMS_LOGGER_PROF_ALL_DEFAULT
        self.enabled = COMMS_LOGGER_ENABLED_DEFAULT

        # DES-LOC: per-tier cumulative communication accounting
        self.desloc_enabled = desloc_enabled
        self.desloc_tier_bytes = {
            DESLOC_TIER_PARAM: 0,
            DESLOC_TIER_MOMENTUM: 0,
            DESLOC_TIER_VARIANCE: 0,
            DESLOC_TIER_LOCAL: 0,
        }
        self.desloc_tier_counts = {t: 0 for t in self.desloc_tier_bytes}
        self.desloc_tier_latency_ms = {t: 0.0 for t in self.desloc_tier_bytes}
        self.desloc_skipped_ops = 0
        self.desloc_total_ops = 0
        self._desloc_Kx = 1
        self._desloc_Ku = 3
        self._desloc_Kv = 6

    def configure(self, comms_config, desloc_config=None):
        """Configure logger from DeepSpeed comms config.

        DES-LOC extension: accepts desloc_config dict to set sync periods
        for tier classification and gating decisions.

        Args:
            comms_config: Standard DeepSpeed comms configuration.
            desloc_config: Optional dict with 'enabled', 'Kx', 'Ku', 'Kv'.
        """
        self.enabled = comms_config.comms_logger_enabled
        if self.enabled:
            self.verbose = comms_config.comms_logger.verbose
            self.debug = comms_config.comms_logger.debug
            self.prof_ops = comms_config.comms_logger.prof_ops
            self.prof_all = comms_config.comms_logger.prof_all
        if desloc_config is not None:
            self.desloc_enabled = desloc_config.get('enabled', False)
            self._desloc_Kx = desloc_config.get('Kx', 1)
            self._desloc_Ku = desloc_config.get('Ku', 3)
            self._desloc_Kv = desloc_config.get('Kv', 6)

    def start_profiling_comms(self):
        """Enable global profiling of all comm operations."""
        self.prof_all = True

    def stop_profiling_comms(self):
        """Disable global profiling."""
        self.prof_all = False

    def start_profiling_op(self, op_name_list):
        """Start profiling specific operation types."""
        self.prof_ops = list(set(self.prof_ops) | set(op_name_list))

    def stop_profiling_op(self, op_name_list):
        """Stop profiling specific operation types."""
        self.prof_ops = [op for op in self.prof_ops if op not in op_name_list]

    def append(self, raw_name, record_name, latency, msg_size, desloc_tier=None):
        """Record a communication operation.

        DES-LOC extension: classifies the op into a sync tier and
        accumulates per-tier byte counts, latencies, and op counts.

        Args:
            raw_name: Raw collective name (e.g. 'all_reduce').
            record_name: Decorated name for logging.
            latency: Operation latency in milliseconds.
            msg_size: Message size in bytes.
            desloc_tier: Explicit tier override; if None, auto-classified.
        """
        algbw, busbw = calc_bw_log(raw_name, msg_size, latency)
        if record_name in self.comms_dict:
            if msg_size in self.comms_dict[record_name]:
                self.comms_dict[record_name][msg_size][0] += 1
                self.comms_dict[record_name][msg_size][1].append(latency)
                self.comms_dict[record_name][msg_size][2].append(algbw)
                self.comms_dict[record_name][msg_size][3].append(busbw)
            else:
                self.comms_dict[record_name][msg_size] = [1, [latency], [algbw], [busbw]]
        else:
            self.comms_dict[record_name] = {msg_size: [1, [latency], [algbw], [busbw]]}

        if self.verbose:
            log_str = (f"comm op: {record_name} | time (ms): {latency:.2f} | "
                       f"msg size: {convert_size(msg_size)} | "
                       f"algbw (Gbps): {algbw:.2f} | busbw (Gbps): {busbw:.2f}")
            log_dist(log_str, [0])

        # DES-LOC: tier-level accounting
        self.desloc_total_ops += 1
        tier = desloc_tier if desloc_tier is not None else desloc_classify_op(raw_name)
        if tier in self.desloc_tier_bytes:
            self.desloc_tier_bytes[tier] += msg_size
            self.desloc_tier_counts[tier] += 1
            self.desloc_tier_latency_ms[tier] += latency

    def record_skipped_op(self, tier=DESLOC_TIER_PARAM, reason='Kx_gating'):
        """Record that a comm op was skipped due to DES-LOC sync period.

        Called when engine.allreduce_gradients() decides to skip because
        the current step is not a Kx boundary.

        Ref: Algorithm 1 line 10 — skip allreduce when step % Kx != 0.
        """
        self.desloc_skipped_ops += 1

    def get_raw_data(self):
        """Get raw communication data dict."""
        return self.comms_dict.copy()

    def has_data(self, include_desloc=True):
        """Check if any communication data has been logged.

        Args:
            include_desloc: If True, also check DES-LOC tier data.
        """
        if self.comms_dict:
            return True
        if include_desloc and self.desloc_total_ops > 0:
            return True
        return False

    def reset_data(self):
        """Clear all logged data including DES-LOC tier counters."""
        self.comms_dict.clear()
        for t in self.desloc_tier_bytes:
            self.desloc_tier_bytes[t] = 0
            self.desloc_tier_counts[t] = 0
            self.desloc_tier_latency_ms[t] = 0.0
        self.desloc_skipped_ops = 0
        self.desloc_total_ops = 0

    def get_operation_names(self, desloc_tier_filter=None):
        """Get list of logged operation names, optionally filtered by tier.

        Args:
            desloc_tier_filter: If provided, only return ops matching this tier.
        """
        names = list(self.comms_dict.keys())
        if desloc_tier_filter is not None:
            names = [n for n in names if desloc_classify_op(n) == desloc_tier_filter]
        return names

    def get_total_operations(self, by_tier=False):
        """Get total operation count, optionally broken down by tier.

        Args:
            by_tier: If True, return dict mapping tier names to counts.
        """
        if by_tier:
            return {DESLOC_TIER_NAMES.get(t, str(t)): c
                    for t, c in self.desloc_tier_counts.items() if c > 0}
        total = 0
        for record_name in self.comms_dict:
            for msg_size in self.comms_dict[record_name]:
                total += self.comms_dict[record_name][msg_size][0]
        return total

    def get_operation_summary(self, operation_name, desloc_tier=None):
        """Get summary statistics for a specific operation type.

        DES-LOC extension: includes auto-classified tier in the summary.
        """
        if operation_name not in self.comms_dict:
            return None

        from deepspeed.utils.timer import trim_mean
        op_data = self.comms_dict[operation_name].copy()
        summary = {}
        auto_tier = desloc_tier if desloc_tier is not None else desloc_classify_op(operation_name)

        for msg_size, vals in op_data.items():
            count = vals[0]
            total_lat = sum(vals[1])
            avg_lat = trim_mean(vals[1], 0.1)
            avg_algbw = trim_mean(vals[2], 0.1)
            avg_busbw = trim_mean(vals[3], 0.1)
            summary[msg_size] = {
                "count": count,
                "total_latency_ms": total_lat,
                "avg_latency_ms": avg_lat,
                "tput_avg_gbps": avg_algbw,
                "busbw_avg_gbps": avg_busbw,
                "msg_size_bytes": msg_size,
                "msg_size_str": convert_size(msg_size),
                "desloc_tier": auto_tier,
                "desloc_tier_name": DESLOC_TIER_NAMES.get(auto_tier, 'unknown'),
            }
        return summary

    def get_desloc_savings(self):
        """Compute actual communication savings from DES-LOC sync gating.

        Returns dict with skip ratio, bytes sent/saved estimates.
        Ref: Section 5.3 — DES-LOC halves comm vs Local Adam.
        """
        total_attempted = self.desloc_total_ops + self.desloc_skipped_ops
        if total_attempted == 0:
            return {'skip_ratio': 0.0, 'bytes_sent': 0, 'bytes_saved_est': 0}
        total_bytes = sum(self.desloc_tier_bytes.values())
        avg_bytes = total_bytes / max(1, self.desloc_total_ops)
        return {
            'skip_ratio': round(self.desloc_skipped_ops / total_attempted, 6),
            'bytes_sent': total_bytes,
            'bytes_saved_est': int(avg_bytes * self.desloc_skipped_ops),
            'ops_executed': self.desloc_total_ops,
            'ops_skipped': self.desloc_skipped_ops,
            'theoretical_reduction': desloc_comm_reduction(
                self._desloc_Kx, self._desloc_Ku, self._desloc_Kv),
        }

    def log_all(self, print_log=True, show_straggler=False, return_dict=False,
                desloc_summary=True):
        """Print and/or return communication statistics.

        DES-LOC extension: when desloc_summary=True, appends a per-tier
        communication breakdown after the standard log output, showing
        bytes, ops, latency, and bandwidth per tier.

        Args:
            print_log: Whether to print to console.
            show_straggler: Whether to include straggler analysis.
            return_dict: Whether to return stats as dict.
            desloc_summary: Whether to include DES-LOC tier breakdown.
        """
        import torch
        from deepspeed.utils.timer import trim_mean
        import deepspeed.comm as dist
        from deepspeed.comm.reduce_op import ReduceOp
        from deepspeed.accelerator import get_accelerator
        from datetime import datetime

        comms_dict_snapshot = self.comms_dict.copy()

        result_dict = {
            "summary": {},
            "straggler_analysis": None,
            "metadata": {
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                "rank": dist.get_rank() if dist.is_initialized() else 0,
                "timestamp": datetime.now().isoformat(),
            },
        } if return_dict else None

        if print_log:
            print(
                f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}"
                f"{'Total Latency(ms)': <20}{'Avg Latency(ms)': <20}"
                f"{'tput_avg (Gbps)': <20}{'busbw_avg (Gbps)': <20}"
            )

        for record_name in comms_dict_snapshot.keys():
            if print_log:
                print(record_name)
            if return_dict:
                result_dict["summary"][record_name] = {}

            for msg_size, vals in sorted(comms_dict_snapshot[record_name].items()):
                count = vals[0]
                total_lat = sum(vals[1])
                avg_lat = trim_mean(vals[1], 0.1)
                avg_algbw = trim_mean(vals[2], 0.1)
                avg_busbw = trim_mean(vals[3], 0.1)

                if return_dict:
                    result_dict["summary"][record_name][msg_size] = {
                        "count": count,
                        "total_latency_ms": total_lat,
                        "avg_latency_ms": avg_lat,
                        "tput_avg_gbps": avg_algbw,
                        "busbw_avg_gbps": avg_busbw,
                        "msg_size_bytes": msg_size,
                        "msg_size_str": convert_size(msg_size),
                    }

                if print_log:
                    print(
                        f"{' ': <20}{convert_size(msg_size): <20}{count: <20}"
                        f"{total_lat: <20.2f}{avg_lat: <20.2f}"
                        f"{avg_algbw: <20.2f}{avg_busbw: <20.2f}"
                    )

        if show_straggler:
            if return_dict:
                result_dict["straggler_analysis"] = {}
            if print_log:
                print("_______________________________")
                print("Breakdown with straggler effect")
                print("-------------------------------")
                print(
                    f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}"
                    f"{'Total comm lat(ms)': <20}{'Total straggler(ms)': <20}"
                    f"{'Avg comm lat(ms)': <20}{'Avg straggler(ms)': <20}"
                )

            device = get_accelerator().current_device_name()
            for record_name in comms_dict_snapshot.keys():
                if print_log:
                    print(record_name)
                if return_dict:
                    result_dict["straggler_analysis"][record_name] = {}

                for msg_size, vals in sorted(comms_dict_snapshot[record_name].items()):
                    count = vals[0]
                    lats = torch.tensor(vals[1], device=device)
                    min_lats = torch.tensor(vals[1], device=device)
                    dist.all_reduce(min_lats, op=ReduceOp.MIN)
                    total_lat = min_lats.sum().item()
                    total_straggler = (lats - min_lats).sum().item()
                    avg_lat = trim_mean(min_lats.tolist(), 0.1)
                    avg_straggler = trim_mean((lats - min_lats).tolist(), 0.1)

                    if return_dict:
                        result_dict["straggler_analysis"][record_name][msg_size] = {
                            "count": count,
                            "total_comm_lat_ms": total_lat,
                            "total_straggler_ms": total_straggler,
                            "avg_comm_lat_ms": avg_lat,
                            "avg_straggler_ms": avg_straggler,
                            "msg_size_bytes": msg_size,
                            "msg_size_str": convert_size(msg_size),
                        }

                    if print_log:
                        print(
                            f"{' ': <20}{convert_size(msg_size): <20}{count: <20}"
                            f"{total_lat: <20.2f}{total_straggler: <20.2f}"
                            f"{avg_lat: <20.2f}{avg_straggler: <20.2f}"
                        )

        # DES-LOC: tier-level communication breakdown
        if desloc_summary and self.desloc_total_ops > 0:
            if print_log:
                print("")
                print("--- DES-LOC Communication Tier Summary ---")
                print(f"{'Tier':<12}{'Bytes':<18}{'Ops':<8}"
                      f"{'Latency(ms)':<14}{'AvgBW(GB/s)':<14}")
                for tid in sorted(self.desloc_tier_bytes.keys()):
                    b = self.desloc_tier_bytes[tid]
                    c = self.desloc_tier_counts[tid]
                    lat = self.desloc_tier_latency_ms[tid]
                    if c > 0:
                        bw = b / (lat * 1e6) if lat > 0 else 0.0
                        nm = DESLOC_TIER_NAMES.get(tid, str(tid))
                        print(f"{nm:<12}{convert_size(b):<18}{c:<8}"
                              f"{lat:<14.4f}{bw:<14.4f}")
                if self.desloc_skipped_ops > 0:
                    total_att = self.desloc_total_ops + self.desloc_skipped_ops
                    pct = 100.0 * self.desloc_skipped_ops / total_att
                    print(f"Skipped ops (Kx/Ku/Kv gating): "
                          f"{self.desloc_skipped_ops} ({pct:.1f}%)")
                savings = self.get_desloc_savings()
                if 'theoretical_reduction' in savings:
                    tr = savings['theoretical_reduction']
                    print(f"Reduction vs DDP: {tr.get('reduction_vs_ddp', 0)*100:.1f}%")
                    print(f"Reduction vs Local Adam: "
                          f"{tr.get('reduction_vs_local_adam', 0)*100:.1f}%")

            if return_dict:
                result_dict['desloc_tiers'] = {
                    DESLOC_TIER_NAMES.get(t, str(t)): {
                        'bytes': self.desloc_tier_bytes[t],
                        'count': self.desloc_tier_counts[t],
                        'latency_ms': round(self.desloc_tier_latency_ms[t], 4),
                    } for t in self.desloc_tier_bytes if self.desloc_tier_bytes[t] > 0
                }
                result_dict['desloc_savings'] = self.get_desloc_savings()

        return result_dict if return_dict else None


class DeslocLogParser:
    """Parse DES-LOC experiment logs into structured records.
    Ref: NKI-FA exp_utils/draw_plot.py — regex-based log parsing."""

    STEP_PATTERN = r'desloc_step:\s*(\d+)\s*\|\s*is_sync:\s*(\d)\s*\|.*?loss:\s*([\d.]+)'

    @staticmethod
    def parse_file(filepath):
        import re
        records = []
        pat = re.compile(DeslocLogParser.STEP_PATTERN)
        with open(filepath) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    records.append({
                        'step': int(m.group(1)),
                        'is_sync': int(m.group(2)),
                        'loss': float(m.group(3)),
                    })
        return records

    @staticmethod
    def parse_throughput(filepath):
        import re
        records = []
        pat = re.compile(r'throughput:\s*([\d.]+)\s*samples/s')
        with open(filepath) as f:
            for line in f:
                m = pat.search(line)
                if m:
                    records.append(float(m.group(1)))
        return records


class DeslocFigureDataPreparer:
    """Prepare parsed experiment data for NeurIPS figures.
    Ref: Section 5 RQ1-RQ6. All data from logs only."""

    def __init__(self, parser):
        self.parser = parser

    def prepare_rq3_comm(self):
        groups = self.parser.group_by('Kx')
        return {kx: {
            'avg_bytes': sum(e['metrics'].get('comm_bytes', 0) for e in exps) / max(1, len(exps)),
            'n': len(exps),
            'theory': desloc_comm_reduction(kx, kx * 3, kx * 6),
        } for kx, exps in groups.items()}

    def prepare_figure1_loss(self, model_size=None):
        """Extract Figure 1 loss curve data grouped by method.

        Returns: dict {method_label: [(step, loss), ...]}

        Ref: Section 5.4 — Figure shows loss vs step for DDP, DES-LOC,
        Local Adam across model scales.
        Ref: NKI-FA draw_plot.py — parsed from log strings.
        """
        groups = self.parser.group_by('Kx')
        result = {}
        for kx, exps in sorted(groups.items()):
            if model_size and any(e.get('config', {}).get('model_size') != model_size for e in exps):
                continue
            label = 'DDP' if kx == 1 else f'DES-LOC (Kx={kx})'
            curves = []
            for exp in exps:
                m = exp.get('metrics', {})
                steps = m.get('steps', [])
                losses = m.get('losses', [])
                if steps and losses:
                    curves.append(list(zip(steps, losses)))
            if curves:
                # Average across seeds
                avg_curve = self._average_curves(curves)
                result[label] = avg_curve
        return result

    def prepare_figure2_comm(self):
        """Extract Figure 2 communication data grouped by Kx.

        Returns: list of dicts for bar chart:
            [{'Kx': 1, 'label': 'DDP', 'comm_gb': 12.456,
              'reduction_vs_ddp': 1.0}, ...]

        Ref: Section 5.3 — comm volume comparison.
        """
        groups = self.parser.group_by('Kx')
        ddp_bytes = None
        results = []
        for kx, exps in sorted(groups.items()):
            avg_bytes = sum(
                e.get('metrics', {}).get('comm_bytes', 0)
                for e in exps) / max(1, len(exps))
            if kx == 1:
                ddp_bytes = avg_bytes
            results.append({
                'Kx': kx,
                'label': 'DDP' if kx == 1 else f'Kx={kx}',
                'comm_gb': round(avg_bytes / (1024**3), 3),
                'comm_bytes': int(avg_bytes),
                'n_runs': len(exps),
            })
        # Compute reduction ratios
        if ddp_bytes and ddp_bytes > 0:
            for r in results:
                r['reduction_vs_ddp'] = round(
                    ddp_bytes / max(1, r['comm_bytes']), 1)
        else:
            for r in results:
                r['reduction_vs_ddp'] = 1.0
        return results

    def _average_curves(self, curves):
        """Average multiple (step, value) curves.

        Aligns by step index, computes mean across seeds.
        Pure python — no numpy.
        """
        if not curves:
            return []
        if len(curves) == 1:
            return curves[0]
        # Find common step range
        min_len = min(len(c) for c in curves)
        averaged = []
        for i in range(min_len):
            step = curves[0][i][0]
            values = [c[i][1] for c in curves if i < len(c)]
            avg_val = sum(values) / len(values)
            averaged.append((step, avg_val))
        return averaged


class DeslocFigure1Formatter:
    """Format Figure 1 data for NKI-FA log output.

    Ref: NKI-FA da964f3 draw_plot.py — each experiment block starts
    with ### config ### header, then metric: value lines.

    Figure 1 = Loss vs Training Step, multiple lines (DDP, DES-LOC variants).
    """

    @staticmethod
    def format_loss_curve(label, curve_data, kx, ku, kv):
        """Format one loss curve as NKI-FA log block.

        Args:
            label: str — legend label
            curve_data: list of (step, loss) tuples
            kx, ku, kv: sync periods

        Returns: str — NKI-FA format block
        """
        lines = [f'### method = {label}, Kx = {kx}, '
                 f'Ku = {ku}, Kv = {kv} ###']
        for step, loss in curve_data:
            lines.append(f'step: {step} | loss: {loss:.6f}')
        if curve_data:
            lines.append('')
            lines.append('--- summary ---')
            losses = [l for _, l in curve_data]
            lines.append(f'first_loss: {losses[0]:.6f}')
            lines.append(f'final_loss: {losses[-1]:.6f}')
            lines.append(f'min_loss: {min(losses):.6f}')
            lines.append(f'total_points: {len(losses)}')
            lines.append('--- end summary ---')
        return '\n'.join(lines)

    @staticmethod
    def format_all_curves(prepared_data, output_path=None):
        """Format all Figure 1 curves into single log file.

        Args:
            prepared_data: dict from prepare_figure1_loss()
            output_path: optional file path to write

        Returns: str — complete log content
        """
        blocks = []
        for label, curve in sorted(prepared_data.items()):
            # Infer Kx from label
            if 'DDP' in label:
                kx, ku, kv = 1, 1, 1
            elif 'Kx=' in label:
                import re
                m = re.search(r'Kx=(\d+)', label)
                kx = int(m.group(1)) if m else 32
                ku, kv = 3 * kx, 6 * kx
            else:
                kx, ku, kv = 32, 96, 192
            block = DeslocFigure1Formatter.format_loss_curve(
                label, curve, kx, ku, kv)
            blocks.append(block)
        content = '\n\n'.join(blocks)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        return content


class DeslocFigure2Formatter:
    """Format Figure 2 data for NKI-FA log output.

    Figure 2 = Communication Volume bar chart (DDP vs DES-LOC at various Kx).
    Annotations show exact GB values and reduction ratios.

    Ref: NKI-FA draw_plot.py:
        g_tflops.annotate(f"{p.get_height():.1f}", ...)
    """

    @staticmethod
    def format_comm_bars(bar_data, output_path=None):
        """Format Figure 2 bar chart data.

        Args:
            bar_data: list from prepare_figure2_comm()
            output_path: optional output file

        Returns: str — NKI-FA format log
        """
        lines = ['### figure = 2, type = comm_reduction ###']
        for item in bar_data:
            lines.append(
                f'Kx: {item["Kx"]} | label: {item["label"]} | '
                f'comm_gb: {item["comm_gb"]:.3f} | '
                f'reduction_vs_ddp: {item["reduction_vs_ddp"]:.1f}')
        lines.append('')
        lines.append('--- summary ---')
        if len(bar_data) >= 2:
            ddp_gb = bar_data[0]['comm_gb']
            best = min(bar_data, key=lambda x: x['comm_gb'])
            lines.append(f'ddp_comm_gb: {ddp_gb:.3f}')
            lines.append(f'best_config: {best["label"]}')
            lines.append(f'best_comm_gb: {best["comm_gb"]:.3f}')
            lines.append(f'max_reduction: {best["reduction_vs_ddp"]:.1f}')
        lines.append('--- end summary ---')
        content = '\n'.join(lines)
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        return content


# =====================================================================
# M244 — Claude-16: DES-LOC Hierarchical Communication Logger
# Intra-node vs inter-node tracking, bandwidth utilization,
# per-tier histograms, NKI-FA log export
# Ref: NCCL transport.cc — P2P/SHM/NET transport classification
# Ref: Megatron-LM — distributed_data_parallel.py bucket logging
# =====================================================================

import time as _m244_time
import math as _m244_math


class DeslocHierarchicalCommLogger:
    """Hierarchical communication logger for DES-LOC.

    Tracks communication at three levels:
    1. Transport level: P2P (NVLink), SHM (shared memory), NET (Ethernet/IB)
    2. Tier level: params (Kx), momentum1 (Ku), momentum2 (Kv)
    3. Operation level: AllReduce, ReduceScatter, AllGather, Broadcast

    Each event records: timestamp, bytes, duration, tier, transport, op_type.
    """

    TRANSPORT_P2P = 'p2p'
    TRANSPORT_SHM = 'shm'
    TRANSPORT_NET = 'net'

    OP_ALLREDUCE = 'all_reduce'
    OP_REDUCE_SCATTER = 'reduce_scatter'
    OP_ALLGATHER = 'all_gather'
    OP_BROADCAST = 'broadcast'

    def __init__(self, max_events=100000):
        self._max_events = max_events
        self._events = []
        self._tier_bytes = {'x': 0, 'u': 0, 'v': 0}
        self._tier_ops = {'x': 0, 'u': 0, 'v': 0}
        self._tier_skipped = {'x': 0, 'u': 0, 'v': 0}
        self._transport_bytes = {'p2p': 0, 'shm': 0, 'net': 0}
        self._op_counts = {}
        self._total_duration_ms = 0.0
        self._peak_bw_gbps = 0.0
        self._start_time = _m244_time.monotonic()

    def record_event(self, nbytes, duration_ms, tier='x',
                     transport='p2p', op_type='all_reduce'):
        """Record a single communication event."""
        if len(self._events) < self._max_events:
            self._events.append({
                't': _m244_time.monotonic() - self._start_time,
                'bytes': nbytes,
                'dur_ms': duration_ms,
                'tier': tier,
                'transport': transport,
                'op': op_type,
            })

        # Update aggregates
        if tier in self._tier_bytes:
            self._tier_bytes[tier] += nbytes
            self._tier_ops[tier] += 1

        if transport in self._transport_bytes:
            self._transport_bytes[transport] += nbytes

        self._op_counts[op_type] = self._op_counts.get(op_type, 0) + 1
        self._total_duration_ms += duration_ms

        # Track peak bandwidth
        if duration_ms > 0:
            bw_gbps = (nbytes * 8) / (duration_ms * 1e6)  # bits/ms → Gbps
            if bw_gbps > self._peak_bw_gbps:
                self._peak_bw_gbps = bw_gbps

    def record_skip(self, tier='x'):
        """Record a skipped communication (DES-LOC gated)."""
        if tier in self._tier_skipped:
            self._tier_skipped[tier] += 1

    def bandwidth_utilization(self, peak_bw_gbps=400.0):
        """Calculate bandwidth utilization as fraction of peak."""
        total_bytes = sum(self._tier_bytes.values())
        total_dur_s = self._total_duration_ms / 1000.0
        if total_dur_s <= 0:
            return 0.0
        actual_gbps = (total_bytes * 8) / (total_dur_s * 1e9)
        return round(actual_gbps / peak_bw_gbps, 6)

    def tier_report(self):
        """Generate per-tier communication report."""
        report = {}
        for tier in ('x', 'u', 'v'):
            total_ops = self._tier_ops[tier] + self._tier_skipped[tier]
            skip_pct = (100.0 * self._tier_skipped[tier] /
                        max(1, total_ops))
            tier_name = {'x': 'params', 'u': 'momentum1', 'v': 'momentum2'}
            report[tier_name[tier]] = {
                'bytes': self._tier_bytes[tier],
                'bytes_gb': round(self._tier_bytes[tier] / 1e9, 4),
                'ops': self._tier_ops[tier],
                'skipped': self._tier_skipped[tier],
                'skip_pct': round(skip_pct, 2),
            }
        return report

    def transport_report(self):
        """Generate per-transport communication report."""
        total = sum(self._transport_bytes.values())
        report = {}
        for transport in ('p2p', 'shm', 'net'):
            frac = self._transport_bytes[transport] / max(1, total)
            transport_name = {'p2p': 'NVLink/PCIe', 'shm': 'SharedMem',
                              'net': 'Ethernet/IB'}
            report[transport_name[transport]] = {
                'bytes': self._transport_bytes[transport],
                'fraction': round(frac, 4),
            }
        return report

    def latency_histogram(self, nbins=10):
        """Build latency histogram from recorded events."""
        if not self._events:
            return {'bins': [], 'counts': []}

        durations = [e['dur_ms'] for e in self._events if e['dur_ms'] > 0]
        if not durations:
            return {'bins': [], 'counts': []}

        min_d = min(durations)
        max_d = max(durations)
        if max_d <= min_d:
            return {'bins': [min_d], 'counts': [len(durations)]}

        bin_width = (max_d - min_d) / nbins
        bins = [min_d + i * bin_width for i in range(nbins + 1)]
        counts = [0] * nbins

        for d in durations:
            idx = min(int((d - min_d) / bin_width), nbins - 1)
            counts[idx] += 1

        return {
            'bins': [round(b, 4) for b in bins],
            'counts': counts,
            'mean_ms': round(sum(durations) / len(durations), 4),
            'p50_ms': round(sorted(durations)[len(durations) // 2], 4),
            'p99_ms': round(sorted(durations)[min(len(durations) - 1,
                            int(len(durations) * 0.99))], 4),
        }

    def export_nkifa_format(self, config_dict=None):
        """Export stats in NKI-FA log format.

        ### Kx = 32, tier = params, transport = NVLink ###
        total_bytes: 15099494400
        total_ops: 156
        ...
        """
        lines = []
        tier_rep = self.tier_report()
        for tier_name, data in tier_rep.items():
            cfg = dict(config_dict or {})
            cfg['tier'] = tier_name
            header_parts = [f'{k} = {v}' for k, v in sorted(cfg.items())]
            lines.append('### ' + ', '.join(header_parts) + ' ###')
            for k, v in sorted(data.items()):
                if isinstance(v, float):
                    lines.append(f'{k}: {v:.6f}')
                else:
                    lines.append(f'{k}: {v}')
            lines.append('')
        return '\n'.join(lines)

    def summary(self):
        """One-line summary."""
        total_bytes = sum(self._tier_bytes.values())
        total_ops = sum(self._tier_ops.values())
        total_skipped = sum(self._tier_skipped.values())
        total_all = total_ops + total_skipped
        skip_pct = 100.0 * total_skipped / max(1, total_all)
        return (f'CommLog: {total_bytes/1e9:.2f}GB in {total_ops} ops, '
                f'{total_skipped} skipped ({skip_pct:.1f}%), '
                f'peak_bw={self._peak_bw_gbps:.1f}Gbps')


class DeslocCommBudgetTracker:
    """Track communication budget relative to DDP baseline.

    Computes running "comm savings" in real-time during training.
    Alerts if actual comm exceeds expected DES-LOC budget.
    """

    def __init__(self, model_params, Kx, Ku, Kv, dtype_bytes=2):
        self._model_params = model_params
        self._Kx = max(1, Kx)
        self._Ku = max(1, Ku)
        self._Kv = max(1, Kv)
        self._dtype_bytes = dtype_bytes
        self._step = 0
        self._actual_bytes = 0
        self._ddp_baseline_bytes = 0

    def step(self, actual_bytes_this_step):
        """Record actual comm for this step and update baseline."""
        self._step += 1
        self._actual_bytes += actual_bytes_this_step
        # DDP would have communicated every step
        ar_bytes = 2 * self._model_params * self._dtype_bytes
        self._ddp_baseline_bytes += ar_bytes

    def savings_ratio(self):
        """Fraction of DDP comm that was saved."""
        if self._ddp_baseline_bytes <= 0:
            return 0.0
        return round(1.0 - self._actual_bytes / self._ddp_baseline_bytes, 6)

    def theoretical_savings(self):
        """Expected savings from DES-LOC config."""
        desloc_freq = (1.0 / self._Kx + 1.0 / self._Ku + 1.0 / self._Kv)
        ddp_freq = 3.0  # params + mom1 + mom2 every step
        return round(1.0 - desloc_freq / ddp_freq, 6)

    def is_on_budget(self, tolerance=0.1):
        """Check if actual savings are within tolerance of theoretical."""
        actual = self.savings_ratio()
        theoretical = self.theoretical_savings()
        return abs(actual - theoretical) < tolerance

    def report(self):
        return {
            'step': self._step,
            'actual_bytes': self._actual_bytes,
            'ddp_baseline_bytes': self._ddp_baseline_bytes,
            'actual_savings': self.savings_ratio(),
            'theoretical_savings': self.theoretical_savings(),
            'on_budget': self.is_on_budget(),
        }


# M244: end of Claude-16 comms_logging integration


# =====================================================================
# M244-TOPUP — Claude-16: DES-LOC Comm Anomaly Detector + Rate Limiter
# Detects: bandwidth drops, latency spikes, asymmetric traffic
# Ref: Nick Joseph — 'a capacitor fault can crash the entire training'
# Ref: NCCL debug.cc — multi-level logging (WARN/INFO/TRACE)
# =====================================================================

import math as _m244t_math


class DeslocCommAnomalyDetector:
    """Detect anomalous communication patterns during DES-LOC training.

    Anomaly types:
    1. Bandwidth drop: sustained < 50% of baseline
    2. Latency spike: > 5x median latency
    3. Asymmetric traffic: one tier sends 10x more than expected
    4. Silent failure: no comm for > 2*Kx steps (possible NCCL hang)
    5. NaN propagation: NaN detected in post-AllReduce buffer
    """

    ANOMALY_BW_DROP = 'bandwidth_drop'
    ANOMALY_LAT_SPIKE = 'latency_spike'
    ANOMALY_ASYMMETRIC = 'asymmetric_traffic'
    ANOMALY_SILENT = 'silent_failure'
    ANOMALY_NAN = 'nan_propagation'

    def __init__(self, Kx=32, baseline_bw_gbps=None,
                 bw_drop_threshold=0.5, lat_spike_factor=5.0,
                 max_anomalies=1000):
        self._Kx = max(1, Kx)
        self._baseline_bw = baseline_bw_gbps
        self._bw_threshold = bw_drop_threshold
        self._lat_factor = lat_spike_factor
        self._max_anomalies = max_anomalies

        self._bw_samples = []
        self._lat_samples = []
        self._tier_bytes = {'x': 0, 'u': 0, 'v': 0}
        self._last_comm_step = 0
        self._anomalies = []
        self._nan_count = 0

    def record_comm_event(self, step, nbytes, duration_ms, tier='x',
                          had_nan=False):
        """Record a communication event and check for anomalies."""
        self._last_comm_step = step

        # Track bandwidth
        if duration_ms > 0:
            bw = (nbytes * 8) / (duration_ms * 1e6)  # Gbps
            self._bw_samples.append(bw)
            if len(self._bw_samples) > 200:
                self._bw_samples = self._bw_samples[-200:]

            # Set baseline from first 20 samples
            if self._baseline_bw is None and len(self._bw_samples) >= 20:
                self._baseline_bw = sum(self._bw_samples[:20]) / 20

            # Check bandwidth drop
            if (self._baseline_bw and len(self._bw_samples) >= 10):
                recent_bw = sum(self._bw_samples[-10:]) / 10
                if recent_bw < self._baseline_bw * self._bw_threshold:
                    self._record_anomaly(step, self.ANOMALY_BW_DROP, {
                        'recent_bw': round(recent_bw, 4),
                        'baseline_bw': round(self._baseline_bw, 4),
                        'ratio': round(recent_bw / self._baseline_bw, 4),
                    })

        # Track latency
        self._lat_samples.append(duration_ms)
        if len(self._lat_samples) > 200:
            self._lat_samples = self._lat_samples[-200:]

        # Check latency spike
        if len(self._lat_samples) >= 20:
            sorted_lats = sorted(self._lat_samples[-50:])
            median_lat = sorted_lats[len(sorted_lats) // 2]
            if median_lat > 0 and duration_ms > median_lat * self._lat_factor:
                self._record_anomaly(step, self.ANOMALY_LAT_SPIKE, {
                    'current_ms': round(duration_ms, 4),
                    'median_ms': round(median_lat, 4),
                    'factor': round(duration_ms / median_lat, 2),
                })

        # Track per-tier traffic
        if tier in self._tier_bytes:
            self._tier_bytes[tier] += nbytes

        # NaN check
        if had_nan:
            self._nan_count += 1
            self._record_anomaly(step, self.ANOMALY_NAN, {
                'total_nans': self._nan_count,
                'tier': tier,
            })

    def check_silent_failure(self, current_step):
        """Check if comm has been silent for too long."""
        gap = current_step - self._last_comm_step
        max_gap = self._Kx * 2
        if gap > max_gap and self._last_comm_step > 0:
            self._record_anomaly(current_step, self.ANOMALY_SILENT, {
                'gap_steps': gap,
                'max_expected': max_gap,
                'last_comm_step': self._last_comm_step,
            })
            return True
        return False

    def check_asymmetric_traffic(self, expected_ratios=None):
        """Check for asymmetric per-tier traffic.

        Expected ratios: x:u:v ≈ Ku/Kx : 1 : Kx/Kv
        (params sync most, mom2 syncs least)
        """
        total = sum(self._tier_bytes.values())
        if total <= 0:
            return False

        if expected_ratios is None:
            expected_ratios = {'x': 0.5, 'u': 0.3, 'v': 0.2}

        for tier, expected_frac in expected_ratios.items():
            actual_frac = self._tier_bytes.get(tier, 0) / total
            if expected_frac > 0 and actual_frac > expected_frac * 10:
                self._record_anomaly(-1, self.ANOMALY_ASYMMETRIC, {
                    'tier': tier,
                    'actual_frac': round(actual_frac, 4),
                    'expected_frac': round(expected_frac, 4),
                })
                return True
        return False

    def _record_anomaly(self, step, anomaly_type, details):
        """Record an anomaly event."""
        self._anomalies.append({
            'step': step,
            'type': anomaly_type,
            'details': details,
        })
        if len(self._anomalies) > self._max_anomalies:
            self._anomalies = self._anomalies[-self._max_anomalies:]

    def get_anomalies(self, last_n=50):
        """Get recent anomalies."""
        return self._anomalies[-last_n:]

    def anomaly_count_by_type(self):
        """Count anomalies by type."""
        counts = {}
        for a in self._anomalies:
            t = a['type']
            counts[t] = counts.get(t, 0) + 1
        return counts

    def health_score(self):
        """0.0 (unhealthy) to 1.0 (healthy) health score."""
        if not self._anomalies:
            return 1.0
        # Recent anomalies weigh more
        recent = self._anomalies[-20:]
        penalty = len(recent) * 0.05
        return round(max(0.0, 1.0 - penalty), 4)

    def report(self):
        """Full anomaly report."""
        return {
            'health_score': self.health_score(),
            'total_anomalies': len(self._anomalies),
            'by_type': self.anomaly_count_by_type(),
            'nan_count': self._nan_count,
            'baseline_bw_gbps': (round(self._baseline_bw, 4)
                                  if self._baseline_bw else None),
            'bw_samples': len(self._bw_samples),
            'lat_samples': len(self._lat_samples),
        }


class DeslocCommRateLimiter:
    """Rate-limit communication to prevent network saturation.

    When multiple tiers sync simultaneously (e.g., step divisible
    by both Kx and Ku), stagger the AllReduces to avoid congestion.

    Ref: NCCL enqueue.cc — async collective queuing
    """

    def __init__(self, max_concurrent_ops=2, stagger_ms=1.0):
        self._max_concurrent = max_concurrent_ops
        self._stagger_ms = stagger_ms
        self._pending_count = 0
        self._total_staggered = 0
        self._total_passed = 0

    def should_stagger(self, n_ops_this_step):
        """Check if operations should be staggered this step."""
        if n_ops_this_step <= self._max_concurrent:
            self._total_passed += n_ops_this_step
            return False
        self._total_staggered += (n_ops_this_step - self._max_concurrent)
        self._total_passed += self._max_concurrent
        return True

    def get_stagger_schedule(self, tiers_to_sync):
        """Create staggered schedule for multiple tier syncs.

        Returns list of (tier, delay_ms) pairs.
        Priority: x > u > v (params first, mom2 last)
        """
        priority = {'x': 0, 'u': 1, 'v': 2}
        sorted_tiers = sorted(tiers_to_sync,
                               key=lambda t: priority.get(t, 99))

        schedule = []
        for i, tier in enumerate(sorted_tiers):
            delay = 0.0 if i < self._max_concurrent else (
                (i - self._max_concurrent + 1) * self._stagger_ms
            )
            schedule.append({'tier': tier, 'delay_ms': delay, 'order': i})

        return schedule

    def stats(self):
        total = self._total_passed + self._total_staggered
        stagger_pct = (100.0 * self._total_staggered /
                       max(1, total))
        return {
            'total_ops': total,
            'passed': self._total_passed,
            'staggered': self._total_staggered,
            'stagger_pct': round(stagger_pct, 2),
        }


# M244-TOPUP: end


class DeslocCommWindowAnalyzer:
    """Sliding window analysis of DES-LOC communication patterns.

    Computes moving averages of bytes/latency/ops per window,
    detects trends, and predicts future comm load.
    """

    def __init__(self, window_size=100):
        self._window_size = window_size
        self._bytes_window = []
        self._lat_window = []
        self._ops_window = []
        self._step_window = []

    def push(self, step, nbytes, latency_ms, n_ops=1):
        """Push one step's comm data into the window."""
        self._bytes_window.append(nbytes)
        self._lat_window.append(latency_ms)
        self._ops_window.append(n_ops)
        self._step_window.append(step)
        if len(self._bytes_window) > self._window_size:
            self._bytes_window.pop(0)
            self._lat_window.pop(0)
            self._ops_window.pop(0)
            self._step_window.pop(0)

    def mean_bytes(self):
        if not self._bytes_window:
            return 0.0
        return sum(self._bytes_window) / len(self._bytes_window)

    def mean_latency_ms(self):
        if not self._lat_window:
            return 0.0
        vals = [v for v in self._lat_window if v > 0]
        return sum(vals) / max(1, len(vals))

    def bytes_trend(self):
        """Positive = increasing comm, negative = decreasing."""
        n = len(self._bytes_window)
        if n < 10:
            return 0.0
        first_half = sum(self._bytes_window[:n//2]) / max(1, n//2)
        second_half = sum(self._bytes_window[n//2:]) / max(1, n - n//2)
        if first_half <= 0:
            return 0.0
        return round((second_half - first_half) / first_half, 6)

    def latency_percentiles(self):
        """Compute p50, p90, p99 latency."""
        vals = sorted(v for v in self._lat_window if v > 0)
        if not vals:
            return {'p50': 0.0, 'p90': 0.0, 'p99': 0.0}
        n = len(vals)
        return {
            'p50': round(vals[n // 2], 4),
            'p90': round(vals[min(n - 1, int(n * 0.9))], 4),
            'p99': round(vals[min(n - 1, int(n * 0.99))], 4),
        }

    def predict_next_window_bytes(self):
        """Simple linear extrapolation of comm bytes."""
        trend = self.bytes_trend()
        current_mean = self.mean_bytes()
        return round(current_mean * (1.0 + trend), 2)

    def comm_burstiness(self):
        """Measure how bursty communication is.
        Burstiness = std/mean of bytes per step.
        DES-LOC should have periodic bursts (every Kx steps)."""
        if len(self._bytes_window) < 5:
            return 0.0
        mean = self.mean_bytes()
        if mean <= 0:
            return 0.0
        variance = sum((b - mean)**2 for b in self._bytes_window) / len(self._bytes_window)
        std = variance ** 0.5
        return round(std / mean, 6)

    def detect_periodicity(self):
        """Detect dominant communication period (should match Kx).
        Uses simple autocorrelation on bytes_window."""
        n = len(self._bytes_window)
        if n < 20:
            return {'detected_period': 0, 'confidence': 0.0}

        mean = self.mean_bytes()
        if mean <= 0:
            return {'detected_period': 0, 'confidence': 0.0}

        centered = [b - mean for b in self._bytes_window]
        var = sum(c * c for c in centered) / n

        if var <= 0:
            return {'detected_period': 0, 'confidence': 0.0}

        best_lag = 0
        best_corr = 0.0

        for lag in range(1, min(n // 2, 256)):
            corr = sum(centered[i] * centered[i + lag]
                       for i in range(n - lag)) / ((n - lag) * var)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        return {
            'detected_period': best_lag,
            'confidence': round(best_corr, 6),
        }

    def full_report(self):
        """Full sliding window report."""
        percs = self.latency_percentiles()
        period = self.detect_periodicity()
        return {
            'window_size': len(self._bytes_window),
            'mean_bytes': round(self.mean_bytes(), 2),
            'mean_latency_ms': round(self.mean_latency_ms(), 4),
            'bytes_trend': self.bytes_trend(),
            'burstiness': self.comm_burstiness(),
            'predicted_next_bytes': self.predict_next_window_bytes(),
            'latency_p50_ms': percs['p50'],
            'latency_p90_ms': percs['p90'],
            'latency_p99_ms': percs['p99'],
            'detected_period': period['detected_period'],
            'period_confidence': period['confidence'],
        }


# M244-TOPUP2: end
