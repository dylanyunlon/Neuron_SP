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
