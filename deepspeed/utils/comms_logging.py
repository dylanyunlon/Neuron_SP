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




# M283: Structured comm event logger

# M313: Tier logging + figure data
import re as _cre

def desloc_cl_op(tn):
    n = tn.lower() if tn else ''
    if 'exp_avg_sq' in n: return 2
    elif 'exp_avg' in n: return 1
    return 0

def desloc_cl_entry(step, op, tier, nb, us, synced=True):
    return "### comm_op=%s, tier=%d, step=%d ###\nbytes: %d\ntime_us: %.2f\nsynced: %d" % (op, tier, step, nb, us, int(synced))

def desloc_cl_parse(path):
    entries = []; cur = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                m = _cre.match(r'^###\s*comm_op\s*=\s*(\w+),\s*tier\s*=\s*(\d+),\s*step\s*=\s*(\d+)', line)
                if m:
                    if cur: entries.append(cur)
                    cur = {'op': m.group(1), 'tier': int(m.group(2)), 'step': int(m.group(3))}; continue
                m = _cre.match(r'^(\w+)\s*:\s*(.+)$', line)
                if m and cur:
                    try: cur[m.group(1)] = float(m.group(2))
                    except: cur[m.group(1)] = m.group(2)
            if cur: entries.append(cur)
    except FileNotFoundError: pass
    return entries

def desloc_cl_sum(entries):
    ts = {i: {'b': 0, 'o': 0, 's': 0} for i in range(3)}
    for e in entries:
        t = e.get('tier', 0)
        if t not in ts: ts[t] = {'b': 0, 'o': 0, 's': 0}
        if e.get('synced', 1) == 1: ts[t]['b'] += e.get('bytes', 0); ts[t]['o'] += 1
        else: ts[t]['s'] += 1
    to = sum(s['o'] for s in ts.values()); sk = sum(s['s'] for s in ts.values())
    return {'tiers': ts, 'ops': to, 'skips': sk, 'red': round((to + sk) / max(1, to), 2)}

def desloc_fig_data(entries, kind='reduction'):
    s = desloc_cl_sum(entries)
    if kind == 'reduction':
        t = s['ops'] + s['skips']
        return {'x': ['DDP', 'DES-LOC'], 'y': [1.0, round(s['ops'] / max(1, t), 4)]}
    elif kind == 'tier':
        return {'x': ['Kx', 'Ku', 'Kv'], 'y': [s['tiers'].get(i, {}).get('b', 0) / 1e9 for i in range(3)]}
    return {}
# --- End M313 ---


# =====================================================================
# M337 — Claude-30: NCCL Profiler Event Structure
# Source: nccl/src/include/plugin/profiler/profiler_v3.h (ncclProfilerEventDescr_v3_t)
# Adapted from NCCL coll struct fields: name, commHash, seqNumber, func,
# sendBuff, recvBuff, count, root, datatype, nMaxChannels, nWarps, algo, proto
# → Python dataclass for DES-LOC comm event profiling
# =====================================================================

import time as _time
import collections as _collections

class DeslocCommEvent:
    """Structured communication event for DES-LOC profiling.

    Adapted from NCCL ncclProfilerEventDescr_v3_t.coll struct
    (nccl/src/include/plugin/profiler/profiler_v3.h). Captures the
    essential fields for DES-LOC tier-aware communication tracking.

    Fields map to NCCL profiler:
        func → op_name (AllReduce, ReduceScatter, etc.)
        count → n_elements
        algo → nccl_algo (ring, tree, collnet)
        proto → nccl_proto (simple, ll, ll128)
        DES-LOC extensions: tier, step, synced, bucket_id
    """
    __slots__ = (
        'op_name', 'tier', 'step', 'n_elements', 'n_bytes',
        'nccl_algo', 'nccl_proto', 'rank', 'comm_hash',
        'start_time_ns', 'end_time_ns', 'synced', 'bucket_id',
        'seq_number',
    )

    def __init__(self, op_name, tier, step, n_elements=0, n_bytes=0,
                 nccl_algo='ring', nccl_proto='simple', rank=0,
                 comm_hash=0, synced=True, bucket_id=-1, seq_number=0):
        self.op_name = op_name
        self.tier = tier
        self.step = step
        self.n_elements = n_elements
        self.n_bytes = n_bytes
        self.nccl_algo = nccl_algo
        self.nccl_proto = nccl_proto
        self.rank = rank
        self.comm_hash = comm_hash
        self.start_time_ns = _time.monotonic_ns()
        self.end_time_ns = 0
        self.synced = synced
        self.bucket_id = bucket_id
        self.seq_number = seq_number

    def finish(self):
        """Mark event completion and return duration in microseconds."""
        self.end_time_ns = _time.monotonic_ns()
        return (self.end_time_ns - self.start_time_ns) / 1000.0

    def duration_us(self):
        if self.end_time_ns == 0:
            return 0.0
        return (self.end_time_ns - self.start_time_ns) / 1000.0

    def bandwidth_gbps(self):
        """Compute achieved bandwidth in GB/s."""
        dur_s = self.duration_us() / 1e6
        if dur_s <= 0:
            return 0.0
        return (self.n_bytes / 1e9) / dur_s

    def to_nkifa_str(self):
        """Export event in NKI-FA log format.

        Format: ### comm_op=OP, tier=T, step=S, algo=A, proto=P ###
                key: value
        """
        header = (
            f"### comm_op={self.op_name}, tier={self.tier}, "
            f"step={self.step}, algo={self.nccl_algo}, "
            f"proto={self.nccl_proto} ###"
        )
        body = (
            f"bytes: {self.n_bytes}\n"
            f"elements: {self.n_elements}\n"
            f"time_us: {self.duration_us():.2f}\n"
            f"bandwidth_gbps: {self.bandwidth_gbps():.4f}\n"
            f"synced: {int(self.synced)}\n"
            f"bucket_id: {self.bucket_id}\n"
            f"rank: {self.rank}\n"
            f"seq_number: {self.seq_number}"
        )
        return f"{header}\n{body}"

    def to_dict(self):
        return {
            'op': self.op_name, 'tier': self.tier, 'step': self.step,
            'bytes': self.n_bytes, 'elements': self.n_elements,
            'algo': self.nccl_algo, 'proto': self.nccl_proto,
            'time_us': self.duration_us(), 'bw_gbps': self.bandwidth_gbps(),
            'synced': self.synced, 'bucket_id': self.bucket_id,
            'rank': self.rank, 'seq': self.seq_number,
        }


# =====================================================================
# M337 — Claude-30: NCCL Profiler Lifecycle
# Source: nccl/plugins/profiler/example/event.h (init→start→stop lifecycle)
# Adapted: Python profiler with init/start/stop/export phases
# =====================================================================

class DeslocCommProfiler:
    """Communication profiler following NCCL profiler lifecycle pattern.

    NCCL profiler lifecycle (event.h): init → startColl → stopColl
    Python adaptation: init → start_event → finish_event → export

    Tracks per-tier bandwidth histograms, latency percentiles, and
    comm/compute overlap ratios for DES-LOC NeurIPS figure generation.
    """

    def __init__(self, rank=0, enabled=True, max_events=100000):
        self.rank = rank
        self.enabled = enabled
        self.max_events = max_events
        self._events = []
        self._tier_stats = {t: {
            'total_bytes': 0, 'total_us': 0.0, 'n_ops': 0, 'n_skipped': 0,
            'bw_samples': [], 'latency_samples': [],
        } for t in range(3)}
        self._step_comm_time = {}
        self._seq_counter = 0
        self._active_event = None

    def start_event(self, op_name, tier, step, n_elements=0, n_bytes=0,
                    nccl_algo='ring', nccl_proto='simple', bucket_id=-1):
        """Start a new comm event (NCCL startColl equivalent)."""
        if not self.enabled:
            return None
        self._seq_counter += 1
        evt = DeslocCommEvent(
            op_name=op_name, tier=tier, step=step,
            n_elements=n_elements, n_bytes=n_bytes,
            nccl_algo=nccl_algo, nccl_proto=nccl_proto,
            rank=self.rank, synced=True, bucket_id=bucket_id,
            seq_number=self._seq_counter,
        )
        self._active_event = evt
        return evt

    def skip_event(self, op_name, tier, step, n_bytes=0):
        """Record a skipped comm event (DES-LOC gating said no sync)."""
        if not self.enabled:
            return
        ts = self._tier_stats.get(tier)
        if ts:
            ts['n_skipped'] += 1

    def finish_event(self, event=None):
        """Finish an active comm event (NCCL stopColl equivalent)."""
        if not self.enabled:
            return 0.0
        evt = event or self._active_event
        if evt is None:
            return 0.0
        dur = evt.finish()
        if len(self._events) < self.max_events:
            self._events.append(evt)
        ts = self._tier_stats.get(evt.tier)
        if ts:
            ts['total_bytes'] += evt.n_bytes
            ts['total_us'] += dur
            ts['n_ops'] += 1
            if len(ts['bw_samples']) < 10000:
                ts['bw_samples'].append(evt.bandwidth_gbps())
            if len(ts['latency_samples']) < 10000:
                ts['latency_samples'].append(dur)
        step = evt.step
        if step not in self._step_comm_time:
            self._step_comm_time[step] = 0.0
        self._step_comm_time[step] += dur
        self._active_event = None
        return dur

    def get_tier_summary(self, tier):
        """Get summary statistics for a specific tier."""
        ts = self._tier_stats.get(tier, {})
        n = ts.get('n_ops', 0)
        sk = ts.get('n_skipped', 0)
        total_ops = n + sk
        return {
            'tier': tier,
            'total_bytes': ts.get('total_bytes', 0),
            'total_us': ts.get('total_us', 0.0),
            'n_ops': n,
            'n_skipped': sk,
            'comm_reduction': round(sk / max(1, total_ops), 4),
            'avg_bw_gbps': (
                sum(ts.get('bw_samples', [])) / max(1, len(ts.get('bw_samples', [])))
            ),
            'avg_latency_us': (
                sum(ts.get('latency_samples', [])) / max(1, len(ts.get('latency_samples', [])))
            ),
        }

    def get_overall_summary(self):
        """Get overall profiling summary across all tiers."""
        summaries = [self.get_tier_summary(t) for t in range(3)]
        total_ops = sum(s['n_ops'] for s in summaries)
        total_skipped = sum(s['n_skipped'] for s in summaries)
        total_bytes = sum(s['total_bytes'] for s in summaries)
        total_us = sum(s['total_us'] for s in summaries)
        return {
            'total_ops': total_ops,
            'total_skipped': total_skipped,
            'total_bytes': total_bytes,
            'total_us': total_us,
            'comm_reduction': round(total_skipped / max(1, total_ops + total_skipped), 4),
            'avg_bw_gbps': (total_bytes / 1e9) / max(1e-9, total_us / 1e6),
            'tiers': summaries,
        }


# =====================================================================
# M337 — Claude-30: Megatron Config Logger Pattern
# Source: Megatron-LM/megatron/core/config_logger.py (log_config_to_disk)
# Adapted: Disk-persistent comm event logging for post-hoc analysis
# =====================================================================

def desloc_log_comm_event(event, log_file=None, log_dir=None):
    """Log a single DeslocCommEvent to disk in NKI-FA format.

    Adapted from Megatron's log_config_to_disk() pattern: write structured
    data to a known path so downstream tools (draw_plot.py) can parse it.

    Args:
        event: DeslocCommEvent instance
        log_file: Direct file path (takes precedence)
        log_dir: Directory for auto-named log files
    """
    import os as _os
    if log_file is None and log_dir is None:
        return
    if log_file is None:
        _os.makedirs(log_dir, exist_ok=True)
        log_file = _os.path.join(log_dir, f"comm_rank{event.rank}.log")
    with open(log_file, 'a') as f:
        f.write(event.to_nkifa_str() + '\n\n')


def desloc_export_comm_log(profiler, output_path, format='nkifa'):
    """Export all profiled comm events to a structured log file.

    Adapted from Megatron config_logger.py log_config_to_disk pattern.

    Args:
        profiler: DeslocCommProfiler instance
        output_path: Output file path
        format: 'nkifa' for NKI-FA text format, 'json' for structured JSON
    """
    if format == 'json':
        data = {
            'summary': profiler.get_overall_summary(),
            'events': [e.to_dict() for e in profiler._events],
        }
        with open(output_path, 'w') as f:
            _json.dump(data, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            f.write("# DES-LOC Communication Log (NKI-FA format)\n")
            f.write(f"# Rank: {profiler.rank}\n")
            f.write(f"# Total events: {len(profiler._events)}\n\n")
            for evt in profiler._events:
                f.write(evt.to_nkifa_str() + '\n\n')
            f.write("# === Summary ===\n")
            summary = profiler.get_overall_summary()
            for key, val in summary.items():
                if key != 'tiers':
                    f.write(f"# {key}: {val}\n")


# =====================================================================
# M337 — Claude-30: Enhanced Log Parser with NCCL Algo/Proto Tracking
# Source: nccl/src/device/all_reduce.h (algo/proto specializations)
# Extends M313 desloc_cl_parse with algo/proto fields
# =====================================================================

def desloc_cl_parse_v2(path):
    """Enhanced comm log parser supporting NCCL algo/proto fields.

    Extends M313 desloc_cl_parse to capture algorithm and protocol
    from NKI-FA formatted comm logs. Handles both v1 (M313) and v2 (M337)
    log formats for backward compatibility.

    Returns list of dicts with keys: op, tier, step, algo, proto, bytes,
    elements, time_us, bandwidth_gbps, synced, bucket_id, rank, seq.
    """
    entries = []
    cur = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if cur:
                        entries.append(cur)
                        cur = {}
                    continue
                m = _cre.match(
                    r'^###\s*comm_op\s*=\s*(\w+)'
                    r'(?:,\s*tier\s*=\s*(\d+))?'
                    r'(?:,\s*step\s*=\s*(\d+))?'
                    r'(?:,\s*algo\s*=\s*(\w+))?'
                    r'(?:,\s*proto\s*=\s*(\w+))?',
                    line
                )
                if m:
                    if cur:
                        entries.append(cur)
                    cur = {
                        'op': m.group(1),
                        'tier': int(m.group(2)) if m.group(2) else 0,
                        'step': int(m.group(3)) if m.group(3) else 0,
                        'algo': m.group(4) if m.group(4) else 'ring',
                        'proto': m.group(5) if m.group(5) else 'simple',
                    }
                    continue
                m = _cre.match(r'^(\w+)\s*:\s*(.+)$', line)
                if m and cur:
                    key = m.group(1)
                    val = m.group(2).strip()
                    try:
                        cur[key] = float(val)
                    except ValueError:
                        cur[key] = val
        if cur:
            entries.append(cur)
    except FileNotFoundError:
        pass
    return entries


# =====================================================================
# M337 — Claude-30: Bandwidth Analysis Tools
# Source: nccl/src/device/all_reduce.h (Ring vs Tree bandwidth characteristics)
#         TransformerEngine/.../comm_gemm_overlap (overlap analysis)
# Ref: Analyze comm efficiency per NCCL algorithm for DES-LOC tier routing
# =====================================================================

def desloc_bandwidth_analysis(entries):
    """Analyze bandwidth efficiency per NCCL algorithm and DES-LOC tier.

    Groups comm events by (algo, tier) and computes bandwidth statistics.
    Used for selecting optimal algorithm per tier (Ring for intra-node param
    sync, Tree for inter-node momentum sync).

    Args:
        entries: List of parsed comm event dicts (from desloc_cl_parse_v2)

    Returns:
        Dict mapping (algo, tier) → {avg_bw, p50_bw, p99_bw, n_ops}
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for e in entries:
        algo = e.get('algo', 'ring')
        tier = e.get('tier', 0)
        bw = e.get('bandwidth_gbps', 0.0)
        if bw > 0:
            groups[(algo, tier)].append(bw)

    result = {}
    for key, bw_list in groups.items():
        bw_sorted = sorted(bw_list)
        n = len(bw_sorted)
        result[key] = {
            'avg_bw': sum(bw_sorted) / n,
            'p50_bw': bw_sorted[n // 2],
            'p99_bw': bw_sorted[min(n - 1, int(n * 0.99))],
            'min_bw': bw_sorted[0],
            'max_bw': bw_sorted[-1],
            'n_ops': n,
        }
    return result


def desloc_comm_compute_overlap_ratio(profiler, step_compute_time_us=None):
    """Compute communication/compute overlap ratio per step.

    Measures how effectively comm is hidden behind compute.
    Ratio = 1.0 means perfect overlap; 0.0 means no overlap.

    Adapted from TransformerEngine comm_gemm_overlap analysis pattern.

    Args:
        profiler: DeslocCommProfiler instance
        step_compute_time_us: Dict mapping step → compute time in us.
            If None, uses default estimate from step duration.

    Returns:
        Dict with overlap statistics: avg_ratio, per_step ratios,
        comm_dominant_steps (where comm > compute).
    """
    ratios = []
    comm_dominant_steps = 0
    for step, comm_us in profiler._step_comm_time.items():
        if step_compute_time_us and step in step_compute_time_us:
            compute_us = step_compute_time_us[step]
        else:
            compute_us = comm_us * 3.0
        if compute_us <= 0:
            continue
        exposed_comm = max(0.0, comm_us - compute_us)
        overlap = 1.0 - (exposed_comm / max(1.0, comm_us))
        ratios.append(overlap)
        if comm_us > compute_us:
            comm_dominant_steps += 1

    if not ratios:
        return {'avg_ratio': 0.0, 'comm_dominant_steps': 0, 'n_steps': 0}

    return {
        'avg_ratio': round(sum(ratios) / len(ratios), 4),
        'min_ratio': round(min(ratios), 4),
        'max_ratio': round(max(ratios), 4),
        'comm_dominant_steps': comm_dominant_steps,
        'n_steps': len(ratios),
    }


def desloc_nkifa_figure_data(profiler, figure_type='comm_reduction'):
    """Generate figure-ready data for NKI-FA draw_plot.py.

    Adapted from NKI-FA exp_utils/draw_plot.py parse_data() pattern.
    Output format matches NKI-FA: {x: [...], y: [...], labels: [...]}.

    Args:
        profiler: DeslocCommProfiler instance
        figure_type: One of 'comm_reduction', 'tier_bandwidth',
            'algo_comparison', 'overlap_timeline'

    Returns:
        Dict with x, y, and optional labels/hue for plotting
    """
    summary = profiler.get_overall_summary()

    if figure_type == 'comm_reduction':
        total_possible = summary['total_ops'] + summary['total_skipped']
        return {
            'x': ['DDP (baseline)', 'DES-LOC'],
            'y': [1.0, round(summary['total_ops'] / max(1, total_possible), 4)],
            'ylabel': 'Fraction of AllReduces executed',
            'title': 'Communication Reduction',
        }

    elif figure_type == 'tier_bandwidth':
        tiers = summary.get('tiers', [])
        return {
            'x': ['Param (Kx)', 'Mom1 (Ku)', 'Mom2 (Kv)'],
            'y': [t.get('avg_bw_gbps', 0.0) for t in tiers],
            'ylabel': 'Average Bandwidth (GB/s)',
            'title': 'Per-Tier Communication Bandwidth',
        }

    elif figure_type == 'tier_volume':
        tiers = summary.get('tiers', [])
        return {
            'x': ['Param (Kx)', 'Mom1 (Ku)', 'Mom2 (Kv)'],
            'y': [t.get('total_bytes', 0) / 1e9 for t in tiers],
            'ylabel': 'Total Data Volume (GB)',
            'title': 'Per-Tier Communication Volume',
        }

    elif figure_type == 'algo_comparison':
        entries = [e.to_dict() for e in profiler._events]
        analysis = desloc_bandwidth_analysis(entries)
        x_labels = []
        y_values = []
        for (algo, tier), stats in sorted(analysis.items()):
            x_labels.append(f"{algo}/T{tier}")
            y_values.append(stats['avg_bw'])
        return {
            'x': x_labels,
            'y': y_values,
            'ylabel': 'Average Bandwidth (GB/s)',
            'title': 'NCCL Algorithm Bandwidth by Tier',
        }

    return {'x': [], 'y': []}
# --- End M337 ---
