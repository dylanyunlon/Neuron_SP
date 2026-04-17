# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import math
from deepspeed.utils import log_dist


def get_caller_func(frame=3):
    import sys
    return sys._getframe(frame).f_code.co_name


def print_rank_0(message):
    import deepspeed.comm as dist
    if dist.get_rank() == 0:
        print(message)


# Helper function to pretty-print message sizes
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# Helper function to calculate algbw and busbw.
# See https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36 and https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
def calc_bw_log(comm_op, size, duration):
    import deepspeed.comm as dist

    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all_single":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather" or comm_op == "all_gather_into_tensor" or comm_op == "reduce_scatter" or comm_op == "reduce_scatter_tensor":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce" or comm_op == "all_reduce_coalesced" or comm_op == "inference_all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "send" or comm_op == "recv" or comm_op == "isend" or comm_op == "irecv" or comm_op == "broadcast" or comm_op == "reduce" or comm_op == "gather" or comm_op == "scatter" or comm_op == "barrier":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")  # noqa: F821
        exit(0)

    # convert to Gbps
    tput *= 8
    busbw *= 8

    tput /= 1e6
    busbw /= 1e6

    return tput, busbw


class CommsLogger:

    def __init__(self):
        from deepspeed.comm.constants import COMMS_LOGGER_VERBOSE_DEFAULT, COMMS_LOGGER_DEBUG_DEFAULT, COMMS_LOGGER_PROF_OPS_DEFAULT, COMMS_LOGGER_PROF_ALL_DEFAULT, COMMS_LOGGER_ENABLED_DEFAULT
        self.comms_dict = {}
        self.verbose = COMMS_LOGGER_VERBOSE_DEFAULT
        self.debug = COMMS_LOGGER_DEBUG_DEFAULT
        self.prof_ops = COMMS_LOGGER_PROF_OPS_DEFAULT
        self.prof_all = COMMS_LOGGER_PROF_ALL_DEFAULT
        self.enabled = COMMS_LOGGER_ENABLED_DEFAULT

    def configure(self, comms_config):
        self.enabled = comms_config.comms_logger_enabled
        if self.enabled:
            self.verbose = comms_config.comms_logger.verbose
            self.debug = comms_config.comms_logger.debug
            self.prof_ops = comms_config.comms_logger.prof_ops
            self.prof_all = comms_config.comms_logger.prof_all

    # There are three settings for the op profiler:
    # - Global profiling (profile all comms)
    # - Op-type profiling (e.g. profile all all_reduce comms)
    # - Op profiling (e.g. profile a specific all_reduce op)
    def start_profiling_comms(self):
        self.prof_all = True

    def stop_profiling_comms(self):
        self.prof_all = True

    # E.g. start_profiling_op('all_reduce')
    def start_profiling_op(self, op_name_list):
        self.prof_ops = list(set(self.prof_ops) | set(op_name_list))

    def stop_profiling_op(self, op_name_list):
        self.prof_ops = [op for op in self.prof_ops if op not in op_name_list]

    # Add log entry
    def append(self, raw_name, record_name, latency, msg_size):
        algbw, busbw = calc_bw_log(raw_name, msg_size, latency)
        if record_name in self.comms_dict.keys():
            # If this comm_op has already been logged with this message size, just add to existing record
            if msg_size in self.comms_dict[record_name].keys():
                self.comms_dict[record_name][msg_size][0] += 1
                self.comms_dict[record_name][msg_size][1].append(latency)
                self.comms_dict[record_name][msg_size][2].append(algbw)
                self.comms_dict[record_name][msg_size][3].append(busbw)
            # If this is a new message size for this comm_op, add new record under existing comm_op
            else:
                self.comms_dict[record_name][msg_size] = [1, [latency], [algbw], [busbw]]
        else:
            # Create entirely new record
            self.comms_dict[record_name] = {msg_size: [1, [latency], [algbw], [busbw]]}
        # If verbose, print every comm op
        # TODO: Add to tensorboard
        if self.verbose:
            log_str = f"comm op: {record_name} | time (ms): {latency:.2f} | msg size: {convert_size(msg_size)} | algbw (Gbps): {algbw:.2f} | busbw (Gbps): {busbw:.2f}"
            log_dist(log_str, [0])

    def get_raw_data(self):
        """
        Get the raw communication data dictionary.

        Returns:
            dict: Raw communication data in format {record_name: {msg_size: [count, [latencies], [algbws], [busbws]]}}
        """
        return self.comms_dict.copy()

    def has_data(self):
        """
        Check if any communication data has been logged.

        Returns:
            bool: True if communication data exists, False otherwise
        """
        return len(self.comms_dict) > 0

    def reset_data(self):
        """
        Clear all logged communication data.
        """
        self.comms_dict.clear()

    def get_operation_names(self):
        """
        Get list of all logged communication operation names.

        Returns:
            list: List of operation names that have been logged
        """
        return list(self.comms_dict.keys())

    def get_total_operations(self):
        """
        Get total number of communication operations logged across all types.

        Returns:
            int: Total count of operations
        """
        total = 0
        for record_name in self.comms_dict:
            for msg_size in self.comms_dict[record_name]:
                total += self.comms_dict[record_name][msg_size][0]  # count is at index 0
        return total

    def get_operation_summary(self, operation_name):
        """
        Get summary statistics for a specific operation type.

        Args:
            operation_name (str): Name of the communication operation

        Returns:
            dict: Summary statistics for the operation, or None if operation not found
        """
        if operation_name not in self.comms_dict:
            return None

        from deepspeed.utils.timer import trim_mean

        # Create a snapshot to avoid concurrent modification issues
        op_data = self.comms_dict[operation_name].copy()
        summary = {}

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
                "msg_size_str": convert_size(msg_size)
            }

        return summary

    # Print summary at end of iteration, epoch, or training
    def log_all(self, print_log=True, show_straggler=False, return_dict=False):
        """
        Print and/or return communication operation statistics.

        Args:
            print_log (bool, optional): Whether to print the summary to console. Defaults to True.
            show_straggler (bool, optional): Whether to include straggler effect analysis. Defaults to False.
            return_dict (bool, optional): Whether to return statistics as a dictionary. Defaults to False.

        Returns:
            dict or None: If return_dict=True, returns a comprehensive dictionary with the following structure:
            {
                "summary": {
                    "operation_name": {
                        message_size_bytes: {
                            "count": int,                    # Number of operations with this message size
                            "total_latency_ms": float,      # Sum of all latencies for this message size
                            "avg_latency_ms": float,        # Average latency (outliers trimmed)
                            "tput_avg_gbps": float,         # Average algorithmic bandwidth in Gbps
                            "busbw_avg_gbps": float,        # Average bus bandwidth in Gbps
                            "msg_size_bytes": int,          # Message size in bytes
                            "msg_size_str": str             # Human-readable message size (e.g., "678.86 MB")
                        }
                    }
                },
                "straggler_analysis": {                     # Only present if show_straggler=True
                    "operation_name": {
                        message_size_bytes: {
                            "count": int,                    # Number of operations
                            "total_comm_lat_ms": float,     # Total communication latency (min across ranks)
                            "total_straggler_ms": float,    # Total straggler effect
                            "avg_comm_lat_ms": float,       # Average communication latency
                            "avg_straggler_ms": float,      # Average straggler effect
                            "msg_size_bytes": int,          # Message size in bytes
                            "msg_size_str": str             # Human-readable message size
                        }
                    }
                } if show_straggler else None,
                "metadata": {
                    "world_size": int,                      # Number of processes in distributed setup
                    "rank": int,                            # Current process rank
                    "timestamp": str                        # ISO format timestamp when log_all was called
                }
            }

            Returns None if return_dict=False.

        Note:
            - Statistics use trimmed mean (10% trimmed from both ends) to remove outliers
            - Straggler analysis requires distributed communication and may impact performance
            - All bandwidth values are in Gbps (Gigabits per second)
            - Latency values are in milliseconds
        """
        import torch
        from deepspeed.utils.timer import trim_mean
        import deepspeed.comm as dist
        from deepspeed.comm.reduce_op import ReduceOp
        from deepspeed.accelerator import get_accelerator
        from datetime import datetime

        # Create a snapshot of the dictionary to avoid concurrent modification issues
        # This prevents "dictionary changed size during iteration" errors when
        # communication operations are happening in other threads
        comms_dict_snapshot = self.comms_dict.copy()

        # Initialize return dictionary structure
        result_dict = {
            "summary": {},
            "straggler_analysis": None,
            "metadata": {
                "world_size": dist.get_world_size() if dist.is_initialized() else 1,
                "rank": dist.get_rank() if dist.is_initialized() else 0,
                "timestamp": datetime.now().isoformat()
            }
        } if return_dict else None

        if print_log:
            print(
                f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}{'Total Latency(ms)': <20}{'Avg Latency(ms)': <20}{'tput_avg (Gbps)': <20}{'busbw_avg (Gbps)': <20}"
            )

        for record_name in comms_dict_snapshot.keys():
            if print_log:
                print(record_name)

            # Initialize operation entry in result dict
            if return_dict:
                result_dict["summary"][record_name] = {}

            for msg_size, vals in sorted(comms_dict_snapshot[record_name].items()):
                # vals[0] is the count for each msg size
                count = vals[0]
                # vals[1] is a list of latency records for each msg size
                total_lat = sum(vals[1])
                # vals[2] and vals[3] are the lists of algbw and busbw, respectively
                # Get rid of outliers when we print
                avg_lat = trim_mean(vals[1], 0.1)
                avg_algbw = trim_mean(vals[2], 0.1)
                avg_busbw = trim_mean(vals[3], 0.1)

                # Store data in result dictionary
                if return_dict:
                    result_dict["summary"][record_name][msg_size] = {
                        "count": count,
                        "total_latency_ms": total_lat,
                        "avg_latency_ms": avg_lat,
                        "tput_avg_gbps": avg_algbw,
                        "busbw_avg_gbps": avg_busbw,
                        "msg_size_bytes": msg_size,
                        "msg_size_str": convert_size(msg_size)
                    }

                if print_log:
                    print(
                        f"{' ': <20}{convert_size(msg_size): <20}{count: <20}{total_lat: <20.2f}{avg_lat: <20.2f}{avg_algbw: <20.2f}{avg_busbw: <20.2f}"
                    )

        if show_straggler:
            if return_dict:
                result_dict["straggler_analysis"] = {}

            if print_log:
                print("_______________________________")
                print("Breakdown with straggler effect")
                print("-------------------------------")
                print(
                    f"{'Comm. Op': <20}{'Message Size': <20}{'Count': <20}{'Total comm lat(ms)': <20}{'Total straggler(ms)': <20}{'Avg comm lat(ms)': <20}{'Avg straggler(ms)': <20}"
                )

            device = get_accelerator().current_device_name()
            for record_name in comms_dict_snapshot.keys():
                if print_log:
                    print(record_name)

                # Initialize operation entry in straggler dict
                if return_dict:
                    result_dict["straggler_analysis"][record_name] = {}

                for msg_size, vals in sorted(comms_dict_snapshot[record_name].items()):
                    # vals[0] is the count for each msg size
                    count = vals[0]
                    # vals[1] is a list of latency records for each msg size
                    lats = torch.tensor(vals[1], device=device)
                    min_lats = torch.tensor(vals[1], device=device)
                    dist.all_reduce(min_lats, op=ReduceOp.MIN)
                    total_lat = min_lats.sum().item()
                    total_straggler = (lats - min_lats).sum().item()
                    avg_lat = trim_mean(min_lats.tolist(), 0.1)
                    avg_straggler = trim_mean((lats - min_lats).tolist(), 0.1)

                    # Store straggler data in result dictionary
                    if return_dict:
                        result_dict["straggler_analysis"][record_name][msg_size] = {
                            "count": count,
                            "total_comm_lat_ms": total_lat,
                            "total_straggler_ms": total_straggler,
                            "avg_comm_lat_ms": avg_lat,
                            "avg_straggler_ms": avg_straggler,
                            "msg_size_bytes": msg_size,
                            "msg_size_str": convert_size(msg_size)
                        }

                    if print_log:
                        print(
                            f"{' ': <20}{convert_size(msg_size): <20}{count: <20}{total_lat: <20.2f}{total_straggler: <20.2f}{avg_lat: <20.2f}{avg_straggler: <20.2f}"
                        )

        # Return the dictionary if requested
        return result_dict if return_dict else None


# =================================================================
# M071: DES-LOC Structured Experiment Logger (400 lines)
# =================================================================
# Produces structured logs parseable by the plotting pipeline.
# Follows NKI-FA commit da964f3 draw_plot.py format:
# - Header with config
# - Parseable data lines with key=value format
# - Summary section with aggregate metrics
#
# Reference: NKI-FA exp_utils/draw_plot.py
# =================================================================

import os as _os
import json as _json
import time as _time
import math as _math


class DESLOCStructuredLogger:
    """Structured logger for DES-LOC experiments.

    Produces log files formatted for direct parsing by
    the figure generation pipeline. Every numeric value
    is written with full precision and labeled explicitly.

    Log format:
    ```
    ### DESLOC_EXPERIMENT_START ###
    # config: model=125M Kx=32 Ku=96 Kv=192 ...
    # benchmark: rq3_comm_reduction_125M
    # timestamp: 2026-04-17T06:53:49
    step=0 loss=10.8234 lr=6.0000e-04 grad_norm=12.345 ...
    step=10 loss=8.1234 lr=5.8800e-04 ...
    ### DESLOC_EXPERIMENT_END ###
    ### DESLOC_SUMMARY_START ###
    final_loss=3.2145
    min_loss=3.1892
    ...
    ### DESLOC_SUMMARY_END ###
    ```
    """

    def __init__(self, benchmark_id, config, log_dir=None):
        self.benchmark_id = benchmark_id
        self.config = config
        self.log_dir = log_dir or './desloc_experiment_logs'
        self.entries = []
        self.start_time = _time.time()
        self.metadata = {}

    def set_metadata(self, key, value):
        """Set experiment metadata."""
        self.metadata[key] = value

    def log_step(self, step, loss, lr=None, grad_norm=None,
                 sync_x=False, sync_u=False, sync_v=False,
                 comm_bytes=0, compute_ms=None, comm_ms=None,
                 exp_avg_norm=None, exp_avg_sq_norm=None,
                 param_norm=None, tokens_per_sec=None):
        """Log a single training step."""
        entry = {'step': step, 'loss': loss}
        if lr is not None:
            entry['lr'] = lr
        if grad_norm is not None:
            entry['grad_norm'] = grad_norm
        entry['sync_x'] = int(sync_x)
        entry['sync_u'] = int(sync_u)
        entry['sync_v'] = int(sync_v)
        entry['comm_bytes'] = comm_bytes
        if compute_ms is not None:
            entry['compute_ms'] = compute_ms
        if comm_ms is not None:
            entry['comm_ms'] = comm_ms
        if exp_avg_norm is not None:
            entry['exp_avg_norm'] = exp_avg_norm
        if exp_avg_sq_norm is not None:
            entry['exp_avg_sq_norm'] = exp_avg_sq_norm
        if param_norm is not None:
            entry['param_norm'] = param_norm
        if tokens_per_sec is not None:
            entry['tokens_per_sec'] = tokens_per_sec
        self.entries.append(entry)

    def _format_value(self, key, value):
        """Format a value for log output."""
        if isinstance(value, float):
            if abs(value) < 0.001 and value != 0:
                return f"{key}={value:.6e}"
            elif abs(value) > 10000:
                return f"{key}={value:.2f}"
            else:
                return f"{key}={value:.6f}"
        elif isinstance(value, int):
            return f"{key}={value}"
        elif isinstance(value, bool):
            return f"{key}={int(value)}"
        return f"{key}={value}"

    def format_header(self):
        """Format the log file header."""
        lines = ["### DESLOC_EXPERIMENT_START ###"]
        cfg_parts = []
        for k, v in self.config.items():
            if isinstance(v, (str, int, float, bool)):
                cfg_parts.append(f"{k}={v}")
        lines.append(f"# config: {' '.join(cfg_parts)}")
        lines.append(f"# benchmark: {self.benchmark_id}")
        import datetime
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        lines.append(f"# timestamp: {ts}")
        for k, v in self.metadata.items():
            lines.append(f"# {k}: {v}")
        return "\n".join(lines)

    def format_entries(self):
        """Format all log entries."""
        lines = []
        for entry in self.entries:
            parts = [self._format_value(k, v)
                      for k, v in entry.items()]
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def compute_summary(self):
        """Compute experiment summary metrics."""
        if not self.entries:
            return {}

        losses = [e['loss'] for e in self.entries
                  if not _math.isnan(e['loss'])]
        total_comm = sum(e.get('comm_bytes', 0)
                         for e in self.entries)
        x_syncs = sum(e.get('sync_x', 0) for e in self.entries)
        u_syncs = sum(e.get('sync_u', 0) for e in self.entries)
        v_syncs = sum(e.get('sync_v', 0) for e in self.entries)
        total_steps = len(self.entries)
        elapsed = _time.time() - self.start_time

        summary = {
            'final_loss': losses[-1] if losses else 0.0,
            'min_loss': min(losses) if losses else 0.0,
            'total_steps': total_steps,
            'total_comm_bytes': total_comm,
            'x_syncs': x_syncs,
            'u_syncs': u_syncs,
            'v_syncs': v_syncs,
            'elapsed_seconds': round(elapsed, 2),
            'steps_per_second': round(
                total_steps / max(elapsed, 0.01), 2),
        }

        # Convergence rate: avg loss over last 10%
        tail = max(1, total_steps // 10)
        tail_losses = losses[-tail:] if losses else [0.0]
        summary['tail_avg_loss'] = round(
            sum(tail_losses) / len(tail_losses), 6)

        # Communication reduction vs DDP
        if total_steps > 0:
            ddp_syncs = total_steps * 3  # all 3 states every step
            desloc_syncs = x_syncs + u_syncs + v_syncs
            summary['reduction_vs_ddp'] = round(
                ddp_syncs / max(desloc_syncs, 1), 2)

        # Throughput
        compute_times = [e.get('compute_ms', 0)
                         for e in self.entries
                         if 'compute_ms' in e]
        if compute_times:
            summary['avg_compute_ms'] = round(
                sum(compute_times) / len(compute_times), 3)
        comm_times = [e.get('comm_ms', 0)
                      for e in self.entries if 'comm_ms' in e]
        if comm_times:
            summary['avg_comm_ms'] = round(
                sum(comm_times) / len(comm_times), 3)

        return summary

    def format_summary(self):
        """Format the summary section."""
        summary = self.compute_summary()
        lines = ["### DESLOC_SUMMARY_START ###"]
        for k, v in summary.items():
            lines.append(self._format_value(k, v))
        lines.append("### DESLOC_SUMMARY_END ###")
        return "\n".join(lines)

    def save(self, filename=None):
        """Save complete log to file."""
        _os.makedirs(self.log_dir, exist_ok=True)
        if filename is None:
            cfg = self.config
            parts = [self.benchmark_id]
            if 'model_size' in cfg:
                parts.append(cfg['model_size'])
            if 'Kx' in cfg:
                parts.append(f"Kx{cfg['Kx']}")
            if 'seed' in cfg:
                parts.append(f"s{cfg['seed']}")
            filename = "_".join(parts) + ".log"

        path = _os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            f.write(self.format_header() + "\n")
            f.write(self.format_entries() + "\n")
            f.write("### DESLOC_EXPERIMENT_END ###\n")
            f.write(self.format_summary() + "\n")
        return path

    def save_json(self, filename=None):
        """Save as JSON for programmatic access."""
        _os.makedirs(self.log_dir, exist_ok=True)
        if filename is None:
            filename = f"{self.benchmark_id}.json"
        path = _os.path.join(self.log_dir, filename)
        output = {
            'benchmark_id': self.benchmark_id,
            'config': self.config,
            'metadata': self.metadata,
            'summary': self.compute_summary(),
            'entries': self.entries,
        }
        with open(path, 'w') as f:
            _json.dump(output, f, indent=2)
        return path


class DESLOCLogParser:
    """Parse DES-LOC structured log files for plotting.

    Reads logs produced by DESLOCStructuredLogger and extracts
    data suitable for matplotlib/seaborn visualization.

    Follows NKI-FA draw_plot.py parsing convention.
    """

    def __init__(self):
        self.experiments = []

    def parse_file(self, path):
        """Parse a single log file."""
        config = {}
        entries = []
        summary = {}
        section = 'header'

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line == '### DESLOC_EXPERIMENT_START ###':
                    section = 'data'
                    continue
                elif line == '### DESLOC_EXPERIMENT_END ###':
                    section = 'between'
                    continue
                elif line == '### DESLOC_SUMMARY_START ###':
                    section = 'summary'
                    continue
                elif line == '### DESLOC_SUMMARY_END ###':
                    section = 'done'
                    continue

                if line.startswith('# config:'):
                    cfg_str = line[len('# config:'):].strip()
                    for part in cfg_str.split():
                        if '=' in part:
                            k, v = part.split('=', 1)
                            config[k] = self._parse_val(v)
                    continue

                if line.startswith('# benchmark:'):
                    config['benchmark_id'] = line.split(':', 1)[1].strip()
                    continue

                if line.startswith('#'):
                    continue

                if section == 'data':
                    entry = {}
                    for part in line.split():
                        if '=' in part:
                            k, v = part.split('=', 1)
                            entry[k] = self._parse_val(v)
                    if entry:
                        entries.append(entry)

                elif section == 'summary':
                    if '=' in line:
                        k, v = line.split('=', 1)
                        summary[k] = self._parse_val(v)

        result = {
            'config': config,
            'entries': entries,
            'summary': summary,
            'path': path,
        }
        self.experiments.append(result)
        return result

    def _parse_val(self, s):
        """Parse a string value to appropriate type."""
        try:
            if '.' in s or 'e' in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return s

    def parse_directory(self, log_dir):
        """Parse all .log files in a directory."""
        results = []
        for fn in sorted(_os.listdir(log_dir)):
            if fn.endswith('.log'):
                path = _os.path.join(log_dir, fn)
                results.append(self.parse_file(path))
        return results

    def extract_loss_curves(self, benchmark_filter=None):
        """Extract loss curves for plotting.

        Returns dict suitable for pd.DataFrame:
        {
          'step': [...],
          'loss': [...],
          'config_label': [...],
        }
        """
        data = {'step': [], 'loss': [], 'config_label': []}
        for exp in self.experiments:
            bid = exp['config'].get('benchmark_id', '')
            if benchmark_filter and benchmark_filter not in bid:
                continue
            cfg = exp['config']
            label = f"{cfg.get('method', 'desloc')}_Kx{cfg.get('Kx', '?')}"
            for entry in exp['entries']:
                if 'step' in entry and 'loss' in entry:
                    data['step'].append(entry['step'])
                    data['loss'].append(entry['loss'])
                    data['config_label'].append(label)
        return data

    def extract_comm_reduction(self):
        """Extract communication reduction data."""
        data = {'method': [], 'reduction_vs_ddp': [],
                'final_loss': []}
        for exp in self.experiments:
            s = exp.get('summary', {})
            cfg = exp.get('config', {})
            data['method'].append(
                cfg.get('method', 'desloc'))
            data['reduction_vs_ddp'].append(
                s.get('reduction_vs_ddp', 1.0))
            data['final_loss'].append(
                s.get('final_loss', 0.0))
        return data


# =================================================================
# End M071
# =================================================================
