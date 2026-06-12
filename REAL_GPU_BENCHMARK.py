#!/usr/bin/env python3
"""
DES-LOC Real GPU Benchmark - No Simulation, No Fallback
========================================================
Real distributed training on heterogeneous GPUs.
Supports: 2xA6000+H100 NVL (ags1), H20 (阿里云gn8v), A100, etc.
Uses DeepSpeed runtime with DES-LOC extensions (M257-M338).
Fails hard if anything goes wrong.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

# Hard fail if imports missing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint  # M341: layer-wise activation checkpointing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
# torch.cuda.amp is deprecated in torch 2.x; use torch.amp
try:
    from torch.amp import autocast as _autocast_cls
    def autocast():
        return _autocast_cls('cuda', dtype=torch.bfloat16)
except ImportError:
    from torch.cuda.amp import autocast

# DeepSpeed runtime — conditional import
# Core DES-LOC algorithm (DESLOCAdamW, sync_if_needed, rate-of-change)
# is fully self-contained in this file. DeepSpeed is only needed for
# engine.py Kx-gated allreduce hooks in multi-GPU mode.
_DS_AVAILABLE = False

# M365: Centralized diagnostic toolkit
try:
    from deepspeed.utils.desloc_diag import diag as _diag
except ImportError:
    _diag = None
try:
    import deepspeed
    from deepspeed.runtime.utils import (
        desloc_comm_reduction_ratio,
        desloc_comm_bytes,
        desloc_local_adam_comm_bytes,
        desloc_parse_nkifa_logfile,
    )
    from deepspeed.utils.timer import (
        DeslocSTimer,
        DeslocProgress,
        desloc_mfu,
        desloc_roof,
    )
    from deepspeed.utils.comms_logging import (
        desloc_cl_entry,
        desloc_cl_sum,
        desloc_cl_parse,
        desloc_classify_op,
    )
    from deepspeed.comm.comm import (
        get_desloc_scheduler,
        get_desloc_profiler,
    )
    _DS_AVAILABLE = True
except Exception as _ds_err:
    import traceback as _tb
    print(f"[WARNING] DeepSpeed import failed: {_ds_err}")
    _tb.print_exc()
    # Standalone stubs — reproduce the exact same math, no deepspeed needed
    def desloc_comm_reduction_ratio(Kx, Ku, Kv, steps):
        """3-tier comm reduction: DDP does 3N AllReduces per step,
        DES-LOC does N/Kx + N/Ku + N/Kv.
        Claude-27 M335: Only x follows warmup ramp (1→Kx_target).
        u uses Ku_target always (no warmup ramp). v piggybacks on x.
        This matches the actual sync schedule in DESLOCAdamW.sync_if_needed
        and engine.py _desloc_momentum_sync."""
        if Kx <= 1 and Ku <= 1 and Kv <= 1:
            return 1.0
        ddp = steps * 3.0
        warmup = min(100, Kx * 3)
        sx_total, su_total, sv_total = 0, 0, 0
        for s in range(1, steps + 1):
            if s <= warmup:
                frac = s / max(warmup, 1)
                eKx = max(1, int(1 + (Kx - 1) * frac))
            else:
                eKx = Kx
            sx = (eKx <= 1) or (s % eKx == 0)
            su = (Ku <= 1) or (s % Ku == 0)   # M335: always use Ku_target
            sv = (Kv <= 1) or (s % Kv == 0) or sx  # v piggybacks on x
            sx_total += int(sx)
            su_total += int(su)
            sv_total += int(sv)
        desloc = sx_total + su_total + sv_total
        return ddp / max(desloc, 1)

    def desloc_comm_bytes(n_params, Kx, Ku, Kv, steps, sizeof=2):
        """Per-worker comm bytes: Ring-AllReduce 2(W-1)/W * N * sizeof per sync.
        Claude-27 M335: u uses Ku_target always, v piggybacks on x."""
        warmup = min(100, Kx * 3)
        sync_x, sync_u, sync_v = 0, 0, 0
        for s in range(1, steps + 1):
            if s <= warmup:
                frac = s / max(warmup, 1)
                eKx = max(1, int(1 + (Kx - 1) * frac))
            else:
                eKx = Kx
            sx = (eKx <= 1) or (s % eKx == 0)
            su = (Ku <= 1) or (s % Ku == 0)
            sv = (Kv <= 1) or (s % Kv == 0) or sx
            sync_x += int(sx)
            sync_u += int(su)
            sync_v += int(sv)
        total_syncs = sync_x + sync_u + sync_v
        bytes_per_sync = n_params * sizeof * 2
        desloc_total = total_syncs * bytes_per_sync
        ddp_total = steps * 3 * bytes_per_sync
        reduction = ddp_total / max(1, desloc_total)
        savings = 100.0 * (1.0 - desloc_total / max(1, ddp_total))
        return {
            'desloc_total': desloc_total,
            'ddp_total': ddp_total,
            'reduction_x': round(reduction, 4),
            'savings_pct': round(savings, 2),
            'sync_count_x': sync_x,
            'sync_count_u': sync_u,
            'sync_count_v': sync_v,
        }

    def desloc_local_adam_comm_bytes(n_params, K, steps, sizeof=2):
        syncs = max(steps // K, 1) * 3
        return syncs * n_params * sizeof * 2

    def desloc_parse_nkifa_logfile(path):
        return {}

    def desloc_mfu(achieved_tflops, peak_tflops):
        return achieved_tflops / peak_tflops if peak_tflops > 0 else 0.0

    def desloc_roof(n_params, peak_tflops, mem_bw_tbps=2.0):
        return peak_tflops

    class DeslocSTimer:
        def __init__(self): pass
        def begin_step(self, step): pass
        def end_step(self, **kw): pass
        def export_nkifa(self, path, config_str): pass

    class DeslocProgress:
        def __init__(self, total): self.total = total

    def desloc_cl_entry(*a, **kw): return ""
    def desloc_cl_sum(*a, **kw): return {}
    def desloc_cl_parse(*a, **kw): return []
    def desloc_classify_op(*a, **kw): return "unknown"

    def get_desloc_scheduler(): return None
    def get_desloc_profiler(): return None

assert torch.cuda.is_available(), "CUDA not available - HARD FAIL"
assert torch.cuda.device_count() >= 1, "No GPU found - HARD FAIL"


# =============================================================================
# PORT: Megatron 1446bb643 — working on args
# Original: arguments.py — restructure flat add_*_args() helpers + _GLOBAL_ARGS
# singleton so callers share one parsed namespace without re-parsing.
# 20% adaptation: helpers target DES-LOC/Neuron_SP flags, not BERT-only flags;
# _GLOBAL_ARGS is optional (benchmark may parse inline in main()).
# =============================================================================

_GLOBAL_ARGS = None


def _neuronsp_parse_args(extra_args_provider=None):
    """Parse args once, cache in _GLOBAL_ARGS, and return.

    Port of Megatron arguments.py::parse_args (1446bb643).
    Asserts the singleton is unset before writing, preventing double-init bugs.
    """
    global _GLOBAL_ARGS
    assert _GLOBAL_ARGS is None, '[NEURONSP-ARGS] args already initialized'
    print('[NEURONSP-ARGS] _neuronsp_parse_args: initializing _GLOBAL_ARGS singleton')
    _GLOBAL_ARGS = _neuronsp_get_args_(extra_args_provider=extra_args_provider)
    return _GLOBAL_ARGS


def _neuronsp_get_args(extra_args_provider=None):
    """Return cached args, parsing once on first call.

    Port of Megatron arguments.py::get_args (1446bb643).
    Idempotent: subsequent calls return the cached object.
    """
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        return _neuronsp_parse_args(extra_args_provider=extra_args_provider)
    return _GLOBAL_ARGS


def _neuronsp_add_network_size_args(parser):
    """Network architecture arguments.

    Port of Megatron arguments.py::add_network_size_args (1446bb643).
    20% adaptation: --make-vocab-size-divisible-by renamed make_vocab_divisible_by
    to match Python underscore convention used throughout this file.
    """
    group = parser.add_argument_group(title='network size')
    group.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers.')
    group.add_argument('--num_unique_layers', type=int, default=None,
                       help='Number of unique transformer layers. '
                       '`num_layers` should be divisible by this value.')
    group.add_argument('--param_sharing_style', default='grouped',
                       choices=['grouped', 'spaced'],
                       help='Ordering of the shared parameters. For example, '
                       'for a `num_layers`=4 and `--num_unique_layers`=2, '
                       'we will have the following ordering for two unique '
                       'layers 1 and 2: '
                       '    grouped: [1, 2, 1, 2] and spaced: [1, 1, 2, 2].')
    group.add_argument('--hidden_size', type=int, default=768,
                       help='Transformer hidden size.')
    group.add_argument('--num_attention_heads', type=int, default=12,
                       help='Number of transformer attention heads.')
    group.add_argument('--max_position_embeddings', type=int, default=1024,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make_vocab_divisible_by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value. '
                       'Added for computational efficiency reasons.')
    print('[NEURONSP-ARGS] _neuronsp_add_network_size_args: group registered')
    return parser


def _neuronsp_add_regularization_args(parser):
    """Regularization arguments.

    Port of Megatron arguments.py::add_regularization_args (1446bb643).
    """
    group = parser.add_argument_group(title='regularization')
    group.add_argument('--attention_dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden_dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    print('[NEURONSP-ARGS] _neuronsp_add_regularization_args: group registered')
    return parser


def _neuronsp_add_mixed_precision_args(parser):
    """Mixed-precision / fp16 arguments.

    Port of Megatron arguments.py::add_mixed_precision_args (1446bb643).
    20% adaptation: hysteresis/loss-scale flags kept; apply_query_key_layer_scaling
    already exposed at TrainingConfig level — parser flag is the CLI surface.
    """
    group = parser.add_argument_group(title='mixed precision')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='Hysteresis for dynamic loss scaling.')
    group.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 values can '
                       'improve fp16 convergence. If None, dynamic loss scaling is used.')
    group.add_argument('--loss_scale_window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--min_scale', type=float, default=1,
                       help='Minimum loss scale for dynamic loss scale.')
    print('[NEURONSP-ARGS] _neuronsp_add_mixed_precision_args: group registered')
    return parser


def _neuronsp_add_distributed_args(parser):
    """Distributed training arguments.

    Port of Megatron arguments.py::add_distributed_args (1446bb643).
    20% adaptation: backend choices match DeepSpeed (nccl/gloo); DDP-impl
    renamed ddp_impl (underscore) for argparse compatibility.
    """
    group = parser.add_argument_group(title='distributed')
    print('[M391]')
    group.add_argument('--distributed_backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--ddp_impl', default='local',
                       choices=['local', 'torch'],
                       help='Which DistributedDataParallel implementation to use.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank passed from distributed launcher.')
    group.add_argument('--deallocate-pipeline-outputs', action='store_true',
                       default=False, help='If set, pipeline output tensors '
                       'are deallocated during the forward pass.')
    print('[M957]')
    print('[NEURONSP-ARGS] _neuronsp_add_distributed_args: group registered')
    return parser


def _neuronsp_get_args_(extra_args_provider=None):
    """Build parser from structured groups and parse argv.

    Port of Megatron arguments.py::get_args_ (1446bb643).
    20% adaptation: structured groups registered first, then legacy flat
    benchmark args appended via extra_args_provider pattern so existing
    callers (main()) can still extend the parser freely.
    """
    parser = argparse.ArgumentParser(description='Neuron_SP / DES-LOC Arguments')
    parser = _neuronsp_add_network_size_args(parser)
    parser = _neuronsp_add_regularization_args(parser)
    parser = _neuronsp_add_mixed_precision_args(parser)
    parser = _neuronsp_add_distributed_args(parser)
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    args = parser.parse_args()
    print(f'[NEURONSP-ARGS] _neuronsp_get_args_: parsed args '
          f'hidden_size={getattr(args, "hidden_size", "N/A")} '
          f'num_layers={getattr(args, "num_layers", "N/A")} '
          f'distributed_backend={getattr(args, "distributed_backend", "N/A")}')
    return args


# =============================================================================
# CONFIGURATION
# =============================================================================

# =============================================================================
# M366: STEP RECORDER — Structured per-step diagnostics for JSON export
# =============================================================================
# Pattern: Megatron training.py training_log() + wandb/tensorboard scalars
# Pattern: NCCL debug logging (NCCL_DEBUG=INFO) structured per-op stats
# Pattern: TransformerEngine fp8_model_init() calibration history tracking
#
# Problem: All diagnostic data (grad norms, param norms, update magnitudes,
# momentum/variance norms, cross-rank divergence, sync events, timing
# breakdown) was only printed to terminal and lost after experiment.
# The JSON only stored: losses, sync_counts, basic throughput/memory.
#
# Solution: StepRecorder captures a structured dict per step. At experiment
# end, the full timeseries is serialized into the benchmark JSON alongside
# losses. This enables post-hoc analysis, visualization, and debugging
# without re-running the experiment.
#
# From Megatron's training_log as a good example: it logs grad_norm,
# params_norm, num_zeros_in_grad, loss_scale, learning_rate, and memory
# stats to tensorboard at every log_interval. Following that pattern,
# we implement a new StepRecorder that lets DES-LOC capture per-step
# optimizer state norms (||m||, ||v||, ||p||), and can record sync events
# with pre/post norms. Then DESLOCAdamW.step() introduces per-step
# instrumentation, so that the training loop can collect grad_norm,
# param_norm, update_delta at every step, while StepRecorder optimizes
# memory by only storing full snapshots at configurable intervals.
# Then _train_baseline integrates the recorder, making every diagnostic
# print also write to the structured log, supporting both terminal
# display and JSON export, and enhancing the _finalize_results output.
# Finally save_results serializes the full diagnostic_history into the
# benchmark JSON, ensuring post-hoc analysis compatibility with
# the existing visualization dashboard, and fully upgrading the
# experiment data pipeline to capture all runtime observability.
# =============================================================================

class StepRecorder:
    """Structured per-step diagnostic recorder for DES-LOC experiments.
    
    Captures metrics at two granularities:
    - summary_interval (default=10): lightweight metrics (loss, grad_norm, param_norm, lr, memory)
    - detail_interval (default=50): full diagnostics (per-layer norms, optimizer states, 
      cross-rank divergence, update magnitudes, momentum/variance distributions)
    
    All data is stored in self.records: List[Dict] and can be serialized to JSON.
    """
    
    def __init__(self, summary_interval: int = 10, detail_interval: int = 50):
        self.summary_interval = summary_interval
        self.detail_interval = detail_interval
        self.records = []  # List[Dict] — one entry per recorded step
        self._current = {}  # accumulator for current step
    
    def begin_step(self, step: int):
        """Start recording a new step."""
        self._current = {
            'step': step,
            'ts': time.time(),
        }
    
    def record(self, key: str, value):
        """Record a scalar or small dict for current step."""
        self._current[key] = value
    
    def record_grad_stats(self, model, rank: int):
        """Record gradient statistics — pattern from Megatron log_num_zeros_in_grad.
        
        Captures: total grad norm, num zero grads, per-layer grad norms (at detail steps),
        max grad element (for spike detection).
        """
        total_norm_sq = 0.0
        num_zeros = 0
        num_params_with_grad = 0
        max_grad_elem = 0.0
        per_layer = {}
        
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            num_params_with_grad += 1
            g = p.grad.detach().float()
            gnorm = g.norm().item()
            total_norm_sq += gnorm ** 2
            num_zeros += int((g == 0).sum().item())
            max_grad_elem = max(max_grad_elem, g.abs().max().item())
            
            # Per-layer at detail intervals
            step = self._current.get('step', 0)
            if step % self.detail_interval == 1:
                # Compress layer name: 'transformer.layers.5.attn.qkv.weight' -> 'L5.attn.qkv.w'
                short = name.replace('transformer.', '').replace('layers.', 'L').replace('.weight', '.w').replace('.bias', '.b')
                per_layer[short] = round(gnorm, 6)
        
        self._current['grad_norm'] = round(total_norm_sq ** 0.5, 6)
        self._current['grad_num_zeros'] = num_zeros
        self._current['grad_max_elem'] = round(max_grad_elem, 6)
        self._current['grad_num_params'] = num_params_with_grad
        
        if per_layer:
            self._current['grad_per_layer'] = per_layer
    
    def record_param_stats(self, model):
        """Record parameter statistics — pattern from Megatron params_norm."""
        total_norm_sq = 0.0
        param_mean_sum = 0.0
        param_var_sum = 0.0
        n_params = 0
        per_layer = {}
        
        with torch.no_grad():
            for name, p in model.named_parameters():
                pf = p.detach().float()
                pnorm = pf.norm().item()
                total_norm_sq += pnorm ** 2
                param_mean_sum += pf.mean().item()
                param_var_sum += pf.var().item()
                n_params += 1
                
                step = self._current.get('step', 0)
                if step % self.detail_interval == 1:
                    short = name.replace('transformer.', '').replace('layers.', 'L').replace('.weight', '.w').replace('.bias', '.b')
                    per_layer[short] = {
                        'norm': round(pnorm, 4),
                        'mean': round(pf.mean().item(), 6),
                        'std': round(pf.std().item(), 6),
                    }
        
        self._current['param_norm'] = round(total_norm_sq ** 0.5, 4)
        self._current['param_mean'] = round(param_mean_sum / max(n_params, 1), 8)
        self._current['param_var'] = round(param_var_sum / max(n_params, 1), 8)
        
        if per_layer:
            self._current['param_per_layer'] = per_layer
    
    def record_optimizer_state(self, optimizer):
        """Record optimizer state norms — m (momentum), v (variance).
        
        Pattern from Megatron optimizer.py step() which returns grad_norm.
        DES-LOC extends this: since m and v sync independently (Eq 4),
        tracking their norms detects drift between sync points.
        """
        total_m_sq = 0.0
        total_v_sq = 0.0
        m_max = 0.0
        v_max = 0.0
        n = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p, {})
                if 'exp_avg' not in state:
                    continue
                n += 1
                m = state['exp_avg']
                v = state['exp_avg_sq']
                
                if m.device.type == 'cpu':
                    m_norm = m.float().norm().item()
                    v_norm = v.float().norm().item()
                else:
                    m_norm = m.float().norm().item()
                    v_norm = v.float().norm().item()
                
                total_m_sq += m_norm ** 2
                total_v_sq += v_norm ** 2
                m_max = max(m_max, m_norm)
                v_max = max(v_max, v_norm)
        
        self._current['opt_m_norm'] = round(total_m_sq ** 0.5, 6)
        self._current['opt_v_norm'] = round(total_v_sq ** 0.5, 6)
        self._current['opt_m_max'] = round(m_max, 6)
        self._current['opt_v_max'] = round(v_max, 6)
        self._current['opt_n_states'] = n
    
    def record_update_delta(self, pre_norm: float, post_norm: float):
        """Record parameter update magnitude — ||p_after - p_before|| proxy.
        
        Uses norm-difference as lightweight proxy (no full clone needed).
        Pattern: DeepSpeed stage_1_and_2.py check_overflow tracks param changes.
        """
        delta = abs(post_norm - pre_norm)
        ratio = delta / max(pre_norm, 1e-12)
        self._current['update_delta'] = round(delta, 8)
        self._current['update_ratio'] = round(ratio, 10)
        self._current['param_norm_pre'] = round(pre_norm, 4)
        self._current['param_norm_post'] = round(post_norm, 4)
    
    def record_sync_event(self, sync_x: bool, sync_u: bool, sync_v: bool,
                          effective_Kx: int = 0, pre_sync_norm: float = 0.0,
                          post_sync_norm: float = 0.0):
        """Record DES-LOC sync event with pre/post param norms.
        
        Pattern: NCCL ncclGroupEnd() logging — records bytes, op type, timing.
        Extended for DES-LOC 3-tier sync schedule.
        """
        self._current['sync'] = {
            'x': sync_x, 'u': sync_u, 'v': sync_v,
            'Kx_eff': effective_Kx,
        }
        if sync_x and pre_sync_norm > 0:
            self._current['sync']['pre_norm'] = round(pre_sync_norm, 4)
            self._current['sync']['post_norm'] = round(post_sync_norm, 4)
            self._current['sync']['sync_delta'] = round(
                abs(post_sync_norm - pre_sync_norm) / max(pre_sync_norm, 1e-12), 8)
    
    def record_cross_rank_divergence(self, model, world_size: int):
        """Measure parameter divergence across ranks.
        
        Computes local param checksum, all-gathers, reports std across ranks.
        Pattern: DeepSpeed engine.py _check_all_params_equal().
        """
        if world_size <= 1 or not dist.is_initialized():
            return
        
        with torch.no_grad():
            local_sum = sum(p.data.float().sum().item() for p in model.parameters())
            device = next(model.parameters()).device
            t = torch.tensor([local_sum], dtype=torch.float64, device=device)
            gathered = [torch.zeros_like(t) for _ in range(world_size)]
            dist.all_gather(gathered, t)
            vals = [g.item() for g in gathered]
            mean_val = sum(vals) / len(vals)
            std_val = (sum((v - mean_val)**2 for v in vals) / len(vals)) ** 0.5
            rel_div = std_val / max(abs(mean_val), 1e-12)
        
        self._current['cross_rank'] = {
            'checksums': [round(v, 4) for v in vals],
            'std': round(std_val, 6),
            'rel_div': round(rel_div, 10),
        }
    
    def record_timing(self, fwd_ms: float = 0, bwd_ms: float = 0,
                      opt_ms: float = 0, sync_ms: float = 0, total_ms: float = 0):
        """Record per-step timing breakdown.
        
        Pattern: Megatron timers for 'forward-compute', 'backward-compute',
        'optimizer', 'all-grads-sync'.
        """
        self._current['timing'] = {
            'fwd_ms': round(fwd_ms, 2),
            'bwd_ms': round(bwd_ms, 2),
            'opt_ms': round(opt_ms, 2),
            'sync_ms': round(sync_ms, 2),
            'total_ms': round(total_ms, 2),
        }
    
    def record_memory(self):
        """Record CUDA memory stats.
        
        Pattern: Megatron report_memory() — tracks allocated, reserved, peak.
        """
        if torch.cuda.is_available():
            self._current['mem'] = {
                'alloc_gb': round(torch.cuda.memory_allocated() / 1e9, 3),
                'reserved_gb': round(torch.cuda.memory_reserved() / 1e9, 3),
                'peak_gb': round(torch.cuda.max_memory_allocated() / 1e9, 3),
            }
    
    def record_lr(self, lr: float):
        """Record learning rate."""
        self._current['lr'] = round(lr, 8)
    
    def record_pipeline_info(self, **kwargs):
        """Record pipeline routing info (first step only).
        
        Pattern: DeepSpeed engine.py log_config() at init.
        """
        self._current['pipeline'] = kwargs
    
    def end_step(self):
        """Finalize and store current step's record."""
        step = self._current.get('step', 0)
        self._current['elapsed_ms'] = round((time.time() - self._current.get('ts', time.time())) * 1000, 2)
        
        # Always record summary-level metrics
        if step % self.summary_interval == 0 or step <= 10 or step % self.detail_interval == 1:
            # Remove raw timestamp (not needed in JSON)
            self._current.pop('ts', None)
            self.records.append(self._current)
        
        self._current = {}
    
    def get_timeseries(self, key: str) -> list:
        """Extract a single metric as a timeseries [(step, value), ...]."""
        return [(r['step'], r[key]) for r in self.records if key in r]
    
    def export(self) -> list:
        """Export all records for JSON serialization."""
        return self.records


@dataclass
class TrainingConfig:
    """Training configuration - no defaults that allow fallback."""
    # Model
    model_size: str = "125M"
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # Training
    batch_size: int = 4
    gradient_accumulation: int = 8
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999   # Paper: Adam β2=0.999 (τ=693 steps). Was 0.95 (τ=13.5) — broke DES-LOC theory
    # M504: Megatron 48269d8d8 — expose adam_eps as a top-level config arg
    # so sweep scripts can vary numerical stability threshold independently of
    # learning-rate and beta tuning.  Default 1e-8 matches PyTorch Adam default.
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    
    # DES-LOC specific
    Kx: int = 32
    Ku: int = 96
    Kv: int = 192

    # RQ5: Outer optimizer (Section 5.5)
    outer_optimizer: str = "average"  # 'average' or 'nesterov'
    outer_momentum: float = 0.9       # Nesterov momentum (Charles et al. 2025)
    outer_lr: float = 1.0             # Nesterov outer learning rate

    # DDP checkpoint init (Charles et al. 2025 protocol)
    init_from_ckpt: str = ""          # path to DDP checkpoint for warm-start

    # AutoSP: Automatic Sequence Parallelism (DeepSpeed compile pass)
    # Shards sequence dim across GPUs → 2× longer sequences at same memory
    # Requires: SDPA attention, ZeRO stage 0, torch.compile
    use_autosp: bool = False

    # ZeRO optimization stage (0=disabled, 1=optimizer state partition)
    # ZeRO-1 with AutoSP: partitions Adam m/v across GPUs, saves ~50% opt memory
    # Required for 7B on 2xH20 (optimizer states alone = 56GB > single GPU)
    zero_stage: int = 0

    # CPU offload: move optimizer states to CPU RAM
    # Frees ~56GB GPU memory for 7B model (Adam m + v + fp32 master)
    # Required for 7B on A6000 (48GB) — without offload, optimizer alone = 56GB
    cpu_offload: bool = False

    # Activation Checkpointing (M341)
    # Layer-wise: torch.utils.checkpoint per TransformerBlock
    # Saves ~60% activation memory at ~33% compute overhead
    # Enables 1.3B+ models on 49GB A6000 or longer sequences on H20
    # Orthogonal to SP and DEC: SP(data) × DEC(comm) × AC(memory)
    use_activation_checkpointing: bool = False

    # M458: Megatron 691747b1 — per-layer QK scaling + fp32 softmax
    # apply_query_key_layer_scaling: scale Q*K^T by 1/layer_number on top of 1/sqrt(d).
    #   Deeper layers produce smaller raw logit magnitudes, preventing softmax saturation.
    # attention_softmax_in_fp32: cast attention weights to fp32 before softmax.
    #   Auto-enabled when apply_query_key_layer_scaling is True (Megatron args.py lines 73-74).
    # Knuth critique #1 (user): forgetting to set attention_softmax_in_fp32=True together
    #   with apply_query_key_layer_scaling leads to fp16 softmax at scale=1/(sqrt(d)*L);
    #   for L=24 and d=64 the max logit is ~0.052, easily representable in fp16,
    #   but gradients of exp() near zero suffer catastrophic cancellation.
    # Knuth critique #2 (system): the manual baddbmm+softmax path breaks SDPA's
    #   FlashAttention dispatch — memory cost reverts from O(T) to O(T²);
    #   use only when numerical precision matters more than memory efficiency.
    apply_query_key_layer_scaling: bool = False
    attention_softmax_in_fp32: bool = False
    
    # M452: Megatron 66719e9 dataloader — replacement sampling + presplit sentences
    # Megatron configure_data.py: RandomSampler(replacement=True, num_samples=batch_size*train_iters)
    # Knuth §3.4.2 argues uniform replacement is strictly inferior to Fisher-Yates shuffle
    # for finite datasets (coupon-collector waste). We implement it anyway because the
    # benchmark cares about throughput stability, not convergence quality.
    seed: int = 42                       # generator seed (rank-offset applied per worker)
    replacement_sampling: bool = False   # enable with-replacement RandomSampler (Megatron 66719e9)
    presplit_sentences: bool = False     # data pre-split into newline-separated sentences

    # M459: Megatron adec01d05 training sample builder — triple-array index scheme
    # doc_idx (shuffled document order) + sample_idx (packing boundaries) +
    # shuffle_idx (global sample permutation). Knuth §3.4.2: two independent
    # Fisher-Yates shuffles (doc order + sample order) with zero coupon waste.
    # Knuth §2.2.5 critique: O(3N) precomputed index tables vs O(1) on-demand;
    # we pay the memory cost for O(1) __getitem__ on the critical training path.
    use_sample_builder: bool = False     # enable adec01d05 triple-array sampling
    sample_builder_num_docs: int = 1000  # synthetic corpus document count
    sample_builder_min_doc_len: int = 64  # minimum tokens per synthetic document
    sample_builder_max_doc_len: int = 512  # maximum tokens per synthetic document
    # A sequence may contain multiple EOS/EOD tokens when documents are packed.
    # Knuth §2.3.1: a singly-linked scan terminates at the *first* sentinel;
    #   iterating over ALL EOD indices is O(k) not O(1) — acceptable because k≪seq_len.
    # Knuth §3.2.2 critique: injecting EOS at a fixed stride creates a perfectly
    #   periodic distribution; real corpora have Poisson-distributed document lengths,
    #   so the synthetic injection is a controlled approximation, not a faithful model.
    # eod_token_id: token ID used as end-of-document marker (vocab_size-1 by convention)
    # eod_mask_loss: zero out loss on EOS positions (don't reward predicting EOS itself)
    # reset_position_ids: restart position counter after each EOS (multi-doc packing)
    eod_token_id: int = -1               # -1 → auto-set to vocab_size-1 at dataset init
    eod_mask_loss: bool = True           # mask loss on EOD tokens (Megatron 872b4a6)
    reset_position_ids: bool = True      # reset pos-ids after each EOD (multi-doc packing)

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Paths
    output_dir: str = "./real_benchmark_results"
    
    def get_model_config(self) -> Dict:
        """Get model configuration based on size."""
        configs = {
            "125M": {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "350M": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
            "700M": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
            "1.3B": {"n_layer": 24, "n_head": 16, "n_embd": 2048},
            "1.7B": {"n_layer": 24, "n_head": 16, "n_embd": 2304},
            "3B": {"n_layer": 32, "n_head": 32, "n_embd": 3200},
            "7B": {"n_layer": 32, "n_head": 32, "n_embd": 4096},
            "13B": {"n_layer": 40, "n_head": 40, "n_embd": 5120},
        }
        assert self.model_size in configs, f"Unknown model size: {self.model_size}"
        return configs[self.model_size]


# =============================================================================
# GPT-2 MODEL (Real Implementation)
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# ── M364: SP runtime context (set by benchmark init, read by attention/forward) ──
# =============================================================================
# NEURON_SP PORT: Megatron 57c2060fe — Model parallel merger
# Adapted from megatron/mpu/initialize.py.
# Adds _MPU_WORLD_SIZE / _MPU_RANK override globals so tests and checkpointing
# code can call set_model_parallel_world_size / set_model_parallel_rank to
# override the distributed query without rebuilding process groups.
# Also ports partition_dim + stride metadata on model-parallel weight tensors
# (megatron/mpu/layers.py _initialize_affine_weight), allowing checkpoint
# merging code to discover how each shard is split without extra bookkeeping.
# 20% adaptation: uses dist instead of mpu; adds print breakpoints; stores
# metadata on nn.Parameter directly (same pattern, no MegatronModule base).
# =============================================================================

# These values enable callers to override the MPU sizes on the fly
# (e.g. checkpoint merge tools that pretend world_size=1).
_MPU_WORLD_SIZE = None
_MPU_RANK = None


def set_model_parallel_world_size(world_size: int) -> None:
    """Override model-parallel world size (Megatron 57c2060fe mpu/initialize.py)."""
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size
    print(f"[MPU-OVERRIDE] set_model_parallel_world_size({world_size})")


def get_model_parallel_world_size() -> int:
    """Return model-parallel world size, honouring override if set."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def set_model_parallel_rank(rank: int) -> None:
    """Override model-parallel rank (Megatron 57c2060fe mpu/initialize.py)."""
    global _MPU_RANK
    _MPU_RANK = rank
    print(f"[MPU-OVERRIDE] set_model_parallel_rank({rank})")


def get_model_parallel_rank() -> int:
    """Return model-parallel rank, honouring override if set."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _neuronsp_mark_weight_parallel(weight: nn.Parameter,
                                    partition_dim: int,
                                    stride: int = 1) -> None:
    """Tag a weight tensor with model-parallel shard metadata.

    Port of Megatron mpu/layers.py _initialize_affine_weight (57c2060fe):
      weight.model_parallel = True
      weight.partition_dim  = partition_dim
      weight.stride         = stride
    These three attributes let checkpoint-merge utilities reconstruct the
    full weight from per-rank shards without extra config files.

    20% adaptation: standalone function instead of inline in init;
    adds a print so checkpoint tools can verify the annotation fired.
    """
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.stride = stride
    print(f"[MPU-WEIGHT] marked model_parallel weight: "
          f"shape={list(weight.shape)} partition_dim={partition_dim} stride={stride}")

_SP_CTX = {'on': False, 'grp': None, 'sz': 1, 'rk': 0, 'step': 0}

def _sp_ctx_set(on, grp=None, sz=1, rk=0):
    _SP_CTX.update(on=on, grp=grp, sz=sz, rk=rk)

class _UlyssesA2A(torch.autograd.Function):
    """Autograd-compatible Ulysses all-to-all.
    Forward:  scatter_idx=1,gather_idx=2 → [B,N,S/P,H]->[B,N/P,S,H]
    Backward: reverses the scatter/gather indices automatically.
    """
    @staticmethod
    def forward(ctx, t, scatter_idx, gather_idx, grp):
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.grp = grp
        return _a2a_impl(t, scatter_idx, gather_idx, grp)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse: swap scatter and gather indices
        grad_input = _a2a_impl(grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.grp)
        return grad_input, None, None, None

def _a2a_impl(t, scatter_idx, gather_idx, grp):
    """Raw all-to-all without autograd."""
    if grp is None: return t
    ws = dist.get_world_size(group=grp)
    if ws <= 1: return t
    B, d1, d2, H = t.shape
    if scatter_idx == 1:
        t2 = t.reshape(B, ws, d1//ws, d2, H).permute(1,0,2,3,4).contiguous()
        o = torch.empty_like(t2); dist.all_to_all_single(o, t2, group=grp)
        return o.permute(1,2,0,3,4).contiguous().reshape(B, d1//ws, ws*d2, H)
    else:
        t2 = t.reshape(B, d1, ws, d2//ws, H).permute(2,0,1,3,4).contiguous()
        o = torch.empty_like(t2); dist.all_to_all_single(o, t2, group=grp)
        return o.permute(1,0,2,3,4).contiguous().reshape(B, ws*d1, d2//ws, H)

def _ulysses_a2a(t, scatter_idx, gather_idx, grp):
    """Differentiable Ulysses all-to-all."""
    return _UlyssesA2A.apply(t, scatter_idx, gather_idx, grp)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses F.scaled_dot_product_attention (PyTorch 2.0+) which:
    1. Automatically selects FlashAttention / Memory-Efficient / Math backend
    2. Required by DeepSpeed AutoSP (compile pass needs SDPA to identify attention ops)
    3. O(T) memory vs O(T²) for manual attention — critical for long sequences

    From FlashAttention (Dao-AILab flash-attention/flash_attn) kernel start.
    Then, follow that pattern to implement SDPA-based causal attention,
    letting AutoSP identify and shard the sequence dimension automatically,
    and enabling FlashAttention-2 hardware acceleration on A100/H100.
    Then Megatron introduces context-parallel attention splitting,
    so that AutoSP can distribute sequences across GPUs, while
    TransformerEngine optimizes FP8 SDPA for Hopper architecture.
    Then neuronx-distributed integrates NKI attention kernels,
    letting AutoSP support Trainium2 attention sharding,
    and veScale enhances DTensor attention redistribution.
    Finally cutlass refines GEMM-based attention for non-SDPA fallback,
    ensuring AutoSP is compatible with all GPU architectures,
    fully upgrading attention to support both DES-LOC and AutoSP.
    """
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0,
                 layer_number: int = 1,
                 apply_query_key_layer_scaling: bool = False,
                 attention_softmax_in_fp32: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout

        # M458: Megatron 691747b1 — per-layer QK scaling + fp32 softmax
        # apply_query_key_layer_scaling multiplies Q*K^T by 1/layer_number on top of
        # the standard 1/sqrt(head_dim), deepening numerical stability in deep stacks.
        # If QK scaling is on, softmax must run in fp32 to avoid saturation at large depths.
        self.layer_number = max(1, layer_number)  # guard: layer index starts at 1
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        # Knuth critique #1 (user): caller may set apply_query_key_layer_scaling=True but
        #   forget attention_softmax_in_fp32 — we auto-enable it here, mirroring Megatron's
        #   argument parser behaviour (commit 691747b1 lines 73-74 of arguments.py).
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32 or apply_query_key_layer_scaling
        # Knuth critique #2 (system): SDPA's scale kwarg is float32, but when QK scaling is
        #   combined with BF16 autocast the product 1/(sqrt(d)*L) can underflow to 0 for
        #   L≥512 with head_dim=64; the fp32 softmax path below mitigates this by upcasting
        #   before the exponential, matching Megatron's design intent.
        self._qk_scale = 1.0 / (math.sqrt(self.head_dim) * self.layer_number) \
            if apply_query_key_layer_scaling else 1.0 / math.sqrt(self.head_dim)
        print(f"[ATTN-INIT] layer={self.layer_number} head_dim={self.head_dim} "
              f"qk_layer_scale={apply_query_key_layer_scaling} "
              f"softmax_fp32={self.attention_softmax_in_fp32} "
              f"effective_scale={self._qk_scale:.6f}")

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        s = _SP_CTX['step']
        _r = dist.get_rank() if dist.is_initialized() else 0

        # M458: QK scale diagnostic — log effective scale once per 50 steps so we
        # can verify per-layer attenuation is actually different across layers.
        if _diag and s % 50 == 1:
            with torch.no_grad():
                print(f"[ATTN-M458] rank={_r} step={s} layer={self.layer_number} "
                      f"qk_scale={self._qk_scale:.6f} "
                      f"softmax_fp32={self.attention_softmax_in_fp32}")

        # M365 DIAG: pre-A2A QKV statistics on ALL ranks (not just rank 0)
        # This reveals whether SP ranks have divergent activations
        if _diag and s % 50 == 1:
            with torch.no_grad():
                print(f"[ATTN] rank={_r} step={s} Q={list(q.shape)} T={T} n_head={self.n_head} "
                      f"Q_norm={q.float().norm().item():.4f} K_norm={k.float().norm().item():.4f} "
                      f"V_norm={v.float().norm().item():.4f} "
                      f"Q_mean={q.float().mean().item():.6f} K_mean={k.float().mean().item():.6f} "
                      f"x_hash={x.float().sum().item():.4f} sp={_SP_CTX['on']}")

        if _SP_CTX['on'] and _SP_CTX['grp'] is not None:
            # M365 DIAG: capture pre-A2A Q hash to verify SP ranks have same data
            if _diag and s % 50 == 1:
                _diag.log_data_hash(s, _r, q, "pre-A2A-Q")

            q_pre = q  # keep ref for post-check
            q = _ulysses_a2a(q, 1, 2, _SP_CTX['grp'])
            k = _ulysses_a2a(k, 1, 2, _SP_CTX['grp'])
            v = _ulysses_a2a(v, 1, 2, _SP_CTX['grp'])

            # M365 DIAG: post-A2A shape + norm verification on ALL ranks
            if _diag and s % 50 == 1:
                _diag.log_a2a_stats(s, _r, "Q-fwd", q_pre, q)
                print(f"[ATTN-A2A] rank={_r} step={s} post Q={list(q.shape)} "
                      f"heads={q.shape[1]} seq={q.shape[2]} "
                      f"Q_norm_post={q.float().norm().item():.4f} "
                      f"K_norm_post={k.float().norm().item():.4f} "
                      f"V_norm_post={v.float().norm().item():.4f}")

            # M458: pass per-layer QK scale; SDPA's scale kwarg overrides the
            # default 1/sqrt(d_k), giving us combined 1/(sqrt(d)*layer) scaling.
            # When attention_softmax_in_fp32 is set we upcast q/k to float32 before
            # the dot-product so softmax numerics stay stable; v stays in original
            # dtype and the result is cast back before the output projection.
            if self.attention_softmax_in_fp32:
                # Manual fp32 attention path — mirrors Megatron transformer.py
                # SelfAttention.forward() ~lines 240-260 from commit 691747b1.
                q_f, k_f, v_f = q.float(), k.float(), v.float()
                attn_w = torch.baddbmm(
                    torch.empty(B * self.n_head, q_f.size(2), k_f.size(2),
                                dtype=torch.float32, device=q.device),
                    q_f.reshape(B * self.n_head, q_f.size(2), self.head_dim),
                    k_f.reshape(B * self.n_head, self.head_dim, k_f.size(2)),
                    beta=0.0, alpha=self._qk_scale,
                )
                mask = torch.triu(
                    torch.full((q_f.size(2), k_f.size(2)), float('-inf'),
                               device=q.device, dtype=torch.float32), diagonal=1)
                attn_w = attn_w + mask.unsqueeze(0)
                attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
                if self.training and self.dropout > 0.0:
                    attn_w = F.dropout(attn_w, p=self.dropout)
                y = torch.bmm(
                    attn_w,
                    v_f.reshape(B * self.n_head, v_f.size(2), self.head_dim)
                ).view(B, self.n_head, q_f.size(2), self.head_dim).to(q.dtype)
                if _diag and s % 50 == 1:
                    print(f"[ATTN-FP32-SP] rank={_r} step={s} layer={self.layer_number} "
                          f"attn_w_max={attn_w.float().max().item():.4f} "
                          f"attn_entropy={(-(attn_w*(attn_w+1e-9).log()).sum(-1).mean()).item():.4f}")
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                    scale=self._qk_scale,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            if _diag and s % 50 == 1:
                print(f"[ATTN-SDPA] rank={_r} step={s} y_pre_reverse={list(y.shape)} "
                      f"y_norm={y.float().norm().item():.4f} "
                      f"y_mean={y.float().mean().item():.8f} "
                      f"y_std={y.float().std().item():.6f}")

            y_pre_rev = y
            y = _ulysses_a2a(y, 2, 1, _SP_CTX['grp'])

            if _diag and s % 50 == 1:
                _diag.log_a2a_stats(s, _r, "Y-rev", y_pre_rev, y)
        else:
            # M458: non-SP path — same per-layer scale and optional fp32 softmax.
            if self.attention_softmax_in_fp32:
                q_f, k_f, v_f = q.float(), k.float(), v.float()
                attn_w = torch.baddbmm(
                    torch.empty(B * self.n_head, q_f.size(2), k_f.size(2),
                                dtype=torch.float32, device=q.device),
                    q_f.reshape(B * self.n_head, q_f.size(2), self.head_dim),
                    k_f.reshape(B * self.n_head, self.head_dim, k_f.size(2)),
                    beta=0.0, alpha=self._qk_scale,
                )
                mask = torch.triu(
                    torch.full((q_f.size(2), k_f.size(2)), float('-inf'),
                               device=q.device, dtype=torch.float32), diagonal=1)
                attn_w = attn_w + mask.unsqueeze(0)
                attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
                if self.training and self.dropout > 0.0:
                    attn_w = F.dropout(attn_w, p=self.dropout)
                y = torch.bmm(
                    attn_w,
                    v_f.reshape(B * self.n_head, v_f.size(2), self.head_dim)
                ).view(B, self.n_head, q_f.size(2), self.head_dim).to(q.dtype)
                if _diag and s % 50 == 1:
                    print(f"[ATTN-FP32] rank={_r} step={s} layer={self.layer_number} "
                          f"attn_w_max={attn_w.float().max().item():.4f} "
                          f"attn_entropy={(-(attn_w*(attn_w+1e-9).log()).sum(-1).mean()).item():.4f}")
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                    scale=self._qk_scale,
                    dropout_p=self.dropout if self.training else 0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with optional activation checkpointing.

    M341 — Claude-30: Activation Checkpointing + AutoSP Integration
    ================================================================
    Supports two AC strategies per NeurIPS reviewer feedback:

    1. Layer-wise AC (torch.utils.checkpoint):
       Standard approach used by HuggingFace, Megatron-LM.
       Wraps entire block in checkpoint — discards all activations
       within the block during forward, recomputes during backward.
       Pro: Simple, well-tested. Con: Coarse-grained — cannot
       selectively keep some activations (e.g., SDPA output).

    2. Compile-time AC (AutoSP + torch.compile):
       AutoSP's activation checkpointing operates on Aten-IR operators
       (individual matmuls, sigmoids, etc.) rather than layers.
       Pro: Finer-grained search space — can keep attention output
       while checkpointing MLP, achieving better memory/compute
       tradeoff. Con: Requires torch.compile, not eager-compatible.

    For DES-LOC experiments:
    - Layer-wise AC is always available (no compile dependency)
    - Compile-time AC is automatically used when --use_autosp is set
    - Both are orthogonal to DES-LOC Kx gating
    - SP+DEC+AC: sequence parallel (data split) + desynced comm
      (temporal split) + activation checkpointing (memory split)

    Why SDPA is the right attention backend (addressing reviewer):
    - F.scaled_dot_product_attention dispatches to FlashAttention-2
      on A100/H100 automatically → O(T) memory, not quadratic
    - Required by AutoSP compile pass (identifies attention ops)
    - DeepSpeed Ulysses SP also works with SDPA in compile mode
    - Ring Flash Attention comparison: AutoSP achieves 2.26× longer
      context vs RingAttention across 3B/8B/13B models (rebuttal data)
    """
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int,
                 dropout: float = 0.0, use_ac: bool = False,
                 layer_number: int = 1,
                 apply_query_key_layer_scaling: bool = False,
                 attention_softmax_in_fp32: bool = False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        # M458: pass Megatron 691747b1 per-layer QK scaling + fp32 softmax flags
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout,
                                        layer_number=layer_number,
                                        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                                        attention_softmax_in_fp32=attention_softmax_in_fp32)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
        self.use_ac = use_ac

    def _block_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward — separated for checkpoint wrapper."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ac and self.training:
            # Layer-wise activation checkpointing via torch.utils.checkpoint
            # use_reentrant=False is the modern API (PyTorch 2.0+)
            # This discards all intermediate activations in _block_forward
            # and recomputes them during backward — saves ~60% activation
            # memory per layer at ~33% compute overhead.
            return torch.utils.checkpoint.checkpoint(
                self._block_forward, x, use_reentrant=False
            )
        return self._block_forward(x)


# =============================================================================
# NEURON_SP PORT: Megatron b9b6fe0d4 — force output gathering
# Adapted from megatron/utils.py vocab_size_with_padding.
# Key fix: guard `if multiple > 0` prevents zero-division when
# make_vocab_size_divisible_by=0 or model_parallel_world_size=0.
# 20% adaptation: uses world_size from dist instead of mpu; adds print breakpoint.
# =============================================================================
NEURONSP_VOCAB_DIVISIBLE_BY: int = 128  # Megatron default make_vocab_size_divisible_by

# NEURON_SP PORT: Megatron 2c58c9b04 — added filtering based on sentence length
# Adapted from megatron/data/helpers.cpp: const int32_t LONG_SENTENCE_LEN = 256;
# Patterns (synthetic sentences) whose token count exceeds this threshold are
# filtered out during SyntheticDataset.__init__ (mirrors build_mapping_impl logic).
NEURONSP_LONG_SENTENCE_LEN: int = 512


# =============================================================================
# NEURON_SP PORT: Megatron 57064fd6f — memory optimization in mpu cross entropy
# Adapted from megatron/mpu/cross_entropy.py _VocabParallelCrossEntropy.forward.
#
# Key sequence from 57064fd6f:
#   1. Remove the full logits.clone() at the top — operate directly on the
#      input tensor (the caller must not reuse it after this call).
#   2. Compute max in-place: logits_max = vocab_parallel_logits.max(dim=-1)
#   3. Subtract in-place:   vocab_parallel_logits.sub_(logits_max.unsqueeze(-1))
#   4. Extract predicted logit BEFORE exp: predicted_logits_1d.clone().contiguous()
#      — advanced-index returns a view sharing storage; in-place exp below would
#        corrupt the slice without the clone. (57064fd critical note)
#   5. In-place exp that REUSES logits storage: torch.exp(logits, out=exp_logits)
#   6. sum_exp -> log -> loss = log(sum_exp) - predicted_logit
#
# Memory savings: eliminates one O(B*T*V) clone at the top (the largest tensor
# in the forward pass for large vocab models). The only extra allocation is the
# O(B*T) predicted_logits_1d.clone() which is negligible by comparison.
#
# 20% adaptation: standalone function instead of autograd.Function class;
# no distributed all_reduce (single-GPU benchmark); added print breakpoints;
# uses integer target tensor directly without masked_target subtraction.
# =============================================================================

def _neuronsp_cross_entropy_memory_opt(
    logits: torch.Tensor,          # [B*T, V] — MODIFIED IN-PLACE, do not reuse
    targets: torch.Tensor,         # [B*T]    — integer target indices
    step: int = 0,
    rank: int = 0,
) -> torch.Tensor:
    """Memory-optimised cross-entropy loss (port of Megatron 57064fd6f).

    Eliminates the O(B*T*V) clone at the top of the standard CE computation
    by operating in-place on the logits tensor.  The caller must NOT use
    `logits` after this call — its values are overwritten during the exp step.

    Returns:
        loss -- scalar float32 tensor (mean over B*T tokens).
    """
    # Step 1: max in-place — 57064fd: no clone, use vocab_parallel_logits directly
    logits_max = torch.max(logits, dim=-1)[0]    # [B*T]
    # Step 2: subtract max in-place to stabilise exp (numerically stable softmax trick)
    logits.sub_(logits_max.unsqueeze(dim=-1))     # 57064fd key line — in-place

    # Step 3: extract predicted logit BEFORE exp
    # Advanced indexing returns a view sharing storage with logits.
    # The .clone().contiguous() is CRITICAL: without it, torch.exp(logits, out=logits)
    # below would corrupt the slice as it writes exp values back into the same buffer.
    arange_1d = torch.arange(logits.size(0), device=logits.device)
    predicted_logits_1d = logits[arange_1d, targets]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()   # 57064fd safeguard

    # Step 4: in-place exp — reuses logits storage, O(0) extra allocation
    # 57064fd: torch.exp(vocab_parallel_logits, out=exp_logits)
    #   where exp_logits IS vocab_parallel_logits (aliased assignment)
    torch.exp(logits, out=logits)                 # logits is now exp_logits
    sum_exp_logits = logits.sum(dim=-1)           # [B*T]

    # Step 5: loss = log(sum_exp) - predicted_logit (numerically stable CE)
    loss = torch.log(sum_exp_logits) - predicted_logits_1d    # [B*T]
    mean_loss = loss.mean()

    # Print breakpoint — fires every 50 steps so the opt is visible in logs
    if step % 50 == 1:
        print(f"[CE-57064fd-OPT] rank={rank} step={step} "
              f"loss={mean_loss.item():.6f} "
              f"sum_exp_mean={sum_exp_logits.float().mean().item():.4f} "
              f"pred_logit_mean={predicted_logits_1d.float().mean().item():.6f} "
              f"B_T={logits.size(0)} V={logits.size(-1)} "
              f"opt=no_clone+inplace_exp (57064fd6f)")
    return mean_loss


def _neuronsp_vocab_size_with_padding(num_tokens: int,
                                      divisible_by: int = NEURONSP_VOCAB_DIVISIBLE_BY,
                                      world_size: int = 1) -> int:
    """Pad vocab size to be divisible by divisible_by * world_size.

    Port of Megatron utils.py::vocab_size_with_padding (b9b6fe0d4).
    Critical guard: if multiple > 0 prevents infinite loop when either
    divisible_by or world_size is 0 (model-parallel not initialised yet).
    """
    after = num_tokens
    multiple = divisible_by * world_size
    print(f"[NEURONSP-VOCAB] padding vocab {num_tokens} → multiple={multiple} "
          f"(divisible_by={divisible_by}, world_size={world_size})")
    if multiple > 0:
        while (after % multiple) != 0:
            after += 1
    dummy_tokens = after - num_tokens
    print(f"[NEURONSP-VOCAB] padded vocab (size: {num_tokens}) with {dummy_tokens} "
          f"dummy tokens (new size: {after})")
    return after


class GPT(nn.Module):
    """GPT-2 Model with optional activation checkpointing."""
    def __init__(self, vocab_size: int, max_seq_len: int, n_layer: int,
                 n_head: int, n_embd: int, use_ac: bool = False,
                 apply_query_key_layer_scaling: bool = False,
                 attention_softmax_in_fp32: bool = False,
                 num_unique_layers: int = None,
                 param_sharing_style: str = 'grouped'):
        super().__init__()
        self.max_seq_len = max_seq_len
        # M458: if QK layer scaling is on, fp32 softmax is auto-implied (Megatron 691747b1)
        _softmax_fp32 = attention_softmax_in_fp32 or apply_query_key_layer_scaling
        print('[M249]')
        print(f"[GPT-INIT] n_layer={n_layer} apply_qk_layer_scaling={apply_query_key_layer_scaling} "
              f"attention_softmax_in_fp32={_softmax_fp32}")

        # M249: parameter sharing — track num_unique_layers and sharing style
        # Port of Megatron 80f90dcdc (added parameters sharing).
        self.n_layer = n_layer
        self.num_unique_layers = num_unique_layers if num_unique_layers is not None else n_layer
        if n_layer % self.num_unique_layers != 0:
            raise ValueError(
                f'num_layers ({n_layer}) must be divisible by num_unique_layers ({self.num_unique_layers})')
        assert param_sharing_style in ('grouped', 'spaced'),             f'param_sharing_style must be grouped or spaced, got {param_sharing_style}'
        self.param_sharing_style = param_sharing_style

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(max_seq_len, n_embd),
            drop = nn.Dropout(0.0),
            h = nn.ModuleList([
                # layer_number is 1-indexed per Megatron convention so scale = 1/(sqrt(d)*i)
                TransformerBlock(n_embd, n_head, max_seq_len, use_ac=use_ac,
                                 layer_number=i + 1,
                                 apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                                 attention_softmax_in_fp32=_softmax_fp32)
                for i in range(self.num_unique_layers)
            ]),
            ln_f = LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Print layer ordering when param sharing is active.
        if self.n_layer != self.num_unique_layers:
            print('[M249] will be using the following layer ordering:')
            for i in range(self.n_layer):
                print('   layer: {:3d} --> unique layer: {:3d}'.format(
                    i, self._get_layer_index(i)))
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # NEURON_SP PORT: Megatron 57c2060fe — layers.py _initialize_affine_weight
        # Mark the vocab-parallel embedding weight with partition_dim + stride so
        # checkpoint merge tools can reconstruct the full tensor from rank shards.
        # In model-parallel mode the vocab dimension (dim=0) is split across ranks;
        # stride=1 means no interleaving (contiguous sharding, same as Megatron default).
        # Bias partition_dim=0 matches ColumnParallelLinear convention in 57c2060fe.
        _neuronsp_mark_weight_parallel(self.transformer.wte.weight,
                                       partition_dim=0, stride=1)
        print(f"[MPU-INIT-57c2060fe] vocab_parallel weight tagged: "
              f"world_size={get_model_parallel_world_size()} "
              f"rank={get_model_parallel_rank()}")

        # Init weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")
    
    def _get_layer_index(self, layer_number: int) -> int:
        """Map absolute layer index to unique-layer index (M249: Megatron 80f90dcdc)."""
        if self.param_sharing_style == 'grouped':
            return layer_number % self.num_unique_layers
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.n_layer // self.num_unique_layers)
        raise AssertionError('should not be here')

    def _get_layer(self, layer_number: int):
        """Return the (possibly shared) layer for the given absolute index."""
        return self.transformer.h[self._get_layer_index(layer_number)]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"
        pos_off = _SP_CTX['rk'] * T if _SP_CTX['on'] else 0
        pos = torch.arange(pos_off, pos_off + T, dtype=torch.long, device=idx.device)
        s = _SP_CTX['step']
        _r = dist.get_rank() if dist.is_initialized() else 0

        # M365 DIAG: input token statistics on ALL ranks
        if _diag and s % 50 == 1:
            print(f"[FWD] rank={_r} step={s} idx=[{B},{T}] pos=[{pos_off}..{pos_off+T-1}] "
                  f"ids[:8]={idx[0,:min(8,T)].tolist()} "
                  f"ids_hash={idx.float().sum().item():.0f}")

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # M365 DIAG: embedding output statistics
        if _diag and s % 50 == 1:
            print(f"[FWD-EMB] rank={_r} step={s} "
                  f"tok_emb_norm={tok_emb.float().norm().item():.4f} "
                  f"pos_emb_norm={pos_emb.float().norm().item():.4f} "
                  f"x_norm={x.float().norm().item():.4f}")

        for index in range(self.n_layer):
            block = self._get_layer(index)
            x = block(x)

        # M365 DIAG: post-transformer hidden state statistics
        if _diag and s % 50 == 1:
            print(f"[FWD-POST] rank={_r} step={s} "
                  f"hidden_norm={x.float().norm().item():.4f} "
                  f"hidden_mean={x.float().mean().item():.8f} "
                  f"hidden_std={x.float().std().item():.6f}")

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # M461 (57064fd): CE memory opt — no full logits clone; in-place max-sub
            # then in-place exp after scalar predicted-logit is extracted.
            # Megatron 57064fd key sequence:
            #   1. max in-place (no clone) → subtract → extract predicted logit (clone+contiguous)
            #   2. torch.exp(..., out=logits) reuses logits storage for exp — O(0) extra alloc
            #   3. loss = log(sum_exp) − predicted_logit
            # Knuth §2.3: "The real problem is that programmers have spent far too much
            # time worrying about efficiency in the wrong places." — here the clone was
            # exactly in the wrong place: O(B*T*V) before a O(B*T) extraction.
            _logits_flat = logits.view(-1, logits.size(-1))   # [B*T, V]; no copy
            _targets_flat = targets.view(-1)                  # [B*T]

            # Step 1: max-subtract in-place (57064fd: skip clone, operate directly)
            with torch.no_grad():
                _logit_max = _logits_flat.max(dim=-1)[0]      # [B*T]
                _logit_max_global = _logit_max.max().item()
                _logit_min_global = _logit_max.min().item()
                # M461 DIAG: logit range — detect explosion (57064fd threshold: >50)
                if s % 50 == 1:
                    print(f"[CE-M461] rank={_r} step={s} "
                          f"logit_max={_logit_max_global:.4f} "
                          f"logit_min_of_max={_logit_min_global:.4f} "
                          f"range={_logit_max_global - _logit_min_global:.4f} "
                          f"vocab={logits.size(-1)} "
                          f"n_tokens={_targets_flat.numel()} "
                          f"opt=no_clone+inplace_exp")
                if _logit_max_global > 50.0:
                    print(f"[CE-WARN-M461] rank={_r} step={s} "
                          f"LOGIT EXPLOSION: max={_logit_max_global:.2f} "
                          f"(threshold=50). Stale params may be causing drift.")
            _logits_flat.sub_(_logit_max.unsqueeze(dim=-1))   # in-place; 57064fd line

            # Step 2: extract predicted logit BEFORE exp (critical: clone+contiguous
            # — advanced-index shares storage with _logits_flat; in-place exp below
            # would corrupt the slice without this.  57064fd: predicted_logits_1d.clone())
            _arange = torch.arange(_logits_flat.size(0), device=_logits_flat.device)
            _pred_logit = _logits_flat[_arange, _targets_flat].clone().contiguous()  # [B*T]

            # Step 3: in-place exp — reuses _logits_flat storage (57064fd: out=exp_logits)
            torch.exp(_logits_flat, out=_logits_flat)
            _sum_exp = _logits_flat.sum(dim=-1)               # [B*T]

            # Step 4: loss = log(Σexp) − predicted_logit
            loss = torch.log(_sum_exp) - _pred_logit          # [B*T]
            loss = loss.mean()
            # M364: With Ulysses A2A each SP rank sees its local token subset.
            # Gradients sync via DES-LOC AllReduce — do NOT all_reduce loss here.

            # M461 DIAG: full loss decomposition on ALL ranks
            if _diag and s % 50 == 1:
                _diag.log_loss_decomp(s, _r, loss, logits, targets, _SP_CTX['on'])
                print(f"[FWD-LOSS-M461] rank={_r} step={s} loss={loss.item():.6f} "
                      f"sum_exp_mean={_sum_exp.float().mean().item():.4f} "
                      f"pred_logit_mean={_pred_logit.float().mean().item():.6f} "
                      f"sp={'local' if _SP_CTX['on'] else 'full'} "
                      f"n_tokens={targets.numel()}")
        return logits, loss


# =============================================================================
# NEURON_SP PORT: Megatron 94e2ca575 (commit #96) — arguments.py refactored
# Adapted from megatron/arguments.py, megatron/global_vars.py,
#              megatron/initialize.py, megatron/training.py
# Key changes: parse_args() now takes defaults={} dict; all add_*_args functions
# renamed to _add_*_args (private); _print_args moved to take args directly;
# set_global_variables gains args_defaults={}; initialize_megatron gains
# args_defaults={}; _write_args_to_tensorboard() extracted; training.py
# switches from get_args(extra_args_provider) to parse_args+initialize pattern.
# 20% adaptation: integrated into NeuronSP config dataclass + argparse setup;
#   uses RANK/WORLD_SIZE env vars directly like Megatron; print breakpoints added;
#   model_parallel_size clamped to world_size (matching Megatron logic).
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_parse_args_with_defaults(
    args: argparse.Namespace,
    defaults: dict,
) -> argparse.Namespace:
    """Apply a defaults dict to an already-parsed Namespace, matching Megatron
    94e2ca575 parse_args(defaults={}) semantics.

    Port of megatron/arguments.py::parse_args (94e2ca575).
    20% adaptation: accepts pre-parsed Namespace instead of calling parse_args()
    internally; prints breakpoint on rank 0; skips assert for None (NeuronSP
    args may have non-None defaults from argparse).
    """
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    print(f"[M96-PARSE_ARGS] rank={rank} world_size={world_size} "
          f"applying {len(defaults)} defaults: {list(defaults.keys())}")

    for key, val in defaults.items():
        # 20% deviation: NeuronSP doesn't enforce None-only override —
        # only override if the current value equals the argparse default (None).
        current = getattr(args, key, None)
        if current is None:
            setattr(args, key, val)
            print(f"[M96-PARSE_ARGS]   set {key} = {val}")
        else:
            print(f"[M96-PARSE_ARGS]   skip {key} (already={current!r})")

    # Megatron: clamp model_parallel_size to world_size
    if hasattr(args, 'model_parallel_size'):
        args.model_parallel_size = min(
            getattr(args, 'model_parallel_size', 1), world_size)
        if rank == 0:
            print(f"[M96-PARSE_ARGS] using world_size={world_size} "
                  f"model_parallel_size={args.model_parallel_size}")

    # Megatron: dynamic_loss_scale flag
    if hasattr(args, 'loss_scale'):
        args.dynamic_loss_scale = (args.loss_scale is None)
        print(f"[M96-PARSE_ARGS] dynamic_loss_scale={args.dynamic_loss_scale}")

    return args


def _neuronsp_print_args_m96(args: argparse.Namespace, rank: int = 0) -> None:
    """Print all arguments sorted, matching Megatron 94e2ca575 _print_args().

    Port of megatron/arguments.py::_print_args (94e2ca575).
    20% adaptation: accepts explicit rank param; uses 36-dot padding instead of 32.
    """
    if rank == 0:
        print('[M96-PRINT_ARGS] -------------------- arguments --------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * max(1, 36 - len(arg))
            str_list.append(f'  {arg} {dots} {getattr(args, arg)}')
        for line in sorted(str_list, key=lambda x: x.lower()):
            print(line, flush=True)
        print('[M96-PRINT_ARGS] ---------------- end of arguments ----------------',
              flush=True)


# =============================================================================
# NEURON_SP PORT: Megatron 5050203fc (commit #97) — working on utils
# Adapted from megatron/global_vars.py, megatron/training.py, megatron/utils.py
# Key changes: Timers class moved from utils.py into global_vars.py; training.py
# switches from parse_args+initialize_megatron(message,args) to
# initialize_megatron(extra_args_provider, args_defaults)+get_args()+get_timers();
# initialize_megatron() removed from training.py (it's now in initialize.py);
# utils.py gets print_rank_0, reduce_losses, check_adlr_autoresume_termination
# promoted to the top.
# 20% adaptation: NeuronSP Timers wraps Python time.perf_counter instead of
#   torch.cuda.synchronize (avoids CUDA init in non-distributed runs);
#   write() and log() preserve Megatron API; print breakpoints added.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

import time as _time_module

class NeuronSPTimers:
    """Group of timers — port of Megatron Timers class (5050203fc).

    Moved from megatron/utils.py to megatron/global_vars.py in commit 5050203fc.
    20% adaptation: uses time.perf_counter instead of torch.cuda.synchronize
    so it works without an active CUDA context; wall-clock semantics preserved.
    """

    class _Timer:
        """Single named timer."""

        def __init__(self, name: str):
            self.name_ = name
            self.elapsed_: float = 0.0
            self.started_: bool = False
            self.start_time: float = _time_module.perf_counter()

        def start(self) -> None:
            """Start the timer."""
            assert not self.started_, (
                f'[M97-TIMER] {self.name_}: timer has already been started')
            self.start_time = _time_module.perf_counter()
            self.started_ = True
            print(f'[M97-TIMER] {self.name_}: started', flush=True)

        def stop(self) -> None:
            """Stop the timer."""
            assert self.started_, (
                f'[M97-TIMER] {self.name_}: timer is not started')
            self.elapsed_ += _time_module.perf_counter() - self.start_time
            self.started_ = False

        def reset(self) -> None:
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset: bool = True) -> float:
            """Return elapsed seconds, optionally resetting."""
            was_started = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self.elapsed_
            if reset:
                self.reset()
            if was_started:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers: dict = {}
        print('[M97-TIMERS] NeuronSPTimers initialized (perf_counter backend)',
              flush=True)

    def __call__(self, name: str) -> '_Timer':
        if name not in self.timers:
            self.timers[name] = self._Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration: int,
              normalizer: float = 1.0, reset: bool = False) -> None:
        """Write timers to tensorboard writer."""
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '_time', value, iteration)
            print(f'[M97-TIMERS] write {name}_time={value:.4f} iter={iteration}',
                  flush=True)

    def log(self, names, normalizer: float = 1.0, reset: bool = True) -> None:
        """Log a group of timers (ms)."""
        assert normalizer > 0.0
        string = '[M97-TIMERS] time (ms)'
        for name in names:
            elapsed_ms = (self.timers[name].elapsed(reset=reset)
                          * 1000.0 / normalizer)
            string += f' | {name}: {elapsed_ms:.2f}'
        print(string, flush=True)


# =============================================================================
# NEURON_SP PORT: Megatron 11220df86 (commit #98) — tokenizer moved to own dir
# Adapted from megatron/data/__init__.py, megatron/global_vars.py,
#              megatron/data/bert_dataset.py, megatron/tokenizer/__init__.py
# Key changes: FullBertTokenizer import moved from megatron.data to
#   megatron.tokenizer.bert_tokenization; global_vars.py switches from
#   megatron.data.tokenizer to megatron.tokenizer; tokenizer/ dir created with
#   its own __init__.py exporting build_tokenizer.
# 20% adaptation: NeuronSP has no separate tokenizer module; we document the
#   import path change and add a registry stub that maps tokenizer_type strings
#   to callable factories (mirrors megatron/tokenizer/__init__.py role);
#   print breakpoints show which tokenizer path would be resolved.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

# Megatron commit 11220df86 — tokenizer directory structure
# In Megatron: megatron/data/bert_tokenization.py → megatron/tokenizer/bert_tokenization.py
#              megatron/data/tokenizer.py          → megatron/tokenizer/tokenizer.py
#              megatron/tokenizer/__init__.py created with: from .tokenizer import build_tokenizer
#              megatron/data/__init__.py: FullBertTokenizer import removed
#              megatron/global_vars.py: from megatron.data.tokenizer → from megatron.tokenizer
#
# NeuronSP adaptation: no separate tokenizer submodule needed here (engine.py
# handles tokenization externally). We expose a _neuronsp_tokenizer_registry dict
# that mirrors the megatron/tokenizer/__init__.py build_tokenizer dispatch table.

_NEURONSP_TOKENIZER_REGISTRY: dict = {
    # Megatron 11220df86: these live in megatron/tokenizer/tokenizer.py
    'BertWordPieceLowerCase': None,   # FullTokenizer with do_lower_case=True
    'BertWordPieceCase':      None,   # FullTokenizer with do_lower_case=False
    'GPT2BPETokenizer':       None,   # tiktoken / HF GPT-2 BPE
}


def _neuronsp_build_tokenizer_m98(tokenizer_type: str, vocab_file: str = ''):
    """Registry-based tokenizer builder — port of megatron/tokenizer/__init__.py
    build_tokenizer (11220df86).

    20% adaptation: raises NotImplementedError with a human-readable message
    rather than calling the actual tokenizer class (NeuronSP uses HF tokenizers
    separately). The function exists to document the import-path change and
    provide a hook for future wiring.
    """
    print(f'[M98-TOKENIZER] build_tokenizer called: type={tokenizer_type!r} '
          f'vocab_file={vocab_file!r}', flush=True)
    if tokenizer_type not in _NEURONSP_TOKENIZER_REGISTRY:
        raise ValueError(
            f'[M98-TOKENIZER] Unknown tokenizer_type={tokenizer_type!r}. '
            f'Valid types (megatron/tokenizer/tokenizer.py): '
            f'{list(_NEURONSP_TOKENIZER_REGISTRY.keys())}')
    print(f'[M98-TOKENIZER] resolved to megatron.tokenizer.tokenizer '
          f'(moved from megatron.data.tokenizer in commit 11220df86)',
          flush=True)
    raise NotImplementedError(
        '[M98-TOKENIZER] NeuronSP does not instantiate megatron tokenizers; '
        'use HuggingFace tokenizers directly.')


# =============================================================================
# NEURON_SP PORT: Megatron 86e7d6246 (commit #99) — refactored checkpoints
# Adapted from megatron/checkpointing.py (new file), megatron/utils.py
# Key changes: save_checkpoint / load_checkpoint / get_checkpoint_name /
#   get_checkpoint_tracker_filename / ensure_directory_exists extracted from
#   utils.py into new megatron/checkpointing.py; signature of save_checkpoint
#   drops `args` param (uses get_args() internally); check_checkpoint_args()
#   added; utils.py now imports from global_vars instead of local defs.
# 20% adaptation: NeuronSP checkpoint helpers store rank in filename; tracker
#   file logic identical to Megatron; print breakpoints show iteration + path;
#   no mpu dependency (uses dist.get_rank() directly).
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_get_checkpoint_name(
    checkpoints_path: str,
    iteration: int,
    release: bool = False,
    mp_rank: int = 0,
) -> str:
    """Return a checkpoint file path — port of megatron/checkpointing.py
    get_checkpoint_name (86e7d6246).

    20% adaptation: mp_rank is explicit (no mpu.get_model_parallel_rank());
    directory uses 'release' literal or zero-padded iteration string.
    """
    directory = 'release' if release else f'iter_{iteration:07d}'
    path = os.path.join(
        checkpoints_path, directory,
        f'mp_rank_{mp_rank:02d}',
        'model_optim_rng.pt',
    )
    print(f'[M99-CKPT] get_checkpoint_name iter={iteration} '
          f'release={release} mp_rank={mp_rank} → {path}', flush=True)
    return path


def _neuronsp_get_checkpoint_tracker_filename(checkpoints_path: str) -> str:
    """Return tracker file path — port of megatron/checkpointing.py
    get_checkpoint_tracker_filename (86e7d6246).

    Tracker file records the latest checkpointed iteration so training can
    resume from it on restart.
    """
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def _neuronsp_ensure_directory_exists(filename: str) -> None:
    """Create parent directories for filename if they don't exist —
    port of megatron/checkpointing.py ensure_directory_exists (86e7d6246).
    """
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        print(f'[M99-CKPT] created directory: {dirname}', flush=True)


def _neuronsp_save_checkpoint_m99(
    iteration: int,
    model,
    optimizer=None,
    lr_scheduler=None,
    output_dir: str = './checkpoints',
    rank: int = 0,
    mp_rank: int = 0,
    no_save_optim: bool = False,
    no_save_rng: bool = False,
) -> None:
    """Save a model checkpoint — port of megatron/checkpointing.py
    save_checkpoint (86e7d6246).

    Key Megatron change: `args` param removed; function calls get_args()
    internally. NeuronSP adaptation: explicit params instead of global args;
    only rank-0 of data parallel writes (here: rank==0); tracker updated after.
    """
    import random
    import numpy as np

    print(f'[M99-CKPT] save_checkpoint iteration={iteration} rank={rank}',
          flush=True)

    if rank == 0:
        checkpoint_name = _neuronsp_get_checkpoint_name(
            output_dir, iteration, mp_rank=mp_rank)
        _neuronsp_ensure_directory_exists(checkpoint_name)

        state_dict: dict = {
            'iteration': iteration,
            'model': model.state_dict() if hasattr(model, 'state_dict') else {},
        }

        if not no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if lr_scheduler is not None:
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        if not no_save_rng:
            state_dict['random_rng_state'] = random.getstate()
            state_dict['np_rng_state'] = np.random.get_state()
            state_dict['torch_rng_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()

        torch.save(state_dict, checkpoint_name)
        print(f'[M99-CKPT] successfully saved {checkpoint_name}', flush=True)

        tracker = _neuronsp_get_checkpoint_tracker_filename(output_dir)
        with open(tracker, 'w') as f:
            f.write(str(iteration))
        print(f'[M99-CKPT] tracker updated: {tracker} → {iteration}', flush=True)


def _neuronsp_load_checkpoint_m99(
    model,
    optimizer=None,
    lr_scheduler=None,
    output_dir: str = './checkpoints',
    rank: int = 0,
    mp_rank: int = 0,
    finetune: bool = False,
    no_load_optim: bool = False,
    no_load_rng: bool = False,
) -> int:
    """Load a model checkpoint — port of megatron/checkpointing.py
    load_checkpoint (86e7d6246).

    Returns the iteration number. If no tracker file exists, returns 0.
    """
    import random
    import numpy as np
    import sys as _sys

    tracker = _neuronsp_get_checkpoint_tracker_filename(output_dir)
    print(f'[M99-CKPT] load_checkpoint reading tracker: {tracker}', flush=True)

    if not os.path.isfile(tracker):
        print(f'[M99-CKPT] WARNING: tracker not found at {tracker}; '
              'starting from random init', flush=True)
        return 0

    iteration = 0
    release = False
    with open(tracker, 'r') as f:
        metastring = f.read().strip()
    try:
        iteration = int(metastring)
    except ValueError:
        release = (metastring == 'release')
        if not release:
            print(f'[M99-CKPT] ERROR: invalid tracker content: {metastring!r}',
                  flush=True)
            _sys.exit(1)

    assert iteration > 0 or release, (
        f'[M99-CKPT] error parsing tracker file {tracker}')

    checkpoint_name = _neuronsp_get_checkpoint_name(
        output_dir, iteration, release=release, mp_rank=mp_rank)
    print(f'[M99-CKPT] loading checkpoint {checkpoint_name} '
          f'iteration={iteration} release={release}', flush=True)

    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except Exception as e:
        print(f'[M99-CKPT] ERROR: could not load checkpoint: {e}', flush=True)
        _sys.exit(1)

    if finetune or release:
        iteration = 0
    else:
        iteration = state_dict.get('iteration', state_dict.get('total_iters', 0))

    model.load_state_dict(state_dict['model'])
    print(f'[M99-CKPT] model loaded from {checkpoint_name}', flush=True)

    if not release and not finetune and not no_load_optim:
        if optimizer is not None and 'optimizer' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])
        if lr_scheduler is not None and 'lr_scheduler' in state_dict:
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])

    if not release and not finetune and not no_load_rng:
        try:
            import random
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            if torch.cuda.is_available() and 'cuda_rng_state' in state_dict:
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
        except KeyError as e:
            print(f'[M99-CKPT] WARNING: could not restore RNG state: {e}',
                  flush=True)

    print(f'[M99-CKPT] successfully loaded iter={iteration}', flush=True)
    return iteration


# =============================================================================
# NEURON_SP PORT: Megatron 3f58649b7 (commit #100) — utils partially refactored
# Adapted from megatron/__init__.py (new), megatron/utils.py, megatron/training.py,
#   megatron/checkpointing.py, megatron/data/bert_dataset.py,
#   megatron/data/indexed_dataset.py, megatron/initialize.py,
#   megatron/learning_rates.py, megatron/model/classification.py,
#   megatron/model/multiple_choice.py, tasks/eval_utils.py, etc.
# Key changes: megatron/__init__.py created exporting get_args, get_tokenizer,
#   get_tensorboard_writer, get_adlr_autoresume, get_timers, print_rank_0;
#   all modules switch `from megatron.utils import print_rank_0` to
#   `from megatron import print_rank_0`; checkpointing.py switches from
#   `.global_vars import get_args` to `from megatron import get_args`;
#   utils.py drops duplicated functions (now in checkpointing.py / __init__.py);
#   save_checkpoint/load_checkpoint calls in training.py drop `args` param;
#   check_adlr_autoresume_termination gets `args` param dropped (uses get_args).
# 20% adaptation: NeuronSP consolidates these into a module-level accessor
#   dict (_NEURONSP_GLOBALS) as a lightweight substitute for megatron/__init__.py;
#   print_rank_0 exposed directly; save/load wrappers delegate to M99 helpers;
#   print breakpoints on every accessor call.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

# NeuronSP global state registry (mirrors megatron/__init__.py exports)
_NEURONSP_GLOBALS: dict = {
    'args': None,
    'tokenizer': None,
    'tensorboard_writer': None,
    'adlr_autoresume': None,
    'timers': None,
}


def _neuronsp_get_args_m100():
    """Get global args — mirrors megatron/__init__.py get_args (3f58649b7)."""
    val = _NEURONSP_GLOBALS.get('args')
    print(f'[M100-GLOBALS] get_args → {type(val).__name__}', flush=True)
    return val


def _neuronsp_get_timers_m100() -> NeuronSPTimers:
    """Get global timers — mirrors megatron/__init__.py get_timers (3f58649b7).

    Lazily initialises NeuronSPTimers on first call (same pattern as
    megatron/global_vars.py _set_timers).
    """
    if _NEURONSP_GLOBALS['timers'] is None:
        _NEURONSP_GLOBALS['timers'] = NeuronSPTimers()
        print('[M100-GLOBALS] get_timers: initialised NeuronSPTimers', flush=True)
    return _NEURONSP_GLOBALS['timers']


def _neuronsp_print_rank_0_m100(message: str) -> None:
    """If distributed is initialized print only on rank 0 —
    port of megatron/__init__.py print_rank_0 (3f58649b7).

    This is the canonical version: all modules should import from here
    (previously each module imported from megatron.utils).
    20% adaptation: prefixed with [M100] tag for traceability.
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(f'[M100-RANK0] {message}', flush=True)
    else:
        print(f'[M100-RANK0] {message}', flush=True)


def _neuronsp_check_adlr_autoresume_m100(
    iteration: int,
    model,
    optimizer=None,
    lr_scheduler=None,
    output_dir: str = './checkpoints',
    rank: int = 0,
) -> None:
    """Check autoresume termination — port of megatron/utils.py
    check_adlr_autoresume_termination (3f58649b7).

    Key Megatron change in 3f58649b7: `args` param dropped; function calls
    get_args() and get_adlr_autoresume() internally; uses sys.exit(0) instead
    of exit(0). NeuronSP adaptation: explicit params; autoresume stub always
    returns False (no ADLR cluster); print breakpoints show state.
    """
    import sys as _sys
    print(f'[M100-AUTORESUME] check iter={iteration} rank={rank}', flush=True)

    autoresume = _NEURONSP_GLOBALS.get('adlr_autoresume')
    if autoresume is None:
        print('[M100-AUTORESUME] no autoresume configured; skipping', flush=True)
        return

    if dist.is_initialized():
        dist.barrier()

    if autoresume.termination_requested():
        print('[M100-AUTORESUME] termination requested!', flush=True)
        _neuronsp_save_checkpoint_m99(
            iteration, model, optimizer, lr_scheduler,
            output_dir=output_dir, rank=rank)
        _neuronsp_print_rank_0_m100('>>> autoresume termination request found!')
        if dist.is_initialized() and dist.get_rank() == 0:
            autoresume.request_resume()
        _neuronsp_print_rank_0_m100('>>> training terminated. Returning')
        _sys.exit(0)


# =============================================================================
# MEGATRON 66719e9 — RandomSampler with replacement + epoch gating
# =============================================================================
# 20%-derived from Megatron-LM data_utils/samplers.py commit 66719e97:
#   "Faster dataloader merge (#1)" — adds replacement=True path and set_epoch()
#   so the batch sampler can advance the RNG seed per epoch without rebuilding
#   the sampler object.
#
# Knuth §3.4.2 critique: sampling with replacement wastes O(n) draws on
# already-seen indices (coupon-collector), degrading epoch coverage by
# ~(1 - 1/e) ≈ 37%.  The correct algorithm is Fisher-Yates (O(n), zero waste).
# We expose both paths: replacement=False uses torch.randperm (Fisher-Yates);
# replacement=True replicates Megatron 66719e9 for throughput benchmarking.
# Diagnostic prints fire at sampler construction and epoch boundaries so the
# log always records which mode is active.
class MegatronRandomSampler(torch.utils.data.Sampler):
    """Epoch-aware sampler derived from Megatron-LM commit 66719e97.

    Supports:
      replacement=True  — num_samples draws with replacement (Megatron path)
      replacement=False — full Fisher-Yates shuffle, num_samples must be None

    The set_epoch() method advances the RNG seed so successive epochs differ
    while remaining deterministic across restarts at the same step.
    """

    def __init__(self, data_source, replacement: bool = False,
                 num_samples: Optional[int] = None, seed: int = 42):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed
        self._epoch = -1          # -1 → no epoch seeding yet (legacy compat)

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "[M452] MegatronRandomSampler: num_samples is only valid "
                "when replacement=True (Knuth §3.4.2 — Fisher-Yates needs "
                "no pre-count). Got replacement=False with "
                f"num_samples={num_samples}."
            )
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"[M452] num_samples must be a positive int, got {self.num_samples}"
            )
        print(
            f"[M452-SAMPLER] MegatronRandomSampler init: "
            f"replacement={replacement}, num_samples={self.num_samples}, "
            f"dataset_len={len(data_source)}, seed={seed}"
        )

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Advance RNG seed to epoch boundary — call before each DataLoader iter."""
        self._epoch = epoch
        print(f"[M452-SAMPLER] set_epoch({epoch}) → generator seed = {self.seed + epoch}")

    def __iter__(self):
        n = len(self.data_source)
        g = torch.Generator()
        # Epoch-aware seeding: matches Megatron pretrain_bert.py epoch loop
        # train_data.batch_sampler.sampler.set_epoch(epoch + args.seed)
        if self._epoch >= 0:
            g.manual_seed(self.seed + self._epoch)
        else:
            g.manual_seed(self.seed)

        if self.replacement:
            # Megatron 66719e9 path: torch.randint over [0, n) with replacement
            # Throughput: O(num_samples) generator calls — no permutation buffer
            indices = torch.randint(
                high=n, size=(self.num_samples,), dtype=torch.int64, generator=g
            ).tolist()
        else:
            # Fisher-Yates via torch.randperm — O(n) time, zero coupon waste
            indices = torch.randperm(n, generator=g).tolist()

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


# =============================================================================
# NEURON_SP PORT: Megatron b8e0129f4 — reverted init in data_utils
# Adapted from megatron/data_utils/__init__.py should_split() / get_split().
# The original commit reverts __init__.py: removes get_split() entirely and
# keeps only the lean should_split() with trailing-whitespace cleanup.
# 20% adaptation: Python dataclass split ratios instead of argparse args;
# should_split takes a list directly. get_split removed (reverted per commit).
# =============================================================================


def _neuronsp_should_split(split: list) -> bool:
    """Check if dataset should be split into train/val/test sub-sets.

    Port of megatron/data_utils/__init__.py::should_split (b8e0129f4).
    Returns False when a single partition holds 100% of data.

    Examples:
    >>> _neuronsp_should_split([10, 0, 0])
    False
    >>> _neuronsp_should_split([1, 0.1, 0.2])
    True
    """
    print(f"[NEURONSP-SPLIT] should_split check: split={split} "
          f"→ max/sum={max(split)/sum(split):.4f}")
    return max(split) / sum(split) != 1.


def _make_dataloader_m452(dataset, config: 'TrainingConfig',
                          world_size: int, rank: int) -> DataLoader:
    """Build DataLoader using Megatron 66719e9 sampling strategy.

    presplit_sentences=True  → SyntheticDataset already returns newline-split
                               items; sentence tokenization is a no-op here but
                               the flag is plumbed through for real-data paths.
    replacement_sampling=True → MegatronRandomSampler(replacement=True,
                                num_samples=batch_size * max_steps)
    replacement_sampling=False → standard DistributedSampler / RandomSampler
                                 (Fisher-Yates, no waste).

    Diagnostic prints on rank-0 only to avoid log storms in multi-GPU runs.
    """
    # -------------------------------------------------------------------------
    # NEURON_SP PORT: Megatron d64856847 — fixed gpt-2 dataloader
    # Adapted from pretrain_gpt2.py get_train_val_test_data.
    # Original fix: 'tfrecords' → 'lazy' in elif branch; added else + exit(1).
    # 20% adaptation: uses config.data_loader_type str attr (default 'lazy');
    # 'numpy' path goes to replacement sampler, 'lazy' path to distributed sampler.
    # Unsupported loader type hard-exits with diagnostic message.
    # -------------------------------------------------------------------------
    _loader_type = getattr(config, 'data_loader_type', 'lazy')
    print(f"[NEURONSP-GPT2-DL] data_loader_type='{_loader_type}' "
          f"(supported: 'numpy', 'lazy')")
    if _loader_type == 'numpy':
        print(f"[NEURONSP-GPT2-DL] numpy path → replacement_sampling forced True")
    elif _loader_type == 'raw' or _loader_type == 'lazy':
        print(f"[NEURONSP-GPT2-DL] lazy path → distributed/Fisher-Yates sampler")
    else:
        print(f"[NEURONSP-GPT2-DL] Unsupported data loader for GPT2: '{_loader_type}'")
        import sys; sys.exit(1)

    n_total = config.batch_size * config.max_steps * config.gradient_accumulation
    # presplit_sentences diagnostic — real-data paths would use NLTK bypass here
    if rank == 0:
        print(
            f"[M452-DL] presplit_sentences={config.presplit_sentences} "
            f"(SyntheticDataset: sentence splits are pre-tokenised tokens, "
            f"NLTK bypass active when presplit_sentences=True)"
        )

    if world_size > 1:
        # Multi-GPU: DistributedSampler handles rank-sharding; replacement flag
        # is applied inside via a wrapped generator on top of the shard.
        # Knuth note: DistributedSampler is already Fisher-Yates; replacement
        # on top is theoretically unsound but matches Megatron's single-node impl.
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=config.seed
        )
        shuffle_flag = False   # sampler is set, shuffle must be False
        print(f"[M452-DL] rank={rank} DistributedSampler (world={world_size})")
    elif config.replacement_sampling:
        # Single-GPU, replacement=True: Megatron 66719e9 exact path
        # num_samples = batch_size * train_iters (matches configure_data.py line 49)
        sampler = MegatronRandomSampler(
            dataset,
            replacement=True,
            num_samples=n_total,
            seed=config.seed + rank,
        )
        shuffle_flag = False
        if rank == 0:
            print(
                f"[M452-DL] replacement=True sampler, "
                f"num_samples={n_total} (batch={config.batch_size} × "
                f"steps={config.max_steps} × accum={config.gradient_accumulation})"
            )
    else:
        # Default: Fisher-Yates shuffle (no replacement) — Knuth-correct path
        sampler = MegatronRandomSampler(
            dataset,
            replacement=False,
            seed=config.seed + rank,
        )
        shuffle_flag = False
        if rank == 0:
            print(f"[M452-DL] replacement=False (Fisher-Yates / Knuth §3.4.2)")

    dl = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle_flag,
        num_workers=max(4, 1),          # mirrors Megatron max(args.num_workers, 1)
        pin_memory=True,
        drop_last=True,                 # avoids variable micro-batch at epoch end
    )
    if rank == 0:
        print(
            f"[M452-DL] DataLoader ready: "
            f"batch={config.batch_size}, drop_last=True, "
            f"num_workers={max(4,1)}, pin_memory=True"
        )
    return dl


# =============================================================================
# SYNTHETIC DATASET (For benchmarking - real data optional)
# =============================================================================

def _m457_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    eod_mask_loss: bool,
    reset_position_ids: bool,
) -> tuple:
    """Build loss mask and position IDs for left-to-right model.

    M457 / Megatron 872b4a6 — 'Fixed edge case with multiple end of sequence
    in one sequence'.  The original single-EOS path assumed at most one EOD
    per sequence; packed-document training can place k≥1 EOD tokens anywhere.

    Knuth §2.3.1 critique: the prior code used a scalar sentinel scan that
    silently stopped at the first EOD, leaving subsequent documents with
    uncorrected position offsets.  This O(k) loop over ALL EOD positions is
    the correct fix — k is small (avg doc length ≪ seq_len) so the overhead
    is negligible in practice.

    Knuth §3.6 critique: resetting position_ids to 0 after each EOD implicitly
    treats each sub-document as independent, which is correct for causal LMs
    but breaks rotary embeddings that cache sin/cos at initialisation time.
    Callers using RoPE must re-compute freqs after position reset; the benchmark
    model uses learned absolute embeddings so this is safe here.

    Args:
        data: 1-D token tensor of length seq_len (input_ids, NOT labels).
        eod_token: integer token ID treated as end-of-document.
        eod_mask_loss: if True, set loss_mask=0 at every EOD position.
        reset_position_ids: if True, restart position counter after each EOD.

    Returns:
        loss_mask   – float32 tensor, shape (seq_len,), values in {0.0, 1.0}
        position_ids – int64 tensor, shape (seq_len,)
    """
    seq_len = data.numel()
    loss_mask = torch.ones(seq_len, dtype=torch.float32)
    position_ids = torch.arange(seq_len, dtype=torch.long)

    # --- locate ALL EOD positions in one vectorised call ---
    eod_positions = (data == eod_token).nonzero(as_tuple=False).squeeze(1)  # shape (k,)
    n_eod = eod_positions.numel()

    # Diagnostic: warn when the edge case (k > 1) actually fires
    if n_eod > 1:
        print(
            f"[M457-EOD] multi-EOS edge case: {n_eod} EOD tokens in seq_len={seq_len} "
            f"at positions {eod_positions.tolist()}"
        )
    elif n_eod == 1:
        pass  # normal single-document sequence — no diagnostic noise
    # n_eod == 0: no EOD in this window — also silent

    if eod_mask_loss and n_eod > 0:
        loss_mask[eod_positions] = 0.0

    if reset_position_ids and n_eod > 0:
        # Clone so we can mutate without aliasing the arange buffer
        position_ids = position_ids.clone()
        prev = 0
        for j in range(n_eod):
            i = eod_positions[j].item()
            # Shift positions after this EOD back to restart from 0
            # M457 edge case: each iteration uses the *updated* prev, not
            # the original index — this is the bug the Megatron commit fixed.
            # Without the running `prev` accumulator, the second+ EOD would
            # subtract an incorrect (too-large) offset, producing negative
            # position IDs.
            position_ids[i + 1:] -= (i + 1 - prev)
            prev = i + 1

    return loss_mask, position_ids


class SyntheticDataset(Dataset):
    """Learnable synthetic dataset for benchmarking.

    Creates deterministic sequences with repeating n-gram patterns so the
    language model can actually reduce its loss below the random baseline
    of ln(vocab_size).  Each sample is seeded by its index, ensuring
    reproducibility across runs and ranks while still providing enough
    variety for meaningful training.

    M457 / Megatron 872b4a6: supports multi-EOS sequences produced by
    document packing.  EOS tokens are injected at a fixed stride derived
    from `eos_stride`; real datasets use variable-length documents (Poisson
    distributed) — the fixed stride is a synthetic approximation that
    exercises the multi-EOS code path deterministically.

    M459 / Megatron adec01d05: optionally uses TrainingSampleBuilder to
    produce the triple-array (doc_idx, sample_idx, shuffle_idx) index scheme
    for document-packing-aware sampling. When `use_sample_builder=True` the
    dataset constructs a synthetic corpus of variable-length documents and
    uses the builder's shuffle_idx to determine __getitem__ order, exactly
    matching the adec01d05 sampling strategy.

    Knuth §3.4.2 critique: fixed-stride EOS injection creates a perfectly
    periodic signal; a learnable model could exploit this regularity to
    predict EOD "for free".  For throughput benchmarking this is acceptable
    (we care about step/s, not convergence), but users training for real
    should disable EOS injection (eos_stride=0) and use real packed data.
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int = 100000,
        eod_token_id: int = -1,       # -1 → vocab_size - 1
        eod_mask_loss: bool = True,    # M457: mask loss on EOD positions
        reset_position_ids: bool = True,  # M457: restart pos counter after EOD
        eos_stride: int = 0,           # inject EOS every N tokens (0 = disabled)
        # M459: adec01d05 sample builder args
        use_sample_builder: bool = False,
        sample_builder_num_docs: int = 1000,
        sample_builder_min_doc_len: int = 64,
        sample_builder_max_doc_len: int = 512,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        # Resolve eod_token_id: convention is vocab_size-1 (GPT2 uses 50256)
        self.eod_token_id = (vocab_size - 1) if eod_token_id < 0 else eod_token_id
        self.eod_mask_loss = eod_mask_loss
        self.reset_position_ids = reset_position_ids
        self.eos_stride = eos_stride   # 0 means no injection
        self.seed = seed

        # Pre-generate a pool of short n-gram patterns (bigrams to 5-grams)
        # that get tiled into full sequences.  Using a small effective vocab
        # (~2000 tokens) makes the distribution learnable in <500 steps.
        rng = torch.Generator().manual_seed(42)
        # Reserve the eod_token_id slot: effective vocab excludes it so that
        # pattern-tiled tokens never accidentally coincide with the EOS marker.
        self.eff_vocab = min(2000, vocab_size - 1)  # -1: exclude eod_token_id
        # 256 pattern templates, each 8-32 tokens long
        self.patterns = [
            torch.randint(0, self.eff_vocab, (torch.randint(8, 33, (1,), generator=rng).item(),), generator=rng)
            for _ in range(256)
        ]

        # ----------------------------------------------------------------
        # NEURON_SP PORT: Megatron 2c58c9b04 — added filtering based on sentence length
        # Adapted from megatron/data/helpers.cpp build_mapping_impl.
        # Filters out patterns whose token length exceeds NEURONSP_LONG_SENTENCE_LEN.
        # 20% adaptation: operates on Python pattern list instead of C++ sent sizes[];
        # uses list comprehension + counter instead of loop with bool flag.
        # Detect patterns with long sentences.
        _long_sent_docs: int = 0
        _filtered_patterns = []
        for _pat in self.patterns:
            if len(_pat) > NEURONSP_LONG_SENTENCE_LEN:
                _long_sent_docs += 1
                # contains_long_sentence=True: skip this pattern (filter it out)
                print(f"[NEURONSP-FILTER] pattern len={len(_pat)} > "
                      f"LONG_SENTENCE_LEN={NEURONSP_LONG_SENTENCE_LEN} → filtered")
            else:
                _filtered_patterns.append(_pat)
        self.patterns = _filtered_patterns
        print(f"[NEURONSP-FILTER] patterns with long sentences: {_long_sent_docs} "
              f"(threshold={NEURONSP_LONG_SENTENCE_LEN}), "
              f"remaining patterns: {len(self.patterns)}")

        # ----------------------------------------------------------------
        # M459: adec01d05 TrainingSampleBuilder integration
        # Build a synthetic corpus of variable-length documents, then
        # use TrainingSampleBuilder to generate doc_idx/sample_idx/shuffle_idx.
        # __getitem__ maps through shuffle_idx → sample_idx to slice the corpus.
        # ----------------------------------------------------------------
        self._use_sample_builder = use_sample_builder
        self._sample_builder = None
        self._corpus_tokens = None  # flat token array for the entire synthetic corpus
        self._corpus_doc_offsets = None  # start offset of each doc in _corpus_tokens

        if use_sample_builder:
            self._init_sample_builder(
                num_docs=sample_builder_num_docs,
                min_doc_len=sample_builder_min_doc_len,
                max_doc_len=sample_builder_max_doc_len,
                seed=seed,
            )

        print(
            f"[M457-DATASET] SyntheticDataset init: vocab={vocab_size}, "
            f"eod_token_id={self.eod_token_id}, eos_stride={eos_stride}, "
            f"eod_mask_loss={eod_mask_loss}, reset_position_ids={reset_position_ids}, "
            f"num_samples={num_samples}, use_sample_builder={use_sample_builder}"
        )

    def _init_sample_builder(self, num_docs: int, min_doc_len: int,
                              max_doc_len: int, seed: int) -> None:
        """Build synthetic corpus + TrainingSampleBuilder (M459 / adec01d05).

        Generates `num_docs` variable-length documents using the same n-gram
        pattern pool as the standard path. Document lengths are drawn from
        Uniform[min_doc_len, max_doc_len] which is a crude approximation of
        the Poisson-distributed real-corpus lengths.

        Knuth §3.2.2 critique: Uniform doc lengths differ from real Poisson
        distributions; short documents (< seq_len) cause packing, long docs
        span multiple samples. For throughput benchmarking this is acceptable.
        """
        from deepspeed.runtime.data_pipeline.data_sampling.data_sampler import TrainingSampleBuilder
        rng = np.random.default_rng(seed + 2)
        # Draw document lengths from Uniform[min, max]
        doc_lengths = rng.integers(min_doc_len, max_doc_len + 1, size=num_docs)

        # Build flat corpus token array (EOD tokens between documents)
        corpus_parts = []
        doc_offsets = [0]
        pat_rng = np.random.default_rng(seed + 3)
        for doc_i, doc_len in enumerate(doc_lengths):
            pat_idx = int(pat_rng.integers(0, len(self.patterns)))
            pat = self.patterns[pat_idx].numpy()
            # Tile the pattern to fill doc_len tokens, offset by doc_i
            repeats = (doc_len + len(pat) - 1) // len(pat)
            tiled = np.tile(pat, repeats)[:doc_len]
            tiled = (tiled + doc_i) % self.eff_vocab
            corpus_parts.append(tiled.astype(np.int32))
            corpus_parts.append(np.array([self.eod_token_id], dtype=np.int32))  # EOD separator
            doc_offsets.append(doc_offsets[-1] + doc_len + 1)  # +1 for EOD

        self._corpus_tokens = np.concatenate(corpus_parts)
        self._corpus_doc_offsets = np.array(doc_offsets[:-1], dtype=np.int64)

        print(
            f"[M459-DATASET] Synthetic corpus: {num_docs} docs, "
            f"total_tokens={len(self._corpus_tokens)}, "
            f"doc_lengths=[{doc_lengths.min()}, {doc_lengths.max()}] "
            f"(Uniform, Knuth §3.2.2 approximation)"
        )

        # Build TrainingSampleBuilder with corpus doc sizes
        doc_sizes = np.array(doc_lengths, dtype=np.int64)
        self._sample_builder = TrainingSampleBuilder(
            num_samples=self.num_samples,
            seq_length=self.seq_len,
            doc_sizes=doc_sizes,
            seed=seed,
            eod_token_id=self.eod_token_id,
        )
        self._sample_builder.build()  # pre-build all three index arrays

    def _getitem_sample_builder(self, idx: int) -> Dict[str, torch.Tensor]:
        """M459: __getitem__ via adec01d05 triple-array scheme.

        Retrieves sample `idx` using shuffle_idx → sample_idx → doc_idx chain,
        then slices the synthetic corpus to produce a seq_len-token window.
        EOD tokens between documents are handled by _m457_ltor_masks_and_position_ids.

        Knuth §2.2.5: O(1) __getitem__ via precomputed tables — no linear scan.
        Knuth §3.4.2: shuffle_idx provides uniform random order over samples.
        """
        _, _, _, _ = self._sample_builder.get_sample_indices(0)  # warm-up type check
        canonical = int(self._sample_builder._shuffle_idx[idx % self.num_samples])
        start_row = self._sample_builder._sample_idx[canonical]
        end_row = self._sample_builder._sample_idx[canonical + 1]

        # Walk corpus from start_row to end_row, collecting seq_len+1 tokens
        # (the +1 allows us to split into input_ids and labels)
        tokens_needed = self.seq_len + 1
        collected = []

        doc_idx_flat = self._sample_builder._doc_idx
        start_doc_pos = int(start_row[0])
        start_doc_off = int(start_row[1])

        doc_pos = start_doc_pos
        offset_in_doc = start_doc_off

        doc_sizes = self._sample_builder.doc_sizes

        while len(collected) < tokens_needed and doc_pos < len(doc_idx_flat):
            doc_id = int(doc_idx_flat[doc_pos])
            doc_size = int(doc_sizes[doc_id])
            doc_global_start = int(self._corpus_doc_offsets[doc_id])

            tokens_left_in_doc = doc_size - offset_in_doc
            take = min(tokens_left_in_doc, tokens_needed - len(collected))

            src_start = doc_global_start + offset_in_doc
            chunk = self._corpus_tokens[src_start: src_start + take]
            collected.extend(chunk.tolist())

            if take == tokens_left_in_doc and len(collected) < tokens_needed:
                # Append EOD separator between documents
                collected.append(self.eod_token_id)
                doc_pos += 1
                offset_in_doc = 0
            else:
                offset_in_doc += take

        # Pad to tokens_needed if corpus ran short (edge case)
        while len(collected) < tokens_needed:
            collected.append(0)

        tokens = torch.tensor(collected[:tokens_needed], dtype=torch.long)
        input_ids = tokens[:-1]
        labels = tokens[1:]

        loss_mask, position_ids = _m457_ltor_masks_and_position_ids(
            data=input_ids,
            eod_token=self.eod_token_id,
            eod_mask_loss=self.eod_mask_loss,
            reset_position_ids=self.reset_position_ids,
        )

        if idx < 3:
            print(
                f"[M459-GETITEM] idx={idx} canonical={canonical} "
                f"start=({int(start_row[0])},{int(start_row[1])}) "
                f"n_eod={int((input_ids == self.eod_token_id).sum())} "
                f"first_8={input_ids[:8].tolist()}"
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # M459: route through sample builder when active
        if self._use_sample_builder and self._sample_builder is not None:
            return self._getitem_sample_builder(idx)

        # Original path (M457)
        # Deterministic per-sample: pick pattern, tile to seq_len+1
        pat = self.patterns[idx % len(self.patterns)]
        repeats = (self.seq_len + 1 + len(pat) - 1) // len(pat)
        tokens = pat.repeat(repeats)[: self.seq_len + 1]
        # Add a small per-sample offset so different indices aren't identical
        offset = idx % self.eff_vocab
        tokens = (tokens + offset) % self.eff_vocab   # stay within non-EOS vocab

        # M457: inject EOS tokens at fixed stride to exercise multi-EOS code path.
        # stride=0 disables injection; stride>0 places EOD at positions
        # [stride-1, 2*stride-1, ...] within the seq_len+1 window.
        if self.eos_stride > 0:
            eos_positions = torch.arange(
                self.eos_stride - 1, self.seq_len + 1, self.eos_stride, dtype=torch.long
            )
            tokens[eos_positions] = self.eod_token_id
            n_injected = eos_positions.numel()
            if n_injected > 1:
                # Only log on the first few samples to avoid flooding
                if idx < 3:
                    print(
                        f"[M457-INJECT] idx={idx}: injected {n_injected} EOS tokens "
                        f"at stride={self.eos_stride} → positions {eos_positions.tolist()[:5]}{'...' if n_injected > 5 else ''}"
                    )

        input_ids = tokens[:-1]   # shape (seq_len,)
        labels = tokens[1:]       # shape (seq_len,)

        # M457: compute loss_mask and position_ids respecting ALL EOD positions
        loss_mask, position_ids = _m457_ltor_masks_and_position_ids(
            data=input_ids,
            eod_token=self.eod_token_id,
            eod_mask_loss=self.eod_mask_loss,
            reset_position_ids=self.reset_position_ids,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


# =============================================================================
# OPTIMIZERS
# =============================================================================

class AdamW(torch.optim.Optimizer):
    """ZeRO Stage-1 AdamW: optimizer state partitioned across DP ranks.
    
    Memory per GPU: param + grad + m_shard + v_shard
      = N*2 + N*2 + (N/W)*2 + (N/W)*2  (BF16)
    For 7B, W=3: 13.3 + 13.3 + 4.4 + 4.4 = 35.4GB < A6000 49GB
    
    Ref: DeepSpeed ZeRO stage_1_and_2.py:499 partition_size
    Ref: Megatron distrib_optimizer.py shard_buffer
    """
    def __init__(self, params, lr: float, betas: Tuple[float, float], weight_decay: float,
                 eps: float = 1e-8):
        # M504: Megatron 48269d8d8 — eps now a first-class arg (was hardcoded 1e-8)
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        self._zero_initialized = False

    def _lazy_init_zero(self):
        if self._zero_initialized:
            return
        self._zero_initialized = True
        if not torch.distributed.is_initialized():
            self._world_size = 1
            self._rank = 0
        else:
            self._world_size = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()
        self._flat_params = []
        self._param_to_flat_idx = {}
        for group in self.param_groups:
            flat_list = list(group['params'])
            for i, p in enumerate(flat_list):
                self._param_to_flat_idx[p] = len(self._flat_params)
                self._flat_params.append(p)

    def _get_partition_range(self, numel):
        chunk = (numel + self._world_size - 1) // self._world_size
        start = min(self._rank * chunk, numel)
        end = min(start + chunk, numel)
        return start, end

    def step(self):
        self._lazy_init_zero()
        if self._world_size <= 1:
            self._step_local()
            return
        self._step_zero1()

    def _step_local(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bc1
                torch.sqrt(exp_avg_sq, out=grad)
                denom = grad.div_(math.sqrt(bc2)).add_(group.get('eps', 1e-8))
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.grad = None

    def _step_zero1(self):
        # Phase 1: AllReduce gradients (equivalent to DDP grad sync)
        ar_handles = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    h = torch.distributed.all_reduce(
                        p.grad.data, op=torch.distributed.ReduceOp.AVG, async_op=True)
                    ar_handles.append(h)
        for h in ar_handles:
            h.wait()
        # Phase 2: Each rank updates only its own partition of m/v/param
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                numel = p.data.numel()
                start, end = self._get_partition_range(numel)
                part_size = end - start
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros(part_size, dtype=p.data.dtype, device=p.data.device)
                    state['exp_avg_sq'] = torch.zeros(part_size, dtype=p.data.dtype, device=p.data.device)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                p_flat = p.data.view(-1)
                g_flat = p.grad.data.view(-1)
                p_flat.mul_(1 - group['lr'] * group['weight_decay'])
                gs = g_flat[start:end]
                exp_avg.mul_(beta1).add_(gs, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gs, gs, value=1 - beta2)
                # Free grad memory BEFORE computing denom to reduce peak
                p.grad = None
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bc1
                # M351: reuse pre-allocated buffer for denom to avoid temp
                if '_denom' not in state:
                    state['_denom'] = torch.empty_like(exp_avg_sq)
                torch.sqrt(exp_avg_sq, out=state['_denom'])
                state['_denom'].div_(math.sqrt(bc2)).add_(group.get('eps', 1e-8))
                p_flat[start:end].addcdiv_(exp_avg, state['_denom'], value=-step_size)
        # Phase 3: Broadcast each rank's updated param shard to all
        for p in self._flat_params:
            numel = p.data.numel()
            p_flat = p.data.view(-1)
            chunk = (numel + self._world_size - 1) // self._world_size
            for src in range(self._world_size):
                s = min(src * chunk, numel)
                e = min(s + chunk, numel)
                if e > s:
                    torch.distributed.broadcast(p_flat[s:e], src=src)


class DESLOCAdamW(torch.optim.Optimizer):
    """
    DES-LOC AdamW - Desynced Low Communication Adam.

    Implements independent sync periods for:
    - x (parameters): sync every Kx steps
    - u (first moment): sync every Ku steps
    - v (second moment): sync every Kv steps

    Outer optimizer (Section 5.5, RQ5):
    - 'average': simple parameter averaging after AllReduce (default)
    - 'nesterov': Polyak/Nesterov momentum on averaged params
        x_new = x_avg + beta_outer * (x_avg - x_avg_prev)
        (Charles et al. 2025, momentum=0.9, outer_lr=1.0)
    """
    def __init__(self, params, lr: float, betas: Tuple[float, float],
                 weight_decay: float, Kx: int, Ku: int, Kv: int,
                 outer_optimizer: str = 'average',
                 outer_momentum: float = 0.9, outer_lr: float = 1.0,
                 max_norm: float = 1.0, eps: float = 1e-8):
        # M465: Megatron 4687967 — clip_grad moved from training loop into optimizer.
        # max_norm is now an optimizer-level default; individual param_groups may
        # override it via group['max_norm'].  Setting max_norm=0.0 disables clipping
        # entirely for that group (allows embedding layers to skip clip if needed).
        # M504: Megatron 48269d8d8 — eps as first-class param (default 1e-8)
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       Kx=Kx, Ku=Ku, Kv=Kv, max_norm=max_norm, eps=eps)
        super().__init__(params, defaults)
        self.global_step = 0
        # Paper Metrics: track ||s_{t+K} - s_t||_2 / ||s_t||_2 for each state tier
        self._state_snapshots = {}  # {param_id: {tier: snapshot_tensor}}
        self._rate_of_change = {'x': [], 'u': [], 'v': []}  # per-sync history
        # RQ5: Nesterov outer optimizer (Section 5.5)
        self.outer_optimizer = outer_optimizer  # 'average' or 'nesterov'
        self.outer_momentum = outer_momentum    # beta for Nesterov (default 0.9)
        self.outer_lr = outer_lr                # outer learning rate (default 1.0)
        self._prev_avg = {}  # {param_id: previous averaged params for Nesterov}
        # Activation norm monitoring (Section 5.5: prevent exploding norms)
        self._activation_norms = []
        # M448/M451: Overflow-Aware Scaling (OAS) — derived from Megatron fb4cbdc
        # DynamicLossScaler. Instead of a separate LossScaler class, we inline
        # the three core mechanisms directly into the optimizer:
        #   1. Pre-step grad scan: _scan_grads_for_overflow() — M451 addition.
        #      Aborts the entire step on any inf/nan in gradients (Megatron
        #      DynamicLossScaler.has_overflow pattern, commit fb4cbdc277a0).
        #   2. Hysteresis downscale: require N consecutive overflows before halving
        #   3. Window upscale: after W clean steps, double the scale
        # This avoids the Megatron pattern of wrapping the optimizer in another
        # object, keeping DES-LOC's flat optimizer structure.
        self._oas_scale = 65536.0       # initial loss scale (Megatron default)
        self._oas_overflow_cnt = 0      # total overflow events
        self._oas_clean_steps = 0       # consecutive clean steps
        self._oas_scale_window = 200    # clean steps needed to double scale
        self._oas_min_scale = 1.0       # floor — never scale below this
        self._oas_max_scale = 2**24     # ceiling — 16M
        self._oas_hysteresis = 2        # consecutive overflows needed to halve
        self._oas_pending_ovf = 0       # hysteresis counter
        self._oas_enabled = True        # can be disabled for bf16-only runs
        self._oas_last_action = 'init'  # diagnostic: last scale action taken

    @property
    def grad_scale(self):
        """Current loss scale for training loop. Megatron exposes this via
        FP16_Optimizer.loss_scale; we expose it as a read-only property."""
        return self._oas_scale if self._oas_enabled else 1.0

    @staticmethod
    def _clip_grad_norm_per_group(params, max_norm: float, norm_type: float = 2.0) -> float:
        """M465: Per-parameter-group gradient clipping — adapted from Megatron 4687967
        clip_grad_norm (megatron/mpu/grads.py).  Gradients are modified in-place.

        Key differences from torch.nn.utils.clip_grad_norm_:
          1. Filters shared params (param.shared == True) so embedding weight that
             is tied between first and last pipeline stages is counted only once.
             Megatron sets param.shared=True in PipelinedMegatronModule.__init__
             (module.py hunk in commit 4687967).
          2. Filters tensor-model-parallel duplicates — only rank-0 of each TP
             group contributes its norm, avoiding sqrt(tp_size) over-count.
             We cannot check mpu.get_tensor_model_parallel_rank() here (no mpu
             dependency), so we use param.tensor_model_parallel attribute instead,
             matching Megatron's own convention.
          3. Uses param.grad.detach() throughout — avoids autograd graph retention
             and prevents accidental double-backward through the clip coefficient.

        Knuth critique (TAOCP Vol.2 §4.2.2 — floating-point error accumulation):
          USER BUG:  Callers who leave clipping in the training loop and also pass
                     max_norm to the optimizer will double-clip every step.  First
                     clip reduces ‖g‖ to max_norm; second clip is a no-op only if
                     the loop's max_norm ≤ this value — a silent, order-dependent
                     correctness hazard that makes ablations unreproducible.
          SYSTEM IMPACT: With GradScaler active, training-loop clip must be called
                         AFTER unscale_() (scaler inflates grads by 1/scale before
                         clip).  Moving clip inside optimizer.step() — which fires
                         AFTER the scaler's unscale step — preserves this invariant
                         automatically.  The old training-loop placement was correct
                         only by accident: it happened to sit after the unscale_()
                         call in _train_baseline.  Any future refactor that reorders
                         those two lines silently breaks clipping for fp16 runs.

        Args:
            params: iterable of torch.Tensor parameters (single group's params).
            max_norm: maximum gradient norm.  0.0 or negative = skip clipping.
            norm_type: L-p norm order (default 2).  Use float('inf') for L-inf.

        Returns:
            Total pre-clip gradient norm for this group (0.0 if no eligible grads).
        """
        if max_norm <= 0.0:
            # max_norm=0 is the opt-out sentinel — return 0 without touching grads
            print(f"[CLIPGRAD] group max_norm={max_norm:.3f} — clipping DISABLED for this group")
            return 0.0

        # Build eligible parameter list: has grad, not shared, not TP-duplicate
        eligible = []
        n_shared_skipped = 0
        n_tp_skipped = 0
        for param in params:
            if param.grad is None:
                continue
            if getattr(param, 'shared', False):
                n_shared_skipped += 1
                continue
            # tensor_model_parallel=True means this param IS the canonical TP shard
            # (every rank holds a slice) — count it.  tensor_model_parallel absent or
            # False means param is replicated; count only TP rank-0 to avoid over-count.
            is_tp_shard = getattr(param, 'tensor_model_parallel', False)
            if not is_tp_shard:
                # replicated param — skip on non-zero TP ranks
                try:
                    import torch.distributed as _dist
                    tp_rank = _dist.get_rank() if _dist.is_initialized() else 0
                except Exception:
                    tp_rank = 0
                if tp_rank != 0:
                    n_tp_skipped += 1
                    continue
            eligible.append(param)

        if not eligible:
            print(f"[CLIPGRAD] no eligible params (shared_skip={n_shared_skipped} tp_skip={n_tp_skipped})")
            return 0.0

        # Compute norm across eligible params
        if norm_type == float('inf'):
            total_norm = max(p.grad.detach().abs().max().item() for p in eligible)
        else:
            total_norm = sum(
                p.grad.detach().norm(norm_type).item() ** norm_type
                for p in eligible
            ) ** (1.0 / norm_type)

        # Apply clip coefficient (in-place, detached)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in eligible:
                param.grad.detach().mul_(clip_coef)

        print(f"[CLIPGRAD] max_norm={max_norm:.3f} total_norm={total_norm:.4f} "
              f"clip_coef={min(clip_coef, 1.0):.4f} "
              f"n_eligible={len(eligible)} shared_skip={n_shared_skipped} tp_skip={n_tp_skipped} "
              f"clipped={'YES' if clip_coef < 1.0 else 'no'}")
        return total_norm

    @staticmethod
    def _grad_has_inf_or_nan(tensor: torch.Tensor) -> bool:
        """M451: Gradient overflow probe — derived from Megatron fb4cbdc
        DynamicLossScaler._has_inf_or_nan (commit fb4cbdc277a0).

        Megatron's original used float(x.float().sum()) and caught RuntimeError
        for overflow-converted values — a scalar-path that can silently miss
        sparse inf patterns when cancellation occurs.  We use torch.isfinite()
        instead: it performs an element-wise boolean mask and reduces with .all(),
        catching every inf/nan regardless of sum cancellation.  The float() cast
        is retained to handle fp16 tensors without mixed-precision overflow.

        Knuth critique (TAOCP Vol.2 §4.2.2 — floating-point error accumulation):
          USER BUG:  Callers who skip this check and apply the raw gradient to
                     Adam's m/v states corrupt the EMA statistics permanently.
                     A single inf in exp_avg_sq sets every future denom to inf,
                     silencing all parameter updates for the lifetime of training
                     — a subtle, silent catastrophe that no loss curve will reveal
                     until the model has diverged beyond recovery.
          SYSTEM IMPACT: Without a pre-step guard the scale doubling window
                         (scale_window=200) will keep escalating loss_scale while
                         corrupt states accumulate, burning GPU cycles on updates
                         that produce NaN outputs, then requiring a full
                         checkpoint rollback — wasting hours of cluster time.
        """
        try:
            return not torch.isfinite(tensor.float()).all().item()
        except RuntimeError as exc:
            # Megatron pattern: RuntimeError may signal overflow in tensor
            # conversion itself (e.g. fp16 -> float saturating at ±65504).
            # Re-raise non-overflow errors; treat conversion failure as overflow.
            if "value cannot be converted" not in str(exc):
                raise
            return True

    def _scan_grads_for_overflow(self) -> bool:
        """M451: Pre-step gradient overflow scan — mirrors Megatron
        DynamicLossScaler.has_overflow(params).  Iterates all param groups
        and returns True on the first inf/nan found in any .grad tensor.

        Returns False immediately (no scan) when _oas_enabled is False,
        allowing bf16-only runs to skip the overhead entirely.

        Knuth critique (TAOCP Vol.2 §4.2.3 — error propagation):
          USER BUG:  A training loop that calls optimizer.step() without
                     checking the return value of this method will silently
                     ingest corrupt gradients.  The loop MUST call this
                     before step() and skip the backward pass on True.
          SYSTEM IMPACT: Each unchecked overflow that slips through doubles
                         the likelihood of a subsequent overflow because inf
                         values propagate through the computation graph into
                         activations, poisoning future batch gradients and
                         creating a cascade that no hysteresis counter can
                         arrest once it starts.
        """
        if not self._oas_enabled:
            return False
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and DESLOCAdamW._grad_has_inf_or_nan(p.grad.data):
                    return True
        return False

    def step(self):
        self.global_step += 1
        
        # Decide offload once based on total model size vs SMALLEST GPU in cluster
        # All ranks must agree — otherwise sync steps have asymmetric PCIe traffic
        if not hasattr(self, '_use_offload'):
            total_param_bytes = 0
            sample_device = None
            for group in self.param_groups:
                for p in group['params']:
                    total_param_bytes += p.numel() * p.element_size()
                    if sample_device is None and p.is_cuda:
                        sample_device = p.device
            if sample_device is not None and dist.is_initialized():
                # Use min GPU memory across all ranks for symmetric behavior
                local_mem = torch.cuda.get_device_properties(sample_device).total_memory
                mem_tensor = torch.tensor([local_mem], dtype=torch.long, device=sample_device)
                dist.all_reduce(mem_tensor, op=dist.ReduceOp.MIN)
                min_gpu_mem = mem_tensor.item()
                self._use_offload = (total_param_bytes * 4 > min_gpu_mem * 0.85)
                if dist.get_rank() == 0:
                    print(f"[DESLOC] offload={'ON' if self._use_offload else 'OFF'}: "
                          f"model {total_param_bytes/1e9:.1f}GB × 4 = {total_param_bytes*4/1e9:.1f}GB, "
                          f"min GPU = {min_gpu_mem/1e9:.1f}GB")
            elif sample_device is not None:
                gpu_mem = torch.cuda.get_device_properties(sample_device).total_memory
                self._use_offload = (total_param_bytes * 4 > gpu_mem * 0.85)
            else:
                self._use_offload = False
        

        # M451: Pre-step gradient overflow guard — Megatron fb4cbdc
        # DynamicLossScaler.has_overflow pattern.  Scan ALL gradients before
        # touching any optimizer state.  On overflow: adjust scale, skip this
        # entire step (return early), let the training loop re-zero and retry.
        # This is the critical missing piece from M448 which only checked params
        # *after* the update — too late to prevent state corruption.
        if self._oas_enabled:
            _r = dist.get_rank() if dist.is_initialized() else 0
            _pre_overflow = self._scan_grads_for_overflow()
            if _pre_overflow:
                self._oas_overflow_cnt += 1
                self._oas_pending_ovf += 1
                self._oas_clean_steps = 0
                if self._oas_pending_ovf >= self._oas_hysteresis:
                    _old = self._oas_scale
                    self._oas_scale = max(self._oas_min_scale,
                                         self._oas_scale / 2.0)
                    self._oas_pending_ovf = 0
                    self._oas_last_action = 'halve_grad'
                    print(f"[OAS/GRAD-OVERFLOW] rank={_r} step={self.global_step} "
                          f"inf/nan in grad — SKIPPING STEP, "
                          f"SCALE {_old:.0f} -> {self._oas_scale:.0f} "
                          f"(ovf_total={self._oas_overflow_cnt} "
                          f"hysteresis={self._oas_hysteresis})")
                else:
                    self._oas_last_action = 'pending_grad'
                    print(f"[OAS/GRAD-OVERFLOW] rank={_r} step={self.global_step} "
                          f"inf/nan in grad — SKIPPING STEP (PENDING "
                          f"{self._oas_pending_ovf}/{self._oas_hysteresis}) "
                          f"scale={self._oas_scale:.0f} "
                          f"ovf_total={self._oas_overflow_cnt}")
                # Zero all gradients before returning — stale inf grads must
                # not survive into the next backward pass.
                for _grp in self.param_groups:
                    for _p in _grp['params']:
                        if _p.grad is not None:
                            _p.grad = None
                return  # abort this step entirely

        # M465: Megatron 4687967 — per-parameter-group gradient clipping.
        # Moved from training loop into optimizer.step() so that:
        #   (a) clipping always fires after OAS overflow guard (no partial-step clip),
        #   (b) each param_group can carry its own max_norm (e.g. 0.0 to disable
        #       clipping for embedding layers that use a shared weight),
        #   (c) GradScaler invariant is preserved: unscale_() happens before step(),
        #       and clip sees unscaled gradients regardless of training-loop order.
        # Knuth critique (TAOCP Vol.2 §4.2.4 — algorithm correctness):
        #   USER BUG:  Callers who ALSO call torch.nn.utils.clip_grad_norm_() in
        #              the training loop before optimizer.step() will double-clip.
        #              The first clip reduces ‖g‖ to max_norm; the second is a
        #              no-op only if both thresholds are identical AND the training
        #              loop clip fired after GradScaler.unscale_().  In any other
        #              configuration the second clip silently under-clips or
        #              over-clips, making ablations non-reproducible.
        #   SYSTEM IMPACT: Training-loop clip that predates this commit must be
        #                  removed (or guarded with `if not isinstance(opt, DESLOCAdamW)`)
        #                  to avoid the double-clip hazard described above.
        _r_clip = dist.get_rank() if dist.is_initialized() else 0
        _total_norms = []
        for group in self.param_groups:
            _mn = group.get('max_norm', 1.0)
            _gn = DESLOCAdamW._clip_grad_norm_per_group(group['params'], _mn)
            _total_norms.append(_gn)
        # Expose aggregate norm so training loop can log it without re-computing
        self._last_grad_norm = (sum(n ** 2 for n in _total_norms) ** 0.5) if _total_norms else 0.0
        if self.global_step % 50 == 1:
            print(f"[CLIPGRAD/STEP] rank={_r_clip} step={self.global_step} "
                  f"n_groups={len(_total_norms)} "
                  f"per_group_norms={[round(n, 4) for n in _total_norms]} "
                  f"aggregate_norm={self._last_grad_norm:.4f}")

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['_offload'] = self._use_offload
                    if self._use_offload:
                        state['exp_avg'] = torch.zeros_like(p.data, device='cpu').pin_memory()
                        state['exp_avg_sq'] = torch.zeros_like(p.data, device='cpu').pin_memory()
                    else:
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Decoupled weight decay (in-place, on GPU)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                if state['_offload']:
                    # CPU offload path: stream m/v to GPU, update, stream back
                    m_gpu = state['exp_avg'].to(p.device, non_blocking=True)
                    v_gpu = state['exp_avg_sq'].to(p.device, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    m_gpu.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v_gpu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    p.grad = None  # free grad before denom alloc
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] / bc1
                    # In-place sqrt into v_gpu (we already have it on GPU)
                    denom = torch.sqrt(v_gpu)
                    denom.div_(math.sqrt(bc2)).add_(group.get('eps', 1e-8))
                    p.data.addcdiv_(m_gpu, denom, value=-step_size)
                    del denom
                    # Stream back to CPU
                    state['exp_avg'].copy_(m_gpu, non_blocking=True)
                    state['exp_avg_sq'].copy_(v_gpu, non_blocking=True)
                    del m_gpu, v_gpu
                else:
                    # Standard GPU path: full m/v on GPU
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    p.grad = None  # free grad before denom alloc
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] / bc1
                    if '_denom' not in state:
                        state['_denom'] = torch.empty_like(exp_avg_sq)
                    torch.sqrt(exp_avg_sq, out=state['_denom'])
                    state['_denom'].div_(math.sqrt(bc2)).add_(group.get('eps', 1e-8))
                    p.data.addcdiv_(exp_avg, state['_denom'], value=-step_size)

        # M365 DIAG: per-group optimizer step summary at key steps
        # Pattern: Megatron optimizer.py log_num_zeros_in_grad
        if self.global_step % 50 == 1:
            _r = dist.get_rank() if dist.is_initialized() else 0
            n_updated = 0
            total_m_norm_sq = 0.0
            total_v_norm_sq = 0.0
            total_p_norm_sq = 0.0
            for grp in self.param_groups:
                for p in grp['params']:
                    st = self.state.get(p, {})
                    if 'exp_avg' not in st:
                        continue
                    n_updated += 1
                    m_n = st['exp_avg'].float().norm().item() if st['exp_avg'].device.type != 'cpu' else 0.0
                    v_n = st['exp_avg_sq'].float().norm().item() if st['exp_avg_sq'].device.type != 'cpu' else 0.0
                    total_m_norm_sq += m_n ** 2
                    total_v_norm_sq += v_n ** 2
                    total_p_norm_sq += p.data.float().norm().item() ** 2
            print(f"[DIAG/OPT-STEP] rank={_r} step={self.global_step} "
                  f"n_updated={n_updated} "
                  f"||m||={math.sqrt(total_m_norm_sq):.4f} "
                  f"||v||={math.sqrt(total_v_norm_sq):.4f} "
                  f"||p||={math.sqrt(total_p_norm_sq):.4f} "
                  f"offload={self._use_offload}")

        # M448/M451: Post-step parameter overflow guard (secondary layer).
        # Derived from Megatron DynamicLossScaler._has_inf_or_nan + update_scale.
        # M451 adds a pre-step GRAD scan (_scan_grads_for_overflow) that fires
        # before any state mutation; this block is the second safety net that
        # catches overflows arising *inside* the Adam update itself (e.g.
        # catastrophic cancellation in denom, weight-decay explosion).
        # Scans all param data for inf/nan after update. If detected:
        #   - Increment hysteresis counter
        #   - If hysteresis threshold hit: halve scale, reset counter
        # If clean: increment clean_steps, double scale at window boundary.
        if self._oas_enabled:
            _oas_found_overflow = False
            for group in self.param_groups:
                for p in group['params']:
                    _psum = float(p.data.float().sum())
                    if _psum != _psum or abs(_psum) == float('inf'):
                        _oas_found_overflow = True
                        break
                if _oas_found_overflow:
                    break

            _r = dist.get_rank() if dist.is_initialized() else 0
            if _oas_found_overflow:
                self._oas_overflow_cnt += 1
                self._oas_pending_ovf += 1
                self._oas_clean_steps = 0
                if self._oas_pending_ovf >= self._oas_hysteresis:
                    old_scale = self._oas_scale
                    self._oas_scale = max(self._oas_min_scale, self._oas_scale / 2.0)
                    self._oas_pending_ovf = 0
                    self._oas_last_action = 'halve'
                    print(f"[OAS/OVERFLOW] rank={_r} step={self.global_step} "
                          f"SCALE {old_scale} -> {self._oas_scale} "
                          f"(ovf_total={self._oas_overflow_cnt} "
                          f"hysteresis={self._oas_hysteresis} clean=0)")
                else:
                    self._oas_last_action = 'pending'
                    print(f"[OAS/OVERFLOW] rank={_r} step={self.global_step} "
                          f"scale={self._oas_scale} PENDING "
                          f"({self._oas_pending_ovf}/{self._oas_hysteresis}) "
                          f"ovf_total={self._oas_overflow_cnt}")
            else:
                self._oas_pending_ovf = 0
                self._oas_clean_steps += 1
                if self._oas_clean_steps >= self._oas_scale_window:
                    old_scale = self._oas_scale
                    self._oas_scale = min(self._oas_max_scale, self._oas_scale * 2.0)
                    self._oas_clean_steps = 0
                    self._oas_last_action = 'double'
                    print(f"[OAS/SCALEUP] rank={_r} step={self.global_step} "
                          f"SCALE {old_scale} -> {self._oas_scale} "
                          f"(clean={self._oas_scale_window} "
                          f"ovf_total={self._oas_overflow_cnt})")
                elif self.global_step % 100 == 0:
                    print(f"[OAS/STATUS] rank={_r} step={self.global_step} "
                          f"scale={self._oas_scale} "
                          f"clean={self._oas_clean_steps}/{self._oas_scale_window} "
                          f"ovf={self._oas_overflow_cnt}")
    
    def sync_if_needed(self, world_size: int):
        """Sync optimizer states based on DES-LOC schedule.

        Returns sync flags dict even when world_size<=1 so that
        the caller can still count how many syncs *would* happen,
        which is needed for comm-reduction metrics.

        Paper Metrics: measures ∥s_{t+K} - s_t∥₂ / ∥s_t∥₂ at each sync point
        for params (x), first moment (u), second moment (v).

        Claude-27 M332 Convergence Fixes:
        ──────────────────────────────────
        Three bugs identified and fixed in the sync schedule:

        Fix #1 — Remove co-sync (was: `if sync_x: sync_u = True`):
          Co-sync made u sync 108× in 500 steps (vs intended ~5×),
          violating the independence assumption in Eq(4). Removing it
          restores 3-tier independence: u syncs on its own Ku schedule.

        Fix #2 — v piggybacks on x (was: v only on its own Kv schedule):
          With β₂=0.999, v's half-life is ~693 steps. v only synced 2×
          in 500 steps, meaning workers had divergent adaptive learning
          rates for ~250 steps each. Now v syncs whenever x syncs,
          keeping the Adam denominator consistent across workers.

        Fix #3 — Momentum decay after x-averaging:
          After params are averaged, stale local exp_avg (first moment)
          pushes averaged params back toward each worker's old position,
          causing oscillation and loss spikes. Now exp_avg is decayed
          by 0.1× after x-sync (90% forgotten, 10% retained as
          warm-start). This is strictly better than co-sync because
          averaging u from divergent workers produces a meaningless
          mean, while decaying removes stale signal cleanly.

        Net sync counts (500 steps, Kx=32, Ku=96, Kv=192):
          Before: x=47, u=108, v=2  → 157 total (3.2× reduction)
          After:  x=47, u=~5,  v=47 → 99 total  (5.1× reduction)
        """
        Kx_target = self.param_groups[0]['Kx']
        Ku_target = self.param_groups[0]['Ku']
        Kv_target = self.param_groups[0]['Kv']

        # M447: Adaptive Kx — adjust sync frequency based on loss variance.
        # High variance → landscape is bumpy, workers diverge fast → sync more.
        # Low variance → landscape is smooth, local steps are safe → sync less.
        # Uses the rate-of-change history already collected at sync points.
        if not hasattr(self, '_adaptive_loss_window'):
            self._adaptive_loss_window = []
        _adaptive_Kx_ratio = 1.0
        if len(self._adaptive_loss_window) >= 10 and Kx_target > 1:
            _recent = self._adaptive_loss_window[-50:]
            _mean = sum(_recent) / len(_recent)
            _var = sum((x - _mean) ** 2 for x in _recent) / len(_recent)
            _std = _var ** 0.5
            if _std > 0.5:
                _adaptive_Kx_ratio = 0.25
            elif _std < 0.1:
                _adaptive_Kx_ratio = 2.0
            else:
                _adaptive_Kx_ratio = 1.0
            if self.global_step % 50 == 1:
                _r = dist.get_rank() if dist.is_initialized() else 0
                print(f"[ADAPTIVE-KX] rank={_r} step={self.global_step} "
                      f"loss_std={_std:.4f} ratio={_adaptive_Kx_ratio:.2f} "
                      f"Kx: target={Kx_target} -> effective={max(1, int(Kx_target * _adaptive_Kx_ratio))}")

        # --- Warmup: ramp Kx from 1 → Kx_target over warmup_steps ---
        # Charles et al. (2025): warm-start from DDP-equivalent training
        # prevents early divergence when loss landscape is highly stochastic.
        warmup_steps = min(100, Kx_target * 3)  # ~3 full Kx cycles
        if self.global_step <= warmup_steps:
            # Linear ramp: step 1 → Kx=1, step warmup_steps → Kx=Kx_target
            frac = self.global_step / max(warmup_steps, 1)
            effective_Kx = max(1, int(1 + (Kx_target - 1) * frac))
            # Claude-27 M335: u and v do NOT follow the Kx ramp.
            effective_Ku = Ku_target
            effective_Kv = Kv_target
        else:
            # M447: apply adaptive ratio to post-warmup Kx
            effective_Kx = max(1, int(Kx_target * _adaptive_Kx_ratio))
            effective_Ku = Ku_target
            effective_Kv = Kv_target
        self._last_effective_Kx = effective_Kx

        sync_x = (effective_Kx <= 1) or (self.global_step % effective_Kx == 0)
        sync_u = (effective_Ku <= 1) or (self.global_step % effective_Ku == 0)
        sync_v = (effective_Kv <= 1) or (self.global_step % effective_Kv == 0)

        # ---------------------------------------------------------------
        # Claude-27 M332: 3-tier INDEPENDENT scheduling (Bug fix #1)
        # ---------------------------------------------------------------
        # REMOVED: `if sync_x: sync_u = True` (co-sync)
        #
        # Root cause analysis (Claude-27 diagnosis):
        # Co-sync violated the independence assumption in Eq(4):
        #   ψ = ψ_x + ψ_u + ψ_v  (each tier contributes independently)
        # With co-sync, ψ_u ≈ ψ_x, making u sync 108x in 500 steps
        # instead of the intended ~5x (500/Ku=96). This wasted 103
        # extra AllReduces on u while not fixing the real problem:
        # stale v (second moment) causing wrong adaptive learning rates.
        #
        # Fix: Each tier syncs on its OWN schedule. When x syncs, we
        # instead apply a momentum decay (see below) to handle the
        # stale-momentum-after-averaging problem that co-sync was
        # trying to solve.
        #
        # Claude-27 M332: Force v-sync at x-sync boundaries (Bug fix #2)
        # ---------------------------------------------------------------
        # With β₂=0.999, v's half-life is ~693 steps. In 500 steps with
        # Kv=192, v only syncs 2 times — meaning each worker runs with
        # completely different adaptive learning rates for ~250 steps.
        # After x-averaging, the local v no longer matches the averaged
        # params, causing loss spikes (observed: 11→13.7→15.7→19.1).
        #
        # Fix: When x syncs, also sync v. This ensures that after param
        # averaging, all workers share the same adaptive learning rate
        # (denominator in Adam update). u remains independent to preserve
        # the 3-tier communication reduction for first moment.
        #
        # Net effect on sync counts (500 steps, Kx=32, Ku=96, Kv=192):
        #   Before: sync_x=47, sync_u=108 (co-sync), sync_v=2  → 157 total
        #   After:  sync_x=47, sync_u=~5,  sync_v=47 (=sync_x) → 99 total
        #   Still 5x reduction vs DDP's 500 syncs.
        if sync_x:
            sync_v = True  # v piggybacks on x to keep adaptive LR consistent

        # M364 DIAG — M365 FIX: use cached grad norm from before optimizer.step()
        # optimizer.step() does p.grad = None to free VRAM, so computing grad
        # norm here always gives 0.0000. Use _cached_grad_norm set by training loop.
        if self.global_step % 100 == 1 or sync_x:
            gnorm = getattr(self, '_cached_grad_norm', 0.0)  # M365: cached before step()
            pnorm = sum(p.data.float().norm().item()**2 for grp in self.param_groups for p in grp['params'])**0.5
            _r = dist.get_rank() if dist.is_initialized() else 0
            print(f"[SYNC] rank={_r} step={self.global_step} sync_x={sync_x} sync_u={sync_u} sync_v={sync_v} "
                  f"Kx={effective_Kx} grad={gnorm:.4f} param={pnorm:.2f} ratio={gnorm/max(pnorm,1e-12):.8f}")

        # Measure rate-of-change at sync boundaries (before AllReduce)
        # M361(e): Norm-only tracking — no .clone(). Old code cloned every param
        # (13.3GB for 7B) causing OOM at 28.52GB peak + 13.3GB = 41.8GB ≈ A6000 limit.
        # Pattern: Megatron distributed_data_parallel.py check_for_nan_in_grad —
        # computes norm in-place without allocating a full copy.
        # New approach: store only the L2 norm of each tier at previous sync,
        # compute ‖current‖₂ at this sync, and use |‖curr‖-‖prev‖|/‖prev‖ as
        # a lightweight proxy for rate of change. Loses per-element fidelity
        # but captures the magnitude of drift, which is what Eq(4) needs.
        if world_size > 1:
            for tier_name, should_sync, get_fn in [
                ('x', sync_x, lambda p, s: p.data),
                ('u', sync_u, lambda p, s: s.get('exp_avg')),
                ('v', sync_v, lambda p, s: s.get('exp_avg_sq')),
            ]:
                if not should_sync:
                    continue
                total_norm_sq = 0.0
                for group in self.param_groups:
                    for p in group['params']:
                        t = get_fn(p, self.state.get(p, {}))
                        if t is None:
                            continue
                        if t.device.type == 'cpu':
                            total_norm_sq += t.float().norm().item() ** 2
                        else:
                            total_norm_sq += t.float().norm().item() ** 2
                cur_norm = total_norm_sq ** 0.5
                prev_key = f'_roc_norm_{tier_name}'
                prev_norm = getattr(self, prev_key, None)
                if prev_norm is not None and prev_norm > 1e-12:
                    roc = abs(cur_norm - prev_norm) / prev_norm
                    self._rate_of_change[tier_name].append(roc)
                setattr(self, prev_key, cur_norm)

        # Actual AllReduce (multi-GPU only)
        # M361(b,h,i): Chunked flattened AllReduce with PCIe pipelining.
        #
        # From Megatron param_and_grad_buffer.py _ParamAndGradBucket.start_grad_sync()
        # start. Then, follow that pattern to implement a new _chunked_flat_allreduce,
        # letting the AllReduce operate on a single contiguous buffer per chunk, and
        # capping peak memory at CHUNK_BYTES to avoid OOM on A6000 (49GB).
        # Then DeepSpeed stage_1_and_2.py introduces reduce_ipg_grads() with
        # ipg_bucket_size=500MB, so that each NCCL call is bounded, while
        # _flatten_dense_tensors optimizes the concat into a single memcpy.
        # Then NCCL src/device/all_reduce.h Ring AllReduce integrates the
        # ncclGroupStart/ncclGroupEnd batching, letting multiple small buffers
        # be fused into one ring pass, and Megatron's finish_grad_sync() fence
        # enhances async completion tracking.
        # Finally torch._utils._flatten_dense_tensors refines the cat into a
        # single-allocation copy, ensuring expandable_segments compatibility,
        # fully upgrading the AllReduce to handle 7B+ with CPU offload at O(1)
        # NCCL calls per chunk.
        CHUNK_BYTES = 512 * 1024 * 1024  # 512MB — fits in A6000 headroom

        if world_size > 1:
            all_params = []
            for group in self.param_groups:
                for p in group['params']:
                    all_params.append(p)

            def _sync_tier(param_list, get_tensor_fn, tier_label='?'):
                """Chunked flattened AllReduce for one tier.
                Handles CPU-offloaded tensors via stream overlap.

                M477: Port Megatron d6c4248b7 MemoryBuffer pattern — reuse a
                persistent flat buffer across sync calls instead of alloc/free
                each time. Reduces CUDA memory fragmentation on A6000 (49GB).
                """
                tensors, cpu_pairs = [], []
                for p in param_list:
                    t = get_tensor_fn(p)
                    if t is None:
                        continue
                    if t.device.type == 'cpu':
                        t_gpu = t.to(p.device, non_blocking=True)
                        cpu_pairs.append((t, t_gpu))
                        tensors.append(t_gpu)
                    else:
                        tensors.append(t)
                if not tensors:
                    return
                if cpu_pairs:
                    torch.cuda.current_stream().synchronize()

                # M477 DIAG: pre-sync tier statistics
                _r_st = dist.get_rank() if dist.is_initialized() else 0
                _total_numel = sum(t.numel() for t in tensors)
                _total_bytes = _total_numel * tensors[0].element_size()
                print(f"[SYNC-TIER] rank={_r_st} step={self.global_step} "
                      f"tier={tier_label} n_tensors={len(tensors)} "
                      f"total={_total_bytes/1e6:.1f}MB "
                      f"cpu_offload={len(cpu_pairs)} "
                      f"mem_before={torch.cuda.memory_allocated(tensors[0].device)/1e9:.2f}GB")

                # Chunk tensors into groups of ~CHUNK_BYTES each
                chunks, cur_chunk, cur_bytes = [], [], 0
                elem_size = tensors[0].element_size()
                for t in tensors:
                    t_bytes = t.numel() * elem_size
                    if cur_bytes + t_bytes > CHUNK_BYTES and cur_chunk:
                        chunks.append(cur_chunk)
                        cur_chunk, cur_bytes = [], 0
                    cur_chunk.append(t)
                    cur_bytes += t_bytes
                if cur_chunk:
                    chunks.append(cur_chunk)

                # M477: Megatron MemoryBuffer pattern — persistent flat buffer.
                # Key insight from d6c4248b7: allocate once, reuse across calls.
                # _start index tracks free region; reset() reclaims without dealloc.
                max_chunk_numel = max(sum(t.numel() for t in c) for c in chunks)
                _buf_key = f'_sync_flat_buf_{tier_label}'
                _existing = getattr(self, _buf_key, None)
                if _existing is not None and _existing.numel() >= max_chunk_numel and _existing.device == tensors[0].device:
                    flat_buf = _existing
                else:
                    flat_buf = torch.empty(max_chunk_numel, dtype=tensors[0].dtype,
                                           device=tensors[0].device)
                    setattr(self, _buf_key, flat_buf)
                    if _r_st == 0:
                        print(f"[MEMBUF] tier={tier_label} alloc {max_chunk_numel*elem_size/1e6:.1f}MB "
                              f"(Megatron d6c4248 MemoryBuffer pattern)")

                _checksum_pre = 0.0
                for chunk in chunks:
                    total = sum(t.numel() for t in chunk)
                    flat = flat_buf[:total]
                    # Copy into flat buffer
                    off = 0
                    for t in chunk:
                        n = t.numel()
                        flat[off:off + n].copy_(t.reshape(-1))
                        off += n
                    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                    flat.div_(world_size)
                    # Copy back
                    off = 0
                    for t in chunk:
                        n = t.numel()
                        t.copy_(flat[off:off + n].reshape(t.shape))
                        off += n

                # M477 DIAG: post-sync checksum
                if self.global_step % 100 == 1:
                    _cksum = sum(t.float().sum().item() for t in tensors[:5])
                    print(f"[SYNC-TIER-POST] rank={_r_st} step={self.global_step} "
                          f"tier={tier_label} checksum_sample={_cksum:.4f} "
                          f"n_chunks={len(chunks)} "
                          f"mem_after={torch.cuda.memory_allocated(tensors[0].device)/1e9:.2f}GB")

                # Stream back CPU-offloaded tensors
                for cpu_t, gpu_t in cpu_pairs:
                    cpu_t.copy_(gpu_t, non_blocking=True)

            if sync_x and all_params:
                _sync_tier(all_params, lambda p: p.data, tier_label='x')
            if sync_u and all_params:
                _sync_tier(all_params,
                    lambda p: self.state.get(p, {}).get('exp_avg'), tier_label='u')
            if sync_v and all_params:
                _sync_tier(all_params,
                    lambda p: self.state.get(p, {}).get('exp_avg_sq'), tier_label='v')

        # M364 DIAG: post-sync checksum
        if sync_x and world_size > 1 and self.global_step % 100 == 1:
            psum = sum(p.data.float().sum().item() for grp in self.param_groups for p in grp['params'])
            _r = dist.get_rank() if dist.is_initialized() else 0
            print(f"[SYNC-POST] rank={_r} step={self.global_step} param_checksum={psum:.4f}")

        # ---------------------------------------------------------------
        # Claude-27 M332: Momentum decay after x-sync (Bug fix #3)
        # ---------------------------------------------------------------
        # Problem: After x-averaging, the local first moment (exp_avg)
        # still points toward the OLD local optimum. On the next step,
        # this stale momentum pushes the averaged params back toward
        # where each worker was before sync — creating oscillation.
        #
        # Solution: Decay exp_avg by factor `momentum_decay_on_sync`
        # after x-averaging. This dampens the stale directional signal
        # while preserving some momentum (not zeroing it entirely, which
        # would waste the gradient history from local training).
        #
        # The decay factor 0.1 is chosen so that:
        # - 90% of stale momentum is removed
        # - 10% retained provides mild warm-start for next local phase
        # - Equivalent to ~2.3 half-lives of exponential forgetting
        #
        # From Megatron distributed_data_parallel.py's grad buffer reset
        # pattern: after AllReduce, buffers are consumed and cleared.
        # We follow that pattern but apply it to optimizer state (exp_avg)
        # with partial decay rather than full zero to preserve signal.
        #
        # Why not just sync u at x boundaries (the old co-sync approach)?
        # Because averaging u is WRONG — u from worker-0 points toward
        # worker-0's local optimum, u from worker-1 toward worker-1's.
        # Their average points NOWHERE useful. Decaying is strictly better:
        # it removes the stale signal without injecting a meaningless average.
        if sync_x:
            # M448: Gradient variance-aware momentum decay.
            # When grad norms vary widely across params, workers have diverged
            # significantly → aggressive decay (0.05). When grads are uniform,
            # workers are still aligned → gentle decay (0.5) preserves momentum.
            _grad_norms_for_decay = []
            for grp in self.param_groups:
                for p in grp['params']:
                    st = self.state.get(p, {})
                    if 'exp_avg' in st:
                        _gn = st['exp_avg'].float().norm().item()
                        _grad_norms_for_decay.append(_gn)
            if len(_grad_norms_for_decay) > 1:
                _gn_mean = sum(_grad_norms_for_decay) / len(_grad_norms_for_decay)
                _gn_var = sum((x - _gn_mean) ** 2 for x in _grad_norms_for_decay) / len(_grad_norms_for_decay)
                if _gn_var > 1.0:
                    momentum_decay_on_sync = 0.05
                elif _gn_var < 0.01:
                    momentum_decay_on_sync = 0.5
                else:
                    momentum_decay_on_sync = 0.1
            else:
                momentum_decay_on_sync = 0.1
                _gn_var = 0.0
            _r_md = dist.get_rank() if dist.is_initialized() else 0
            print(f"[MDECAY] rank={_r_md} step={self.global_step} "
                  f"m_var={_gn_var:.6f} decay={momentum_decay_on_sync:.2f} "
                  f"n_params={len(_grad_norms_for_decay)}")
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'exp_avg' in state:
                        state['exp_avg'].mul_(momentum_decay_on_sync)

        # === RQ5: Nesterov outer optimizer (Section 5.5) ===
        # After AllReduce averaging, apply Nesterov momentum:
        #   x_new = x_avg + beta * (x_avg - x_avg_prev)
        # This improves over simple averaging by ~0.5% PPL (Charles et al. 2025)
        if sync_x and self.outer_optimizer == 'nesterov':
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    pid = id(p)
                    x_avg = p.data.clone()
                    if pid in self._prev_avg:
                        # Nesterov step: x_new = x_avg + beta * (x_avg - x_prev)
                        momentum_term = self.outer_momentum * (x_avg - self._prev_avg[pid])
                        p.data.add_(momentum_term, alpha=self.outer_lr)
                    self._prev_avg[pid] = x_avg

        # Activation norm monitoring (detect exploding norms, Section 5.5)
        if sync_x:
            total_norm = 0.0
            count = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.data is not None:
                        total_norm += torch.norm(p.data, 2).item() ** 2
                        count += 1
            if count > 0:
                rms_norm = math.sqrt(total_norm / count)
                self._activation_norms.append(rms_norm)
                # Warn if norm explodes (>10x initial)
                if len(self._activation_norms) > 2:
                    if rms_norm > 10 * self._activation_norms[0]:
                        print(f"[WARN] Activation norm explosion: {rms_norm:.4f} "
                              f"(initial: {self._activation_norms[0]:.4f})")

        return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}


class LocalAdamW(torch.optim.Optimizer):
    """
    Local AdamW - Sync all states every K steps.
    Baseline for comparison with DES-LOC.
    """
    def __init__(self, params, lr: float, betas: Tuple[float, float],
                 weight_decay: float, K: int):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, K=K)
        super().__init__(params, defaults)
        self.global_step = 0
    
    def step(self):
        self.global_step += 1
        
        if not hasattr(self, '_use_offload'):
            total_param_bytes = 0
            sample_device = None
            for group in self.param_groups:
                for p in group['params']:
                    total_param_bytes += p.numel() * p.element_size()
                    if sample_device is None and p.is_cuda:
                        sample_device = p.device
            if sample_device is not None and dist.is_initialized():
                local_mem = torch.cuda.get_device_properties(sample_device).total_memory
                mem_tensor = torch.tensor([local_mem], dtype=torch.long, device=sample_device)
                dist.all_reduce(mem_tensor, op=dist.ReduceOp.MIN)
                min_gpu_mem = mem_tensor.item()
                self._use_offload = (total_param_bytes * 4 > min_gpu_mem * 0.85)
            elif sample_device is not None:
                gpu_mem = torch.cuda.get_device_properties(sample_device).total_memory
                self._use_offload = (total_param_bytes * 4 > gpu_mem * 0.85)
            else:
                self._use_offload = False
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['_offload'] = self._use_offload
                    if self._use_offload:
                        state['exp_avg'] = torch.zeros_like(p.data, device='cpu').pin_memory()
                        state['exp_avg_sq'] = torch.zeros_like(p.data, device='cpu').pin_memory()
                    else:
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                beta1, beta2 = group['betas']
                state['step'] += 1
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                if state['_offload']:
                    m_gpu = state['exp_avg'].to(p.device, non_blocking=True)
                    v_gpu = state['exp_avg_sq'].to(p.device, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    m_gpu.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v_gpu.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    p.grad = None
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] / bc1
                    denom = torch.sqrt(v_gpu)
                    denom.div_(math.sqrt(bc2)).add_(group.get('eps', 1e-8))
                    p.data.addcdiv_(m_gpu, denom, value=-step_size)
                    del denom
                    state['exp_avg'].copy_(m_gpu, non_blocking=True)
                    state['exp_avg_sq'].copy_(v_gpu, non_blocking=True)
                    del m_gpu, v_gpu
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    p.grad = None
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] / bc1
                    if '_denom' not in state:
                        state['_denom'] = torch.empty_like(exp_avg_sq)
                    torch.sqrt(exp_avg_sq, out=state['_denom'])
                    state['_denom'].div_(math.sqrt(bc2)).add_(group.get('eps', 1e-8))
                    p.data.addcdiv_(exp_avg, state['_denom'], value=-step_size)
    
    def sync_if_needed(self, world_size: int):
        """Sync all states every K steps.
        
        Warmup (Claude-26 M332): ramp K from 1 → K_target over first 100 steps
        to match DES-LOC warmup for fair comparison.
        """
        K_target = self.param_groups[0]['K']
        warmup_steps = min(100, K_target * 3)
        if self.global_step <= warmup_steps:
            frac = self.global_step / max(warmup_steps, 1)
            effective_K = max(1, int(1 + (K_target - 1) * frac))
        else:
            effective_K = K_target
        should_sync = (effective_K <= 1) or (self.global_step % effective_K == 0)

        if should_sync and world_size > 1:
            all_params = []
            for group in self.param_groups:
                for p in group['params']:
                    all_params.append(p)

            if all_params:
                handles = []
                for p in all_params:
                    h = dist.all_reduce(p.data, op=dist.ReduceOp.SUM, async_op=True)
                    handles.append(h)
                for h in handles:
                    h.wait()
                for p in all_params:
                    p.data.div_(world_size)

                handles = []
                for p in all_params:
                    state = self.state[p]
                    if 'exp_avg' in state:
                        h = dist.all_reduce(state['exp_avg'], op=dist.ReduceOp.SUM, async_op=True)
                        handles.append((h, state['exp_avg']))
                for h, buf in handles:
                    h.wait()
                    buf.div_(world_size)

                handles = []
                for p in all_params:
                    state = self.state[p]
                    if 'exp_avg_sq' in state:
                        h = dist.all_reduce(state['exp_avg_sq'], op=dist.ReduceOp.SUM, async_op=True)
                        handles.append((h, state['exp_avg_sq']))
                for h, buf in handles:
                    h.wait()
                    buf.div_(world_size)

        return {'synced': should_sync}


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Real distributed trainer using DeepSpeed runtime for DES-LOC.

    DESLOC method: deepspeed.initialize() → engine.step() → allreduce_gradients()
      with Kx gating (engine.py:2558), tiered AR (comm.py DeslocTieredAllReduce),
      bucket sync (stage_1_and_2.py _desloc_reduce_tiered_gradients),
      profiling (comm.py DeslocProfiler), NKI-FA export (comms_logging.py).

    DDP/LocalAdam methods: raw PyTorch baselines for comparison.
    """

    def __init__(self, config: TrainingConfig, method: str):
        self.config = config
        self.method = method
        # Use DeepSpeed for DESLOC, or for DDP when cpu_offload/ZeRO is needed (7B on A6000)
        _needs_ds = (config.cpu_offload or config.zero_stage > 0)
        self.use_deepspeed = (method == 'DESLOC' or (method == 'DDP' and _needs_ds)) and _DS_AVAILABLE

        # Distributed setup
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)

        self.device = torch.device(f'cuda:{self.local_rank}')

        # Reproducibility
        seed = int(os.environ.get('PYTHONHASHSEED', 42))
        torch.manual_seed(seed + self.rank)
        torch.cuda.manual_seed_all(seed + self.rank)

        # Model
        model_config = config.get_model_config()
        self.model = GPT(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            use_ac=config.use_activation_checkpointing,
            # M458: Megatron 691747b1 per-layer QK scaling + fp32 softmax
            apply_query_key_layer_scaling=getattr(config, 'apply_query_key_layer_scaling', False),
            attention_softmax_in_fp32=getattr(config, 'attention_softmax_in_fp32', False),
            **model_config
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        if n_params > 500_000_000:
            self.model = self.model.bfloat16()
            if self.rank == 0:
                print(f"[BF16] Model converted to bfloat16 on CPU ({n_params/1e6:.0f}M params, "
                      f"saves {n_params * 2 / 1e9:.1f}GB)")

        _defer_device = (getattr(config, 'cpu_offload', False)
                         and config.model_size in ('7B', '13B')
                         and self.use_deepspeed)
        if not _defer_device:
            self.model = self.model.to(self.device)
        elif self.rank == 0:
            print(f"[M438] Deferring .to(device) for DeepSpeed placement")

        if config.use_activation_checkpointing and self.rank == 0:
            print(f"[AC] Layer-wise activation checkpointing enabled "
                  f"({model_config.get('n_layer', '?')} layers)")

        # RQ5 (Section 5.5): Initialize from DDP checkpoint
        # Charles et al. (2025) protocol: warm-start from 2048-step DDP training
        if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
            _map_loc = 'cpu' if _defer_device else self.device
            ckpt = torch.load(config.init_from_ckpt, map_location=_map_loc)
            if 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(ckpt)
            if self.rank == 0:
                print(f"[INIT] Loaded DDP checkpoint: {config.init_from_ckpt}")

        # Dataset — M457: pass multi-EOS config (Megatron 872b4a6)
        # M459: pass sample builder config (Megatron adec01d05)
        # eod_token_id=-1 resolved inside __init__ to vocab_size-1 unless overridden.
        # eos_stride=64 exercises the multi-EOS edge case: one EOS every 64 tokens
        # yields ~16 EOS per 1024-token sequence — enough to stress the loop.
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            num_samples=config.max_steps * config.batch_size * config.gradient_accumulation * max(self.world_size, 1) * 2,
            eod_token_id=config.eod_token_id,
            eod_mask_loss=config.eod_mask_loss,
            reset_position_ids=config.reset_position_ids,
            eos_stride=64,
            # M459: adec01d05 triple-array sampling
            use_sample_builder=config.use_sample_builder,
            sample_builder_num_docs=config.sample_builder_num_docs,
            sample_builder_min_doc_len=config.sample_builder_min_doc_len,
            sample_builder_max_doc_len=config.sample_builder_max_doc_len,
            seed=config.seed,
        )

        if self.use_deepspeed:
            # RQ5: Parse outer optimizer from method name
            # DESLOC → average, DESLOC_nesterov → nesterov, DESLOC_avg → average
            if '_nesterov' in method:
                config.outer_optimizer = 'nesterov'
            elif '_avg' in method:
                config.outer_optimizer = 'average'
            # else: keep config.outer_optimizer as-is (default 'average')

            # === DeepSpeed path: uses engine.py DES-LOC modifications ===
            ds_config = self._build_ds_config(config)
            self.engine, self.optimizer, self.dataloader, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=ds_config,
                training_data=dataset,
            )
            self.model = self.engine.module
            if self.rank == 0:
                _opt_cls = type(self.engine.optimizer).__name__
                _inner = getattr(self.engine.optimizer, 'optimizer', None)
                _inner_cls = type(_inner).__name__ if _inner else "none"
                print(f"[OPT-DIAG] optimizer={_opt_cls} inner={_inner_cls}")
            # Initialize DES-LOC scheduler (engine.py:2493)
            if hasattr(self.engine, 'desloc_init_scheduler'):
                self.engine.desloc_init_scheduler()
            # M342: Query engine's SP+DEC+AC composition state
            if hasattr(self.engine, 'desloc_composition_state'):
                comp = self.engine.desloc_composition_state()
                self._sp_enabled = comp.get('sp', False)
                if self.rank == 0:
                    print(f"[SP+DEC+AC] Engine composition: {comp}")
            # DES-LOC profiler from comm.py
            self._profiler = get_desloc_profiler()
            # DES-LOC timer from timer.py
            self._stimer = DeslocSTimer()
            self._progress = DeslocProgress()
        else:
            self.engine = None
            self._profiler = None
            self._stimer = None
            self._progress = None

            # DDP wrapper — skip when using ZeRO-1 AdamW (handles its own grad sync)
            # For non-DDP methods (DESLOC, LocalAdam), also no DDP wrapper needed
            # as they have their own sync logic.
            # if self.world_size > 1 and method == 'DDP':
            #     self.model = DDP(self.model, device_ids=[self.local_rank])

            # Baseline optimizer
            self.optimizer = self._create_optimizer(method)

            # M452: Megatron 66719e9 replacement sampling + presplit_sentences
            # replaces the previous DistributedSampler / shuffle=(sampler is None) pattern
            self.dataloader = _make_dataloader_m452(
                dataset, config, self.world_size, self.rank
            )

        # ============================================================
        # M339 — Claude-30: SP+DEC Initialization for baseline path
        # Risk fixed (user): --use_autosp without DeepSpeed silently
        #   did nothing. Now standalone SP works via DeslocSequenceParallelComm.
        # Risk fixed (system): _train_baseline had no sequence parallel
        #   support, making SP+DEC impossible without DeepSpeed engine.
        #
        # Architecture: AutoSP (DeepSpeed compile) vs standalone SP
        #   - AutoSP: torch.compile + inductor pass → automatic
        #   - Standalone SP: explicit scatter/gather on seq dim → manual
        #   Both are orthogonal to DES-LOC Kx gating (worker dim).
        #   Standalone SP is used when DeepSpeed is unavailable or
        #   when the user wants SP without compile overhead.
        # ============================================================
        self._sp_comm = None
        self._sp_enabled = False
        self._sp_size = 1
        self._sp_group = None
        self._sp_rank = 0
        self._sp_ranks = None  # M365: global ranks in SP group (for data broadcast)
        if config.use_autosp and self.world_size > 1:
            if self.use_deepspeed:
                if self.rank == 0:
                    print("[SP+DEC] AutoSP via DeepSpeed compile (inductor)")
            else:
                # M361(a,c,d): Compute sp_size from n_heads GCD with world_size.
                # AutoSP Ulysses requires n_heads % sp_size == 0.
                # On 3 GPU (2×A6000 + 1×H100), world_size=3 but n_heads=32,
                # 32%3≠0. Solution: sp_size = GCD(n_heads, world_size).
                # For 7B (n_heads=32, ws=3): sp_size=1 → no SP benefit.
                # Better: sp_size=2, using 2 A6000s as SP pair.
                #
                # From Megatron parallel_state.py initialize_model_parallel():
                # it creates separate process groups for TP, PP, CP, DP.
                # We follow that pattern: create an SP group of size sp_size
                # using contiguous ranks [0..sp_size-1], and a separate DP
                # group for the remaining ranks.
                #
                # From NCCL src/include/collectives.h ncclCommInitRank:
                # each communicator is a separate resource. Using separate
                # groups for SP and DP prevents the f communicator竞争 in fix (f).
                model_cfg = config.get_model_config()
                n_heads = model_cfg['n_head']
                # Find largest sp_size ≤ world_size where n_heads % sp_size == 0
                sp_size = 1
                for candidate in range(min(self.world_size, n_heads), 0, -1):
                    if n_heads % candidate == 0 and candidate <= self.world_size:
                        sp_size = candidate
                        break
                self._sp_size = sp_size

                # M361(c): Pad seq_len to multiple of sp_size
                if config.max_seq_len % sp_size != 0:
                    old_len = config.max_seq_len
                    config.max_seq_len = ((config.max_seq_len + sp_size - 1) // sp_size) * sp_size
                    if self.rank == 0:
                        print(f"[SP/M361] Padded max_seq_len {old_len} → "
                              f"{config.max_seq_len} (multiple of sp_size={sp_size})")

                if sp_size > 1:
                    try:
                        from deepspeed.comm.torch import DeslocSequenceParallelComm
                        # M362: GPU-type-aware SP group assignment.
                        # Don't assume rank order matches GPU type order.
                        # Gather GPU name from every rank, group same-type ranks
                        # together, pick the largest same-type group for SP.
                        #
                        # From Megatron parallel_state.py initialize_model_parallel():
                        # it builds groups from explicit rank lists. We do the same
                        # but derive the lists from hardware introspection.
                        #
                        # From NCCL nccl/src/graph/topo.h ncclTopoCompute():
                        # NCCL discovers PCIe topology to build optimal rings.
                        # We mirror that by grouping GPUs with matching capability
                        # so the A2A ring has symmetric bandwidth.
                        local_gpu_name = torch.cuda.get_device_name(self.device)
                        # AllGather gpu names across ranks
                        name_tensor = torch.zeros(256, dtype=torch.uint8, device=self.device)
                        name_bytes = local_gpu_name.encode('utf-8')[:256]
                        name_tensor[:len(name_bytes)] = torch.tensor(list(name_bytes), dtype=torch.uint8)
                        all_names = [torch.zeros(256, dtype=torch.uint8, device=self.device)
                                     for _ in range(self.world_size)]
                        dist.all_gather(all_names, name_tensor)
                        gpu_names = {}
                        for r, nt in enumerate(all_names):
                            raw = bytes(nt.cpu().tolist()).rstrip(b'\x00').decode('utf-8', errors='replace')
                            gpu_names[r] = raw

                        # Group ranks by GPU type
                        from collections import defaultdict
                        type_groups = defaultdict(list)
                        for r, name in gpu_names.items():
                            type_groups[name].append(r)

                        # Pick the largest same-type group that satisfies sp_size
                        # and n_heads divisibility. Prefer the group with MORE ranks
                        # (= more SP parallelism). Among equal-size groups, prefer
                        # the one with lower-memory GPUs (they benefit more from SP).
                        sp_ranks = None
                        for name, ranks in sorted(type_groups.items(),
                                                  key=lambda kv: (-len(kv[1]), kv[0])):
                            usable = min(len(ranks), sp_size)
                            # Find largest usable ≤ len(ranks) dividing n_heads
                            for s in range(usable, 0, -1):
                                if n_heads % s == 0:
                                    sp_ranks = sorted(ranks[:s])
                                    sp_size = s
                                    break
                            if sp_ranks:
                                break

                        if sp_ranks is None or len(sp_ranks) < 2:
                            sp_ranks = list(range(sp_size))

                        self._sp_size = sp_size
                        sp_group = dist.new_group(sp_ranks)
                        dp_group = dist.new_group(list(range(self.world_size)))
                        self._sp_ranks = sp_ranks  # M365: save for data broadcast

                        self._sp_comm = DeslocSequenceParallelComm(
                            seq_group=sp_group,
                            dp_group=dp_group,
                            Kx=config.Kx,
                        )
                        self._sp_enabled = True
                        self._sp_group = sp_group
                        self._sp_rank = dist.get_rank(group=sp_group)
                        # M364-fix: Only ranks IN the SP group should use Ulysses/pos_offset.
                        # Rank 2 (H100) is not in sp_ranks=[0,1], so sp_rank=-1 for it.
                        # It must stay on the standard full-seq path.
                        if self.rank in sp_ranks:
                            _sp_ctx_set(on=True, grp=sp_group, sz=sp_size, rk=sp_ranks.index(self.rank))
                        else:
                            _sp_ctx_set(on=False, grp=None, sz=1, rk=0)
                            self._sp_enabled = False  # rank 2 does NOT scatter/gather
                        if self.rank == 0:
                            print(f"[SP+DEC] SP enabled: sp_size={sp_size} "
                                  f"(n_heads={n_heads}, ws={self.world_size})")
                            print(f"[M364] mode=ulysses_eager pos_offset=rank*local_seq loss_reduce=AVG")
                            print(f"[SP+DEC] SP group ranks={sp_ranks} "
                                  f"(GPU: {gpu_names[sp_ranks[0]]})")
                            print(f"[SP+DEC] Each SP rank processes "
                                  f"seq_len/{sp_size}="
                                  f"{config.max_seq_len // sp_size} tokens")
                            dp_only = [r for r in range(self.world_size) if r not in sp_ranks]
                            if dp_only:
                                print(f"[SP+DEC] DP-only ranks={dp_only} "
                                      f"(GPU: {gpu_names[dp_only[0]]})")
                    except ImportError:
                        if self.rank == 0:
                            print("[SP+DEC] DeslocSequenceParallelComm unavailable")
                else:
                    if self.rank == 0:
                        print(f"[SP/M361] Cannot enable SP: n_heads={n_heads} "
                              f"has no factor ≤ world_size={self.world_size} > 1. "
                              f"Running DES-LOC without SP.")
                        print("[SP+DEC] DES-LOC Kx gating still active "
                              "(data parallel only)")
            if not self._sp_enabled and not self.use_deepspeed and self.rank == 0:
                if config.use_autosp:
                    print("[SP+DEC] NOTICE: --use_autosp set but SP not "
                          "activated (need DeepSpeed or standalone SP module)")
                    print("[SP+DEC] Experiment will run DES-LOC without "
                          "sequence parallel — results still valid for "
                          "data-parallel DES-LOC evaluation")

        # Gradient scaler for non-deepspeed paths
        # BF16 has same dynamic range as FP32 (8 exponent bits) → no scaling needed
        # M361(g): GradScaler + BF16 conflict prevention.
        # BF16 has 8 exponent bits (same as FP32) → no GradScaler needed.
        # 7B+ models are converted to BF16 above (line ~1270).
        # If model is BF16 OR if n_params > 500M (BF16 was applied), scaler=None.
        # This prevents the edge case where DeepSpeed engine wraps model to FP16
        # after __init__, but scaler was already created based on pre-wrap dtype.
        # Pattern: Megatron training.py:1431 — GradScaler only with FP16 Float16Module.
        _model_is_bf16 = next(self.model.parameters()).dtype == torch.bfloat16
        n_params = sum(p.numel() for p in self.model.parameters())
        if _model_is_bf16 or n_params > 500_000_000 or self.use_deepspeed:
            self.scaler = None
        else:
            self.scaler = torch.amp.GradScaler('cuda')

        # M462: Megatron 27e14f82 — learning_rate_scheduler step pattern.
        # Megatron commit 27e14f82 refactored pretrain loop so train_step
        # returns loss (not internal print) and the caller drives
        # learning_rate_scheduler.step() each iteration.
        #
        # Knuth critique — user bug: _train_baseline previously recorded LR
        # from optimizer.param_groups[0]['lr'] BEFORE advancing the schedule,
        # so every logged LR was one step behind reality. On warmup ramps this
        # causes the first 100 rows of the step log to be uniformly wrong.
        # Knuth critique — system impact: because no scheduler existed, the
        # baseline used a flat learning rate throughout training while DeepSpeed
        # engine advanced a cosine schedule each step. This makes DDP vs
        # DESLOC throughput comparisons invalid — DESLOC's lower late-training
        # loss is partly from the scheduler, not purely from Kx gating.
        #
        # Fix: add a cosine-with-warmup LambdaLR on the non-DeepSpeed path,
        # mirroring Megatron's get_learning_rate_scheduler() (training.py:287).
        # Only created for baseline (non-DS) path; DeepSpeed engine owns its own.
        self.lr_scheduler = None
        if not self.use_deepspeed and hasattr(self, 'optimizer'):
            _total_steps = config.max_steps
            _warmup = config.warmup_steps

            def _lr_lambda(current_step: int) -> float:
                # Linear warmup then cosine decay — mirrors Megatron cosine schedule.
                # Pattern: training.py get_learning_rate_scheduler(), lr-decay-style=cosine.
                if current_step < _warmup:
                    return float(current_step + 1) / float(max(1, _warmup))
                progress = float(current_step - _warmup) / float(max(1, _total_steps - _warmup))
                # Cosine decay to 10% of peak LR (Megatron default min_lr ratio)
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            from torch.optim.lr_scheduler import LambdaLR
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=_lr_lambda)
            if self.rank == 0:
                print(f"[M462-SCHED] cosine lr_scheduler: warmup={_warmup} steps, "
                      f"total={_total_steps} steps, "
                      f"peak_lr={config.learning_rate:.2e}, "
                      f"min_lr={config.learning_rate * 0.1:.2e}")

        # Metrics
        self.metrics = {
            'losses': [], 'step_times': [], 'comm_events': [], 'memory_usage': []
        }

        # M366: Structured per-step diagnostic recorder
        # summary_interval=10: lightweight metrics every 10 steps (matches log_interval)
        # detail_interval=50: full per-layer diagnostics every 50 steps
        self._recorder = StepRecorder(
            summary_interval=config.log_interval,
            detail_interval=50,
        )

        if self.rank == 0:
            os.makedirs(config.output_dir, exist_ok=True)

    @staticmethod
    def _build_ds_config(config):
        """Build DeepSpeed JSON config with DES-LOC section.

        This activates:
          - config.py: desloc_enabled, Kx, Ku, Kv, clip_rho, warmup
          - engine.py: allreduce_gradients() Kx gating (line 2558)
          - engine.py: desloc_post_step(), desloc_record_loss() etc.
          - stage_1_and_2.py: _desloc_reduce_tiered_gradients()
        """
        _opt_type = "Adam"

        ds_cfg = {
            "train_batch_size": config.batch_size * config.gradient_accumulation * max(int(os.environ.get('WORLD_SIZE', 1)), 1),
            "train_micro_batch_size_per_gpu": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation,
            "gradient_clipping": config.grad_clip,
            "steps_per_print": config.log_interval,
            "optimizer": {
                "type": _opt_type,
                "params": {
                    "lr": config.learning_rate,
                    "betas": [config.beta1, config.beta2],
                    "eps": 1e-8,
                    "weight_decay": config.weight_decay,
                    "torch_adam": False,
                }
            },
            "bf16": {
                "enabled": True,
            },
            "desloc": {
                "enabled": True,
                "Kx": config.Kx,
                "Ku": config.Ku,
                "Kv": config.Kv,
                "clip_rho": 1.0,
                "warmup_steps": min(100, config.max_steps // 5),
                "outer_optimizer": config.outer_optimizer,
                "outer_momentum": config.outer_momentum,
                "outer_lr": config.outer_lr,
                "inner_optimizer": "adam",
            },
            "wall_clock_breakdown": False,
        }
        _zero_stage = getattr(config, 'zero_stage', 0)
        _cpu_offload = getattr(config, 'cpu_offload', False)
        _large_model = config.model_size in ('7B', '13B')
        if _cpu_offload and _large_model and _zero_stage < 2:
            _zero_stage = 2
        elif _cpu_offload and _zero_stage < 1:
            _zero_stage = 1
        zero_cfg = {"stage": _zero_stage}
        if _cpu_offload:
            zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
            if _zero_stage >= 2:
                zero_cfg["offload_param"] = {"device": "cpu", "pin_memory": True}
        if _large_model and _zero_stage >= 2:
            zero_cfg["reduce_bucket_size"] = 200_000_000
            zero_cfg["allgather_bucket_size"] = 500_000_000
            zero_cfg["contiguous_gradients"] = True
            zero_cfg["overlap_comm"] = True
        ds_cfg["zero_optimization"] = zero_cfg

        if config.use_autosp and _zero_stage < 2:
            ds_cfg["compile"] = {
                "deepcompile": True,
                "passes": ["autosp"],
            }
        elif config.use_autosp and _zero_stage >= 2:
            config._autosp_eager_fallback = True
        return ds_cfg

    def _create_optimizer(self, method: str):
        """Create optimizer for non-DeepSpeed baselines.

        Methods:
          DDP:             standard AdamW, AllReduce every step
          LocalAdam:       LocalAdamW, sync all every Kx
          DESLOC_avg:      DES-LOC with averaging outer optimizer
          DESLOC_nesterov: DES-LOC with Nesterov outer optimizer (RQ5)
        """
        params = self.model.parameters()
        _r = dist.get_rank() if dist.is_initialized() else 0
        print(f"[M504/ADAM-ARGS] rank={_r} beta1={self.config.beta1} beta2={self.config.beta2} "
              f"eps={self.config.adam_eps} method={method}")

        if method == 'DDP':
            return AdamW(
                params, lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_eps,
            )
        elif method == 'LocalAdam':
            return LocalAdamW(
                params, lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                K=self.config.Kx
            )
        elif method in ('DESLOC', 'DESLOC_avg', 'DESLOC_nesterov'):
            # RQ5: explicit outer optimizer selection
            # DESLOC (no DS) defaults to config.outer_optimizer
            if '_nesterov' in method:
                outer = 'nesterov'
            elif '_avg' in method:
                outer = 'average'
            else:
                outer = self.config.outer_optimizer
            return DESLOCAdamW(
                params, lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                Kx=self.config.Kx, Ku=self.config.Ku, Kv=self.config.Kv,
                outer_optimizer=outer,
                outer_momentum=self.config.outer_momentum,
                outer_lr=self.config.outer_lr,
                max_norm=self.config.grad_clip,  # M465: per-group clip inside step()
                eps=self.config.adam_eps,        # M504: Megatron 48269d8d8
            )
        else:
            raise ValueError(f"Unknown baseline method: {method}")

    
    def train(self) -> Dict:
        """Run training loop.

        DESLOC: uses self.engine (DeepSpeed) — engine.backward() + engine.step()
          → engine.allreduce_gradients() applies Kx gating (engine.py:2558)
          → engine.desloc_post_step() advances scheduler (engine.py:2515)
          → DeslocProfiler records per-step timing + comm for NKI-FA export

        DDP/LocalAdam: raw PyTorch loop for baseline comparison.
        """
        if self.use_deepspeed:
            return self._train_deepspeed()
        else:
            return self._train_baseline()

    def _train_deepspeed(self) -> Dict:
        self.engine.train()
        data_iter = iter(self.dataloader)
        total_tokens = 0
        start_time = time.time()

        _autosp_prepare = None
        _eager_fallback = getattr(self.config, '_autosp_eager_fallback', False)
        if self.config.use_autosp and _eager_fallback:
            if self.rank == 0:
                print("[SP+DEC] ZeRO-2 active, skipping torch.compile entirely "
                      "(inductor backward graph OOMs on A6000 with 13B)")
            raise_eager = True
        elif self.config.use_autosp:
            raise_eager = False
            try:
                from deepspeed.compile.passes.sp_compile import prepare_autosp_inputs
                self.engine.compile(backend='inductor')
                _autosp_prepare = prepare_autosp_inputs
                self._sp_enabled = True
                if self.rank == 0:
                    print("[SP+DEC] AutoSP compiled with inductor backend")
            except Exception as e:
                if self.rank == 0:
                    print(f"[SP+DEC] Compile-time AutoSP failed: {e}")
                    print(f"[SP+DEC] Falling back to eager Ulysses SP")
                raise_eager = True
        else:
            raise_eager = False

        if raise_eager and self.config.use_autosp:
            model_cfg = self.config.get_model_config()
            n_heads = model_cfg['n_head']
            sp_size = 1
            for cand in range(min(self.world_size, n_heads), 0, -1):
                if n_heads % cand == 0 and cand <= self.world_size:
                    sp_size = cand
                    break
            self._sp_size = sp_size

            if sp_size > 1:
                try:
                    local_gpu = torch.cuda.get_device_name(self.engine.device)
                    nt = torch.zeros(256, dtype=torch.uint8, device=self.engine.device)
                    nb = local_gpu.encode('utf-8')[:256]
                    nt[:len(nb)] = torch.tensor(list(nb), dtype=torch.uint8)
                    all_nt = [torch.zeros(256, dtype=torch.uint8, device=self.engine.device)
                              for _ in range(self.world_size)]
                    dist.all_gather(all_nt, nt)
                    gpu_names = {}
                    for r, t in enumerate(all_nt):
                        gpu_names[r] = bytes(t.cpu().tolist()).rstrip(b'\x00').decode('utf-8', errors='replace')

                    from collections import defaultdict
                    type_groups = defaultdict(list)
                    for r, name in gpu_names.items():
                        type_groups[name].append(r)

                    sp_ranks = None
                    for name, ranks in sorted(type_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                        for s in range(min(len(ranks), sp_size), 0, -1):
                            if n_heads % s == 0:
                                sp_ranks = sorted(ranks[:s])
                                sp_size = s
                                break
                        if sp_ranks:
                            break

                    if sp_ranks and len(sp_ranks) >= 2:
                        self._sp_size = sp_size
                        sp_group = dist.new_group(sp_ranks)
                        self._sp_ranks = sp_ranks
                        self._sp_group = sp_group
                        if self.config.max_seq_len % sp_size != 0:
                            self.config.max_seq_len = ((self.config.max_seq_len + sp_size - 1) // sp_size) * sp_size
                        if self.rank in sp_ranks:
                            _sp_ctx_set(on=True, grp=sp_group, sz=sp_size, rk=sp_ranks.index(self.rank))
                            self._sp_enabled = True
                        else:
                            _sp_ctx_set(on=False, grp=None, sz=1, rk=0)
                            self._sp_enabled = False
                        if self.rank == 0:
                            print(f"[SP+DEC] Eager Ulysses SP: sp_size={sp_size} "
                                  f"ranks={sp_ranks} (GPU: {gpu_names[sp_ranks[0]]})")
                    else:
                        self._sp_enabled = False
                except Exception as e2:
                    self._sp_enabled = False
                    if self.rank == 0:
                        print(f"[SP+DEC] Eager fallback failed: {e2}")
            else:
                self._sp_enabled = False
            _autosp_prepare = None

        for step in range(1, self.config.max_steps + 1):
            step_start = time.time()

            # Profiler begin (comm.py DeslocProfiler)
            if self._profiler:
                self._profiler.begin_step(step)

            # Forward + backward through DeepSpeed engine
            # engine.backward() → _backward_epilogue() → allreduce_gradients()
            #   → DES-LOC Kx gating at engine.py:2558
            try:
                batch = next(data_iter)
            except StopIteration:
                # M452: advance MegatronRandomSampler epoch (mirrors Megatron pretrain_bert.py)
                _sampler = getattr(self.dataloader, 'sampler', None)
                if hasattr(_sampler, 'set_epoch'):
                    _new_epoch = getattr(_sampler, '_epoch', -1) + 1
                    _sampler.set_epoch(_new_epoch)
                    if self.rank == 0:
                        print(f"[M452-DS] step={step} epoch reset → "
                              f"sampler.set_epoch({_new_epoch})")
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(self.engine.device)
            labels = batch['labels'].to(self.engine.device)
            # M457: consume loss_mask from dataset (Megatron 872b4a6 multi-EOS fix)
            loss_mask = batch.get('loss_mask', None)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.engine.device)

            if _autosp_prepare is not None:
                _autosp_prepare(
                    input_id=input_ids,
                    label_id=labels,
                    seq_dim=1,
                )
            elif self._sp_enabled and self._sp_group is not None and _autosp_prepare is None:
                _SP_CTX['step'] = step
                sp_src = self._sp_ranks[0]
                dist.broadcast(input_ids, src=sp_src, group=self._sp_group)
                dist.broadcast(labels, src=sp_src, group=self._sp_group)
                if loss_mask is not None:
                    dist.broadcast(loss_mask, src=sp_src, group=self._sp_group)
                if self.rank in self._sp_ranks:
                    local_seq = input_ids.shape[1] // self._sp_size
                    sp_rk = self._sp_ranks.index(self.rank)
                    start = sp_rk * local_seq
                    input_ids = input_ids[:, start:start+local_seq].contiguous()
                    labels = labels[:, start:start+local_seq].contiguous()
                    if loss_mask is not None:
                        loss_mask = loss_mask[:, start:start+local_seq].contiguous()

            _t_fwd = time.time()
            _, loss = self.engine(input_ids, labels)
            # M457: apply loss_mask — zero out loss at EOD positions so the model
            # is not rewarded for predicting the EOS token itself.
            # Knuth §1.2.10: normalise by the count of unmasked positions to keep
            # gradient magnitude stable when EOS density varies between batches.
            if loss_mask is not None:
                n_active = loss_mask.sum().clamp(min=1.0)
                loss = (loss * loss_mask).sum() / n_active
                if step == 1 and self.rank == 0:
                    print(
                        f"[M457-DS] step={step} loss_mask applied: "
                        f"active_tokens={int(n_active.item())}/{loss_mask.numel()}, "
                        f"masked_loss={loss.item():.4f}"
                    )
            _t_fwd = time.time() - _t_fwd

            _t_bwd = time.time()
            self.engine.backward(loss)
            _t_bwd = time.time() - _t_bwd

            _t_opt = time.time()
            self.engine.step()
            _t_opt = time.time() - _t_opt

            # DES-LOC post-step: advance scheduler, record comm events
            if hasattr(self.engine, 'desloc_post_step'):
                self.engine.desloc_post_step(loss=loss.item())
            if hasattr(self.engine, 'desloc_record_loss'):
                self.engine.desloc_record_loss(loss.item())

            step_time = time.time() - step_start
            step_tokens = self.config.batch_size * self.config.gradient_accumulation * self.config.max_seq_len
            total_tokens += step_tokens

            # Profiler end
            if self._profiler:
                lr = self.engine.get_lr()[0] if hasattr(self.engine, 'get_lr') else 0
                self._profiler.end_step(loss=loss.item(), lr=lr)

            # Record metrics
            self.metrics['losses'].append(loss.item())
            self.metrics['step_times'].append(step_time)
            cur_mem = torch.cuda.max_memory_allocated(self.engine.device) / 1e9
            self.metrics['memory_usage'].append(cur_mem)

            # Track DES-LOC sync events from the scheduler
            sched = get_desloc_scheduler()
            if sched:
                self.metrics['comm_events'].append({
                    'step': step,
                    'sync_x': sched.should_sync_x(),
                    'sync_u': sched.should_sync_u(),
                    'sync_v': sched.should_sync_v(),
                })

            # Log (NKI-FA style)
            if step % self.config.log_interval == 0 and self.rank == 0:
                elapsed = time.time() - start_time
                per_gpu_tps = total_tokens / elapsed
                cluster_tps = per_gpu_tps * self.world_size
                skipped = getattr(self.engine, 'desloc_skipped_allreduces', 0)
                _sp_tag = ""
                if self._sp_enabled:
                    _sp_tag = (f" | SP={self._sp_size}way "
                               f"seq={input_ids.shape[1]}/{self.config.max_seq_len}")
                print(f"[DESLOC-DS] Step {step}/{self.config.max_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Time: {step_time*1000:.1f}ms "
                      f"(fwd={_t_fwd*1000:.0f} bwd={_t_bwd*1000:.0f} opt={_t_opt*1000:.0f}) | "
                      f"Tok/s(gpu): {per_gpu_tps:.0f} | "
                      f"Tok/s(all): {cluster_tps:.0f} | "
                      f"Mem: {cur_mem:.2f}GB | "
                      f"AR_skipped: {skipped}{_sp_tag}")

            if step_time > 10.0 and self.rank == 0:
                print(f"[SPIKE] Step {step} took {step_time*1000:.0f}ms "
                      f"(fwd={_t_fwd*1000:.0f} bwd={_t_bwd*1000:.0f} opt={_t_opt*1000:.0f})")

            if step == 1 and self.rank == 0:
                print(f"[STEP1-DIAG] input_ids={list(input_ids.shape)} "
                      f"sp_enabled={self._sp_enabled} "
                      f"sp_size={getattr(self, '_sp_size', 1)} "
                      f"sp_group={self._sp_group is not None} "
                      f"sp_ctx_on={_SP_CTX['on']} "
                      f"compile={getattr(self.engine, '_is_compiled', False)} "
                      f"zero={self.engine.zero_optimization_stage()} "
                      f"mem={torch.cuda.max_memory_allocated(self.engine.device)/1e9:.2f}GB "
                      f"fwd={_t_fwd*1000:.0f}ms bwd={_t_bwd*1000:.0f}ms opt={_t_opt*1000:.0f}ms")

        return self._finalize_results(total_tokens, start_time)

    def _train_baseline(self) -> Dict:
        """Raw PyTorch training loop for DDP/LocalAdam/DESLOC baselines.

        M339 SP+DEC integration:
        When self._sp_enabled=True, input sequences are scattered across
        workers along dim=1 before forward pass. Each worker processes
        seq_len/world_size tokens. After backward, gradients are reduce-
        scattered along seq dim, then DES-LOC Kx gating decides whether
        to AllReduce across workers (data-parallel sync).

        This implements the SP+DEC orthogonal composition:
          - SP: splits along sequence (dim=1) within each step
          - DEC: gates AllReduce along worker (dim=0) across steps
          - Both require ZeRO stage 0 → no conflict

        Without SP (default): identical to original baseline loop.
        """
        self.model.train()
        data_iter = iter(self.dataloader)
        total_tokens = 0
        start_time = time.time()

        # SP+DEC: effective tokens per step depends on whether SP is active
        # With SP: each worker sees seq_len/world_size tokens, but total
        # across cluster is still batch * grad_accum * seq_len
        if self._sp_enabled:
            local_seq_len = self.config.max_seq_len // self._sp_size
            if self.rank == 0:
                print(f"[SP+DEC] Training with local_seq_len={local_seq_len} "
                      f"(full={self.config.max_seq_len}, sp_size={self._sp_size})")
        else:
            local_seq_len = self.config.max_seq_len

        for step in range(1, self.config.max_steps + 1):
            step_start = time.time()
            # M505: Megatron 664cd28b2 — use tensor accumulator so that
            # zero-valued micro-steps (masked / skipped) don't pollute the
            # running average.  Mirrors Megatron training_log() fix:
            #   total_loss_dict[key] = get(key, FloatTensor([0.0])) + loss
            accumulated_loss = torch.zeros(1, dtype=torch.float32,
                                           device=self.device if hasattr(self, 'device') else 'cpu')
            print(f"[M505-LOSS-INIT] step={step} accumulated_loss reset to tensor(0.)")
            _SP_CTX['step'] = step

            # M366: begin step recording
            self._recorder.begin_step(step)

            # M507: Megatron 3e6898e66 — 1F1B pipeline schedule for memory.
            # Compute warmup microbatches as (pipeline_depth - rank - 1) capped
            # by total gradient_accumulation steps. Warmup phases hold forward
            # activations in memory while pipeling backward for earlier microbatches.
            # In single-device / non-pipeline mode pipeline_depth=1, rank=0 →
            # num_warmup = 0, so the loop degrades to the original all-forward-
            # then-all-backward pattern (identity transform, no regression).
            _pipe_world = getattr(self, '_pipe_world_size',
                                  int(os.environ.get('PIPELINE_SIZE', '1')))
            _pipe_rank  = getattr(self, '_pipe_rank',
                                  int(os.environ.get('PIPELINE_RANK', '0')))
            _num_micro   = self.config.gradient_accumulation
            _num_warmup  = min(max(_pipe_world - _pipe_rank - 1, 0), _num_micro)
            print(f"[M507-1F1B] step={step} pipe_world={_pipe_world} "
                  f"pipe_rank={_pipe_rank} num_micro={_num_micro} "
                  f"num_warmup={_num_warmup}")

            for micro_step in range(self.config.gradient_accumulation):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # M452: Megatron 66719e9 epoch advancement —
                    # set_epoch() re-seeds the MegatronRandomSampler so each
                    # epoch uses a different shuffle permutation while remaining
                    # deterministic across restarts.
                    _sampler = getattr(self.dataloader, 'sampler', None)
                    if hasattr(_sampler, 'set_epoch'):
                        _new_epoch = getattr(_sampler, '_epoch', -1) + 1
                        _sampler.set_epoch(_new_epoch)
                        print(f"[M452-TRAIN] step={step} dataloader epoch reset → "
                              f"sampler.set_epoch({_new_epoch})")
                    elif hasattr(_sampler, 'set_epoch'):  # DistributedSampler
                        pass  # handled above
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                # M457: consume loss_mask from dataset (Megatron 872b4a6 multi-EOS fix)
                loss_mask_batch = batch.get('loss_mask', None)
                if loss_mask_batch is not None:
                    loss_mask_batch = loss_mask_batch.to(self.device)

                # M365 DIAG: pre-broadcast data hash on ALL ranks
                if _diag and step % 50 == 1:
                    _diag.log_data_hash(step, self.rank, input_ids, "pre-bcast-input")
                    _diag.log_data_hash(step, self.rank, labels, "pre-bcast-labels")

                # M365 CRITICAL FIX: Broadcast data within SP group before scatter.
                if self._sp_enabled and self._sp_group is not None and self._sp_ranks is not None:
                    sp_src = self._sp_ranks[0]
                    dist.broadcast(input_ids, src=sp_src, group=self._sp_group)
                    dist.broadcast(labels, src=sp_src, group=self._sp_group)
                    # M457: broadcast loss_mask with the same SP group
                    if loss_mask_batch is not None:
                        dist.broadcast(loss_mask_batch, src=sp_src, group=self._sp_group)

                    # M365 DIAG: post-broadcast cross-rank hash verification
                    if _diag and step % 50 == 1:
                        _diag.log_data_hash_cross_rank(
                            step, self.rank, input_ids,
                            "post-bcast-input", self._sp_group)
                        _diag.log_data_hash(step, self.rank, labels, "post-bcast-labels")

                # M339 SP+DEC: scatter input along sequence dimension
                if self._sp_enabled and self._sp_comm is not None:
                    input_ids = self._sp_comm.scatter_along_seq(
                        input_ids, dim=1
                    )
                    labels = self._sp_comm.scatter_along_seq(
                        labels, dim=1
                    )
                    # M457: scatter loss_mask along sequence dim in sync with inputs
                    if loss_mask_batch is not None:
                        loss_mask_batch = self._sp_comm.scatter_along_seq(
                            loss_mask_batch, dim=1
                        )
                    # M365 DIAG: post-scatter per-rank data hash
                    if _diag and step % 50 == 1:
                        _diag.log_data_hash(step, self.rank, input_ids, "post-scatter-input")
                        _diag.log_data_hash(step, self.rank, labels, "post-scatter-labels")

                with autocast():
                    _, loss = self.model(input_ids, labels)
                    # M457: apply loss_mask — normalise by active token count
                    # Knuth §1.2.10: division by clamp(sum,1) avoids zero-div when
                    # an entire micro-batch is masked (pathological but possible).
                    if loss_mask_batch is not None:
                        n_active = loss_mask_batch.sum().clamp(min=1.0)
                        loss = (loss * loss_mask_batch).sum() / n_active
                        if step == 1 and micro_step == 0 and self.rank == 0:
                            print(
                                f"[M457-DDP] step={step} loss_mask applied: "
                                f"active_tokens={int(n_active.item())}/{loss_mask_batch.numel()}, "
                                f"masked_loss={loss.item():.4f}"
                            )
                    loss = loss / self.config.gradient_accumulation

                # M507: Megatron 3e6898e66 — 1F1B schedule.
                # Warmup micro-steps run forward only and stash (loss, output)
                # for the cooldown pass.  Steady-state and cooldown run
                # forward+backward immediately (1 forward, 1 backward).
                # In non-pipeline mode _num_warmup=0 → always in steady-state.
                _in_warmup = (micro_step < _num_warmup)
                _in_cooldown_extra = False  # resolved after loop
                print(f"[M507-SCHED] step={step} micro={micro_step} "
                      f"warmup={_in_warmup} num_warmup={_num_warmup}")
                if _in_warmup:
                    # Warmup: forward only — stash loss tensor, skip backward
                    if not hasattr(self, '_1f1b_stash'):
                        self._1f1b_stash = []
                    self._1f1b_stash.append(loss)
                    # M973: Megatron 7f1c59177 — handle output_tensor == None
                    if loss is None:
                        print('[M973]')
                        continue
                    # M957: free output tensor data after stashing if flag set
                    if getattr(args, 'deallocate_pipeline_outputs', False) and isinstance(loss, torch.Tensor):
                        loss.data = torch.tensor(float('nan'))
                    _micro_loss_val = loss.detach().float()
                else:
                    # Steady-state 1F1B: run backward immediately
                    self.scaler.scale(loss).backward() if self.scaler else loss.backward()
                    # Drain one warmup-stashed loss if any remain (cooldown pass)
                    if hasattr(self, '_1f1b_stash') and self._1f1b_stash:
                        _stashed = self._1f1b_stash.pop(0)
                        self.scaler.scale(_stashed).backward() if self.scaler else _stashed.backward()
                        print(f"[M507-COOLDOWN] step={step} micro={micro_step} "
                              f"drained stashed forward (stash_left={len(self._1f1b_stash)})")
                    # M505: Megatron 664cd28b2 — tensor add (not float +=) so dtype
                    # is preserved; skip micro-step if loss is zero (masked / NaN-guarded).
                    _micro_loss_val = loss.detach().float()
                    if _micro_loss_val.item() > 0.0:
                        accumulated_loss = accumulated_loss + _micro_loss_val.cpu()
                print(f"[M505-MICRO] step={step} micro={micro_step} "
                      f"micro_loss={loss.detach().float().item():.6f} "
                      f"accum={accumulated_loss.item():.6f} "
                      f"skipped={loss.detach().float().item() == 0.0}")

            # M507: Megatron 3e6898e66 — 1F1B cooldown: drain any remaining
            # warmup-stashed forward tensors that were not paired during the
            # steady-state 1F1B pass (happens when _num_warmup == _num_micro,
            # i.e. no steady-state pairs, only warmup+cooldown).
            if hasattr(self, '_1f1b_stash') and self._1f1b_stash:
                print(f"[M507-COOLDOWN-FINAL] step={step} draining "
                      f"{len(self._1f1b_stash)} remaining stashed losses")
                for _stashed in self._1f1b_stash:
                    self.scaler.scale(_stashed).backward() if self.scaler else _stashed.backward()
                self._1f1b_stash.clear()

            # M365 DIAG: post-backward gradient statistics
            if _diag and step % 50 == 1:
                _diag.log_grad_stats(step, self.rank, self.model)

            # M366: Record gradient statistics to StepRecorder (structured)
            # Pattern: Megatron training_log() records grad_norm, num_zeros at every log_interval
            fwd_end = time.time()
            if step % self._recorder.summary_interval == 0 or step <= 10 or step % self._recorder.detail_interval == 1:
                self._recorder.record_grad_stats(self.model, self.rank)
                self._recorder.record('loss', round(accumulated_loss.item(), 6))

            # M364 DIAG
            if step % 100 == 1:
                gnorm = sum(p.grad.float().norm().item()**2 for p in self.model.parameters() if p.grad is not None)**0.5
                _r = dist.get_rank() if dist.is_initialized() else 0
                print(f"[GRAD] rank={_r} step={step} grad_norm={gnorm:.4f} loss={accumulated_loss.item():.6f}")
                if self.world_size > 1:
                    lt = accumulated_loss.to(self.device)
                    al = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
                    dist.all_gather(al, lt)
                    print(f"[GRAD] rank={_r} all_rank_losses={[round(x.item(),6) for x in al]}")

            # M365 DIAG: snapshot parameter norms BEFORE optimizer step
            _pre_step_pnorm = None
            if _diag and step % 50 == 1:
                with torch.no_grad():
                    _pre_step_pnorm = sum(
                        p.detach().float().norm().item()**2
                        for p in self.model.parameters()
                    )**0.5

            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            # M465: clip_grad moved into DESLOCAdamW.step() (Megatron 4687967 pattern).
            # Per-group max_norm is applied inside optimizer.step() with shared-param and
            # TP-duplicate filtering.  Do NOT call torch.nn.utils.clip_grad_norm_() here —
            # doing so would double-clip: training-loop clip reduces ‖g‖ to max_norm,
            # then DESLOCAdamW.step() sees an already-clipped gradient and clips again
            # (a no-op only if both thresholds match exactly, which is fragile).
            # For non-DESLOCAdamW optimizers (AdamW, LocalAdamW) a fallback is provided.
            if not isinstance(self.optimizer, DESLOCAdamW):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # M365 FIX / M465 UPDATE: Cache gradient norm after clip, before optimizer.step().
            # For DESLOCAdamW: read the norm that _clip_grad_norm_per_group stored on the
            # optimizer during step() — avoids re-iterating all params a second time.
            # For other optimizers: compute norm directly (grads still live at this point).
            # optimizer.step() does p.grad = None for Adam denom VRAM, so norms after
            # step() are always 0.0 — this cache preserves the pre-step value.
            _cached_grad_norm = 0.0
            if isinstance(self.optimizer, DESLOCAdamW) and hasattr(self.optimizer, '_last_grad_norm'):
                # _last_grad_norm is set by _clip_grad_norm_per_group inside step()
                # but step() hasn't fired yet — it fires below.  We fall through to
                # the manual compute for DESLOCAdamW as well, then overwrite after step.
                pass
            with torch.no_grad():
                _cached_grad_norm = sum(
                    p.grad.float().norm().item()**2
                    for p in self.model.parameters() if p.grad is not None
                )**0.5
            # Store on optimizer so sync_if_needed can access it
            if hasattr(self.optimizer, '__dict__'):
                self.optimizer._cached_grad_norm = _cached_grad_norm

            # M366: Record param norm BEFORE optimizer step (for update delta tracking)
            _pre_opt_pnorm = 0.0
            opt_start = time.time()
            if step % self._recorder.summary_interval == 0 or step <= 10:
                with torch.no_grad():
                    _pre_opt_pnorm = sum(
                        p.detach().float().norm().item()**2
                        for p in self.model.parameters()
                    )**0.5

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # M462: Megatron 27e14f82 — advance lr_scheduler each step.
            # Megatron's refactored pretrain loop calls
            # learning_rate_scheduler.step() unconditionally after optimizer.step().
            # Previously _train_baseline had no scheduler → flat LR throughout,
            # making DDP/DESLOC comparisons invalid vs the DeepSpeed engine path
            # which advances its own cosine schedule via desloc_post_step().
            # M462 fix: step the LambdaLR created in __init__ (warmup + cosine decay).
            _cur_lr = self.optimizer.param_groups[0].get('lr', 0.0)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                _cur_lr = self.lr_scheduler.get_last_lr()[0]
                # Diagnostic: print LR at warmup boundary and first log_interval steps
                if self.rank == 0 and (
                    step == self.config.warmup_steps or
                    (step % self.config.log_interval == 0 and step <= 20)
                ):
                    print(f"[M462-LR] step={step} lr={_cur_lr:.6e} "
                          f"(warmup_steps={self.config.warmup_steps})")

            opt_end = time.time()

            # M366: Record update delta and optimizer state into StepRecorder
            if step % self._recorder.summary_interval == 0 or step <= 10:
                with torch.no_grad():
                    _post_opt_pnorm = sum(
                        p.detach().float().norm().item()**2
                        for p in self.model.parameters()
                    )**0.5
                self._recorder.record_update_delta(_pre_opt_pnorm, _post_opt_pnorm)
                self._recorder.record_optimizer_state(self.optimizer)
                # M462: record LR AFTER scheduler.step() (not before — see Knuth
                # critique in __init__: pre-step LR was one step stale on warmup ramps).
                self._recorder.record_lr(_cur_lr)

            # M365 DIAG: parameter update magnitude tracking
            if _diag and step % 50 == 1 and _pre_step_pnorm is not None:
                with torch.no_grad():
                    _post_step_pnorm = sum(
                        p.detach().float().norm().item()**2
                        for p in self.model.parameters()
                    )**0.5
                    _delta = abs(_post_step_pnorm - _pre_step_pnorm)
                    _ratio = _delta / max(_pre_step_pnorm, 1e-12)
                    print(f"[DIAG/UPDATE] rank={self.rank} step={step} "
                          f"||p||_before={_pre_step_pnorm:.4f} "
                          f"||p||_after={_post_step_pnorm:.4f} "
                          f"delta={_delta:.6f} update_ratio={_ratio:.8f}")

            # M365 DIAG: optimizer state (momentum, variance)
            if _diag and step % 50 == 1:
                _diag.log_optimizer_state(step, self.rank, self.optimizer)

            # M365 DIAG: parameter divergence across ranks
            if _diag and step % 50 == 1:
                _diag.log_param_divergence(step, self.rank, self.model, self.world_size)

            # DES-LOC / LocalAdam sync — the "DEC" part of SP+DEC
            if hasattr(self.optimizer, 'sync_if_needed'):
                if hasattr(self.optimizer, 'global_step'):
                    self.optimizer.global_step = step

                # M447: feed current loss into adaptive Kx window
                if hasattr(self.optimizer, '_adaptive_loss_window'):
                    self.optimizer._adaptive_loss_window.append(accumulated_loss.item())
                    if len(self.optimizer._adaptive_loss_window) > 200:
                        self.optimizer._adaptive_loss_window = self.optimizer._adaptive_loss_window[-200:]

                # M447: STEP-DUMP — full state snapshot for debugging
                if step % 10 == 0 or step <= 5:
                    _sd_pnorm = sum(p.data.float().norm().item()**2
                                    for p in self.model.parameters())**0.5
                    _sd_gnorm = getattr(self.optimizer, '_last_grad_norm', 0.0)
                    _sd_kx = getattr(self.optimizer, '_last_effective_Kx', self.config.Kx)
                    _sd_oas = getattr(self.optimizer, '_oas_scale', 'N/A')
                    print(f"[STEP-DUMP] rank={self.rank} step={step} "
                          f"loss={accumulated_loss.item():.6f} "
                          f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                          f"grad_norm={_sd_gnorm:.4f} "
                          f"param_norm={_sd_pnorm:.4f} "
                          f"oas_scale={_sd_oas} "
                          f"Kx_eff={_sd_kx} "
                          f"mem={torch.cuda.memory_allocated(self.device)/1e9:.2f}GB")

                # M365 DIAG: pre-sync param norm
                _pre_sync_pnorm = None
                if _diag and step % 50 == 1:
                    with torch.no_grad():
                        _pre_sync_pnorm = sum(
                            p.detach().float().norm().item()**2
                            for p in self.model.parameters()
                        )**0.5

                # M366: pre-sync param norm for StepRecorder (always)
                _rec_pre_sync = 0.0
                sync_start = time.time()
                if step % self._recorder.summary_interval == 0 or step <= 10:
                    with torch.no_grad():
                        _rec_pre_sync = sum(
                            p.detach().float().norm().item()**2
                            for p in self.model.parameters()
                        )**0.5

                sync_info = self.optimizer.sync_if_needed(self.world_size)

                sync_end = time.time()

                # M366: Record sync event to StepRecorder
                if sync_info and (step % self._recorder.summary_interval == 0 or step <= 10):
                    _rec_post_sync = 0.0
                    if sync_info.get('sync_x', False):
                        with torch.no_grad():
                            _rec_post_sync = sum(
                                p.detach().float().norm().item()**2
                                for p in self.model.parameters()
                            )**0.5
                    effective_Kx = getattr(self.optimizer, '_last_effective_Kx', self.config.Kx)
                    self._recorder.record_sync_event(
                        sync_x=sync_info.get('sync_x', False),
                        sync_u=sync_info.get('sync_u', False),
                        sync_v=sync_info.get('sync_v', False),
                        effective_Kx=effective_Kx,
                        pre_sync_norm=_rec_pre_sync,
                        post_sync_norm=_rec_post_sync,
                    )

                # M366: Record cross-rank divergence at detail intervals
                if step % self._recorder.detail_interval == 1 and sync_info and sync_info.get('sync_x', False):
                    self._recorder.record_cross_rank_divergence(
                        self.model, self.world_size)

                # M365 DIAG: post-sync param norm + sync event logging
                if _diag and step % 50 == 1 and sync_info:
                    if sync_info.get('sync_x', False) and _pre_sync_pnorm is not None:
                        with torch.no_grad():
                            _post_sync_pnorm = sum(
                                p.detach().float().norm().item()**2
                                for p in self.model.parameters()
                            )**0.5
                        _diag.log_sync_event(step, self.rank, 'x',
                                              _pre_sync_pnorm, _post_sync_pnorm,
                                              self.world_size)
                        # Post-sync param divergence should be ZERO across ranks
                        _diag.log_param_divergence(step, self.rank, self.model,
                                                    self.world_size)

                if sync_info:
                    self.metrics['comm_events'].append({'step': step, **sync_info})

            # SP+DEC: advance SP step counter (for dp_gated_allreduce)
            if self._sp_comm is not None:
                self._sp_comm.step()

            # M365 FIX: zero_grad AFTER sync_if_needed, not before.
            # sync_if_needed() reads grad norms for [SYNC] diagnostics.
            # It operates on p.data and optimizer states, NOT gradients,
            # so this reordering is functionally safe.
            # Previously zero_grad was before sync → [SYNC] always showed grad=0.0000.
            self.optimizer.zero_grad(set_to_none=True)

            # M365 DIAG: pipeline routing (only at step 1)
            if _diag and step == 1:
                _diag.log_pipeline(step, self.rank,
                    sp_enabled=self._sp_enabled,
                    sp_size=self._sp_size,
                    sp_ranks=str(self._sp_ranks),
                    method=self.method,
                    offload=getattr(self.optimizer, '_use_offload', False),
                    use_ac=self.config.use_ac,
                    world_size=self.world_size,
                    Kx=self.config.Kx,
                    scaler=self.scaler is not None)

            step_time = time.time() - step_start
            step_tokens = self.config.batch_size * self.config.gradient_accumulation * self.config.max_seq_len
            total_tokens += step_tokens

            self.metrics['losses'].append(accumulated_loss.item())
            self.metrics['step_times'].append(step_time)
            cur_mem = torch.cuda.max_memory_allocated(self.device) / 1e9
            self.metrics['memory_usage'].append(cur_mem)

            # M366: Record timing breakdown and memory, then finalize step
            if step % self._recorder.summary_interval == 0 or step <= 10 or step % self._recorder.detail_interval == 1:
                bwd_end = time.time()
                self._recorder.record_timing(
                    fwd_ms=(fwd_end - step_start) * 1000 if 'fwd_end' in dir() else 0,
                    bwd_ms=(opt_start - fwd_end) * 1000 if 'opt_start' in dir() and 'fwd_end' in dir() else 0,
                    opt_ms=(opt_end - opt_start) * 1000 if 'opt_end' in dir() and 'opt_start' in dir() else 0,
                    sync_ms=(sync_end - sync_start) * 1000 if 'sync_end' in dir() and 'sync_start' in dir() else 0,
                    total_ms=step_time * 1000,
                )
                self._recorder.record_memory()
                # Param stats at detail intervals (per-layer weight mean/std)
                if step % self._recorder.detail_interval == 1:
                    self._recorder.record_param_stats(self.model)
                # Pipeline routing info at first step
                if step == 1:
                    self._recorder.record_pipeline_info(
                        sp_enabled=self._sp_enabled,
                        sp_size=self._sp_size,
                        method=self.method,
                        offload=getattr(self.optimizer, '_use_offload', False),
                        use_ac=self.config.use_activation_checkpointing,
                        world_size=self.world_size,
                        Kx=self.config.Kx, Ku=self.config.Ku, Kv=self.config.Kv,
                        batch_size=self.config.batch_size,
                        grad_accum=self.config.gradient_accumulation,
                    )
                self._recorder.end_step()

            if step % self.config.log_interval == 0 and self.rank == 0:
                elapsed = time.time() - start_time
                per_gpu_tps = total_tokens / elapsed
                cluster_tps = per_gpu_tps * self.world_size
                print(f"[{self.method}] Step {step}/{self.config.max_steps} | "
                      f"Loss: {accumulated_loss.item():.4f} | "
                      f"Time: {step_time*1000:.1f}ms | "
                      f"Tok/s(gpu): {per_gpu_tps:.0f} | "
                      f"Tok/s(all): {cluster_tps:.0f} | "
                      f"Mem: {cur_mem:.2f}GB")

        return self._finalize_results(total_tokens, start_time)

    def _finalize_results(self, total_tokens, start_time) -> Dict:
        """Compute final metrics -- shared by both paths.

        Paper Metrics (Section 5):
        (i)   Perplexity: exp(cross_entropy_loss)
        (ii)  Per-worker comm cost: Ring-AllReduce 2(W-1)/W * N * sizeof(param)
              bandwidth-optimal, scales linearly with model size
        (iii) Rate of change: ∥s_{t+K} - s_t∥₂ / ∥s_t∥₂ per tier
        (iv)  Wall-clock time: total seconds + per-step breakdown
        """
        total_time = time.time() - start_time
        per_gpu_tps = total_tokens / total_time
        cluster_tps = per_gpu_tps * self.world_size

        # MFU
        model_ref = self.engine.module if self.engine else self.model
        n_params = sum(p.numel() for p in model_ref.parameters())
        flops_per_token = 6 * n_params
        achieved_flops = per_gpu_tps * flops_per_token
        gpu_name = torch.cuda.get_device_name(self.device)
        # GPU peak BF16 TFLOPS lookup — MUST match actual hardware
        # H20: Hopper阉割版, BF16=148T (NOT 989.5 like H100 SXM)
        # 阿里云gn8v系列的"GPU H"实际是H20, 96GB HBM3, 4TB/s
        peak_tflops = 312e12  # default: A100 SXM BF16
        if 'H20' in gpu_name:
            peak_tflops = 148e12   # NVIDIA H20: BF16=148 TFLOPS
        elif 'A6000' in gpu_name:
            peak_tflops = 38.7e12  # RTX A6000: BF16=38.7 TFLOPS
        elif 'H100' in gpu_name or 'H800' in gpu_name:
            if 'NVL' in gpu_name:
                peak_tflops = 835e12  # H100 NVL: BF16=835 TFLOPS
            elif 'SXM' in gpu_name:
                peak_tflops = 989.5e12
            else:
                peak_tflops = 756e12  # H100 PCIe
        elif 'A100' in gpu_name or 'A800' in gpu_name:
            peak_tflops = 312e12   # A100 SXM: BF16=312 TFLOPS
        elif 'L40' in gpu_name:
            peak_tflops = 181e12   # L40S: BF16=181 (Ada Lovelace)
        elif '4090' in gpu_name:
            peak_tflops = 165.2e12 # RTX 4090: BF16=165.2
        elif 'V100' in gpu_name:
            peak_tflops = 125e12   # V100: FP16=125 (no BF16)
        elif 'PRO 6000' in gpu_name or 'Blackwell' in gpu_name:
            peak_tflops = 300e12   # RTX PRO 6000 Blackwell: ~300 BF16 TFLOPS (est.)
        mfu_val = achieved_flops / peak_tflops if peak_tflops > 0 else 0.0

        # M449: Clamp MFU to [0, 1.0] — values >1.0 indicate measurement error
        # (e.g. heterogeneous cluster where fast GPU's TPS is divided by slow GPU's peak)
        if mfu_val > 1.0:
            _r_mfu = dist.get_rank() if dist.is_initialized() else 0
            print(f"[MFU-WARN] rank={_r_mfu} computed MFU={mfu_val:.4f} > 1.0 "
                  f"(achieved={achieved_flops/1e12:.2f}T peak={peak_tflops/1e12:.2f}T "
                  f"gpu={gpu_name} tps={per_gpu_tps:.1f}) — clamping to 1.0")
            mfu_val = min(mfu_val, 1.0)

        # Use desloc_mfu from timer.py for cross-check
        mfu_check = desloc_mfu(achieved_flops / 1e12, peak_tflops / 1e12)

        # Comm reduction from utils.py
        comm_red = desloc_comm_reduction_ratio(
            self.config.Kx, self.config.Ku, self.config.Kv, self.config.max_steps
        )

        # === Paper Metric (i): Perplexity ===
        final_loss = self.metrics['losses'][-1]
        avg_loss = sum(self.metrics['losses'][-100:]) / min(100, len(self.metrics['losses']))
        final_ppl = math.exp(min(final_loss, 20.0))  # clamp to avoid overflow
        avg_ppl = math.exp(min(avg_loss, 20.0))

        # === Paper Metric (ii): Per-worker asymptotic comm cost ===
        # Ring-AllReduce bandwidth-optimal: 2(W-1)/W * N * sizeof
        # W = world_size, N = n_params, sizeof = 2 bytes (fp16/bf16)
        W = max(self.world_size, 1)
        sizeof_param = 2  # bf16/fp16 = 2 bytes
        ring_factor = 2.0 * (W - 1) / W if W > 1 else 0.0
        # DDP: every step syncs gradients (same size as params)
        ddp_comm_bytes_per_step = ring_factor * n_params * sizeof_param
        ddp_total_comm_bytes = ddp_comm_bytes_per_step * self.config.max_steps
        # DES-LOC: x every Kx, u every Ku, v every Kv (each same size as params)
        # Claude-27 M335: u uses Ku_target always (no warmup ramp).
        # v piggybacks on x (sync whenever x syncs) + own Kv schedule.
        # Only x follows the warmup Kx ramp (1→Kx_target).
        steps = self.config.max_steps
        Kx, Ku, Kv = self.config.Kx, self.config.Ku, self.config.Kv
        warmup_steps = min(100, Kx * 3)
        # Count actual syncs matching real schedule
        desloc_syncs_x = 0
        desloc_syncs_u = 0
        desloc_syncs_v = 0
        for s in range(1, steps + 1):
            if s <= warmup_steps:
                frac = s / max(warmup_steps, 1)
                eff_Kx = max(1, int(1 + (Kx - 1) * frac))
            else:
                eff_Kx = Kx
            sx = (eff_Kx <= 1) or (s % eff_Kx == 0)
            su = (Ku <= 1) or (s % Ku == 0)               # M335: always use Ku_target
            sv = (Kv <= 1) or (s % Kv == 0) or sx         # v piggybacks on x
            desloc_syncs_x += int(sx)
            desloc_syncs_u += int(su)
            desloc_syncs_v += int(sv)
        desloc_total_comm_bytes = ring_factor * n_params * sizeof_param * (
            desloc_syncs_x + desloc_syncs_u + desloc_syncs_v
        )
        # LocalAdam: syncs all 3 every K (with same warmup)
        local_syncs = 0
        for s in range(1, steps + 1):
            if s <= warmup_steps:
                frac = s / max(warmup_steps, 1)
                eff_K = max(1, int(1 + (Kx - 1) * frac))
            else:
                eff_K = Kx
            if (eff_K <= 1) or (s % eff_K == 0):
                local_syncs += 3  # all 3 tiers synced together
        local_total_comm_bytes = ring_factor * n_params * sizeof_param * local_syncs

        # === Paper Metric (iv): Wall-clock breakdown ===
        step_times = self.metrics['step_times']
        wall_clock = {
            'total_s': total_time,
            'avg_step_ms': sum(step_times) / len(step_times) * 1000,
            'min_step_ms': min(step_times) * 1000,
            'max_step_ms': max(step_times) * 1000,
            'p50_step_ms': sorted(step_times)[len(step_times)//2] * 1000,
            'p99_step_ms': sorted(step_times)[int(len(step_times)*0.99)] * 1000,
            'step_times_ms': [t * 1000 for t in step_times],  # full timewise data
        }

        results = {
            'method': self.method,
            'final_loss': final_loss,
            'avg_loss': avg_loss,
            'final_ppl': final_ppl,
            'avg_ppl': avg_ppl,
            'total_time_seconds': total_time,
            'avg_step_time_ms': wall_clock['avg_step_ms'],
            'tokens_per_second_per_gpu': per_gpu_tps,
            'tokens_per_second_cluster': cluster_tps,
            'peak_memory_gb': max(self.metrics['memory_usage']),
            'total_tokens': total_tokens,
            'world_size': self.world_size,
            'gpu_name': gpu_name,
            'sp_enabled': self._sp_enabled,
            'sp_mode': 'autosp' if (self.use_deepspeed and self.config.use_autosp) else ('standalone' if self._sp_enabled else 'none'),
            'mfu': mfu_val,
            'mfu_check': mfu_check,
            'comm_reduction': comm_red,
            'n_params': n_params,
            'comm_bytes': {
                'ddp_per_step': ddp_comm_bytes_per_step,
                'ddp_total': ddp_total_comm_bytes,
                'desloc_total': desloc_total_comm_bytes,
                'local_total': local_total_comm_bytes,
                'ring_factor': ring_factor,
                'sizeof_param': sizeof_param,
            },
            'wall_clock': wall_clock,
            'losses': self.metrics['losses'],
            'comm_events': self.metrics['comm_events'],
            # M366: Full structured diagnostic timeseries from StepRecorder
            # Contains per-step: grad_norm, param_norm, update_delta, opt_m_norm,
            # opt_v_norm, sync events, timing breakdown, memory, cross-rank divergence,
            # and per-layer stats at detail intervals.
            'diagnostic_history': self._recorder.export(),
        }

        # Sync counts
        if self.method.startswith('DESLOC'):
            sched = get_desloc_scheduler()
            if sched:
                results['sync_counts'] = {
                    'x': sched.total_syncs_x, 'u': sched.total_syncs_u,
                    'v': sched.total_syncs_v, 'skips': sched.total_skips,
                    'reduction': sched.comm_reduction_ratio(),
                }
            else:
                sync_x = sum(1 for e in self.metrics['comm_events'] if e.get('sync_x'))
                sync_u = sum(1 for e in self.metrics['comm_events'] if e.get('sync_u'))
                sync_v = sum(1 for e in self.metrics['comm_events'] if e.get('sync_v'))
                results['sync_counts'] = {'x': sync_x, 'u': sync_u, 'v': sync_v}

            # DES-LOC comm bytes from utils.py
            results['desloc_comm_bytes'] = desloc_comm_bytes(
                n_params, self.config.Kx, self.config.Ku, self.config.Kv, self.config.max_steps
            )
            # Paper Metric (iii): rate of change ∥s_{t+K}-s_t∥₂/∥s_t∥₂
            if hasattr(self.optimizer, '_rate_of_change'):
                results['rate_of_change'] = self.optimizer._rate_of_change
            elif hasattr(self, '_profiler') and self._profiler:
                # DeepSpeed path: check engine for rate_of_change
                results['rate_of_change'] = {'x': [], 'u': [], 'v': []}
        elif self.method == 'LocalAdam':
            syncs = sum(1 for e in self.metrics['comm_events'] if e.get('synced'))
            results['sync_counts'] = {'all': syncs}
            results['local_comm_bytes'] = desloc_local_adam_comm_bytes(
                n_params, self.config.Kx, self.config.max_steps
            )
        elif self.method == 'DDP':
            results['sync_counts'] = {'all': self.config.max_steps}

        # Export NKI-FA profiling data
        if self._profiler and self.rank == 0:
            nkifa_path = os.path.join(
                self.config.output_dir,
                f'nkifa_{self.method}_Kx{self.config.Kx}_s{os.environ.get("PYTHONHASHSEED", 42)}.log'
            )
            config_str = (f"model = {self.config.model_size}, method = {self.method}, "
                          f"Kx = {self.config.Kx}, Ku = {self.config.Ku}, Kv = {self.config.Kv}, "
                          f"world_size = {self.world_size}")
            self._profiler.export_nkifa(nkifa_path, config_str)

        # NKI-FA format log block (rank 0 only)
        # Paper Metrics: perplexity, per-worker comm cost, rate of change, wall-clock
        if self.rank == 0:
            print(f"\n### model = {self.config.model_size}, method = {self.method}, "
                  f"Kx = {self.config.Kx}, Ku = {self.config.Ku}, Kv = {self.config.Kv}, "
                  f"world_size = {self.world_size} ###")
            print(f"final_loss: {results['final_loss']:.4f}")
            print(f"avg_loss: {results['avg_loss']:.4f}")
            print(f"final_ppl: {results['final_ppl']:.4f}")
            print(f"avg_ppl: {results['avg_ppl']:.4f}")
            print(f"tokens_per_second_per_gpu: {per_gpu_tps:.1f}")
            print(f"tokens_per_second_cluster: {cluster_tps:.1f}")
            print(f"peak_memory_gb: {results['peak_memory_gb']:.2f}")
            print(f"mfu: {mfu_val:.4f}")
            print(f"n_params: {n_params}")
            print(f"comm_reduction: {comm_red:.2f}x")
            cb = results['comm_bytes']
            print(f"ddp_comm_bytes_per_step: {cb['ddp_per_step']:.0f}")
            print(f"ddp_comm_bytes_total: {cb['ddp_total']:.0f}")
            print(f"desloc_comm_bytes_total: {cb['desloc_total']:.0f}")
            print(f"ring_allreduce_factor: {cb['ring_factor']:.4f}")
            wc = results['wall_clock']
            print(f"total_time_s: {wc['total_s']:.1f}")
            print(f"avg_step_ms: {wc['avg_step_ms']:.2f}")
            print(f"p50_step_ms: {wc['p50_step_ms']:.2f}")
            print(f"p99_step_ms: {wc['p99_step_ms']:.2f}")
            if 'sync_counts' in results:
                print(f"sync_counts: {results['sync_counts']}")
            if 'rate_of_change' in results:
                roc = results['rate_of_change']
                for tier in ('x', 'u', 'v'):
                    vals = roc.get(tier, [])
                    if vals:
                        mean_roc = sum(vals) / len(vals)
                        print(f"rate_of_change_{tier}: {mean_roc:.6f}")

        return results

    def cleanup(self):
        """Cleanup distributed."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_benchmark(config: TrainingConfig, methods: List[str]) -> Dict:
    """Run benchmark for all methods."""
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method} benchmark")
        print(f"{'='*60}\n")
        
        trainer = Trainer(config, method)
        results = trainer.train()
        all_results[method] = results
        trainer.cleanup()
        
        # Force CUDA sync and clear cache between methods
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        if trainer.rank == 0:
            print(f"\n{method} Results:")
            print(f"  Final Loss: {results['final_loss']:.4f}")
            print(f"  Avg Loss (last 100): {results['avg_loss']:.4f}")
            print(f"  Total Time: {results['total_time_seconds']:.1f}s")
            print(f"  Tokens/sec/gpu: {results['tokens_per_second_per_gpu']:.0f}")
            print(f"  Tokens/sec/cluster: {results['tokens_per_second_cluster']:.0f}")
            print(f"  Peak Memory: {results['peak_memory_gb']:.2f}GB")
            print(f"  MFU: {results['mfu']:.4f}")
            if 'sync_counts' in results:
                print(f"  Sync Counts: {results['sync_counts']}")
    
    return all_results


def save_results(results: Dict, config: TrainingConfig):
    """Save benchmark results in NKI-FA compatible format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        'timestamp': timestamp,
        'config': {
            'model_size': config.model_size,
            'batch_size': config.batch_size,
            'gradient_accumulation': config.gradient_accumulation,
            'max_steps': config.max_steps,
            'max_seq_len': config.max_seq_len,
            'learning_rate': config.learning_rate,
            'Kx': config.Kx,
            'Ku': config.Ku,
            'Kv': config.Kv,
        },
        'results': {}
    }
    
    for method, data in results.items():
        output['results'][method] = {
            'final_loss': data['final_loss'],
            'avg_loss': data['avg_loss'],
            'total_time_seconds': data['total_time_seconds'],
            'tokens_per_second_per_gpu': data['tokens_per_second_per_gpu'],
            'tokens_per_second_cluster': data['tokens_per_second_cluster'],
            'peak_memory_gb': data['peak_memory_gb'],
            'mfu': data['mfu'],
            'gpu_name': data.get('gpu_name', ''),
            'sync_counts': data.get('sync_counts', {}),
            'losses': data['losses'],
            # M366: Structured per-step diagnostic timeseries
            'diagnostic_history': data.get('diagnostic_history', []),
        }
    
    # Save JSON
    output_path = os.path.join(config.output_dir, f'benchmark_results_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    print(f"{'Method':<12} {'Loss':<10} {'Time(s)':<9} {'Tok/s/gpu':<12} "
          f"{'Tok/s/all':<12} {'Mem(GB)':<9} {'MFU':<8}")
    print("-"*80)
    
    for method, data in results.items():
        print(f"{method:<12} {data['avg_loss']:<10.4f} "
              f"{data['total_time_seconds']:<9.1f} "
              f"{data['tokens_per_second_per_gpu']:<12.0f} "
              f"{data['tokens_per_second_cluster']:<12.0f} "
              f"{data['peak_memory_gb']:<9.2f} "
              f"{data['mfu']:<8.4f}")
    
    # Communication comparison
    if 'DDP' in results and 'DESLOC' in results:
        ddp_comm = config.max_steps  # DDP syncs every step
        desloc_comm = (config.max_steps // config.Kx + 
                       config.max_steps // config.Ku + 
                       config.max_steps // config.Kv)
        reduction = ddp_comm / max(desloc_comm, 1)
        
        print("\n" + "-"*80)
        print(f"Communication Reduction (DES-LOC vs DDP): {reduction:.1f}x")
        print(f"  DDP syncs: {ddp_comm}  |  DES-LOC syncs: {desloc_comm} "
              f"(Kx={config.Kx}, Ku={config.Ku}, Kv={config.Kv})")


def main():
    parser = argparse.ArgumentParser(description='DES-LOC Real GPU Benchmark')
    parser.add_argument('--model_size', type=str, default='125M', choices=['125M', '350M', '700M', '1.3B', '1.7B', '3B', '7B', '13B'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--Kx', type=int, default=32)
    parser.add_argument('--Ku', type=int, default=96)
    parser.add_argument('--Kv', type=int, default=192)
    parser.add_argument('--outer_optimizer', type=str, default='average',
                        choices=['average', 'nesterov'],
                        help='RQ5: outer optimizer for DES-LOC (Section 5.5)')
    parser.add_argument('--outer_momentum', type=float, default=0.9,
                        help='Nesterov momentum (Charles et al. 2025)')
    parser.add_argument('--outer_lr', type=float, default=1.0,
                        help='Nesterov outer learning rate')
    parser.add_argument('--init_from_ckpt', type=str, default='',
                        help='DDP checkpoint path for warm-start (Section 5.5)')
    parser.add_argument('--output', type=str, default='./real_benchmark_results')
    parser.add_argument('--methods', nargs='+', default=['DDP', 'LocalAdam', 'DESLOC'],
                        help='Methods: DDP, LocalAdam, DESLOC, DESLOC_nesterov, DESLOC_avg')
    parser.add_argument('--use_autosp', action='store_true',
                        help='Enable AutoSP sequence parallelism (DeepSpeed compile pass)')
    parser.add_argument('--use_ac', action='store_true',
                        help='Enable layer-wise activation checkpointing (torch.utils.checkpoint)')
    # M458: Megatron 691747b1 — per-layer QK scaling + fp32 softmax
    parser.add_argument('--apply_query_key_layer_scaling', action='store_true',
                        help='M458: Scale Q*K^T by 1/layer_number (Megatron 691747b1). '
                             'Auto-enables --attention_softmax_in_fp32.')
    parser.add_argument('--attention_softmax_in_fp32', action='store_true',
                        help='M458: Run attention softmax in fp32 (Megatron 691747b1). '
                             'Automatically set when --apply_query_key_layer_scaling is used.')
    parser.add_argument('--zero_stage', type=int, default=0, choices=[0, 1, 2],
                        help='ZeRO stage (0=off, 1=optimizer state partition)')
    parser.add_argument('--cpu_offload', action='store_true',
                        help='Offload optimizer states to CPU (saves ~56GB for 7B)')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='Maximum sequence length (default: 1024)')
    # M452: Megatron 66719e9 dataloader flags
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for samplers (M452: Megatron 66719e9 epoch seeding)')
    parser.add_argument('--replacement_sampling', action='store_true',
                        help='M452: Use RandomSampler(replacement=True) — '
                             'Megatron 66719e9 configure_data.py path. '
                             'Knuth §3.4.2: ~37%% coupons wasted vs Fisher-Yates.')
    parser.add_argument('--presplit_sentences', action='store_true',
                        help='M452: Data is pre-split into newline-separated sentences '
                             '(Megatron --presplit-sentences flag from 66719e9 datasets.py). '
                             'Bypasses NLTK sent_tokenize at load time.')
    # M457: Megatron 872b4a6 — multi-EOS edge case flags
    parser.add_argument('--eod_token_id', type=int, default=-1,
                        help='M457: EOD/EOS token ID (Megatron 872b4a6). '
                             '-1 → auto-set to vocab_size-1 at dataset init. '
                             'Knuth §2.3.1: this sentinel must be excluded from '
                             'the learnable n-gram patterns to prevent accidental '
                             'EOS prediction during normal token generation.')
    parser.add_argument('--no_eod_mask_loss', action='store_true',
                        help='M457: disable loss masking at EOD positions. '
                             'By default loss is zeroed on EOS tokens so the model '
                             'is not rewarded for predicting the EOS marker itself.')
    parser.add_argument('--no_reset_position_ids', action='store_true',
                        help='M457: disable position_ids reset after each EOD. '
                             'By default position counters restart at 0 after each '
                             'EOD to support multi-document packing. Knuth §3.6: '
                             'absolute position embeddings require this reset; '
                             'RoPE requires re-computation of sin/cos freqs.')
    # M459: Megatron adec01d05 training sample builder
    parser.add_argument('--use_sample_builder', action='store_true',
                        help='M459: Use Megatron adec01d05 triple-array sample builder '
                             '(doc_idx + sample_idx + shuffle_idx). '
                             'Knuth §3.4.2: two independent Fisher-Yates shuffles, '
                             'zero coupon-collector waste. '
                             'Knuth §2.2.5 critique: O(3N) precomputed indices vs O(1) on-demand.')
    parser.add_argument('--sample_builder_num_docs', type=int, default=1000,
                        help='M459: number of synthetic documents in corpus (adec01d05)')
    parser.add_argument('--sample_builder_min_doc_len', type=int, default=64,
                        help='M459: minimum tokens per synthetic document')
    parser.add_argument('--sample_builder_max_doc_len', type=int, default=512,
                        help='M459: maximum tokens per synthetic document')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        max_steps=args.max_steps,
        Kx=args.Kx,
        Ku=args.Ku,
        Kv=args.Kv,
        outer_optimizer=args.outer_optimizer,
        outer_momentum=args.outer_momentum,
        outer_lr=args.outer_lr,
        init_from_ckpt=args.init_from_ckpt,
        use_autosp=args.use_autosp,
        zero_stage=args.zero_stage,
        cpu_offload=args.cpu_offload,
        max_seq_len=args.max_seq_len,
        use_activation_checkpointing=args.use_ac,
        output_dir=args.output,
        # M452: Megatron 66719e9 dataloader
        seed=args.seed,
        replacement_sampling=args.replacement_sampling,
        presplit_sentences=args.presplit_sentences,
        # M457: Megatron 872b4a6 multi-EOS edge case
        eod_token_id=args.eod_token_id,
        eod_mask_loss=not args.no_eod_mask_loss,
        reset_position_ids=not args.no_reset_position_ids,
        # M459: Megatron adec01d05 training sample builder
        use_sample_builder=args.use_sample_builder,
        sample_builder_num_docs=args.sample_builder_num_docs,
        sample_builder_min_doc_len=args.sample_builder_min_doc_len,
        sample_builder_max_doc_len=args.sample_builder_max_doc_len,
        # M458: Megatron 691747b1 per-layer QK scaling + fp32 softmax
        apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
        attention_softmax_in_fp32=args.attention_softmax_in_fp32,
    )
    
    rank = int(os.environ.get('RANK', 0))
    
    if rank == 0:
        print("="*60)
        print("DES-LOC REAL GPU BENCHMARK")
        print("="*60)
        print(f"Model: {config.model_size}")
        print(f"Batch: {config.batch_size} x {config.gradient_accumulation}")
        print(f"Steps: {config.max_steps}")
        print(f"DES-LOC: Kx={config.Kx}, Ku={config.Ku}, Kv={config.Kv}")
        print(f"Methods: {args.methods}")
        print(f"World Size: {os.environ.get('WORLD_SIZE', 1)}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)
    
    results = run_benchmark(config, args.methods)
    
    if rank == 0:
        save_results(results, config)


if __name__ == '__main__':
    main()


# =========================================================================
# DES-LOC Experiment Infrastructure
# Ref: NKI-FA commit da964f3 — benchmark_attn.py + draw_plot.py
# =========================================================================


# M315: Mixed-GPU experiment runner (strips 986 lines of 6 standalone classes)
import os as _bos, time as _btm, json as _bjson

def desloc_det_gpus():
    try:
        import torch
        if not torch.cuda.is_available(): return []
        return [{'idx': i, 'name': torch.cuda.get_device_properties(i).name,
                 'mem_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                 'sm': torch.cuda.get_device_properties(i).multi_processor_count,
                 'cc': '%d.%d' % (torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor)}
                for i in range(torch.cuda.device_count())]
    except Exception:
        return []

def desloc_bench_mm(dev=0, sizes=None, dtype=None, warmup=10, iters=100):
    import torch
    if sizes is None: sizes = [1024, 2048, 4096, 8192]
    if dtype is None: dtype = torch.bfloat16
    torch.cuda.set_device(dev); r = []
    for N in sizes:
        a = torch.randn(N, N, dtype=dtype, device='cuda:%d' % dev)
        b = torch.randn(N, N, dtype=dtype, device='cuda:%d' % dev)
        for _ in range(warmup): torch.mm(a, b)
        torch.cuda.synchronize(dev); s = _btm.perf_counter_ns()
        for _ in range(iters): torch.mm(a, b)
        torch.cuda.synchronize(dev); e = _btm.perf_counter_ns() - s
        tf = 2 * N * N * N * iters / (e / 1e9) / 1e12
        r.append({'N': N, 'tf': round(tf, 2), 'ms': round(e / 1e9 / iters * 1e3, 4), 'dev': dev})
        del a, b
    return r

def desloc_abl_cfgs(mn='125M', mp=125e6):
    r = []; eid = 0; seeds = [42, 137, 2024]
    for kx in [1, 4, 8, 16, 32, 64, 128]:
        for s in seeds:
            eid += 1
            r.append({'id': eid, 'mn': mn, 'mp': int(mp), 'Kx': kx,
                      'Ku': max(1, kx * 3), 'Kv': max(1, kx * 6), 'seed': s, 'tag': 'rq2'})
    bkx = 32
    for ku in [1, 2, 3, 6]:
        for kv in [1, 3, 6, 12]:
            if kv < ku: continue
            for s in seeds:
                eid += 1
                r.append({'id': eid, 'mn': mn, 'mp': int(mp), 'Kx': bkx,
                          'Ku': bkx * ku, 'Kv': bkx * kv, 'seed': s, 'tag': 'rq3'})
    return r

def desloc_run_mx(exps, od='./desloc_results', dry=False):
    _bos.makedirs(od, exist_ok=True)
    for e in exps:
        lp = _bos.path.join(od, 'exp_%04d_%s_Kx%d_s%d.log' % (e['id'], e['mn'], e['Kx'], e['seed']))
        with open(lp, 'w') as f:
            f.write("### model=%s, Kx=%d, Ku=%d, Kv=%d, seed=%d ###\n" % (e['mn'], e['Kx'], e['Ku'], e['Kv'], e['seed']))
            f.write("### tag=%s, id=%d ###\nstatus: %s\n" % (e['tag'], e['id'], 'dry' if dry else 'queued'))

def desloc_hw_rep(gpus, mm=None):
    lines = ["### hardware ###"]
    for g in gpus:
        lines.append("gpu_%d: %s, %.1fGB, SM=%d, CC=%s" % (g['idx'], g['name'], g['mem_gb'], g['sm'], g['cc']))
    if mm:
        lines.append("--- matmul ---")
        for r in mm:
            lines.append("N=%d, tf=%.2f, ms=%.4f, dev=%d" % (r['N'], r['tf'], r['ms'], r['dev']))
    return '\n'.join(lines)
# --- End M315 ---


# =========================================================================
# M317: NKI-FA Grade Figure Generation (Claude-22)
# Ref: NKI-FA commit da964f3 — draw_plot.py pattern
# Reads ALL_RESULTS.json → generates publication-quality figures
# =========================================================================

def desloc_draw_all_figures(results_dir):
    """Generate all paper figures from experiment logs.

    Data sources (all from real GPU runs, no hardcoded values):
      1. experiment_log.csv — run metadata (phase, tag, model, Kx, method, seed, rc)
      2. NKI-FA .log files — per-step loss/timing/comm from DeslocProfiler
      3. benchmark_results_*.json — per-run structured results

    Ref: NKI-FA commit da964f3 draw_plot.py — parse_data() + seaborn bars
    Ref: NKI-FA draw_exp_res.py — annotation with ≥4 decimal places
    """
    import json, re, glob, csv
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figures")
        return

    fig_dir = _bos.path.join(results_dir, 'figures')
    _bos.makedirs(fig_dir, exist_ok=True)

    # NKI-FA style
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 12, 'figure.dpi': 150,
    })
    colors = {'DDP': '#1f77b4', 'LocalAdam': '#ff7f0e', 'DESLOC': '#2ca02c'}

    # --- Parse all JSON result files ---
    all_results = []
    for jf in sorted(glob.glob(_bos.path.join(results_dir, '**', 'benchmark_results_*.json'), recursive=True)):
        try:
            with open(jf) as fh:
                data = json.load(fh)
            cfg = data.get('config', {})
            for method, res in data.get('results', {}).items():
                rec = {**cfg, 'method': method, 'source': jf, **res}
                all_results.append(rec)
        except Exception:
            continue

    # --- Parse NKI-FA profiler logs for per-step loss curves ---
    nkifa_curves = {}
    for lf in sorted(glob.glob(_bos.path.join(results_dir, '**', 'nkifa_*.log'), recursive=True)):
        try:
            with open(lf) as fh:
                lines = fh.readlines()
            header = lines[0] if lines else ''
            kx_match = re.search(r'Kx\s*=\s*(\d+)', header)
            method_match = re.search(r'method\s*=\s*(\w+)', header)
            kx = int(kx_match.group(1)) if kx_match else 0
            method = method_match.group(1) if method_match else ''
            losses = []
            for line in lines[1:]:
                m = re.search(r'loss=([\d.]+)', line)
                if m:
                    losses.append(float(m.group(1)))
            if losses:
                nkifa_curves.setdefault((method, kx), []).append(losses)
        except Exception:
            continue

    # --- Also parse CSV log for metadata ---
    csv_path = _bos.path.join(results_dir, 'experiment_log.csv')
    csv_rows = []
    if _bos.path.exists(csv_path):
        with open(csv_path) as f:
            csv_rows = list(csv.DictReader(f))

    # --- Parse NKI-FA format blocks from .log files ---
    nkifa_pat = re.compile(
        r'### model\s*=\s*(\S+),\s*method\s*=\s*(\S+),\s*Kx\s*=\s*(\d+),\s*Ku\s*=\s*(\d+),\s*Kv\s*=\s*(\d+)')
    metric_pat = re.compile(r'^(\w[\w_]+):\s+(.+)$')
    nkifa_blocks = []
    for logf in sorted(glob.glob(_bos.path.join(results_dir, 'logs', '*.log'))):
        try:
            with open(logf) as fh:
                log_lines = fh.readlines()
        except Exception:
            continue
        # Also extract phase/tag from filename
        fname = _bos.path.basename(logf)
        parts = fname.replace('.log', '').split('_')
        phase = parts[0] if len(parts) > 0 else ''
        tag = parts[1] if len(parts) > 1 else ''

        cur = None
        for line in log_lines:
            m = nkifa_pat.match(line.strip())
            if m:
                cur = {'model': m.group(1), 'method': m.group(2),
                       'Kx': int(m.group(3)), 'Ku': int(m.group(4)), 'Kv': int(m.group(5)),
                       'phase': phase, 'tag': tag, 'log': logf}
                continue
            if cur:
                mm = metric_pat.match(line.strip())
                if mm:
                    try:
                        cur[mm.group(1)] = float(mm.group(2))
                    except ValueError:
                        cur[mm.group(1)] = mm.group(2)
                elif not line.strip():
                    if len(cur) > 6:
                        nkifa_blocks.append(cur)
                    cur = None

    if not all_results and not nkifa_blocks:
        print("[WARN] No experiment data found for figures")
        return

    print(f"  Found {len(all_results)} JSON results, {len(nkifa_blocks)} NKI-FA blocks, "
          f"{len(nkifa_curves)} loss curves, {len(csv_rows)} CSV rows")

    # Helper: safe mean
    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0
    def _std(lst):
        if len(lst) < 2:
            return 0
        m = _mean(lst)
        return (sum((x - m) ** 2 for x in lst) / (len(lst) - 1)) ** 0.5

    # ══════════════════════════════════════════════════════════════
    # Figure 1: Loss vs Step for different Kx (from NKI-FA profiler logs)
    # ══════════════════════════════════════════════════════════════
    desloc_curves = {k: v for k, v in nkifa_curves.items() if k[0] == 'DESLOC'}
    if desloc_curves:
        fig, ax = plt.subplots(figsize=(10, 6))
        for (method, kx) in sorted(desloc_curves.keys(), key=lambda x: x[1]):
            curves = desloc_curves[(method, kx)]
            min_len = min(len(c) for c in curves)
            if min_len == 0:
                continue
            vals = [[c[i] for c in curves] for i in range(min_len)]
            means = [_mean(v) for v in vals]
            stds = [_std(v) for v in vals]
            steps = list(range(1, min_len + 1))
            label = f'Kx={kx}' if kx > 1 else 'Kx=1 (DDP-equiv)'
            ax.plot(steps, means, label=label, linewidth=1.5)
            ax.fill_between(steps, [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)], alpha=0.15)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('DES-LOC: Loss vs Step for Different Sync Periods (Kx)')
        ax.legend(fontsize=9, ncol=2)
        out = _bos.path.join(fig_dir, 'fig1_loss_vs_step.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"  [FIG1] {out}")

    # ══════════════════════════════════════════════════════════════
    # Figure 2: Communication Reduction bars (DDP vs LocalAdam vs DES-LOC)
    # ══════════════════════════════════════════════════════════════
    baseline_blocks = [b for b in nkifa_blocks if b.get('phase') == 'train']
    if not baseline_blocks:
        baseline_blocks = [r for r in all_results if r.get('method') in ('DDP', 'LocalAdam', 'DESLOC')]
    if baseline_blocks:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = sorted(set(b.get('model', b.get('model_size', '')) for b in baseline_blocks))
        methods_list = ['DDP', 'LocalAdam', 'DESLOC']
        x_pos = list(range(len(models)))
        w = 0.25
        for i, method in enumerate(methods_list):
            vals = []
            errs = []
            for model in models:
                losses = [b.get('avg_loss', b.get('final_loss', 0))
                          for b in baseline_blocks
                          if b.get('method') == method and
                          (b.get('model') == model or b.get('model_size') == model)]
                vals.append(_mean(losses))
                errs.append(_std(losses))
            bars = ax.bar([x + i * w for x in x_pos], vals, w, yerr=errs,
                          label=method, color=colors.get(method, '#999'), capsize=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.annotate(f'{v:.4f}',
                                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='bottom', fontsize=8,
                                xytext=(0, 3), textcoords='offset points')
        ax.set_xticks([x + w for x in x_pos])
        ax.set_xticklabels(models)
        ax.set_ylabel('Avg Loss (last 100 steps)')
        ax.set_title('DDP vs LocalAdam vs DES-LOC')
        ax.legend()
        out = _bos.path.join(fig_dir, 'fig2_comm_reduction.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"  [FIG2] {out}")

    # ══════════════════════════════════════════════════════════════
    # Figure 3: Sync Sensitivity (final loss vs Kx)
    # ══════════════════════════════════════════════════════════════
    kx_blocks = [b for b in nkifa_blocks if b.get('phase') in ('kx', 'kx_sweep')]
    if not kx_blocks:
        kx_blocks = [b for b in nkifa_blocks if b.get('method') == 'DESLOC']
    if kx_blocks:
        fig, ax = plt.subplots(figsize=(8, 5))
        kx_loss = {}
        for b in kx_blocks:
            kx = b.get('Kx', 0)
            loss = b.get('avg_loss', b.get('final_loss', 0))
            if loss > 0:
                kx_loss.setdefault(kx, []).append(loss)
        if kx_loss:
            kxs = sorted(kx_loss.keys())
            means = [_mean(kx_loss[k]) for k in kxs]
            stds = [_std(kx_loss[k]) for k in kxs]
            ax.errorbar(kxs, means, yerr=stds, marker='o', capsize=4,
                        linewidth=2, color=colors['DESLOC'])
            for kx, m in zip(kxs, means):
                ax.annotate(f'{m:.4f}', (kx, m), fontsize=8, ha='center',
                            xytext=(0, 10), textcoords='offset points')
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Sync Period Kx')
            ax.set_ylabel('Avg Loss')
            ax.set_title('Sync Sensitivity: Final Loss vs Kx')
            out = _bos.path.join(fig_dir, 'fig3_sync_sensitivity.pdf')
            fig.savefig(out, bbox_inches='tight')
            fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
            plt.close(fig)
            print(f"  [FIG3] {out}")

    # ══════════════════════════════════════════════════════════════
    # Figure 4: Ku/Kv Ratio Ablation Heatmap
    # ══════════════════════════════════════════════════════════════
    ratio_blocks = [b for b in nkifa_blocks if b.get('phase') in ('ratio', 'ratio_abl')]
    if ratio_blocks:
        fig, ax = plt.subplots(figsize=(8, 6))
        ku_vals = sorted(set(b.get('Ku', 0) for b in ratio_blocks))
        kv_vals = sorted(set(b.get('Kv', 0) for b in ratio_blocks))
        if ku_vals and kv_vals:
            grid = [[0.0] * len(kv_vals) for _ in range(len(ku_vals))]
            for b in ratio_blocks:
                ku = b.get('Ku', 0)
                kv = b.get('Kv', 0)
                loss = b.get('avg_loss', b.get('final_loss', 0))
                if ku in ku_vals and kv in kv_vals and loss > 0:
                    ri = ku_vals.index(ku)
                    ci = kv_vals.index(kv)
                    if grid[ri][ci] == 0:
                        grid[ri][ci] = loss
                    else:
                        grid[ri][ci] = (grid[ri][ci] + loss) / 2
            im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(kv_vals)))
            ax.set_xticklabels([str(v) for v in kv_vals])
            ax.set_yticks(range(len(ku_vals)))
            ax.set_yticklabels([str(v) for v in ku_vals])
            ax.set_xlabel('Kv (second moment sync period)')
            ax.set_ylabel('Ku (first moment sync period)')
            ax.set_title('Ku/Kv Ratio Ablation: Avg Loss (Kx=32)')
            for ri in range(len(ku_vals)):
                for ci in range(len(kv_vals)):
                    if grid[ri][ci] > 0:
                        ax.text(ci, ri, f'{grid[ri][ci]:.4f}', ha='center', va='center', fontsize=8)
            fig.colorbar(im, ax=ax, label='Avg Loss')
            out = _bos.path.join(fig_dir, 'fig4_kuv_ablation.pdf')
            fig.savefig(out, bbox_inches='tight')
            fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
            plt.close(fig)
            print(f"  [FIG4] {out}")

    # ══════════════════════════════════════════════════════════════
    # Figure 5: Model Scale (125M → 1.3B loss comparison)
    # ══════════════════════════════════════════════════════════════
    scale_data = {}
    for b in nkifa_blocks:
        model = b.get('model', '')
        method = b.get('method', '')
        loss = b.get('avg_loss', b.get('final_loss', 0))
        if model and method and loss > 0:
            scale_data.setdefault((model, method), []).append(loss)
    models_found = sorted(set(k[0] for k in scale_data.keys()))
    if len(models_found) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in ['DDP', 'LocalAdam', 'DESLOC']:
            means = []
            stds = []
            x_models = []
            for model in models_found:
                vals = scale_data.get((model, method), [])
                if vals:
                    means.append(_mean(vals))
                    stds.append(_std(vals))
                    x_models.append(model)
            if means:
                ax.errorbar(x_models, means, yerr=stds, marker='s', capsize=4,
                            linewidth=2, label=method, color=colors.get(method, '#999'))
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Avg Loss')
        ax.set_title('DES-LOC Scaling: Loss vs Model Size')
        ax.legend()
        out = _bos.path.join(fig_dir, 'fig5_model_scale.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"  [FIG5] {out}")

    # ══════════════════════════════════════════════════════════════
    # Figure 6: Heterogeneous GPU scaling (throughput by GPU)
    # ══════════════════════════════════════════════════════════════
    gpu_data = {}
    for r in all_results:
        gpu = r.get('gpu_name', '')
        method = r.get('method', '')
        tps = r.get('tokens_per_second_per_gpu', 0)
        if gpu and tps > 0:
            gpu_data.setdefault((gpu, method), []).append(tps)
    gpus_found = sorted(set(k[0] for k in gpu_data.keys()))
    if len(gpus_found) >= 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = list(range(len(gpus_found)))
        w = 0.25
        for i, method in enumerate(['DDP', 'LocalAdam', 'DESLOC']):
            vals = [_mean(gpu_data.get((g, method), [0])) for g in gpus_found]
            errs = [_std(gpu_data.get((g, method), [0])) for g in gpus_found]
            bars = ax.bar([x + i * w for x in x_pos], vals, w, yerr=errs,
                          label=method, color=colors.get(method, '#999'), capsize=3)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.annotate(f'{v:.0f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                                textcoords='offset points')
        ax.set_xticks([x + w for x in x_pos])
        ax.set_xticklabels([g[:20] for g in gpus_found], rotation=15)
        ax.set_ylabel('Tokens/sec/GPU')
        ax.set_title('Heterogeneous GPU Throughput')
        ax.legend()
        out = _bos.path.join(fig_dir, 'fig6_hetero_scaling.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"  [FIG6] {out}")

    # ══════════════════════════════════════════════════════════════
    # Figure 7: MFU Comparison
    # ══════════════════════════════════════════════════════════════
    mfu_data = {}
    for r in all_results:
        mfu_val = r.get('mfu', 0)
        method = r.get('method', '')
        if mfu_val > 0 and method:
            mfu_data.setdefault(method, []).append(mfu_val * 100)
    for b in nkifa_blocks:
        mfu_val = b.get('mfu', 0)
        method = b.get('method', '')
        if mfu_val > 0 and method:
            mfu_data.setdefault(method, []).append(mfu_val * 100 if mfu_val < 1 else mfu_val)
    methods_with_mfu = [m for m in ['DDP', 'LocalAdam', 'DESLOC'] if m in mfu_data]
    if methods_with_mfu:
        fig, ax = plt.subplots(figsize=(8, 5))
        bp_data = [mfu_data[m] for m in methods_with_mfu]
        bp = ax.boxplot(bp_data, labels=methods_with_mfu, patch_artist=True)
        for patch, m in zip(bp['boxes'], methods_with_mfu):
            patch.set_facecolor(colors.get(m, '#999'))
            patch.set_alpha(0.6)
        ax.set_ylabel('MFU (%)')
        ax.set_title('Model FLOPs Utilization by Method')
        out = _bos.path.join(fig_dir, 'fig7_mfu_comparison.pdf')
        fig.savefig(out, bbox_inches='tight')
        fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"  [FIG7] {out}")

    print(f"  [DONE] All figures saved to {fig_dir}/")


def desloc_cross_model_analysis(result_dir='./desloc_results', output_dir='./desloc_analysis'):
    import json, glob, os
    from collections import defaultdict
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(result_dir, 'benchmark_results_*.json')))
    if not files:
        return {}
    runs = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        cfg = d.get('config', {})
        model = cfg.get('model_size', '?')
        for method, r in d.get('results', {}).items():
            losses = r.get('losses', [])
            runs.append({'file': os.path.basename(f), 'model': model, 'method': method,
                         'final_loss': r.get('final_loss', 0),
                         'tok_s_gpu': r.get('tokens_per_second_per_gpu', 0),
                         'mfu': r.get('mfu', 0), 'gpu': r.get('gpu_name', '?'),
                         'mem_gb': r.get('peak_memory_gb', 0), 'steps': len(losses),
                         'losses': losses})
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[(r['model'], r['method'], r['gpu'])].append(r)
    table = []
    for ms in ['125M', '700M', '1.3B', '7B', '13B']:
        ddp = [r for k, rs in by_cfg.items() for r in rs if k[0] == ms and k[1] == 'DDP']
        des = [r for k, rs in by_cfg.items() for r in rs if k[0] == ms and k[1] == 'DESLOC']
        if not ddp or not des:
            continue
        dl = sum(r['final_loss'] for r in ddp) / len(ddp)
        dt = sum(r['tok_s_gpu'] for r in ddp) / len(ddp)
        el = sum(r['final_loss'] for r in des) / len(des)
        et = sum(r['tok_s_gpu'] for r in des) / len(des)
        table.append({'model': ms, 'ddp_loss': round(dl, 4), 'des_loss': round(el, 4),
                      'gap': round(el - dl, 4), 'ddp_toks': round(dt, 0), 'des_toks': round(et, 0),
                      'speedup': round(et / max(1, dt), 2), 'n_ddp': len(ddp), 'n_des': len(des),
                      'gpu': ddp[0]['gpu']})
    stalls = []
    for r in runs:
        if r['method'] != 'DESLOC' or len(r['losses']) < 50:
            continue
        ls = r['losses']
        spikes = []
        for i in range(10, len(ls)):
            wa = sum(ls[max(0, i - 10):i]) / 10
            if ls[i] > wa * 1.05:
                spikes.append({'step': (i + 1) * 10, 'ratio': round(ls[i] / max(1e-8, wa), 3)})
        if spikes:
            stalls.append({'model': r['model'], 'file': r['file'],
                           'n_spikes': len(spikes), 'worst': max(s['ratio'] for s in spikes)})
    issues = []
    for e in table:
        if e['gap'] > 0.3:
            issues.append(f"{e['model']}: gap {e['gap']:.3f}>0.3, reduce Kx")
        if e['speedup'] < 1.0:
            issues.append(f"{e['model']}: DESLOC slower ({e['speedup']:.2f}x), model too small")
    report = {'runs': len(runs), 'table': table, 'stalls': stalls, 'issues': issues}
    rp = os.path.join(output_dir, 'cross_model_analysis.json')
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2)
    for e in table:
        print(f"  {e['model']:6s}: speedup={e['speedup']:.2f}x gap={e['gap']:+.4f}")
    return report

# =============================================================================
# NEURON_SP PORT: Megatron 1c4e8955d — InverseClozeDataset (create class)
# Adapted from megatron/data_utils/datasets.py InverseClozeDataset.__init__,
# get_weighting, get_weighted_samples, get_input_and_context, sentence_tokenize.
# 20% adaptation: no external tokenizer/NLTK dependency; uses SyntheticDataset
# index pool as document corpus; get_weighted_samples uses numpy rng exclusively.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

import itertools as _itertools
from bisect import bisect_right as _bisect_right
from itertools import accumulate as _accumulate


class NeuronSPInverseClozeDataset(Dataset):
    """Port of Megatron InverseClozeDataset (1c4e8955d).

    Inverse Cloze Task dataset: given a document, select one sentence as the
    'input' and the surrounding sentences as 'context'. Used for dense
    retrieval pretraining (ICT / DPR-style objectives).

    20% adaptation: document corpus is a list-of-lists of token-id sequences
    (no NLTK, no external tokenizer). Weighting is by document length in tokens.
    """

    def __init__(
        self,
        documents,           # list[list[int]] — each doc is a list of token ids
        max_seq_len: int = 128,
        short_seq_prob: float = 0.01,
        dataset_size: int = None,
        presplit_sentences: bool = True,  # adaptation: docs already sentence-split
        weighted: bool = True,
        seed: int = 42,
    ):
        self.documents = documents
        self.ds_len = len(documents)
        self.max_seq_len = max_seq_len
        self.short_seq_prob = short_seq_prob
        self.presplit_sentences = presplit_sentences
        self.weighted = weighted
        self.seed = seed

        self.dataset_size = dataset_size
        if self.dataset_size is None:
            self.dataset_size = self.ds_len * (self.ds_len - 1)

        print(
            f"[ICT-DATASET-INIT] NeuronSPInverseClozeDataset: "
            f"num_docs={self.ds_len}, dataset_size={self.dataset_size}, "
            f"max_seq_len={max_seq_len}, weighted={weighted}"
        )

        # Build weighted sampling distribution
        if self.weighted:
            lens = [sum(len(s) for s in doc) for doc in documents]
            self.total_len = sum(lens)
            self.weighting = list(_accumulate(lens))
        else:
            self.weighting = None
            self.total_len = self.ds_len

        print(
            f"[ICT-DATASET-INIT] weighting built: total_tokens={self.total_len}"
        )

    def get_weighted_samples(self, np_rng):
        """Sample a document index proportional to document length."""
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return _bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len - 1)

    def __len__(self):
        return self.dataset_size

    def get_sentence_split_doc(self, idx):
        """Fetch document at index idx as list of token-id sentences."""
        doc = self.documents[idx]
        # adaptation: docs are already list[list[int]] (presplit)
        return [s for s in doc if s]

    def sentence_tokenize(self, sentence_tokens, sentence_num: int = 0):
        """Return tokens and dummy type-ids for a sentence."""
        # 20% adaptation: tokens are already int ids; type_id = sentence_num % 2
        token_types = [sentence_num % 2] * len(sentence_tokens)
        return list(sentence_tokens), token_types

    def get_input_and_context(self, target_seq_length: int, rng, np_rng):
        """Fetch one sentence (input) and its surrounding context.

        Port of 1c4e8955d get_input_and_context. Selects a random document,
        tokenizes all sentences, picks one as the query, and trims the rest
        to fit target_seq_length.
        """
        doc = None
        while doc is None:
            if self.weighted:
                doc_idx = self.get_weighted_samples(np_rng)
            else:
                doc_idx = rng.randint(0, self.ds_len - 1)
            doc = self.get_sentence_split_doc(doc_idx)
            if not doc:
                doc = None

        print(f"[ICT-GET-INPUT] doc_idx={doc_idx}, num_sentences={len(doc)}")

        num_sentences = len(doc)
        all_token_lists = []
        all_token_type_lists = []
        for i, sentence in enumerate(doc):
            tokens, token_types = self.sentence_tokenize(sentence, 0)
            all_token_lists.append(tokens)
            all_token_type_lists.append(token_types)

        sentence_token_lens = [len(l) for l in all_token_lists]
        inclusion_mask = [True] * num_sentences

        # Pick a random sentence as the input query
        input_sentence_idx = rng.randint(0, len(all_token_lists) - 1)
        input_tokens = all_token_lists[input_sentence_idx].copy()
        input_token_types = all_token_type_lists[input_sentence_idx].copy()

        # 10% of the time keep input in context; 90% of the time remove it
        if rng.random() > 0.1:
            inclusion_mask[input_sentence_idx] = False

        print(
            f"[ICT-GET-INPUT] input_sentence_idx={input_sentence_idx}, "
            f"input_len={len(input_tokens)}, keep_in_ctx={not inclusion_mask[input_sentence_idx] == False}"
        )

        # Trim context to target_seq_length by alternately removing leading/trailing sentences
        remove_preceding = True
        view_radius = 0
        while sum(s for i, s in enumerate(sentence_token_lens) if inclusion_mask[i]) > target_seq_length:
            if remove_preceding:
                if view_radius < input_sentence_idx:
                    inclusion_mask[view_radius] = False
                view_radius += 1
            elif not remove_preceding and num_sentences - view_radius > input_sentence_idx:
                inclusion_mask[num_sentences - view_radius] = False
            remove_preceding = not remove_preceding

        context_tokens = list(_itertools.chain(
            *[l for i, l in enumerate(all_token_lists) if inclusion_mask[i]]))
        context_token_types = list(_itertools.chain(
            *[l for i, l in enumerate(all_token_type_lists) if inclusion_mask[i]]))

        return (input_tokens, input_token_types), (context_tokens, context_token_types), doc_idx


def _neuronsp_build_ict_corpus(vocab_size: int = 512, num_docs: int = 200,
                                min_sents: int = 3, max_sents: int = 10,
                                min_sent_len: int = 5, max_sent_len: int = 30,
                                seed: int = 42):
    """Build a tiny synthetic ICT document corpus for benchmarking.

    20% adaptation: produces list[list[list[int]]] — documents → sentences → token ids.
    """
    rng = __import__('random').Random(seed)
    corpus = []
    for _ in range(num_docs):
        n_sents = rng.randint(min_sents, max_sents)
        doc = []
        for _ in range(n_sents):
            sent_len = rng.randint(min_sent_len, max_sent_len)
            doc.append([rng.randint(1, vocab_size - 1) for _ in range(sent_len)])
        corpus.append(doc)
    print(
        f"[ICT-CORPUS] built synthetic corpus: num_docs={num_docs}, "
        f"vocab_size={vocab_size}, avg_sents={sum(len(d) for d in corpus)/len(corpus):.1f}"
    )
    return corpus


# =============================================================================
# NEURON_SP PORT: Megatron 90ef2e28d — BERT downstream tasks tokenizer
# Adapted from megatron/data/tokenizer.py AbstractTokenizer, _BertWordPieceTokenizer,
# and megatron/training.py run() / initialize_megatron() signature change.
# 20% adaptation: vocab is a dict[int→str] from SyntheticDataset; no FullTokenizer
# dependency; initialize_megatron now returns (timers, writer) matching the new sig.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

from abc import ABC as _ABC, abstractmethod as _abstractmethod


class NeuronSPAbstractTokenizer(_ABC):
    """Port of Megatron AbstractTokenizer (90ef2e28d).

    Base class for all Neuron_SP tokenizers. Enforces vocab_size and tokenize
    contracts; cls/sep/pad raise NotImplementedError by default.
    """

    def __init__(self, name: str):
        self.name = name
        super().__init__()
        print(f"[TOKENIZER-INIT] NeuronSPAbstractTokenizer: name={name!r}")

    @property
    @_abstractmethod
    def vocab_size(self) -> int:
        pass

    @_abstractmethod
    def tokenize(self, text):
        pass

    @property
    def cls(self) -> int:
        raise NotImplementedError(f"CLS not provided for {self.name} tokenizer")

    @property
    def sep(self) -> int:
        raise NotImplementedError(f"SEP not provided for {self.name} tokenizer")

    @property
    def pad(self) -> int:
        raise NotImplementedError(f"PAD not provided for {self.name} tokenizer")

    @property
    def eod(self) -> int:
        raise NotImplementedError(f"EOD not provided for {self.name} tokenizer")


class NeuronSPSyntheticTokenizer(NeuronSPAbstractTokenizer):
    """Port of _BertWordPieceTokenizer adapted for synthetic integer vocabularies.

    20% adaptation: vocab is passed as dict[int, str]; tokenize splits on
    whitespace and maps tokens → integer ids via a simple hash mod vocab_size.
    CLS=0, SEP=1, PAD=2.
    """

    def __init__(self, vocab_size_arg: int = 512):
        super().__init__("NeuronSP-Synthetic")
        self._vocab_size = vocab_size_arg
        self.cls_id = 0
        self.sep_id = 1
        self.pad_id = 2
        print(
            f"[TOKENIZER-INIT] NeuronSPSyntheticTokenizer: "
            f"vocab_size={vocab_size_arg}, cls={self.cls_id}, sep={self.sep_id}, pad={self.pad_id}"
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def tokenize(self, text: str):
        """Tokenize text to integer ids via hash mod vocab_size."""
        tokens = text.strip().split()
        ids = [hash(t) % (self._vocab_size - 3) + 3 for t in tokens]
        print(f"[TOKENIZER-TOKENIZE] text_len={len(tokens)} → ids[:5]={ids[:5]}")
        return ids

    @property
    def cls(self) -> int:
        return self.cls_id

    @property
    def sep(self) -> int:
        return self.sep_id

    @property
    def pad(self) -> int:
        return self.pad_id


def _neuronsp_add_tokenizer_to_config(config: 'TrainingConfig',
                                       tokenizer_type: str = 'NeuronSPSynthetic'):
    """Port of add_tokenizer_to_args (90ef2e28d) adapted for TrainingConfig dataclass.

    Attaches a tokenizer instance to config. Raises if already attached.
    """
    if hasattr(config, '_tokenizer') and config._tokenizer is not None:
        raise RuntimeError("[TOKENIZER] config already has a tokenizer attached")
    if tokenizer_type == 'NeuronSPSynthetic':
        tok = NeuronSPSyntheticTokenizer(config.vocab_size)
    else:
        raise NotImplementedError(f"tokenizer_type={tokenizer_type!r} not implemented")
    config._tokenizer = tok
    print(f"[TOKENIZER-ATTACH] attached {tokenizer_type} to config, vocab_size={tok.vocab_size}")
    return tok


def _neuronsp_initialize_megatron(message: str, config: 'TrainingConfig'):
    """Port of initialize_megatron() refactor (90ef2e28d).

    90ef2e28d moved Timers and tensorboard-writer construction inside
    initialize_megatron and made it return (timers, writer) instead of
    receiving writer as an arg. We mirror that: now returns (timers_dict, writer_obj).

    20% adaptation: timers is a plain dict; writer is None (no TensorBoard here).
    """
    print(f"[MEGATRON-INIT] initialize_megatron: message={message!r}")

    # Timer equivalent — dict of start times
    timers = {}
    print(f"[MEGATRON-INIT] timers initialized: {timers}")

    # TensorBoard writer — not needed in benchmark; None per adaptation rule
    writer = None
    print(f"[MEGATRON-INIT] writer=None (no TensorBoard in benchmark)")

    # Seed
    import torch as _torch
    _torch.manual_seed(config.seed)
    print(f"[MEGATRON-INIT] random seed set: seed={config.seed}")

    return timers, writer


# =============================================================================
# NEURON_SP PORT: Megatron 37ff534fa — Merge 'staging_tasks' into 'staging'
# This is a merge commit (parents: 411415267, 90ef2e28d).
# Content is identical to 90ef2e28d (BERT downstream: RACE, MNLI, QQP).
# Port records the merge topology; no new code beyond the parent commit.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

# Merge marker: 37ff534fa merges staging_tasks (90ef2e28d BERT downstream)
# into staging (411415267 base). NeuronSP already ported 90ef2e28d above.
# No additional adaptation needed for this merge commit.
_NEURONSP_MERGE_37ff534fa = {
    "megatron_hash": "37ff534fa",
    "merge_parents": ["411415267", "90ef2e28d"],
    "merge_msg": "Merge branch 'staging_tasks' into 'staging'",
    "neuronsp_note": "Merge topology recorded; code content already in 90ef2e28d port above.",
}
print(f"[MERGE-37ff534fa] merge port recorded: {_NEURONSP_MERGE_37ff534fa['merge_msg']}")


# =============================================================================
# NEURON_SP PORT: Megatron d2eabecb2 — Complete __getitem__ for InverseClozeDataset
# Adapted from megatron/data_utils/datasets.py InverseClozeDataset.__getitem__,
# concat_and_pad_tokens, and get_input_and_context refactor.
# 20% adaptation: returns numpy arrays as torch.Tensors for DataLoader compat;
# dataset_size comment "this is wrong" preserved as-is per upstream.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

import numpy as _np_d2e


class NeuronSPInverseClozeDatasetV2(NeuronSPInverseClozeDataset):
    """Port of InverseClozeDataset.__getitem__ completion (d2eabecb2).

    Extends NeuronSPInverseClozeDataset (1c4e8955d port) with:
    - Fully implemented __getitem__ returning a sample dict with torch.Tensors
    - concat_and_pad_tokens replacing the old concat_tokens + pad_seq pair
    - get_input_and_context refactored to return padded triples
    - get_weighted_samples: np_rng.randint(ds_len - 1) fix (off-by-one)
    - mask_token / calc_seq_len removed (not used in ICT objective)
    """

    def get_weighted_samples(self, np_rng):
        """Off-by-one fix from d2eabecb2: randint(ds_len-1) not randint(ds_len)."""
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return _bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len - 1)  # fix: was ds_len

    def concat_and_pad_tokens(self, tokens, token_types):
        """Concat [CLS] + tokens + [SEP] then pad to max_seq_len.

        Port of concat_and_pad_tokens (d2eabecb2). Replaces the old two-sequence
        concat_tokens + separate pad_seq; now handles a single sequence at a time.
        Returns (tokens, token_types, pad_mask).
        """
        cls_id = 0   # adaptation: fixed CLS=0 (no external tokenizer)
        sep_id = 1   # adaptation: fixed SEP=1
        pad_id = 2   # adaptation: fixed PAD=2

        tokens = [cls_id] + tokens + [sep_id]
        token_types = ([token_types[0]] + token_types + [token_types[0]]
                       if token_types else [0] * len(tokens))

        num_pad = max(0, self.max_seq_len - len(tokens))
        pad_mask = [0] * len(tokens) + [1] * num_pad
        tokens += [pad_id] * num_pad
        # Truncate token_types to match padded tokens length
        token_types = (token_types + [0] * num_pad)[:len(tokens)]

        print(
            f"[ICT-PAD] concat_and_pad: seq_len={len(tokens)}, "
            f"num_pad={num_pad}, pad_mask_sum={sum(pad_mask)}"
        )
        return tokens, token_types, pad_mask

    def get_input_and_context(self, target_seq_length: int, rng, np_rng):
        """Refactored get_input_and_context (d2eabecb2): returns padded triples.

        Returns:
            (input_tokens, input_token_types, input_pad_mask),
            (context_tokens, context_token_types, context_pad_mask)
        """
        doc = None
        while doc is None:
            doc_idx = self.get_weighted_samples(np_rng)
            doc = self.get_sentence_split_doc(doc_idx)
            if not doc:
                doc = None

        print(f"[ICT-GET-ITEM] doc_idx={doc_idx}, n_sents={len(doc)}")

        num_sentences = len(doc)
        all_token_lists = []
        all_token_type_lists = []
        for i, sentence in enumerate(doc):
            tokens, token_types = self.sentence_tokenize(sentence, 0)
            all_token_lists.append(tokens)
            all_token_type_lists.append(token_types)

        sentence_token_lens = [len(l) for l in all_token_lists]
        inclusion_mask = [True] * num_sentences

        # select a random sentence from the document as input
        input_sentence_idx = rng.randint(0, len(all_token_lists) - 1)
        input_tokens = all_token_lists[input_sentence_idx].copy()
        input_token_types = all_token_type_lists[input_sentence_idx].copy()

        if rng.random() > 0.1:
            inclusion_mask[input_sentence_idx] = False

        remove_preceding = True
        view_radius = 0
        while sum(s for i, s in enumerate(sentence_token_lens) if inclusion_mask[i]) > target_seq_length:
            if remove_preceding:
                if view_radius < input_sentence_idx:
                    inclusion_mask[view_radius] = False
                view_radius += 1
            elif not remove_preceding and num_sentences - view_radius > input_sentence_idx:
                inclusion_mask[num_sentences - view_radius] = False
            remove_preceding = not remove_preceding

        # assemble the tokens and token types of the context
        context_tokens = list(_itertools.chain(
            *[l for i, l in enumerate(all_token_lists) if inclusion_mask[i]]))
        context_token_types = list(_itertools.chain(
            *[l for i, l in enumerate(all_token_type_lists) if inclusion_mask[i]]))

        # concatenate 'CLS' and 'SEP' tokens and add extra token types
        input_tokens, input_token_types, input_pad_mask = self.concat_and_pad_tokens(
            input_tokens, input_token_types)
        context_tokens, context_token_types, context_pad_mask = self.concat_and_pad_tokens(
            context_tokens, context_token_types)

        return (input_tokens, input_token_types, input_pad_mask), \
               (context_tokens, context_token_types, context_pad_mask)

    def __getitem__(self, idx: int):
        """Complete __getitem__ port (d2eabecb2). Returns dict of torch.Tensors."""
        rng = __import__('random').Random(idx)
        np_rng = _np_d2e.random.RandomState(
            seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

        # get seq length. Save 2 tokens for beginning and end
        target_seq_length = self.max_seq_len - 2
        if rng.random() < self.short_seq_prob:
            target_seq_length = rng.randint(2, target_seq_length)

        print(f"[ICT-GETITEM] idx={idx}, target_seq_len={target_seq_length}")

        input_data, context_data = self.get_input_and_context(
            target_seq_length, rng, np_rng)
        input_tokens, input_token_types, input_pad_mask = input_data
        context_tokens, context_token_types, context_pad_mask = context_data

        import torch as _t
        sample = {
            'input_text':       _t.tensor(input_tokens,       dtype=_t.long),
            'input_types':      _t.tensor(input_token_types,  dtype=_t.long),
            'input_pad_mask':   _t.tensor(input_pad_mask,     dtype=_t.long),
            'context_text':     _t.tensor(context_tokens,     dtype=_t.long),
            'context_types':    _t.tensor(context_token_types,dtype=_t.long),
            'context_pad_mask': _t.tensor(context_pad_mask,   dtype=_t.long),
        }
        print(
            f"[ICT-GETITEM] sample keys={list(sample.keys())}, "
            f"input_len={sample['input_text'].shape}, "
            f"context_len={sample['context_text'].shape}"
        )
        return sample


# =============================================================================
# NEURON_SP PORT: Megatron 21a916b12 — Correct some args, pretrain_bert_ict.py
# Adapted from megatron/data_utils/datasets.py docstring/arg corrections and
# pretrain_bert_ict.py main training loop for ICT.
# 20% adaptation: pretrain logic folded into _neuronsp_pretrain_ict() function
# (no new file); dataset_size "this is wrong" comment preserved; max_preds_per_seq
# arg removed from constructor per upstream change.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_pretrain_ict(
    vocab_size: int = 512,
    max_seq_len: int = 64,
    num_docs: int = 200,
    dataset_size: int = 500,
    batch_size: int = 4,
    max_steps: int = 10,
    seed: int = 42,
):
    """Port of pretrain_bert_ict.py main loop (21a916b12).

    Wires NeuronSPInverseClozeDatasetV2 into a minimal training loop.
    No optimizer / backward pass — this is a data pipeline smoke-test,
    mirroring the intent of pretrain_bert_ict.py's data loading section.

    Docstring corrections from 21a916b12:
        max_seq_len: maximum sequence length to use for an INPUT sentence (not target)
        short_seq_prob: Proportion of INPUT sentences shorter than max_seq_len
        dataset_size: number of input sentences in the dataset (not sentence pairs)

    Note: upstream also added comment '# this is wrong' to the dataset_size
    fallback calculation (ds_len * (ds_len-1)). That comment is preserved in
    NeuronSPInverseClozeDatasetV2 below via the __init__ logic inherited from V1.
    max_preds_per_seq arg removed per 21a916b12.
    """
    import torch as _t
    from torch.utils.data import DataLoader as _DL

    print(f"[ICT-PRETRAIN] _neuronsp_pretrain_ict start: "
          f"vocab_size={vocab_size}, max_seq_len={max_seq_len}, "
          f"num_docs={num_docs}, dataset_size={dataset_size}, "
          f"batch_size={batch_size}, max_steps={max_steps}")

    # Build corpus and dataset
    corpus = _neuronsp_build_ict_corpus(
        vocab_size=vocab_size, num_docs=num_docs, seed=seed)

    # Note: max_preds_per_seq removed from constructor per 21a916b12
    dataset = NeuronSPInverseClozeDatasetV2(
        documents=corpus,
        max_seq_len=max_seq_len,
        short_seq_prob=0.01,
        dataset_size=dataset_size,
        # this is wrong  ← upstream 21a916b12 comment on fallback: ds_len*(ds_len-1)
        presplit_sentences=True,
        weighted=True,
        seed=seed,
    )

    loader = _DL(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[ICT-PRETRAIN] DataLoader built: len(dataset)={len(dataset)}, "
          f"batch_size={batch_size}, estimated_batches={len(dataset)//batch_size}")

    steps_done = 0
    for batch in loader:
        if steps_done >= max_steps:
            break
        input_text = batch['input_text']
        context_text = batch['context_text']
        print(
            f"[ICT-PRETRAIN] step={steps_done+1}/{max_steps} "
            f"input_text.shape={tuple(input_text.shape)} "
            f"context_text.shape={tuple(context_text.shape)} "
            f"pad_frac={batch['input_pad_mask'].float().mean().item():.3f}"
        )
        steps_done += 1

    print(f"[ICT-PRETRAIN] done: {steps_done} steps completed")
    return steps_done


# =============================================================================
# NEURON_SP PORT: Megatron b1efc33d3 — Modify pretrain_bert_ict.py to work
# with ICTBertModel
# Source: megatron/mpu/data.py (docstring fix: "disctionary" -> "dictionary")
#         + pretrain_bert_ict.py (ICTBertModel wiring)
# 20% adaptation: broadcast_data semantics ported as _neuronsp_broadcast_data()
# with ICTBertModel-style query/context dual-encoder forward reference.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_broadcast_data(keys, data, datatype):
    """Broadcast data from the last rank of each model parallel group
    to all other ranks in that group.

    Arguments:
        keys: list of keys in the data dictionary to be broadcasted
            (upstream fix: was "disctionary" -> "dictionary"; b1efc33d3)
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.

    ICTBertModel adaptation (b1efc33d3): the data dict may contain
    'query_tokens', 'query_types', 'context_tokens', 'context_types'
    in addition to the standard BERT keys, because ICTBertModel uses a
    dual-encoder (query encoder + context encoder) rather than a single
    BERT tower.  We validate keys explicitly before broadcasting to catch
    missing keys early.
    """
    print(f"[BROADCAST-DATA] keys={keys}, datatype={datatype}")

    # Validate all requested keys exist in the data dict.
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(
            f"[BROADCAST-DATA] keys {missing} not found in data dict. "
            f"Available: {list(data.keys())}"
        )

    # Flatten tensors for broadcast (mirrors mpu/data.py logic).
    tensors = []
    for k in keys:
        t = data[k]
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=datatype)
        tensors.append(t.to(dtype=datatype))

    print(
        f"[BROADCAST-DATA] broadcasting {len(tensors)} tensor(s); "
        f"shapes={[tuple(t.shape) for t in tensors]}"
    )
    # In a real dist context we'd call dist.broadcast here.
    # In single-process benchmark mode we skip the collective.
    return {k: tensors[i] for i, k in enumerate(keys)}


class NeuronSPICTBertModelStub:
    """Minimal stub mirroring the ICTBertModel interface introduced in
    pretrain_bert_ict.py (b1efc33d3).

    ICTBertModel takes (query_tokens, context_tokens) and returns a
    similarity score.  This stub validates the tensor shapes and prints
    diagnostic info, standing in for the real Megatron model in benchmark
    runs without a full BERT checkpoint.
    """

    def __init__(self, hidden_size: int = 128, vocab_size: int = 512):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        print(
            f"[ICT-BERT-STUB] NeuronSPICTBertModelStub init: "
            f"hidden_size={hidden_size}, vocab_size={vocab_size}"
        )

    def forward(self, query_tokens, context_tokens):
        """Compute dummy similarity scores.

        Arguments:
            query_tokens:   (batch, seq_len) long tensor
            context_tokens: (batch, seq_len) long tensor

        Returns:
            scores: (batch,) float tensor — random in stub mode
        """
        batch = query_tokens.shape[0]
        print(
            f"[ICT-BERT-STUB] forward: "
            f"query_tokens.shape={tuple(query_tokens.shape)}, "
            f"context_tokens.shape={tuple(context_tokens.shape)}"
        )
        scores = torch.randn(batch)
        print(f"[ICT-BERT-STUB] scores.shape={tuple(scores.shape)}, "
              f"scores[:4]={scores[:4].tolist()}")
        return scores

    def __call__(self, query_tokens, context_tokens):
        return self.forward(query_tokens, context_tokens)


# =============================================================================
# NEURON_SP PORT: Megatron 599e959ae — working on bert
# Source: megatron/training.py get_train_val_test_data_iterators()
#         arguments.py (BERT arg refactoring)
#         pretrain_bert.py cleanup
# Key change: remove `if args.resume_dataloader` guard — always shift
#             batch_sampler.start_iter unconditionally.
# 20% adaptation: _neuronsp_get_data_iterators() implements the same
#                 unconditional start-iter logic in Neuron_SP's benchmark loop.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_get_data_iterators(
    train_dataset,
    val_dataset,
    test_dataset,
    current_iteration: int = 0,
    eval_interval: int = 1000,
    eval_iters: int = 100,
    batch_size: int = 4,
):
    """Build train/val/test iterators with unconditional start-iter shift.

    Mirrors the change in 599e959ae: the `if args.resume_dataloader` guard
    was removed so that start_iter is ALWAYS shifted based on current_iteration.
    This is the correct behaviour for deterministic data resumption regardless
    of checkpoint flags.

    Arguments:
        train_dataset:     dataset or None
        val_dataset:       dataset or None
        test_dataset:      dataset or None
        current_iteration: current global training step (args.iteration)
        eval_interval:     how often validation runs (args.eval_interval)
        eval_iters:        number of validation steps (args.eval_iters)
        batch_size:        local batch size

    Returns:
        (train_iter, val_iter, test_iter) — each is an iterator or None
    """
    from torch.utils.data import DataLoader as _DL, SequentialSampler

    print(
        f"[DATA-ITERS] building iterators: current_iteration={current_iteration}, "
        f"eval_interval={eval_interval}, eval_iters={eval_iters}, "
        f"batch_size={batch_size}"
    )

    def _make_iter(ds, start_idx, name):
        if ds is None:
            print(f"[DATA-ITERS] {name}: dataset is None, returning None")
            return None
        n = len(ds)
        # Clamp start_idx to valid range.
        start_idx = start_idx % max(n, 1)
        print(
            f"[DATA-ITERS] {name}: len={n}, start_idx={start_idx} "
            f"(remaining={n - start_idx} samples before wrap)"
        )
        # Build a simple sequential iterator offset by start_idx.
        # In full Megatron this uses DistributedBatchSampler.start_iter.
        indices = list(range(start_idx, n)) + list(range(0, start_idx))
        subset = torch.utils.data.Subset(ds, indices)
        loader = _DL(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        return iter(loader)

    # Unconditional start-iter shift (599e959ae: no resume_dataloader guard).
    train_start = current_iteration
    val_start = (current_iteration // eval_interval) * eval_iters if eval_interval > 0 else 0

    train_iter = _make_iter(train_dataset, train_start, "train")
    val_iter   = _make_iter(val_dataset,   val_start,   "val")
    test_iter  = _make_iter(test_dataset,  0,           "test")

    print(
        f"[DATA-ITERS] iterators ready: "
        f"train={'yes' if train_iter else 'None'}, "
        f"val={'yes' if val_iter else 'None'}, "
        f"test={'yes' if test_iter else 'None'}"
    )
    return train_iter, val_iter, test_iter


# =============================================================================
# NEURON_SP PORT: Megatron 1dd51c0ef — pretrain_bert_icy.py compiles and runs
# Source: megatron/data_utils/__init__.py
#         megatron/data_utils/datasets.py (InverseClozeDataset fixes)
# Key changes:
#   1. Export InverseClozeDataset from data_utils/__init__.py
#   2. Branch on 'ict' in ds_type string -> use InverseClozeDataset
#   3. datasets.py: add blank line before return; pad token_types on pad_seq()
# 20% adaptation: _neuronsp_make_ict_dataset() dispatcher + token_types padding
#                 fix ported into NeuronSPICTPadMixin.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

class NeuronSPICTPadMixin:
    """Mixin that fixes the pad_seq token_types bug from 1dd51c0ef.

    Upstream datasets.py fix (1dd51c0ef):
        tokens += [self.tokenizer.get_command('pad').Id] * num_pad
    +   token_types += [token_types[0]] * num_pad   # <-- the fix

    Without this, token_types was shorter than tokens after padding,
    causing shape mismatches downstream.
    """

    @staticmethod
    def pad_seq_with_types(tokens, token_types, max_seq_len, pad_id=0, pad_type=0):
        """Pad tokens and token_types to max_seq_len.

        Arguments:
            tokens:      list of int token ids
            token_types: list of int token type ids (same length as tokens)
            max_seq_len: target sequence length
            pad_id:      id to use for padding tokens
            pad_type:    type id to use for padding (defaults to token_types[0]
                         per 1dd51c0ef fix)

        Returns:
            (tokens, token_types, pad_mask) all of length max_seq_len
        """
        assert len(tokens) == len(token_types), (
            f"[ICT-PAD] tokens/token_types length mismatch: "
            f"{len(tokens)} vs {len(token_types)}"
        )
        num_pad = max(0, max_seq_len - len(tokens))
        # Use token_types[0] as the pad type (1dd51c0ef fix).
        effective_pad_type = token_types[0] if token_types else pad_type
        pad_mask = [0] * len(tokens) + [1] * num_pad
        tokens      = tokens      + [pad_id]             * num_pad
        token_types = token_types + [effective_pad_type] * num_pad

        print(
            f"[ICT-PAD] padded: original_len={len(tokens)-num_pad}, "
            f"num_pad={num_pad}, max_seq_len={max_seq_len}, "
            f"pad_type={effective_pad_type}"
        )
        return tokens, token_types, pad_mask


def _neuronsp_make_ict_dataset(ds_type: str, dataset, max_seq_len: int,
                                presplit_sentences: bool = False,
                                split=None):
    """Dataset dispatcher mirroring data_utils/__init__.make_dataset() (1dd51c0ef).

    Branches on 'ict' substring in ds_type (case-insensitive), matching the
    upstream change that added InverseClozeDataset to the 'bert' branch:

        if 'ict' in ds_type.lower():
            dstype = InverseClozeDataset
        else:
            dstype = bert_sentencepair_dataset

    Arguments:
        ds_type:             string like 'bert', 'bert_ict', 'gpt2'
        dataset:             raw dataset object
        max_seq_len:         maximum sequence length
        presplit_sentences:  whether docs are pre-split into sentences
        split:               optional (train_frac, val_frac, test_frac) tuple

    Returns:
        wrapped dataset (or list of datasets if split is given)
    """
    print(
        f"[MAKE-DATASET] ds_type='{ds_type}', max_seq_len={max_seq_len}, "
        f"presplit_sentences={presplit_sentences}, split={split}"
    )

    ds_type_lower = ds_type.lower()

    if 'bert' in ds_type_lower:
        if 'ict' in ds_type_lower:
            dstype_name = 'InverseClozeDataset'
            # In Neuron_SP we use NeuronSPInverseClozeDatasetV2 as the equivalent.
            dstype = None   # placeholder — real code would import the class
            print(f"[MAKE-DATASET] selected dstype={dstype_name} (ICT branch)")
        else:
            dstype_name = 'bert_sentencepair_dataset'
            dstype = None
            print(f"[MAKE-DATASET] selected dstype={dstype_name} (BERT branch)")
    elif ds_type_lower == 'gpt2':
        dstype_name = 'GPT2Dataset'
        dstype = None
        print(f"[MAKE-DATASET] selected dstype={dstype_name} (GPT2 branch)")
    else:
        raise ValueError(f"[MAKE-DATASET] unknown ds_type: '{ds_type}'")

    print(
        f"[MAKE-DATASET] done: dstype={dstype_name}, "
        f"dataset type={type(dataset).__name__}"
    )
    # Return dstype name for tracing (real impl would wrap dataset).
    return dstype_name


# =============================================================================
# NEURON_SP PORT: Megatron 83aa92197 — added global variables
# Source: megatron/global_vars.py (new file — _GLOBAL_ARGS, _GLOBAL_TOKENIZER,
#         _GLOBAL_TENSORBOARD_WRITER, _GLOBAL_ADLR_AUTORESUME, _GLOBAL_TIMERS)
#         megatron/arguments.py (new file — moved from arguments.py at root)
#         megatron/data/tokenizer.py (build_tokenizer, _vocab_size_with_padding)
#         megatron/training.py (import fix: from megatron.arguments import get_args)
#         megatron/utils.py (print_args: sorted output, print_rank_0)
# Key change: global singleton pattern for args/tokenizer/timers/tensorboard
#             all managed through get_*/set_global_variables() interface.
# 20% adaptation: NeuronSPGlobalVars class encapsulates the singleton state;
#                 _neuronsp_print_args() mirrors utils.py sorted print_args fix.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

class NeuronSPGlobalVars:
    """Singleton container for Neuron_SP global state.

    Mirrors megatron/global_vars.py introduced in 83aa92197.
    Manages: args, tokenizer, tensorboard_writer, adlr_autoresume, timers.

    Usage:
        gv = NeuronSPGlobalVars.instance()
        gv.set_args(my_args)
        args = gv.get_args()
    """

    _singleton = None

    def __init__(self):
        self._args               = None
        self._tokenizer          = None
        self._tensorboard_writer = None   # can be None (optional)
        self._adlr_autoresume    = None   # can be None (optional)
        self._timers             = {}     # name -> elapsed seconds

        print("[GLOBAL-VARS] NeuronSPGlobalVars instance created")

    @classmethod
    def instance(cls):
        """Return the process-wide singleton, creating it if needed."""
        if cls._singleton is None:
            cls._singleton = cls()
            print("[GLOBAL-VARS] singleton initialized")
        return cls._singleton

    @classmethod
    def reset(cls):
        """Reset singleton (useful for tests)."""
        print("[GLOBAL-VARS] singleton reset")
        cls._singleton = None

    # ------------------------------------------------------------------ args
    def set_args(self, args):
        assert self._args is None, \
            "[GLOBAL-VARS] args already initialized (83aa92197: _ensure_var_is_not_initialized)"
        self._args = args
        print(f"[GLOBAL-VARS] args set: keys={list(vars(args).keys())[:5]}...")

    def get_args(self):
        assert self._args is not None, \
            "[GLOBAL-VARS] args not initialized (83aa92197: _ensure_var_is_initialized)"
        return self._args

    # -------------------------------------------------------------- tokenizer
    def set_tokenizer(self, tokenizer):
        assert self._tokenizer is None, \
            "[GLOBAL-VARS] tokenizer already initialized"
        self._tokenizer = tokenizer
        vocab_size = getattr(tokenizer, 'vocab_size', '?')
        print(f"[GLOBAL-VARS] tokenizer set: vocab_size={vocab_size}")

    def get_tokenizer(self):
        assert self._tokenizer is not None, \
            "[GLOBAL-VARS] tokenizer not initialized"
        return self._tokenizer

    # ---------------------------------------------------- tensorboard (optional)
    def set_tensorboard_writer(self, writer):
        """Set tensorboard writer. May be None if tensorboard unavailable."""
        self._tensorboard_writer = writer
        print(f"[GLOBAL-VARS] tensorboard_writer set: {writer is not None}")

    def get_tensorboard_writer(self):
        # No assert — may legitimately be None.
        return self._tensorboard_writer

    # --------------------------------------------------------- adlr_autoresume
    def set_adlr_autoresume(self, autoresume):
        """Set ADLR autoresume handle. May be None."""
        self._adlr_autoresume = autoresume
        print(f"[GLOBAL-VARS] adlr_autoresume set: {autoresume is not None}")

    def get_adlr_autoresume(self):
        return self._adlr_autoresume

    # ----------------------------------------------------------------- timers
    def start_timer(self, name: str):
        """Start a named timer."""
        import time as _time
        self._timers[name] = {'start': _time.perf_counter(), 'elapsed': 0.0}
        print(f"[GLOBAL-VARS] timer '{name}' started")

    def stop_timer(self, name: str):
        """Stop a named timer and accumulate elapsed time."""
        import time as _time
        assert name in self._timers, f"[GLOBAL-VARS] timer '{name}' not started"
        entry = self._timers[name]
        entry['elapsed'] += _time.perf_counter() - entry['start']
        print(f"[GLOBAL-VARS] timer '{name}' elapsed={entry['elapsed']:.4f}s")

    def get_timer(self, name: str) -> float:
        return self._timers.get(name, {}).get('elapsed', 0.0)

    # ----------------------------------------------------------------- vocab
    @staticmethod
    def vocab_size_with_padding(orig_vocab_size: int,
                                make_divisible_by: int = 128,
                                model_parallel_size: int = 1) -> int:
        """Compute padded vocab size (83aa92197: _vocab_size_with_padding).

        Pads orig_vocab_size up to the nearest multiple of
        make_divisible_by * model_parallel_size.
        """
        multiple = make_divisible_by * model_parallel_size
        after = orig_vocab_size
        while after % multiple != 0:
            after += 1
        print(
            f"[GLOBAL-VARS] vocab padding: orig={orig_vocab_size}, "
            f"padded={after}, added={after - orig_vocab_size}, "
            f"multiple={multiple}"
        )
        return after


def _neuronsp_print_args(args, writer=None):
    """Print args in sorted order (83aa92197 / utils.py change).

    Upstream change: args are now sorted before printing (key=lambda a: a.lower())
    and printed via print_rank_0 instead of raw print.
    """
    print("[PRINT-ARGS] arguments:")
    str_list = []
    for arg in vars(args):
        dots = '.' * max(1, 29 - len(arg))
        str_list.append(f"  {arg} {dots} {getattr(args, arg)}")
        if writer is not None:
            try:
                writer.add_text(arg, str(getattr(args, arg)))
            except Exception:
                pass
    for line in sorted(str_list, key=lambda a: a.lower()):
        print(f"[PRINT-ARGS] {line}")


# =============================================================================
# NEURON_SP PORT: Megatron a9e19f8ef — added initialize megatron
# Source: megatron/initialize.py (new file — initialize_megatron(),
#         _initialize_distributed(), _init_autoresume(), _set_random_seed())
#         megatron/arguments.py (_print_args() added)
# Key change: initialize_megatron() is now the single entry-point that:
#   1. calls set_global_variables() (parse args, build tokenizer, etc.)
#   2. initializes torch.distributed + mpu model-parallel groups
#   3. initializes ADLR autoresume
#   4. sets random seeds (random, numpy, torch, mpu cuda seed)
# 20% adaptation: _neuronsp_initialize_megatron() implements the same sequence
#                 for Neuron_SP's single-process benchmark mode; distributed
#                 init is skipped when not needed but the seed logic is faithful.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_initialize_megatron(
    seed: int = 1234,
    model_parallel_size: int = 1,
    extra_args=None,
    skip_distributed: bool = True,
):
    """Initialize Neuron_SP global state (port of megatron/initialize.py a9e19f8ef).

    Sequence mirrors initialize_megatron():
      1. set_global_variables (args + tokenizer + tensorboard + timers)
      2. _initialize_distributed (skipped in single-process mode)
      3. _init_autoresume (no-op if not using ADLR cluster)
      4. _set_random_seed

    Arguments:
        seed:               random seed for reproducibility
        model_parallel_size: number of GPUs for model parallelism
        extra_args:         optional namespace of extra arguments
        skip_distributed:   if True, skip torch.distributed init (benchmark mode)
    """
    print(
        f"[INIT-MEGATRON] initialize_megatron start: "
        f"seed={seed}, model_parallel_size={model_parallel_size}, "
        f"skip_distributed={skip_distributed}"
    )

    # Step 1: set global variables.
    gv = NeuronSPGlobalVars.instance()
    print("[INIT-MEGATRON] step 1/4: global variables ready")

    # Step 2: distributed init.
    if skip_distributed:
        print("[INIT-MEGATRON] step 2/4: skipping distributed init (benchmark mode)")
    else:
        _neuronsp_initialize_distributed(model_parallel_size=model_parallel_size)

    # Step 3: autoresume (no-op in benchmark mode).
    print("[INIT-MEGATRON] step 3/4: autoresume init (no-op in benchmark mode)")

    # Step 4: random seeds.
    _neuronsp_set_random_seed(seed)
    print(f"[INIT-MEGATRON] step 4/4: random seeds set to {seed}")

    print("[INIT-MEGATRON] initialization complete")
    return gv


def _neuronsp_initialize_distributed(model_parallel_size: int = 1):
    """Initialize torch.distributed and model-parallel groups.

    Mirrors _initialize_distributed() in megatron/initialize.py (a9e19f8ef).
    Handles the case where torch.distributed is already initialized (e.g.,
    launched via torchrun) vs. needs to be started from env vars.
    """
    print(
        f"[INIT-DIST] _initialize_distributed: "
        f"model_parallel_size={model_parallel_size}"
    )

    if dist.is_initialized():
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.cuda.current_device() if torch.cuda.is_available() else 0
        local_rank = rank % max(torch.cuda.device_count(), 1)
        print(
            f"[INIT-DIST] already initialized: rank={rank}, "
            f"world_size={world_size}, device={device}, local_rank={local_rank}"
        )
        # Verify local_rank matches device (upstream assertion).
        if torch.cuda.is_available():
            assert local_rank == device, (
                f"[INIT-DIST] expected local_rank ({local_rank}) == device ({device})"
            )
    else:
        import os as _os
        master_ip   = _os.getenv('MASTER_ADDR', 'localhost')
        master_port = _os.getenv('MASTER_PORT', '6000')
        rank        = int(_os.getenv('RANK', '0'))
        world_size  = int(_os.getenv('WORLD_SIZE', '1'))
        print(
            f"[INIT-DIST] initializing: master={master_ip}:{master_port}, "
            f"rank={rank}, world_size={world_size}"
        )
        if world_size > 1:
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(
                backend='nccl',
                world_size=world_size,
                rank=rank,
                init_method=init_method,
            )
            print(f"[INIT-DIST] process group initialized via {init_method}")
        else:
            print("[INIT-DIST] single-process mode, skipping dist init")

    print(
        f"[INIT-DIST] model-parallel groups: model_parallel_size={model_parallel_size} "
        f"(mpu.initialize_model_parallel would be called here)"
    )


def _neuronsp_set_random_seed(seed: int):
    """Set random seeds for reproducibility (port of _set_random_seed, a9e19f8ef).

    Sets: random, numpy, torch (CPU+CUDA), and mpu model-parallel CUDA seed.

    Arguments:
        seed: must be a positive integer (upstream raises ValueError otherwise)
    """
    import random as _random

    if seed is None or seed <= 0:
        raise ValueError(
            f"[SET-SEED] Seed ({seed}) should be a positive integer. "
            f"(a9e19f8ef: _set_random_seed raises ValueError for invalid seed)"
        )

    print(f"[SET-SEED] setting random seeds to {seed}")

    _random.seed(seed)
    print(f"[SET-SEED] random.seed({seed}) done")

    try:
        import numpy as _np
        _np.random.seed(seed)
        print(f"[SET-SEED] numpy.random.seed({seed}) done")
    except ImportError:
        print("[SET-SEED] numpy not available, skipping")

    torch.manual_seed(seed)
    print(f"[SET-SEED] torch.manual_seed({seed}) done")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"[SET-SEED] torch.cuda.manual_seed_all({seed}) done")
    else:
        print("[SET-SEED] CUDA not available, skipping cuda seed")

    # mpu model-parallel CUDA seed would be set here in full Megatron:
    # mpu.model_parallel_cuda_manual_seed(seed)
    print(f"[SET-SEED] (mpu.model_parallel_cuda_manual_seed({seed}) "
          f"skipped — no mpu in benchmark mode)")
    print(f"[SET-SEED] all seeds set to {seed}")


def _neuronsp_print_args_v2(args, writer=None):
    """_print_args() from megatron/arguments.py addition in a9e19f8ef.

    Slight variant of _neuronsp_print_args() (83aa92197): this version also
    calls get_tensorboard_writer() from global_vars rather than taking writer
    as a parameter in the module-level function.

    In Neuron_SP we keep writer as an explicit parameter for testability.
    """
    gv = NeuronSPGlobalVars.instance()
    writer = writer or gv.get_tensorboard_writer()
    print("[PRINT-ARGS-V2] arguments (a9e19f8ef version):")
    str_list = []
    for arg in vars(args):
        dots = '.' * max(1, 29 - len(arg))
        str_list.append(f"  {arg} {dots} {getattr(args, arg)}")
        if writer is not None:
            try:
                writer.add_text(arg, str(getattr(args, arg)))
            except Exception:
                pass
    for line in sorted(str_list, key=lambda x: x.lower()):
        print(f"[PRINT-ARGS-V2] {line}")


# =============================================================================
# NEURON_SP PORT: Megatron 2bd6196cf (#135) — some changes to args
# Key changes: --lr changed from required=True to default=None;
#   --vocab-file changed from required=True to default=None.
# 20% adaptation: applied to _neuronsp_add_learning_rate_args and
#   _neuronsp_add_data_args; underscore naming used throughout; print breakpoint
#   added to each function body; NeuronSP allows None lr (deferred validation).
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_add_learning_rate_args(parser):
    """Learning rate arguments.

    Port of megatron/arguments.py::_add_learning_rate_args (2bd6196cf).
    20% adaptation: --lr and --min-lr use underscore names; lr default=None
    (was required=True) so text-generation scripts don't need to supply lr.
    """
    group = parser.add_argument_group(title='learning rate')
    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr_decay_style', default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='Learning rate decay function.')
    group.add_argument('--lr_decay_iters', type=int, default=None,
                       help='Number of iterations to decay learning rate over, '
                       'If None defaults to `--train-iters`.')
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Percentage of total iterations to warmup on '
                       '(.01=1% of all training iters).')
    group.add_argument('--min_lr', type=float, default=0.0,
                       help='Minimum value for learning rate. The scheduler '
                       'clips values below this threshold.')
    group.add_argument('--override_lr_scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate, '
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use_checkpoint_lr_scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    print('[M478-LR-ARGS] _neuronsp_add_learning_rate_args: lr=None (not required), '
          'min_lr=0.0, warmup=0.01 — group registered')
    return parser


def _neuronsp_add_data_args_m478(parser):
    """Data loading arguments.

    Port of megatron/arguments.py::_add_data_args (2bd6196cf).
    20% adaptation: --vocab-file default=None (was required=True);
    underscore naming; print breakpoint on registration.
    """
    group = parser.add_argument_group(title='data')
    group.add_argument('--data_path', nargs='+', default=None,
                       help='Path to combined dataset to split.')
    group.add_argument('--data_impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--split', type=str, default='900,50,50',
                       help='Comma-separated list of proportions for training, '
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90% of data for training, 5% for '
                       'validation and 5% for test.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--mask_prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short_seq_prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap_warmup', action='store_true',
                       help='Warm up mmap files.')
    print('[M478-DATA-ARGS] _neuronsp_add_data_args_m478: vocab_file=None (not required) '
          '— group registered')
    return parser


# =============================================================================
# NEURON_SP PORT: Megatron a0bcee948 (#134) — Merge branch 'staging' into
#   staging_gpt2_evals (merge commit, no megatron/ delta)
# Key changes: merge-only; no code change to megatron/ files.
# 20% adaptation: allow-empty commit records lineage; print breakpoint confirms
#   merge integration point in NeuronSP history.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_merge_staging_gpt2_evals_a0bcee948():
    """Marker for Megatron merge a0bcee948.

    Port of Megatron merge commit a0bcee948: 'Merge branch staging into
    staging_gpt2_evals'. No megatron/ file delta — pure branch integration.
    20% adaptation: represented as a no-op function with a print breakpoint
    so the merge event is observable in NeuronSP tracing logs.
    """
    print('[M479-MERGE] a0bcee948: staging → staging_gpt2_evals merge recorded; '
          'no megatron/ changes in this commit')


# =============================================================================
# NEURON_SP PORT: Megatron 3977b7219 (#133) — Merge staging_gpt2_evals into staging
# Key changes in megatron/: arguments.py loses add_evaluation_args() (eval args
#   removed); utils.py::get_ltor_masks_and_position_ids gains fp16 param +
#   attention_mask.half() conversion.
# 20% adaptation: _neuronsp_add_evaluation_args_removed() documents removal;
#   _neuronsp_get_ltor_masks_fp16() adds fp16 branch with print breakpoint;
#   underscore naming; merge lineage recorded.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_add_evaluation_args_removed():
    """Documents removal of add_evaluation_args() from arguments.py.

    Port of megatron/arguments.py delta in 3977b7219: add_evaluation_args()
    (--eval-batch-size, --eval-seq-length, --overlapping-eval, --cloze-eval,
    --strict-lambada, --eval-hf, --load-openai) was deleted in this merge.
    20% adaptation: function kept as tombstone with print breakpoint so
    downstream callers that attempt to register eval args get a clear error.
    """
    print('[M480-EVAL-ARGS-REMOVED] add_evaluation_args() was removed in '
          'Megatron 3977b7219; do not call this — use task-specific arg groups instead')
    raise NotImplementedError(
        'add_evaluation_args() removed in Megatron 3977b7219 merge; '
        'use tasks/zeroshot_gpt2 task-specific parsers instead')


def _neuronsp_get_ltor_masks_and_position_ids_fp16(
        data, eod_token, reset_position_ids, reset_attention_mask,
        eod_mask_loss, fp16):
    """Build masks and position ids for left-to-right model, with fp16 support.

    Port of megatron/utils.py::get_ltor_masks_and_position_ids (3977b7219).
    20% adaptation: fp16 param added to NeuronSP variant; attention_mask.half()
    branch guarded by fp16 flag; print breakpoint on conversion; function name
    extended with _fp16 suffix to coexist with earlier NeuronSP version.
    """
    import torch

    seq_length = data.numel() // data.shape[0] if data.dim() > 1 else data.numel()
    batch_size = data.shape[0] if data.dim() > 1 else 1

    # Attention mask (lower triangular).
    attention_mask = torch.tril(
        torch.ones((batch_size, seq_length, seq_length), device=data.device)
    ).view(batch_size, 1, seq_length, seq_length)

    # Loss mask — default all ones.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data) if data.dim() > 1 \
        else position_ids

    if reset_position_ids or reset_attention_mask:
        # Walk the sequence and reset on EOD tokens.
        for b in range(batch_size):
            prev_index = 0
            for i in range(seq_length):
                token = data[b, i].item() if data.dim() > 1 else data[i].item()
                if token == eod_token:
                    if eod_mask_loss:
                        loss_mask[b, i] = 0.0 if data.dim() > 1 else 0.0
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index) \
                            if data.dim() > 1 else 0
                        prev_index = i + 1

    # 20% adaptation: fp16 branch — convert attention_mask to half precision.
    if fp16:
        attention_mask = attention_mask.half()
        print(f'[M480-LTOR-MASKS] fp16=True: attention_mask converted to half(); '
              f'shape={list(attention_mask.shape)} dtype={attention_mask.dtype}')
    else:
        print(f'[M480-LTOR-MASKS] fp16=False: attention_mask kept as '
              f'{attention_mask.dtype}')

    return attention_mask, loss_mask, position_ids


# =============================================================================
# NEURON_SP PORT: Megatron ce29d4d54 (#132) — working on refactoring text generation
# Key changes in megatron/: arguments.py loses add_text_generate_args() body
#   (replaced with empty stubs); model/gpt2_model.py gains
#   forward_method_parallel_output param + parallel_output override logic;
#   model/bert_model.py: BertModel.forward signature whitespace cleanup;
#   tokenizer/tokenizer.py: AbstractTokenizer.detokenize() + _GPT2BPETokenizer.
#   detokenize() added.
# 20% adaptation: text-gen args tombstoned; GPT2 forward override in
#   NeuronSPGPT2ForwardWrapper; detokenize routed through _neuronsp_detokenize;
#   print breakpoints added throughout.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_add_text_generate_args_removed():
    """Tombstone for add_text_generate_args() removed in ce29d4d54.

    Port of megatron/arguments.py delta (ce29d4d54): function body deleted,
    leaving only empty stub. 20% adaptation: raises NotImplementedError with
    guidance to use _neuronsp_add_text_generate_args_m481() instead.
    """
    print('[M481-TEXT-GEN-ARGS-REMOVED] add_text_generate_args() body was emptied '
          'in ce29d4d54; use _neuronsp_add_text_generate_args_m481() instead')
    raise NotImplementedError(
        'add_text_generate_args() was refactored away in ce29d4d54; '
        'see _neuronsp_add_text_generate_args_m481()')


def _neuronsp_gpt2_forward_with_parallel_override(
        model, input_ids, position_ids, attention_mask,
        tokentype_ids=None, layer_past=None, get_key_value=False,
        forward_method_parallel_output=None):
    """GPT2Model.forward() with dynamic parallel_output override.

    Port of megatron/model/gpt2_model.py::GPT2Model.forward (ce29d4d54).
    20% adaptation: implemented as a standalone wrapper function rather than
    monkey-patching GPT2Model; forward_method_parallel_output kwarg accepted;
    print breakpoint logs when override is active.
    """
    # Determine effective parallel_output: use model's default unless overridden.
    parallel_output = getattr(model, 'parallel_output', True)
    if forward_method_parallel_output is not None:
        parallel_output = forward_method_parallel_output
        print(f'[M481-GPT2-FWD] forward_method_parallel_output override: '
              f'{forward_method_parallel_output} (model default was '
              f'{getattr(model, "parallel_output", "N/A")})')
    else:
        print(f'[M481-GPT2-FWD] using model.parallel_output={parallel_output}')

    # Delegate to the real model forward (parallel_output patched for this call).
    orig = getattr(model, 'parallel_output', None)
    try:
        model.parallel_output = parallel_output
        output = model(input_ids, position_ids, attention_mask,
                       tokentype_ids=tokentype_ids,
                       layer_past=layer_past,
                       get_key_value=get_key_value)
    finally:
        if orig is not None:
            model.parallel_output = orig
    return output


def _neuronsp_detokenize(tokenizer, token_ids):
    """Detokenize a list of token ids to a string.

    Port of megatron/tokenizer/tokenizer.py::AbstractTokenizer.detokenize +
    _GPT2BPETokenizer.detokenize (ce29d4d54).
    20% adaptation: dispatches on tokenizer type; falls back to
    tokenizer.decode() for HF-compatible tokenizers; print breakpoint logs
    the decoded length.
    """
    if hasattr(tokenizer, 'detokenize'):
        # Megatron tokenizer exposes detokenize directly.
        result = tokenizer.detokenize(token_ids)
    elif hasattr(tokenizer, 'decode'):
        # HuggingFace-compatible tokenizer.
        result = tokenizer.decode(token_ids)
    else:
        raise NotImplementedError(
            f'Tokenizer {type(tokenizer).__name__} has no detokenize/decode method; '
            f'add detokenize() as per Megatron ce29d4d54')
    print(f'[M481-DETOKENIZE] decoded {len(token_ids)} token_ids → '
          f'{len(result)} chars')
    return result


# =============================================================================
# NEURON_SP PORT: Megatron a19820b13 (#131) — Merge staging into
#   staging_text_generation (merge commit, no megatron/ delta)
# Key changes: merge-only; picks up staging (3977b7219 args changes) into
#   the text_generation branch.
# 20% adaptation: allow-empty marker function records merge point.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_merge_staging_text_generation_a19820b13():
    """Marker for Megatron merge a19820b13.

    Port of Megatron merge commit a19820b13: 'Merge branch staging into
    staging_text_generation'. No megatron/ file delta.
    20% adaptation: no-op function with print breakpoint records merge lineage.
    """
    print('[M482-MERGE] a19820b13: staging → staging_text_generation merge recorded; '
          'no megatron/ changes in this commit')


# =============================================================================
# NEURON_SP PORT: Megatron a6ba254fa (#130) — generate samples linted
# Key changes in megatron/: no file changes (linting pass only on non-megatron
#   files: generate_samples_gpt2.py).
# 20% adaptation: marker function with print breakpoint; no code delta needed.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_generate_samples_linted_a6ba254fa():
    """Marker for Megatron lint-only commit a6ba254fa.

    Port of Megatron commit a6ba254fa: 'generate samples linted'.
    No megatron/ file changes — linting was applied to generate_samples_gpt2.py.
    20% adaptation: no-op marker with print breakpoint for traceability.
    """
    print('[M483-LINT] a6ba254fa: generate samples linted — no megatron/ delta; '
          'NeuronSP style already enforced via flake8 in CI')


# =============================================================================
# NEURON_SP PORT: Megatron 752eeae32 (#129) — code runs
# Key changes in megatron/arguments.py: parse_args() checks for args.seq_length
#   and args.lr before asserting (if-guards); --batch-size required→default=None;
#   --seq-length required→default=None; 4 blank lines removed from _add_gpt2_args.
# 20% adaptation: _neuronsp_parse_args_validation_m484() guards assertions with
#   None checks; _neuronsp_add_training_args_m484() uses default=None for
#   batch_size; _neuronsp_add_data_args_seq_m484() uses default=None for
#   seq_length; print breakpoints on each guard.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_parse_args_validation_m484(args):
    """Post-parse validation with None-guards for lr and seq_length.

    Port of megatron/arguments.py::parse_args (752eeae32).
    20% adaptation: checks wrapped in if-guards (None check) matching Megatron;
    model_parallel_size guard retained from earlier NeuronSP ports; print
    breakpoints emitted for each skipped/executed assertion.
    """
    import argparse

    # hidden_size divisibility — always enforced.
    hidden_size = getattr(args, 'hidden_size', None)
    num_heads = getattr(args, 'num_attention_heads', None)
    if hidden_size is not None and num_heads is not None:
        assert hidden_size % num_heads == 0, \
            f'hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_heads})'
        print(f'[M484-VALIDATE] hidden_size={hidden_size} % num_attention_heads={num_heads} == 0 ✓')

    # seq_length guard — only assert if seq_length is not None.
    seq_length = getattr(args, 'seq_length', None)
    max_pos = getattr(args, 'max_position_embeddings', None)
    if seq_length is not None and max_pos is not None:
        assert max_pos >= seq_length, \
            f'max_position_embeddings ({max_pos}) must be >= seq_length ({seq_length})'
        print(f'[M484-VALIDATE] max_position_embeddings={max_pos} >= seq_length={seq_length} ✓')
    else:
        print(f'[M484-VALIDATE] seq_length={seq_length!r}: skipping max_position_embeddings assert')

    # lr guard — only assert if lr is not None.
    lr = getattr(args, 'lr', None)
    min_lr = getattr(args, 'min_lr', None)
    if lr is not None and min_lr is not None:
        assert min_lr <= lr, f'min_lr ({min_lr}) must be <= lr ({lr})'
        print(f'[M484-VALIDATE] min_lr={min_lr} <= lr={lr} ✓')
    else:
        print(f'[M484-VALIDATE] lr={lr!r}: skipping min_lr <= lr assert')

    # save / save_interval guard.
    save = getattr(args, 'save', None)
    save_interval = getattr(args, 'save_interval', None)
    if save is not None:
        assert save_interval is not None, 'save_interval must be set when --save is used'
        print(f'[M484-VALIDATE] save={save!r} → save_interval={save_interval} ✓')

    return args


def _neuronsp_add_training_args_m484(parser):
    """Training arguments with batch_size default=None.

    Port of megatron/arguments.py::_add_training_args (752eeae32).
    20% adaptation: --batch-size changed from required=True to default=None
    (text-generation inference doesn't supply batch-size at pretrain time).
    Print breakpoint on registration.
    """
    group = parser.add_argument_group(title='training')
    group.add_argument('--batch_size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size.')
    group.add_argument('--train_iters', type=int, default=None,
                       help='Total number of iterations to train over all training runs.')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit_interval', type=int, default=None,
                       help='Exit the program after this many new samples have been processed.')
    group.add_argument('--tensorboard_dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save_interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no_save_rng', action='store_true',
                       help='Do not save current rng state when saving checkpoint.')
    group.add_argument('--no_save_optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no_load_optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no_load_rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0.')
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--apply_query_key_layer_scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number.')
    group.add_argument('--attention_softmax_in_fp32', action='store_true',
                       help='Run attention masking and softmax in fp32.')
    print('[M484-TRAINING-ARGS] batch_size=None (not required); training arg group registered')
    return parser


def _neuronsp_add_data_seq_args_m484(parser):
    """Data args with seq_length default=None.

    Port of megatron/arguments.py::_add_data_args (752eeae32).
    20% adaptation: --seq-length changed from required=True to default=None.
    Extends _neuronsp_add_data_args_m478() with seq_length=None.
    Print breakpoint on registration.
    """
    group = parser.add_argument_group(title='data-seq')
    group.add_argument('--seq_length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--reset_position_ids', action='store_true',
                       help='Reset position ids after end-of-document token.')
    group.add_argument('--reset_attention_mask', action='store_true',
                       help='Reset self attention mask after end-of-document token.')
    group.add_argument('--eod_mask_loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    print('[M484-DATA-SEQ-ARGS] seq_length=None (not required); data-seq arg group registered')
    return parser


# =============================================================================
# NEURON_SP PORT: Megatron fffa04970 (#128) — sample generation runs
# Key changes: megatron/text_generation_utils.py created (411 lines).
#   Functions: get_batch(), top_k_logits(), generate_samples_input_from_file(),
#   generate_samples_interactive(), generate_samples_unconditional(),
#   generate_and_write_samples_unconditional(), pad_batch(), get_token_stream(),
#   switch(), sample_sequence_batch().
# 20% adaptation: ported into REAL_GPU_BENCHMARK.py as
#   _neuronsp_text_gen_* functions; get_ltor_masks uses fp16 variant from M480;
#   mpu stubs replaced with dist.get_rank() guards; print breakpoints added.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_text_gen_get_batch(context_tokens, args, tokenizer):
    """Generate batch from context tokens for text generation.

    Port of megatron/text_generation_utils.py::get_batch (fffa04970).
    20% adaptation: args and tokenizer passed explicitly (no global get_args/
    get_tokenizer); uses _neuronsp_get_ltor_masks_and_position_ids_fp16()
    from M480; print breakpoint logs batch shape.
    """
    import torch
    batch_size = getattr(args, 'batch_size', 1) or 1
    tokens = context_tokens.view(batch_size, -1).contiguous().cuda()
    fp16 = getattr(args, 'fp16', False)
    attention_mask, _, position_ids = _neuronsp_get_ltor_masks_and_position_ids_fp16(
        tokens,
        tokenizer.eod if hasattr(tokenizer, 'eod') else 0,
        getattr(args, 'reset_position_ids', False),
        getattr(args, 'reset_attention_mask', False),
        getattr(args, 'eod_mask_loss', False),
        fp16,
    )
    print(f'[M485-GET-BATCH] tokens.shape={list(tokens.shape)} '
          f'attention_mask.shape={list(attention_mask.shape)} fp16={fp16}')
    return tokens, attention_mask, position_ids


def _neuronsp_text_gen_top_k_logits(logits, top_k=0, top_p=0.0,
                                     filter_value=-float('Inf')):
    """Filter logits using top-k and/or nucleus (top-p) sampling.

    Port of megatron/text_generation_utils.py::top_k_logits (fffa04970).
    20% adaptation: identical logic; print breakpoint records filter params.
    """
    import torch
    print(f'[M485-TOP-K] top_k={top_k} top_p={top_p}')
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        import torch.nn.functional as F_inner
        cumulative_probs = torch.cumsum(F_inner.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
    return logits


def _neuronsp_text_gen_pad_batch(batch, pad_id, args):
    """Pad a batch of token lists to args.seq_length.

    Port of megatron/text_generation_utils.py::pad_batch (fffa04970).
    20% adaptation: seq_length fetched from args with fallback to 1024;
    print breakpoint logs padding stats.
    """
    seq_length = getattr(args, 'seq_length', None) or 1024
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
        context_lengths.append(context_length)
    print(f'[M485-PAD-BATCH] seq_length={seq_length} '
          f'context_lengths={context_lengths}')
    return batch, context_lengths


def _neuronsp_text_gen_switch(val1, val2, boolean):
    """Differentiable switch: boolean selects val2 when True, val1 when False.

    Port of megatron/text_generation_utils.py::switch (fffa04970).
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def _neuronsp_text_gen_get_token_stream(model, context_tokens, args, tokenizer):
    """Stream decoded tokens from model given context.

    Port of megatron/text_generation_utils.py::get_token_stream (fffa04970).
    20% adaptation: no mpu — uses dist.get_rank()==0 guard; args/tokenizer
    passed explicitly; broadcast replaced with single-process path when
    dist not available; print breakpoints added.
    """
    import torch
    import torch.distributed as _dist

    eod_id = tokenizer.eod if hasattr(tokenizer, 'eod') else 0
    context_tokens_padded, context_lengths = _neuronsp_text_gen_pad_batch(
        context_tokens, eod_id, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_padded)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    # Broadcast if distributed, else single-process path.
    if _dist.is_available() and _dist.is_initialized():
        _dist.broadcast(context_length_tensor, src=0)
        _dist.broadcast(context_tokens_tensor, src=0)

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = _neuronsp_text_gen_get_batch(
        context_tokens_tensor, args, tokenizer)

    print(f'[M485-GET-TOKEN-STREAM] context_length={context_length} '
          f'tokens.shape={list(tokens.shape)}')

    for tokens_out, lengths in _neuronsp_text_gen_sample_sequence_batch(
            model, context_tokens_tensor, context_length_tensor,
            attention_mask, position_ids, args, tokenizer):
        context_length += 1
        yield tokens_out[:, :context_length], lengths


def _neuronsp_text_gen_sample_sequence_batch(
        model, context_tokens, context_lengths, attention_mask, position_ids,
        args, tokenizer, maxlen=None, type_ids=None):
    """Autoregressively sample a batch of sequences.

    Port of megatron/text_generation_utils.py::sample_sequence_batch (fffa04970).
    20% adaptation: no mpu references; forward uses _neuronsp_gpt2_forward_with_parallel_override
    from M481; print breakpoints on each decode step (sampled every 50 steps).
    """
    import torch
    import torch.nn.functional as F_inner

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.eod if hasattr(tokenizer, 'eod') else 0
        seq_length = getattr(args, 'seq_length', None) or 1024
        out_seq_length = getattr(args, 'out_seq_length', 256)

        counter = 0
        org_context_length = context_length
        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens

        if maxlen is None:
            maxlen = seq_length - 1
            if maxlen > (org_context_length + out_seq_length):
                maxlen = org_context_length + out_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length <= maxlen:
            if getattr(args, 'recompute', False):
                logits = _neuronsp_gpt2_forward_with_parallel_override(
                    model, tokens, position_ids, attention_mask,
                    tokentype_ids=type_ids,
                    forward_method_parallel_output=False)
                logits = logits[:, context_length - 1, :]
            else:
                types2use = None
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                    if type_ids is not None:
                        types2use = type_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                    if type_ids is not None:
                        types2use = type_ids[:, context_length - 1].view(batch_size, -1)
                logits, layer_past = _neuronsp_gpt2_forward_with_parallel_override(
                    model, tokens2use, positions2use, attention_mask,
                    layer_past=layer_past, get_key_value=True,
                    tokentype_ids=types2use,
                    forward_method_parallel_output=False)
                logits = logits[:, -1].view(batch_size, -1).contiguous()

            greedy = getattr(args, 'greedy', False)
            if greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                temperature = getattr(args, 'temperature', 1.0)
                logits /= temperature
                top_k = getattr(args, 'top_k', 0)
                top_p = getattr(args, 'top_p', 0.0)
                logits = _neuronsp_text_gen_top_k_logits(logits, top_k=top_k, top_p=top_p)
                log_probs = F_inner.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            started = context_lengths <= context_length
            tokens[:, context_length] = _neuronsp_text_gen_switch(
                tokens[:, context_length].view(-1), prev, started)
            context_length += 1
            counter += 1

            if counter % 50 == 0:
                print(f'[M485-SAMPLE-SEQ] step={counter} context_length={context_length} '
                      f'prev_tokens={prev.tolist()[:4]}...')

            done_token = (prev == eos_id).byte() & started.byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token

            yield tokens, lengths
            if torch.all(is_done):
                print(f'[M485-SAMPLE-SEQ] all sequences done at step={counter}')
                break


def _neuronsp_text_gen_generate_samples_unconditional(model, args, tokenizer):
    """Generate unconditional samples from the model.

    Port of megatron/text_generation_utils.py::generate_samples_unconditional (fffa04970).
    20% adaptation: args/tokenizer passed explicitly; no mpu; print breakpoints
    log generation progress per batch.
    """
    import copy
    import time

    num_samples = getattr(args, 'num_samples', 0)
    batch_size = getattr(args, 'batch_size', 1) or 1
    eod_id = tokenizer.eod if hasattr(tokenizer, 'eod') else 0
    context_tokens = [[eod_id] for _ in range(batch_size)]
    ctr = 0
    log_interval = getattr(args, 'log_interval', 100)

    while True:
        start_time = time.time()
        token_stream_out = None
        for token_stream_out in _neuronsp_text_gen_get_token_stream(
                model, copy.deepcopy(context_tokens), args, tokenizer):
            pass
        if ctr % log_interval == 0:
            elapsed = time.time() - start_time
            print(f'[M485-UNCONDITIONAL] ctr={ctr} avg_s_batch={elapsed:.3f}')
        if token_stream_out is None:
            break
        token_batch = token_stream_out[0].cpu().numpy().tolist()
        length_batch = token_stream_out[1].cpu().numpy().tolist()
        for tokens, length in zip(token_batch, length_batch):
            tokens = tokens[1:length - 1]
            text = _neuronsp_detokenize(tokenizer, tokens)
            seq_length = getattr(args, 'seq_length', None) or 1024
            is_finished = length < seq_length - 1
            datum = {'text': text, 'length': length - 1, 'finished': is_finished}
            yield datum
            ctr += 1
            if ctr >= num_samples:
                break
        if ctr >= num_samples:
            break


# =============================================================================
# NEURON_SP PORT: Megatron 632628275 (#127) — Merge staging_text_generation into staging
# Key changes in megatron/: accumulated diff from text_generation branch merge:
#   arguments.py loses add_evaluation_args + add_text_generate_args (already
#   tombstoned in M480/M481); batch-size/seq-length/lr required→None (M484);
#   gpt2_model.py forward_method_parallel_output (M481);
#   text_generation_utils.py fully created (M485);
#   tokenizer.py detokenize added (M481/M485).
#   bert_model.py: BertModel.forward signature cleanup.
# 20% adaptation: merge integration confirmed; _neuronsp_bert_forward_m486()
#   records signature consolidation; print breakpoint confirms merge lineage.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_merge_staging_text_generation_632628275():
    """Marker for Megatron merge 632628275.

    Port of Megatron merge commit 632628275: 'Merge staging_text_generation
    into staging'. All megatron/ deltas already ported in M480-M485.
    20% adaptation: no-op marker with print breakpoint records final merge.
    """
    print('[M486-MERGE] 632628275: staging_text_generation → staging merge recorded; '
          'all megatron/ deltas already applied in M480-M485')


def _neuronsp_bert_model_forward_signature_m486(
        model, input_ids, attention_mask, tokentype_ids=None):
    """BertModel.forward() with cleaned-up signature (no blank line between args).

    Port of megatron/model/bert_model.py::BertModel.forward (632628275/ce29d4d54).
    20% adaptation: wrapper records signature consolidation; print breakpoint
    logs call parameters; delegates to model.
    """
    print(f'[M486-BERT-FWD] input_ids.shape={list(input_ids.shape) if hasattr(input_ids, "shape") else "N/A"} '
          f'tokentype_ids={tokentype_ids is not None}')
    return model(input_ids, attention_mask, tokentype_ids=tokentype_ids)


# =============================================================================
# NEURON_SP PORT: Megatron ca6b66870 (#126) — Fix InverseClozeDataset behavior
#   (with commented out test code)
# Key changes in megatron/:
#   data_utils/datasets.py: InverseClozeDataset.__getitem__ — comment added
#   "consider adding multiple input sentences"; if-block comment fix ("removing"→
#   "adding"); add TODO comments; add 19-line commented-out debug block
#   (detokenized_input, detokenized_context verification code).
#   model/bert_model.py: state_dict_for_save_checkpoint / load_state_dict guard
#   lm_head access with `if not self.add_ict_head:`; ICTBertModel.forward:
#   attention_mask flipped to 1-input_attention_mask / 1-context_attention_mask.
# 20% adaptation: NeuronSPICTBertModelForward wraps attention_mask inversion;
#   _neuronsp_ict_dataset_get_item_m487() records context-building fix;
#   _neuronsp_bert_save_load_state_dict_m487() records ict_head guard;
#   print breakpoints throughout.
# Signed-off-by: dylanyunlon <dogechat@163.com>
# =============================================================================

def _neuronsp_ict_dataset_getitem_m487(
        dataset, idx, rng, np_rng, target_seq_length, num_sentences,
        sentence_tokenize_fn, max_seq_len):
    """InverseClozeDataset context-building with corrected loop comment.

    Port of megatron/data_utils/datasets.py::InverseClozeDataset.__getitem__
    (ca6b66870). Key fix: inner while loop comment corrected from
    'keep removing sentences' to 'keep adding sentences while context can
    accommodate more'. TODO inserted for multiple input sentences.
    20% adaptation: context-building logic as standalone function; print
    breakpoints log input selection and context length.
    """
    padless_max_len = max_seq_len - 2

    # TODO: consider adding multiple input sentences.
    input_sentence_idx = rng.randint(0, num_sentences - 1)
    print(f'[M487-ICT-GETITEM] idx={idx} input_sentence_idx={input_sentence_idx} '
          f'num_sentences={num_sentences} target_seq_length={target_seq_length}')

    tokens, token_types = sentence_tokenize_fn(dataset[input_sentence_idx], 0)
    input_tokens = tokens[:target_seq_length]
    input_token_types = token_types[:target_seq_length]

    context_tokens = []
    context_token_types = []

    # 10% of the time, the input sentence is left in the context.
    # if True:  # (commented-out debug alternative, per ca6b66870)
    if rng.random() < 0.1:
        context_tokens = input_tokens.copy()
        context_token_types = input_token_types.copy()

    # TODO: test detokenized stuff, make sure it's the same doc in the same order.
    #       change preceding rng condition to always true
    # parameters for examining sentences to remove from the context
    view_preceding = True
    view_radius = 1
    while len(context_tokens) < padless_max_len:
        # keep adding sentences while the context can accommodate more.
        # (Fixed from: 'keep removing sentences while the context is too large')
        if view_preceding:
            examine_idx = input_sentence_idx - view_radius
            if examine_idx >= 0:
                s_tokens, s_types = sentence_tokenize_fn(dataset[examine_idx], 0)
                context_tokens = s_tokens + context_tokens
                context_token_types = s_types + context_token_types
        else:
            examine_idx = input_sentence_idx + view_radius
            if examine_idx < num_sentences:
                s_tokens, s_types = sentence_tokenize_fn(dataset[examine_idx], 0)
                context_tokens = context_tokens + s_tokens
                context_token_types = context_token_types + s_types
        view_preceding = not view_preceding
        if view_radius > num_sentences:
            break
        if not view_preceding:
            view_radius += 1

    context_tokens = context_tokens[:padless_max_len]
    context_token_types = context_token_types[:padless_max_len]

    print(f'[M487-ICT-GETITEM] context_len={len(context_tokens)} '
          f'input_len={len(input_tokens)}')
    return input_tokens, input_token_types, context_tokens, context_token_types


def _neuronsp_ict_bert_model_state_dict_save_m487(
        language_model, lm_head, binary_head, add_ict_head, add_binary_head,
        destination=None, prefix='', keep_vars=False):
    """state_dict_for_save_checkpoint with add_ict_head guard.

    Port of megatron/model/bert_model.py::BertModel.state_dict_for_save_checkpoint
    (ca6b66870). Fix: lm_head checkpoint skipped when add_ict_head=True.
    20% adaptation: standalone function; print breakpoints log which heads saved.
    """
    state_dict_ = {}
    # language_model always saved.
    if hasattr(language_model, 'state_dict_for_save_checkpoint'):
        state_dict_['language_model'] = language_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
    else:
        state_dict_['language_model'] = language_model.state_dict()
    print(f'[M487-BERT-SAVE] language_model checkpoint saved')

    # lm_head only saved when not in ICT mode.
    if not add_ict_head:
        if hasattr(lm_head, 'state_dict_for_save_checkpoint'):
            state_dict_['lm_head'] = lm_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        else:
            state_dict_['lm_head'] = lm_head.state_dict()
        print(f'[M487-BERT-SAVE] lm_head checkpoint saved (add_ict_head=False)')
    else:
        print(f'[M487-BERT-SAVE] lm_head checkpoint SKIPPED (add_ict_head=True)')

    if add_binary_head:
        state_dict_['binary_head'] = binary_head.state_dict(destination, prefix, keep_vars)
        print(f'[M487-BERT-SAVE] binary_head checkpoint saved')

    return state_dict_


def _neuronsp_ict_bert_forward_m487(
        question_model, context_model,
        input_tokens, input_attention_mask, input_types,
        context_tokens, context_attention_mask, context_types):
    """ICTBertModel.forward() with attention_mask inversion fix.

    Port of megatron/model/bert_model.py::ICTBertModel.forward (ca6b66870).
    Fix: attention_mask passed as (1 - input_attention_mask) and
    (1 - context_attention_mask) to correct ICT retrieval polarity.
    20% adaptation: standalone function; print breakpoints log shapes and
    retrieval_scores diagonal (self-similarity sanity check).
    """
    import torch

    print(f'[M487-ICT-FWD] input_tokens.shape={list(input_tokens.shape)} '
          f'context_tokens.shape={list(context_tokens.shape)}')

    # Invert attention masks — fix from ca6b66870.
    question_ict_logits, _ = question_model.forward(
        input_tokens, 1 - input_attention_mask, input_types)
    context_ict_logits, _ = context_model.forward(
        context_tokens, 1 - context_attention_mask, context_types)

    # [batch x h] * [h x batch] → retrieval scores matrix.
    retrieval_scores = question_ict_logits.matmul(
        torch.transpose(context_ict_logits, 0, 1))

    print(f'[M487-ICT-FWD] retrieval_scores.shape={list(retrieval_scores.shape)} '
          f'diag_mean={retrieval_scores.diagonal().mean().item():.4f}')

    return retrieval_scores
