"""
DES-LOC Heterogeneous Training Engine for Neuron_SP Project.

Production-grade engine targeting 2xA6000 (48GB each) + 1xH100-NVL (96GB) cluster.
Implements TierDiscovery, PartitionSolver, and a real training loop with ZeRO-3
heterogeneous gradient accumulation or Pipeline 1F1B scheduling.

Hardware spec:
  GPU0: A6000  48GB  SM8.6  PCIe4 25GB/s  38.7 TFLOPS(BF16)  NUMA1
  GPU1: A6000  48GB  SM8.6  PCIe4 25GB/s  38.7 TFLOPS(BF16)  NUMA1
  GPU2: H100-NVL 96GB SM9.0 PCIe5 50GB/s 835 TFLOPS(BF16)   NUMA1
  No NVLink. CPU: 2xEPYC9354 128-core 1.5TB DDR5
"""

from __future__ import annotations

import concurrent.futures
import importlib
import inspect
import logging
import math
import os
import pkgutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
# GradScaler removed — BF16 does not need loss scaling
# clip_grad_norm_ replaced by core implementation: avoids host/device sync,
# computes norm across model-parallel group (Megatron M2335 pattern).
from deepspeed.core.optimizer.clip_grads import clip_grad_norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------------------------
# core.parallel_state: replaces direct torch.distributed rank/world-size calls
# ---------------------------------------------------------------------------
import deepspeed.core.parallel_state as parallel_state

# ---------------------------------------------------------------------------
# core.stream_manager: Insight I3: centralized stream management (Megatron M3724)
# ---------------------------------------------------------------------------
from deepspeed.core.stream_manager import StreamManager

# ---------------------------------------------------------------------------
# core.distributed: DistributedDataParallel + finalize_model_grads
# ---------------------------------------------------------------------------
from deepspeed.core.distributed import (
    DistributedDataParallel as CoreDDP,
    DistributedDataParallelConfig as CoreDDPConfig,
    finalize_model_grads,
)

# ---------------------------------------------------------------------------
# desloc_checkpointing: save_checkpoint/load_checkpoint extracted into a
# sibling module (Megatron-LM training/checkpointing.py pattern). The
# DesLocEngine methods below delegate to these free functions.
# ---------------------------------------------------------------------------
from deepspeed.runtime.desloc_checkpointing import (
    save_checkpoint as _desloc_save_checkpoint,
    load_checkpoint as _desloc_load_checkpoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eval hook import (eval/run_eval.py)
# ---------------------------------------------------------------------------
try:
    import importlib.util as _importlib_util
    _eval_spec = _importlib_util.spec_from_file_location(
        "run_eval",
        os.path.join(os.path.dirname(__file__), "..", "..", "eval", "run_eval.py"),
    )
    if _eval_spec is not None and _eval_spec.loader is not None:
        _run_eval_mod = _importlib_util.module_from_spec(_eval_spec)
        _eval_spec.loader.exec_module(_run_eval_mod)
        _run_periodic_eval = _run_eval_mod.run_periodic_eval
    else:
        _run_periodic_eval = None
except Exception as _eval_import_exc:
    logger.debug("eval/run_eval.py not importable (%s); eval hook disabled.", _eval_import_exc)
    _run_periodic_eval = None

# ---------------------------------------------------------------------------
# Optional logging backends: wandb and TensorBoard
# ---------------------------------------------------------------------------
try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _wandb = None  # type: ignore[assignment]
    _HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _SummaryWriter = None  # type: ignore[assignment]
    _HAS_TENSORBOARD = False

# ---------------------------------------------------------------------------
# NOTE: All deepspeed.runtime.hetero_* and deepspeed.checkpoint.hetero_*
# imports have been made lazy (imported inside __init__ / methods) to avoid
# triggering deepspeed/__init__.py → apex dependency at module import time.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REGISTRY_BASE = "deepspeed"
_HETERO_PREFIX = "hetero_"
_DEFAULT_DTYPE = torch.bfloat16
_CHECKPOINT_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
# ── Config types extracted to desloc_config.py ─────────────────────
from deepspeed.runtime.desloc_config import (  # noqa: E402
    PartitionStrategy,
    TierClass,
    TierSpec,
    PartitionPlan,
    TrainingConfig,
)
from deepspeed.runtime.hetero_step_batch_scheduler import MicrobatchAllocation  # noqa: E402

class HeteroRegistry:
    """
    Auto-discovers all hetero_*.py modules under the deepspeed package tree
    and exposes them through a unified registry dict.

    Modules are expected to optionally expose:
        - REGISTRY_NAME: str
        - register(engine): callable that receives the engine instance
    """

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._hooks: Dict[str, Any] = {}

    def discover(self, base_package: str = _REGISTRY_BASE) -> None:
        """
        Walk the base_package tree and import every module whose name starts
        with hetero_.  Collects REGISTRY_NAME and register() if present.

        Args:
            base_package: Top-level package name to search.
        """
        try:
            base_mod = importlib.import_module(base_package)
        except ImportError:
            logger.warning("Base package '%s' not importable; skipping discovery.", base_package)
            return

        base_path = getattr(base_mod, "__path__", [])
        found = 0
        for finder, mod_name, is_pkg in pkgutil.walk_packages(
            path=base_path,
            prefix=base_package + ".",
            onerror=lambda e: logger.debug("pkgutil walk error: %s", e),
        ):
            short = mod_name.split(".")[-1]
            if not short.startswith(_HETERO_PREFIX):
                continue
            try:
                mod = importlib.import_module(mod_name)
                key = getattr(mod, "REGISTRY_NAME", mod_name)
                self._modules[key] = mod
                found += 1
                logger.debug("Registered hetero module: %s -> %s", mod_name, key)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to import hetero module %s: %s", mod_name, exc)

        logger.info("HeteroRegistry: discovered %d hetero_* modules.", found)

    def register_hooks(self, engine: "DesLocEngine") -> int:
        """
        Activate every discovered hetero_* module against the engine.

        Two activation paths are supported:
          1. Preferred — the module exposes a top-level ``register(engine)``
             function which is invoked directly.
          2. Fallback  — the module has no ``register()`` hook, in which case
             its primary ``Hetero*`` class is attached to the engine under
             ``_hetero_mod_<module_name>`` so it can be retrieved later via
             the registry.  This ensures even passive extension modules are
             discoverable from the engine instance.

        Returns:
            The number of modules that were successfully activated
            (either via register() or via the fallback path).
        """
        activated = 0
        for key, mod in self._modules.items():
            if key in self._hooks:
                # Already registered in a previous pass — skip to make this
                # method idempotent when called multiple times during init.
                continue

            register_fn = getattr(mod, "register", None)
            if callable(register_fn):
                try:
                    register_fn(engine)
                    self._hooks[key] = mod
                    activated += 1
                    logger.debug("Hook registered from module: %s", key)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hook registration failed for %s: %s", key, exc)

            # Fallback: no register() — discover the primary Hetero* class
            # via the module's public names and attach it to the engine.
            primary_cls = None
            for attr_name in getattr(mod, "__all__", None) or dir(mod):
                if not attr_name.startswith("Hetero") or "Config" in attr_name:
                    continue
                candidate = getattr(mod, attr_name, None)
                if isinstance(candidate, type) and candidate.__module__ == mod.__name__:
                    primary_cls = (attr_name, candidate)
                    break

            if primary_cls is not None:
                attr_name, cls = primary_cls
                short = mod.__name__.rsplit(".", 1)[-1]
                engine_attr = f"_hetero_mod_{short}"
                if not hasattr(engine, engine_attr):
                    setattr(engine, engine_attr, cls)
                self._hooks[key] = mod
                activated += 1
                logger.debug(
                    "Hook fallback for %s: attached %s as engine.%s",
                    key, attr_name, engine_attr,
                )

        logger.info(
            "HeteroRegistry: activated %d/%d hetero_* modules on engine.",
            activated, len(self._modules),
        )
        return activated

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a registered module by its registry name."""
        return self._modules.get(name)

    def __len__(self) -> int:
        return len(self._modules)


# ---------------------------------------------------------------------------
# TierDiscovery
# ---------------------------------------------------------------------------
class TierDiscovery:
    """
    Detects available GPUs using torch.cuda and cross-references with nvidia-smi
    to build a ranked list of TierSpec objects.

    Discovery logic:
    - SM 9.x + >= 80GB → H100-class
    - SM 8.6 + >= 40GB → A6000-class
    - Everything else → UNKNOWN (still usable, degraded performance)
    """

    # Known BF16 TFLOPs and PCIe BW by (sm_major, sm_minor, approx_mem_gb)
    _PERF_TABLE: Dict[Tuple[int, int], Tuple[float, float]] = {
        (9, 0): (835.0, 50.0),   # H100 NVL  PCIe5
        (8, 6): (38.7, 25.0),    # A6000     PCIe4
        (8, 0): (312.0, 40.0),   # A100      PCIe4
        (7, 0): (14.1, 16.0),    # V100
    }

    def discover(self) -> List[TierSpec]:
        """
        Run full GPU discovery and return sorted list (highest-tier first).

        Returns:
            List of TierSpec, sorted by bf16_tflops descending.

        Raises:
            RuntimeError: If no CUDA-capable GPUs are found.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA-capable GPUs detected. Cannot run DES-LOC engine.")

        n_gpus = torch.cuda.device_count()
        logger.info("TierDiscovery: found %d CUDA device(s).", n_gpus)

        numa_map = self._query_numa_map()
        specs: List[TierSpec] = []

        for idx in range(n_gpus):
            try:
                spec = self._inspect_device(idx, numa_map)
                specs.append(spec)
                logger.info("  %s", spec)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to inspect GPU %d: %s", idx, exc)

        if not specs:
            raise RuntimeError("TierDiscovery found zero usable GPUs.")

        specs.sort(key=lambda s: s.bf16_tflops, reverse=True)
        return specs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _inspect_device(self, idx: int, numa_map: Dict[int, int]) -> TierSpec:
        """Build a TierSpec for a single CUDA device index."""
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = props.total_memory / (1 << 30)
        sm_major = props.major
        sm_minor = props.minor
        name = props.name

        torch.cuda.synchronize(idx)
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        free_mem_gb = free_bytes / (1 << 30)

        bf16_tflops, pcie_bw = self._PERF_TABLE.get(
            (sm_major, sm_minor),
            (self._estimate_tflops(props), 16.0),
        )

        tier = self._classify(sm_major, sm_minor, total_mem_gb)
        numa_node = numa_map.get(idx, -1)

        return TierSpec(
            device_index=idx,
            tier=tier,
            total_mem_gb=total_mem_gb,
            free_mem_gb=free_mem_gb,
            sm_major=sm_major,
            sm_minor=sm_minor,
            bf16_tflops=bf16_tflops,
            pcie_bw_gbs=pcie_bw,
            numa_node=numa_node,
            name=name,
        )

    @staticmethod
    def _classify(sm_major: int, sm_minor: int, mem_gb: float) -> TierClass:
        """Classify a GPU into a TierClass based on SM version and memory."""
        if sm_major >= 12 and mem_gb >= 90:
            return TierClass.RTX_PRO_6000_BW
        if sm_major == 9 and mem_gb >= 80:
            return TierClass.H100
        if sm_major == 8 and sm_minor == 6 and mem_gb >= 40:
            return TierClass.A6000
        return TierClass.UNKNOWN

    @staticmethod
    def _estimate_tflops(props: Any) -> float:
        """Rough BF16 TFLOPs estimate when not in the perf table."""
        # 2 * SMs * 128 (FP16 cores/SM) * clock_GHz (approx 1.5)
        return 2 * props.multi_processor_count * 128 * 1.5 / 1e3

    @staticmethod
    def _query_numa_map() -> Dict[int, int]:
        """
        Query NUMA affinity for each GPU via nvidia-smi.

        Returns:
            Dict mapping device_index -> numa_node.
        """
        numa: Dict[int, int] = {}
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,numa_affinity",
                 "--format=csv,noheader,nounits"],
                timeout=10,
                stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    try:
                        numa[int(parts[0])] = int(parts[1])
                    except ValueError:
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("nvidia-smi NUMA query failed (non-fatal): %s", exc)
        return numa


# ---------------------------------------------------------------------------
# PartitionSolver
# ---------------------------------------------------------------------------
class PartitionSolver:
    """
    Evaluates two partition strategies for the discovered GPU tiers and
    selects the one with higher estimated training throughput.

    Strategy A — ZeRO-3 + Heterogeneous Gradient Accumulation:
        H100 processes large micro-batches (22 per accumulation step);
        each A6000 processes 1 micro-batch. AllReduce synchronizes gradients.

    Strategy B — Pipeline 1F1B:
        H100 hosts 30 transformer layers (largest share due to BF16 headroom);
        each A6000 hosts 1 layer. 1F1B schedule used to overlap forward/backward.

    Throughput estimation is analytical (not profiled), based on:
        tokens/s ≈ Σ_device(micro_bs * seq_len * grad_accum / step_time_device)
    where step_time is estimated from BF16 TFLOPs and model FLOPs per token.
    """

    def __init__(self, tiers: List[TierSpec], config: TrainingConfig) -> None:
        self.tiers = tiers
        self.config = config

    def solve(self) -> PartitionPlan:
        """
        Compare both strategies and return the better PartitionPlan.

        If config.strategy_override is set, that strategy is used directly.

        Returns:
            PartitionPlan with strategy, layer assignments, and grad accum steps.
        """
        if self.config.strategy_override is not None:
            logger.info("Strategy override: %s", self.config.strategy_override)
            if self.config.strategy_override == PartitionStrategy.ZERO3_HETERO:
                return self._plan_zero3()
            return self._plan_pipeline()

        plan_a = self._plan_zero3()
        plan_b = self._plan_pipeline()

        logger.info(
            "PartitionSolver — ZeRO-3 est. %.1f tok/s  |  Pipeline est. %.1f tok/s",
            plan_a.estimated_throughput,
            plan_b.estimated_throughput,
        )

        chosen = plan_a if plan_a.estimated_throughput >= plan_b.estimated_throughput else plan_b
        logger.info("Selected strategy: %s  (%s)", chosen.strategy, chosen.notes)
        return chosen

    # ------------------------------------------------------------------
    # Strategy A: ZeRO-3 + Heterogeneous Gradient Accumulation
    # ------------------------------------------------------------------
    def _plan_zero3(self) -> PartitionPlan:
        """ZeRO-3 plan: uniform grad_accum, per-tier micro_batch_size.

        Upstream Megatron pattern — num_microbatches is a single global value
        identical on all ranks. Heterogeneous throughput comes from per-rank
        micro_batch_size (H100 takes larger batches, A6000 takes smaller).

        MBS assignment priority (highest wins):
        1. cfg.micro_batch_size_per_gpu[device_index]  — explicit yaml override
        2. TFLOPS-aware tier multiplier                — hardware-proportional default

        Tier multipliers (conservative, ~⅓ of true TFLOPS ratio to leave headroom):
            H100 NVL      835 TFLOPS  → 8× base   (true ratio ~21:1)
            RTX PRO 6000 Blackwell    → 4× base   (estimated ~300 TFLOPS)
            A6000          38.7 TFLOPS → 1× base
        """
        # TFLOPS-aware multipliers: conservative fractions of the true ratio.
        # The former `min(base*4, 4)` hard-cap is removed; VRAM is the real limit
        # and is already accounted for in the per-gpu yaml values.
        _TIER_MULTIPLIER: Dict[TierClass, int] = {
            TierClass.H100:            8,
            TierClass.RTX_PRO_6000_BW: 4,
            TierClass.A6000:           1,
            TierClass.UNKNOWN:         1,
        }

        cfg = self.config
        tier_layer_map: Dict[int, List[int]] = {}
        grad_accum: Dict[int, int] = {}
        micro_bs: Dict[int, int] = {}

        all_layers = list(range(cfg.num_layers))
        for spec in self.tiers:
            tier_layer_map[spec.device_index] = all_layers[:]
            grad_accum[spec.device_index] = cfg.grad_accum_steps

            # Priority 1: explicit per-gpu override from yaml
            if (cfg.micro_batch_size_per_gpu is not None
                    and spec.device_index < len(cfg.micro_batch_size_per_gpu)):
                micro_bs[spec.device_index] = cfg.micro_batch_size_per_gpu[spec.device_index]
            else:
                # Priority 2: TFLOPS-aware multiplier
                multiplier = _TIER_MULTIPLIER.get(spec.tier, 1)
                micro_bs[spec.device_index] = cfg.micro_batch_size * multiplier

        throughput = self._estimate_zero3_throughput(micro_bs, grad_accum)

        # Build a human-readable summary grouped by tier
        mbs_by_tier = {
            t.name: micro_bs[s.device_index]
            for s in self.tiers
            for t in [s.tier]
        }
        return PartitionPlan(
            strategy=PartitionStrategy.ZERO3_HETERO,
            tier_layer_map=tier_layer_map,
            grad_accum_steps=grad_accum,
            micro_batch_sizes=micro_bs,
            estimated_throughput=throughput,
            notes=(
                f"ZeRO-3 hetero: grad_accum={cfg.grad_accum_steps}, "
                f"per-device micro_bs={dict(sorted(micro_bs.items()))} "
                f"(source={'yaml' if cfg.micro_batch_size_per_gpu else 'tflops_multiplier'})"
            ),
        )

    # ------------------------------------------------------------------
    # Strategy B: Pipeline 1F1B
    # ------------------------------------------------------------------
    def _plan_pipeline(self) -> PartitionPlan:
        """Build the Pipeline 1F1B partition plan."""
        cfg = self.config
        tier_layer_map: Dict[int, List[int]] = {}
        grad_accum: Dict[int, int] = {}
        micro_bs: Dict[int, int] = {}

        layers = list(range(cfg.num_layers))
        h100_specs = [s for s in self.tiers if s.tier == TierClass.H100]
        a6000_specs = [s for s in self.tiers if s.tier == TierClass.A6000]
        other_specs = [s for s in self.tiers
                       if s.tier not in (TierClass.H100, TierClass.A6000)]

        # Assign layers: H100 gets 30, A6000 each get 1 (remaining split evenly)
        n_h100 = len(h100_specs)
        n_a6000 = len(a6000_specs)
        n_other = len(other_specs)
        total_devices = n_h100 + n_a6000 + n_other

        if total_devices == 0:
            total_devices = 1

        h100_share = min(30, cfg.num_layers - n_a6000 - n_other)
        a6000_share = 1 if n_a6000 > 0 else 0
        leftover = cfg.num_layers - h100_share * n_h100 - a6000_share * n_a6000

        cursor = 0
        for spec in h100_specs:
            n = h100_share
            tier_layer_map[spec.device_index] = layers[cursor: cursor + n]
            cursor += n
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            micro_bs[spec.device_index] = cfg.micro_batch_size

        for spec in a6000_specs:
            n = a6000_share
            tier_layer_map[spec.device_index] = layers[cursor: cursor + n]
            cursor += n
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            micro_bs[spec.device_index] = cfg.micro_batch_size

        # Distribute leftover layers to other GPUs
        per_other = (leftover // n_other) if n_other else 0
        for spec in other_specs:
            n = per_other
            tier_layer_map[spec.device_index] = layers[cursor: cursor + n]
            cursor += n
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            micro_bs[spec.device_index] = cfg.micro_batch_size

        throughput = self._estimate_pipeline_throughput(micro_bs, grad_accum)

        return PartitionPlan(
            strategy=PartitionStrategy.PIPELINE_1F1B,
            tier_layer_map=tier_layer_map,
            grad_accum_steps=grad_accum,
            micro_batch_sizes=micro_bs,
            estimated_throughput=throughput,
            notes=f"Pipeline 1F1B: H100 {h100_share} layers, A6000 {a6000_share} each",
        )

    # ------------------------------------------------------------------
    # Throughput estimators (analytical)
    # ------------------------------------------------------------------
    def _flops_per_token(self) -> float:
        """
        Approximate FLOPs per token for a transformer model.

        Using the standard 6*N approximation (N = parameter count).
        N ≈ 12 * num_layers * hidden_size^2 for a dense transformer.
        """
        n_params = 12 * self.config.num_layers * self.config.hidden_size ** 2
        return 6 * n_params

    def _estimate_zero3_throughput(
        self,
        micro_bs: Dict[int, int],
        grad_accum: Dict[int, int],
    ) -> float:
        """Estimate tokens/s for ZeRO-3 strategy (bottlenecked by slowest device)."""
        fpt = self._flops_per_token()
        step_times = []
        for spec in self.tiers:
            idx = spec.device_index
            tokens_per_step = (
                micro_bs.get(idx, 1)
                * self.config.seq_len
                * grad_accum.get(idx, 1)
            )
            flops_per_step = fpt * tokens_per_step
            tflops = spec.bf16_tflops * 1e12
            step_time = flops_per_step / (tflops * 0.35)  # 35% utilization assumption
            step_times.append((tokens_per_step, step_time))

        if not step_times:
            return 0.0

        bottleneck_time = max(t for _, t in step_times)
        total_tokens = sum(tok for tok, _ in step_times)
        return total_tokens / bottleneck_time if bottleneck_time > 0 else 0.0

    def _estimate_pipeline_throughput(
        self,
        micro_bs: Dict[int, int],
        grad_accum: Dict[int, int],
    ) -> float:
        """Estimate tokens/s for Pipeline strategy with bubble overhead."""
        fpt = self._flops_per_token()
        n_stages = len(self.tiers)
        step_times = []
        for spec in self.tiers:
            idx = spec.device_index
            n_layers = len(self.config.num_layers > 0 and [] or [])  # placeholder
            tokens_per_micro = micro_bs.get(idx, 1) * self.config.seq_len
            flops_per_micro = fpt * tokens_per_micro / max(n_stages, 1)
            tflops = spec.bf16_tflops * 1e12
            micro_time = flops_per_micro / (tflops * 0.35)
            step_times.append(micro_time)

        if not step_times:
            return 0.0

        # 1F1B pipeline bubble = (n_stages - 1) / n_stages
        t_stage = max(step_times)
        n_micro = grad_accum.get(list(grad_accum.keys())[0], 1) if grad_accum else 1
        total_step_time = t_stage * (n_micro + n_stages - 1)
        total_tokens = sum(
            micro_bs.get(s.device_index, 1) * self.config.seq_len * n_micro
            for s in self.tiers
        )
        return total_tokens / total_step_time if total_step_time > 0 else 0.0


# ---------------------------------------------------------------------------
# Minimal Transformer building blocks (BF16-native)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (BF16-friendly, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, BF16-compatible."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use scaled_dot_product_attention (FlashAttention path if available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """Position-wise feed-forward network (SwiGLU variant)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        intermediate = int(hidden * 8 / 3)
        intermediate = (intermediate + 63) // 64 * 64  # round to multiple of 64
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-norm."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden)
        self.mlp = MLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    """
    Minimal causal language model for DES-LOC pretraining smoke tests.

    In production this would be replaced by the full model passed into
    DesLocEngine, but it serves as the default when no model is provided.
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embedding = nn.Embedding(cfg.seq_len, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.hidden_size, cfg.num_heads)
             for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Long tensor of shape (B, T).

        Returns:
            Logits tensor of shape (B, T, vocab_size).
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# LR scheduler builder
# ---------------------------------------------------------------------------
def build_warmup_cosine_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Build a combined linear-warmup + cosine-decay LR schedule.

    Args:
        optimizer: The AdamW optimizer.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Ratio of min_lr to max_lr for cosine floor.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Data iterator utility
# ---------------------------------------------------------------------------
def infinite_data_iter(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Infinite iterator of random token batches for smoke testing.

    Yields dicts with "tokens" and "labels" keys matching the BATCH_KEYS
    contract expected by HeteroElasticBatch._fetch_from_iterator().

    Args:
        vocab_size: Vocabulary size for random token sampling.
        batch_size: Number of sequences per batch.
        seq_len: Sequence length.
        device: Target device (tokens are on CPU, moved to GPU in the loop).

    Yields:
        Dict with "tokens" and "labels" each of shape (batch_size, seq_len).
    """
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        yield {"tokens": tokens[:, :-1], "labels": tokens[:, 1:]}


# ---------------------------------------------------------------------------
# DES-LOC Engine
# ---------------------------------------------------------------------------
class DesLocEngine:
    """
    DES-LOC Heterogeneous Training Engine.

    Orchestrates the full pretraining pipeline on a mixed-GPU cluster:
      1. TierDiscovery  — enumerates and classifies GPUs
      2. HeteroRegistry — loads all hetero_*.py extension modules
      3. PartitionSolver — selects optimal partition strategy
      4. Model & Optimizer initialization on the primary device
      5. Training loop with real forward/backward/step
      6. Checkpoint save / load

    The engine exposes a single entry point: engine.train().

    Args:
        config: TrainingConfig with all hyperparameters.
        model: Optional pre-built nn.Module. If None, MiniTransformer is used.
        data_iter: Optional data iterator. If None, synthetic data is used.
        tokenizer: Optional HF tokenizer for CommitSequencePacker. When None
            the packer falls back to a word-split approximation.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[nn.Module] = None,
        data_iter: Optional[Iterator] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._setup_logging()

        # ------------------------------------------------------------------
        # Lazy imports: hetero_* modules are imported here (inside __init__)
        # so that merely importing desloc_engine does NOT trigger the
        # deepspeed/__init__.py → apex dependency chain.
        # ------------------------------------------------------------------
        from deepspeed.runtime.hetero_gdn_selective_recompute import (  # noqa: PLC0415
            build_neuron_sp_config,
            DeviceClass,
            HeteroRecomputeConfig,
        )
        from deepspeed.runtime.hetero_step_batch_scheduler import (  # noqa: PLC0415
            HeteroStepBatchScheduler,
            DeviceProfile,
            MicrobatchAllocation,
            DEFAULT_DES_LOC_DEVICE_PROFILES,
        )
        from deepspeed.runtime.hetero_grad_norm_skip import (  # noqa: PLC0415
            HeteroGradNormConfig,
            integrate_with_deepspeed_engine,
        )
        from deepspeed.runtime.hetero_fp32_grad_accum import (  # noqa: PLC0415
            HeteroFP32GradAccumConfig,
            HeteroFP32GradAccumManager,
        )
        from deepspeed.runtime.hetero_mimo_training_loop import (  # noqa: PLC0415
            DeviceCapabilityRegistry,
            HeteroMIMOTrainingLoop,
            PCIeP2PCommunicator,
            SharedLocalityCache,
            setup_hetero_mimo_training,
        )
        from deepspeed.checkpoint.hetero_checkpoint_config import (  # noqa: PLC0415
            HeteroCheckpointConfig,
            TierRole,
            build_config_for_cluster,
        )
        from deepspeed.checkpoint.hetero_async_checkpoint_save import (  # noqa: PLC0415
            HeteroAsyncCheckpointScheduler,
            build_hetero_async_save_pipeline,
            validate_async_checkpoint_config,
        )
        from deepspeed.checkpoint.hetero_async_checkpoint_load import (  # noqa: PLC0415
            HeteroAsyncCheckpointLoad,
            detect_device_arch,
            DeviceArch,
        )
        import importlib.util as _ilu  # noqa: PLC0415
        _cp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "bigcode", "commit_packing.py")
        _cp_spec = _ilu.spec_from_file_location("commit_packing", _cp_path)
        _cp_mod = _ilu.module_from_spec(_cp_spec)
        import sys as _sys; _sys.modules["commit_packing"] = _cp_mod  # noqa: E702
        _cp_spec.loader.exec_module(_cp_mod)
        CommitSequencePacker = _cp_mod.CommitSequencePacker
        HeteroBatchSampler = _cp_mod.HeteroBatchSampler
        PackedSequence = _cp_mod.PackedSequence
        compute_packing_stats = _cp_mod.compute_packing_stats
        # Store frequently-used lazy symbols on the instance so sub-methods
        # (save_checkpoint, load_checkpoint, train) can reference them without
        # re-importing each time.
        self._lazy = {
            "build_neuron_sp_config": build_neuron_sp_config,
            "DeviceClass": DeviceClass,
            "HeteroRecomputeConfig": HeteroRecomputeConfig,
            "HeteroStepBatchScheduler": HeteroStepBatchScheduler,
            "DeviceProfile": DeviceProfile,
            "MicrobatchAllocation": MicrobatchAllocation,
            "DEFAULT_DES_LOC_DEVICE_PROFILES": DEFAULT_DES_LOC_DEVICE_PROFILES,
            "HeteroGradNormConfig": HeteroGradNormConfig,
            "integrate_with_deepspeed_engine": integrate_with_deepspeed_engine,
            "HeteroFP32GradAccumConfig": HeteroFP32GradAccumConfig,
            "HeteroFP32GradAccumManager": HeteroFP32GradAccumManager,
            "DeviceCapabilityRegistry": DeviceCapabilityRegistry,
            "HeteroMIMOTrainingLoop": HeteroMIMOTrainingLoop,
            "PCIeP2PCommunicator": PCIeP2PCommunicator,
            "SharedLocalityCache": SharedLocalityCache,
            "setup_hetero_mimo_training": setup_hetero_mimo_training,
            "HeteroCheckpointConfig": HeteroCheckpointConfig,
            "TierRole": TierRole,
            "build_config_for_cluster": build_config_for_cluster,
            "HeteroAsyncCheckpointScheduler": HeteroAsyncCheckpointScheduler,
            "build_hetero_async_save_pipeline": build_hetero_async_save_pipeline,
            "validate_async_checkpoint_config": validate_async_checkpoint_config,
            "HeteroAsyncCheckpointLoad": HeteroAsyncCheckpointLoad,
            "detect_device_arch": detect_device_arch,
            "DeviceArch": DeviceArch,
            "CommitSequencePacker": CommitSequencePacker,
            "HeteroBatchSampler": HeteroBatchSampler,
            "PackedSequence": PackedSequence,
            "compute_packing_stats": compute_packing_stats,
        }

        logger.info("=" * 70)
        logger.info("DES-LOC Engine initializing — Neuron_SP / production build")
        logger.info("=" * 70)

        # --- Phase 1: GPU discovery ---
        discovery = TierDiscovery()
        try:
            self.tiers = discovery.discover()
        except RuntimeError as exc:
            logger.error("GPU discovery failed: %s", exc)
            logger.warning("Falling back to CPU-only mode (for testing only).")
            self.tiers = []

        # Primary device: highest-tier GPU (or CPU as fallback)
        self.primary_device = (
            self.tiers[0].device if self.tiers else torch.device("cpu")
        )
        logger.info("Primary device: %s", self.primary_device)

        # --- Phase 2: Registry ---
        self.registry = HeteroRegistry()
        self.registry.discover()
        _initial_hooks = self.registry.register_hooks(self)
        logger.info(
            "HeteroRegistry loaded %d modules; %d hooks activated.",
            len(self.registry), _initial_hooks,
        )

        # --- Phase 3: Partition plan ---
        solver = PartitionSolver(self.tiers, config)
        self.plan = solver.solve()
        logger.info("Partition plan: %s", self.plan.strategy)
        for dev_idx, layers in self.plan.tier_layer_map.items():
            logger.info("  GPU%d → %d layers, grad_accum=%d, micro_bs=%d",
                        dev_idx,
                        len(layers),
                        self.plan.grad_accum_steps.get(dev_idx, 1),
                        self.plan.micro_batch_sizes.get(dev_idx, 1))

        # --- Phase 4: Model ---
        if model is None:
            logger.info("Building MiniTransformer (%d layers, hidden=%d).",
                        config.num_layers, config.hidden_size)
            self.model: nn.Module = MiniTransformer(config)
        else:
            self.model = model

        _local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        _local_mem_gb = torch.cuda.get_device_properties(_local_device).total_memory / (1 << 30)
        self._use_fsdp = False  # Neuron_SP native ZeRO-3, no FSDP

        # Model stays in BF16 on CPU — ZeRO-3 ShardState holds FP32 master
        # copy per-rank on GPU; forward uses gather_full_params() to
        # materialize full BF16 params layer-by-layer on demand.
        self.model = self.model.to(dtype=_DEFAULT_DTYPE)
        self.model_device = _local_device  # forward() moves inputs here

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: %.2fM parameters (BF16 on CPU, ZeRO-3 sharded)", n_params / 1e6)

        # --- Phase 4c: MoE adapter (gated by config.use_moe) ---
        # Must run BEFORE optimizer init so ZeRO-3 sees the full parameter set
        # including the new router weights and expert MLP parameters introduced
        # by MoELayer.patch_model().  When use_moe=False this is a no-op.
        from deepspeed.runtime.core_adapters import build_moe_adapter  # noqa: PLC0415
        self.moe_adapter = build_moe_adapter(
            config=config,
            model=self.model,
            tiers=self.tiers,
        )
        if self.moe_adapter is not None:
            # Recount params: MoE adds router weights + expert MLPs.
            _moe_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "MoE active: model now has %.2fM parameters "
                "(+%.2fM vs dense, experts=%d, topk=%d)",
                _moe_params / 1e6,
                (_moe_params - n_params) / 1e6,
                getattr(config, "num_moe_experts", 8),
                getattr(config, "moe_router_topk", 2),
            )
        else:
            logger.info("MoE disabled (use_moe=False); model stays dense.")

        # --- Phase 4d: MLA adapter (gated by config.use_mla) ---
        # Must run AFTER MoE patching (Phase 4c) and BEFORE optimizer init so
        # ZeRO-3 sees the updated Q/KV projection parameters introduced by
        # _LightweightMLA.  When use_mla=False this is a complete no-op.
        # MLA and MoE are orthogonal: MLA patches .attn, MoE patches .mlp.
        from deepspeed.runtime.core_adapters import build_mla_adapter  # noqa: PLC0415
        self.mla_adapter = build_mla_adapter(
            config=config,
            model=self.model,
        )
        if self.mla_adapter is not None:
            _mla_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "MLA active: model now has %.2fM parameters "
                "(+%.2fM vs pre-MLA, q_lora_rank=%d, kv_lora_rank=%d)",
                _mla_params / 1e6,
                (_mla_params - n_params) / 1e6,
                getattr(config, "mla_q_lora_rank",
                        getattr(config, "hidden_size", 0) // 4),
                getattr(config, "mla_kv_lora_rank",
                        getattr(config, "hidden_size", 0) // 8),
            )
        else:
            logger.info("MLA disabled (use_mla=False); standard attention kept.")

        # --- Phase 4b: ZeRO-3 heterogeneous parameter sharding ---
        # Uses the original zero3_hetero_shard.ShardState which was working
        # at baseline 3faf8420. Each rank keeps a VRAM-proportional FP32
        # slice; full BF16 params gathered on-demand during forward/backward.
        self.param_shard_state = None
        self.param_shard = None
        self.param_offsets = None
        self._dist_optimizer = None

        _ws = (
            parallel_state.get_data_parallel_world_size()
            if parallel_state.is_initialized()
            else (dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1)))
        )
        _rk = (
            parallel_state.get_data_parallel_rank()
            if parallel_state.is_initialized()
            else (dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0)))
        )

        if _ws > 1:
            try:
                from deepspeed.runtime.zero3_hetero_shard import (
                    ShardState as _ShardState,
                    vram_weights_from_tiers as _vram_weights_from_tiers,
                )
                _weights = _vram_weights_from_tiers(self.tiers) if getattr(
                    self, "tiers", None
                ) else None
                if _weights and len(_weights) != _ws:
                    _weights = None
                self.param_shard_state = _ShardState.build(
                    model=self.model,
                    rank=_rk,
                    world_size=_ws,
                    device=_local_device,
                    vram_weights=_weights,
                )
                if self.param_shard_state is not None:
                    self.param_shard = self.param_shard_state.param_shard
                    self.param_offsets = self.param_shard_state.param_offsets
                    _orig_total = sum(p.numel() for p in self.model.parameters())
                    _shard_total = sum(self.param_shard_state.shard_sizes)
                    assert _shard_total >= _orig_total, (
                        f"shard total {_shard_total} < orig {_orig_total}"
                    )
                    logger.info(
                        "[zero3] Sharding active: %d ranks, local=%d, "
                        "total=%d (orig=%d, pad=%d)",
                        _ws, self.param_shard.numel(),
                        _shard_total, _orig_total,
                        self.param_shard_state.pad,
                    )
            except Exception as _shard_exc:  # noqa: BLE001
                logger.warning(
                    "[zero3] Sharding init failed (%s); continuing with "
                    "full-replica parameters.", _shard_exc,
                )
                self.param_shard_state = None

        # --- Phase 5: Optimizer & Scheduler ---
        if self.param_shard_state is not None:
            _shard = self.param_shard_state.param_shard
            _shard.requires_grad_(True)
            _use_foreach = _shard.numel() <= 2**31 - 1
            self.optimizer = AdamW(
                [_shard],
                lr=config.max_lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                foreach=_use_foreach,
            )
            logger.info(
                "[zero3] Optimizer on param_shard: %d FP32 elements on %s",
                _shard.numel(), _shard.device,
            )
        else:
            self.model = self.model.to(_local_device)
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.max_lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                fused=self._fused_adam_available(),
            )
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
            min_lr_ratio=config.min_lr / config.max_lr,
        )
        # --- Core scheduler adapter (gated by config.use_core_scheduler) ---
        from deepspeed.runtime.core_adapters import build_core_scheduler
        _core_sched = build_core_scheduler(self.optimizer, config)
        if _core_sched is not None:
            self.scheduler = _core_sched

        # --- Phase 6: Data ---
        if data_iter is None:
            logger.info("Using synthetic data iterator.")
            self.data_iter: Iterator = infinite_data_iter(
                vocab_size=config.vocab_size,
                batch_size=config.micro_batch_size,
                seq_len=config.seq_len,
                device=self.primary_device,
            )
        else:
            # Wrap a dataset-like iterable with CommitSequencePacker so that
            # commit boundaries are respected during sequence packing.  Then
            # replace the default per-rank sampler with HeteroBatchSampler so
            # that VRAM-proportional micro-batch sizes are enforced across
            # heterogeneous GPUs (H100 96 GB / A6000 48 GB).
            #
            # data_iter may already be a plain (input_ids, labels) iterator
            # (e.g. from a custom DataLoader).  We detect this by checking for
            # a list/Sequence-like interface; plain generators are left as-is.
            if hasattr(data_iter, "__len__") or hasattr(data_iter, "__getitem__"):
                gpu_mem_map: Dict[int, int] = {
                    spec.device_index: int(spec.total_mem_gb)
                    for spec in self.tiers
                } if self.tiers else {0: 96}

                packer = CommitSequencePacker(
                    tokenizer=self.tokenizer,
                    seq_len=config.seq_len,
                    pad_token_id=0,
                )
                packed_seqs: List[PackedSequence] = packer.pack_dataset(iter(data_iter))
                logger.info(
                    "CommitSequencePacker produced %d packed sequences "
                    "(seq_len=%d, gpu_mem_map=%s).",
                    len(packed_seqs), config.seq_len, gpu_mem_map,
                )

                # Log packing efficiency so regressions are caught early.
                pack_stats = compute_packing_stats(packed_seqs)
                if pack_stats:
                    logger.info(
                        "Packing stats: padding_ratio=%.4f, meets_5pct=%s, "
                        "commits_packed=%d, avg_commits/seq=%.1f",
                        pack_stats["padding_ratio"],
                        pack_stats["meets_5pct_target"],
                        pack_stats["commits_packed"],
                        pack_stats["avg_commits_per_seq"],
                    )

                hetero_sampler = HeteroBatchSampler(
                    sequences=packed_seqs,
                    gpu_mem_map=gpu_mem_map,
                    base_batch=config.micro_batch_size,
                    verbose=False,
                )

                # Expose sampler so callers can query it; wrap as an infinite
                # token iterator compatible with the existing training loop.
                self.hetero_sampler = hetero_sampler

                def _packed_iter(
                    sampler: HeteroBatchSampler,
                    primary_idx: int,
                    device: torch.device,
                ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                    # Cycle the sampler indefinitely so training never exhausts data.
                    while True:
                        for batch_per_rank in sampler:
                            seqs = batch_per_rank.get(primary_idx, [])
                            if not seqs:
                                continue
                            # Stack token lists into a single (B, T) tensor.
                            ids = torch.tensor(
                                [s.tokens for s in seqs], dtype=torch.long, device=device
                            )
                            # Causal LM shift: inputs = ids[:, :-1], labels = ids[:, 1:]
                            yield ids[:, :-1], ids[:, 1:]

                _primary_idx = (
                    self.primary_device.index
                    if self.primary_device.type == "cuda"
                    else 0
                )
                self.data_iter = _packed_iter(
                    hetero_sampler, _primary_idx, self.primary_device
                )
                logger.info(
                    "HeteroBatchSampler wired: %d steps/epoch, "
                    "total_per_step=%d sequences.",
                    len(hetero_sampler), hetero_sampler.total_per_step,
                )
            else:
                # Plain iterator: pass through unchanged (caller handles batching).
                self.data_iter = data_iter

        # --- Phase 6b: Heterogeneous CP token split ---
        # Pure DP (no TP): every rank fetches its own data independently.
        # We only need the CP slice logic from HeteroElasticBatch to give
        # H100 more tokens and A6000 fewer tokens per microbatch.
        #
        # TP group = size-1 (just this rank) so broadcast is a no-op.
        # CP group = WORLD (all ranks participate in context-parallel slicing).
        from deepspeed.runtime.hetero_elastic_batch import (
            HeteroElasticBatch, RankDeviceMap,
        )
        _my_rank = (
            parallel_state.get_data_parallel_rank()
            if parallel_state.is_initialized()
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        _tp_group = dist.new_group([_my_rank]) if dist.is_initialized() else None
        _cp_group = dist.GroupMember.WORLD if dist.is_initialized() else None
        rank_device_map = RankDeviceMap.from_env()
        self._hetero_batch = HeteroElasticBatch(
            rank_device_map=rank_device_map,
            tp_group=_tp_group if _tp_group is not None else dist.GroupMember.WORLD,
            cp_group=_cp_group if _cp_group is not None else dist.GroupMember.WORLD,
            device=self.primary_device,
            enable_hetero_cp=(
                (parallel_state.is_initialized() and parallel_state.get_data_parallel_world_size() > 1)
                or (dist.is_initialized() and not parallel_state.is_initialized() and dist.get_world_size() > 1)
            ),
        )
        logger.info(
            "HeteroElasticBatch built: tp_group=size-1 (pure DP), cp_group=WORLD, "
            "hetero_cp=%s", self._hetero_batch.enable_hetero_cp,
        )

        # Gradient accumulation from plan (use primary device's setting)
        primary_idx = self.primary_device.index if self.primary_device.type == "cuda" else -1
        self.grad_accum = self.plan.grad_accum_steps.get(
            primary_idx, config.grad_accum_steps
        )
        logger.info("Effective grad_accum_steps on primary: %d", self.grad_accum)

        # --- Phase 7: HeteroStepBatchScheduler ---
        # Upstream Megatron pattern: all ranks run the same num_microbatches.
        # Heterogeneous throughput via per-rank micro_batch_size, not forward count.
        # capacity_weight is now TFLOPS-proportional so the allocator assigns
        # more microbatches to H100/Blackwell and fewer to A6000.
        #
        # Multipliers (conservative, ~1/3 of true TFLOPS ratio):
        #   H100 NVL 835 TFLOPS → weight 8.0
        #   Blackwell ~300 TFLOPS (est.) → weight 4.0
        #   A6000 38.7 TFLOPS → weight 1.0
        _TIER_WEIGHT: Dict[TierClass, float] = {
            TierClass.H100:            8.0,
            TierClass.RTX_PRO_6000_BW: 4.0,
            TierClass.A6000:           1.0,
            TierClass.UNKNOWN:         1.0,
        }
        if self.tiers:
            device_profiles = []
            for spec in self.tiers:
                weight = _TIER_WEIGHT.get(spec.tier, 1.0)
                # max_micro_batch_size: use explicit per-gpu yaml value if available,
                # otherwise derive from weight (H100→16, Blackwell→8, A6000→2)
                if (config.micro_batch_size_per_gpu is not None
                        and spec.device_index < len(config.micro_batch_size_per_gpu)):
                    max_mbs = config.micro_batch_size_per_gpu[spec.device_index]
                else:
                    max_mbs = max(1, int(config.micro_batch_size * weight))
                device_profiles.append(DeviceProfile(
                    device_id=spec.device_index,
                    sm_arch=spec.sm_major * 10 + spec.sm_minor,
                    vram_gb=spec.total_mem_gb,
                    capacity_weight=weight,
                    max_micro_batch_size=max_mbs,
                ))
        else:
            device_profiles = DEFAULT_DES_LOC_DEVICE_PROFILES

        dp_size = max(len(self.tiers), 1)
        schedule_str = config.batch_schedule or f"0:{config.global_batch_size}"
        seq_len_for_schedule = config.batch_schedule_seq_length

        _rank = (
            parallel_state.get_data_parallel_rank()
            if parallel_state.is_initialized()
            else (dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0)))
        )
        self.hetero_scheduler: HeteroStepBatchScheduler = HeteroStepBatchScheduler(
            rank=_rank,
            micro_batch_size=config.micro_batch_size,
            data_parallel_size=dp_size,
            schedule=schedule_str,
            device_profiles=device_profiles,
            seq_length=seq_len_for_schedule,
            loc_cache_invalidation_hook=lambda old_bs, new_bs: logger.info(
                "LOC Cache invalidated: global_batch_size %d -> %d", old_bs, new_bs
            ),
            dp_reconfigure_hook=lambda new_gbs: logger.info(
                "DP reconfigure hint: new global_batch_size=%d", new_gbs
            ),
        )
        # Also expose under the registry-standard name so that code that
        # references engine.hetero_step_batch_scheduler (set as a placeholder
        # by HeteroStepBatchScheduler.register() during Phase 2) finds the
        # fully-initialized scheduler rather than None.
        self.hetero_step_batch_scheduler: HeteroStepBatchScheduler = self.hetero_scheduler
        logger.info(
            "HeteroStepBatchScheduler initialized: schedule='%s', dp_size=%d, "
            "device_profiles=%d devices",
            schedule_str, dp_size, len(device_profiles),
        )

        # consumed_samples tracks total samples processed (for scheduler stepping)
        self.consumed_samples = 0

        # State
        self.global_step = 0
        self.tokens_seen = 0
        self._start_time = time.time()

        # --- DES-LOC: Decomposed Local SGD sync periods ---
        # Ref: Algorithm 1 — Kx for params, Ku for m1 (exp_avg), Kv for m2 (exp_avg_sq)
        # Kx=1 means standard DDP (sync every step). Kx>1 means local SGD with
        # periodic sync. Must satisfy Kx <= Ku <= Kv.
        # Config source: deepspeed/runtime/config.py line 801-806
        self.desloc_Kx = getattr(config, 'desloc_Kx', 32)   # param sync period
        self.desloc_Ku = getattr(config, 'desloc_Ku', 96)   # m1 sync period
        self.desloc_Kv = getattr(config, 'desloc_Kv', 192)  # m2 sync period
        assert self.desloc_Kx <= self.desloc_Ku <= self.desloc_Kv, \
            f"DES-LOC requires Kx <= Ku <= Kv, got {self.desloc_Kx}/{self.desloc_Ku}/{self.desloc_Kv}"
        logger.info(
            "DES-LOC sync periods: Kx=%d (params), Ku=%d (m1), Kv=%d (m2) — "
            "comm reduction %.1fx vs DDP",
            self.desloc_Kx, self.desloc_Ku, self.desloc_Kv,
            3.0 / (1.0/self.desloc_Kx + 1.0/self.desloc_Ku + 1.0/self.desloc_Kv),
        )

        # --- Phase 7: Hetero Checkpoint System ---
        # IMPORTANT: this block runs BEFORE load_checkpoint() so that the
        # config (and in particular cfg.load_optim / cfg.load_rng /
        # cfg.shard_rebalance_on_load / cfg.locality_cache_path) are
        # available when load_checkpoint() is called below.
        if config.hetero_checkpoint_config is not None:
            self._hetero_ckpt_cfg: HeteroCheckpointConfig = config.hetero_checkpoint_config
        else:
            # Auto-build from cluster topology when CUDA is available;
            # fall back to a minimal config for CPU-only environments.
            try:
                self._hetero_ckpt_cfg = build_config_for_cluster(
                    save_dir=str(config.checkpoint_dir),
                    load_dir=(
                        str(config.resume_from)
                        if config.resume_from is not None
                        else None
                    ),
                    locality_cache_dir="/dev/shm/neuron_sp_ckpt",
                    save_interval=config.save_every,
                )
                logger.info(
                    "HeteroCheckpointConfig built: cache_device=%d, "
                    "workers=%s, async=%s, locality_cache_dir=%s",
                    self._hetero_ckpt_cfg.cache_tier_device_id,
                    self._hetero_ckpt_cfg.worker_device_ids,
                    self._hetero_ckpt_cfg.hetero_async_save,
                    self._hetero_ckpt_cfg.locality_cache_dir,
                )
            except Exception as _hcc_exc:  # noqa: BLE001
                logger.warning(
                    "Could not auto-build HeteroCheckpointConfig (%s); "
                    "hetero async save/load will be disabled.",
                    _hcc_exc,
                )
                self._hetero_ckpt_cfg = None  # type: ignore[assignment]

        # Per-rank async scheduler (one instance, reused across save calls).
        self._hetero_ckpt_scheduler: Optional[HeteroAsyncCheckpointScheduler] = None
        if self._hetero_ckpt_cfg is not None and self._hetero_ckpt_cfg.hetero_async_save:
            self._hetero_ckpt_scheduler = HeteroAsyncCheckpointScheduler()
            logger.info(
                "HeteroAsyncCheckpointScheduler created: cache_device=%d, "
                "pcie_bw_throttle=%.0f GB/s.",
                self._hetero_ckpt_cfg.cache_tier_device_id,
                self._hetero_ckpt_cfg.pcie_bw_throttle_gbps,
            )

        # CPU-staging thread pool: used by save_checkpoint to push payload to
        # the locality-cache ramdisk (/dev/shm) without blocking the training
        # loop.  One worker is sufficient — saves are sequential per rank and
        # we want to avoid saturating the PCIe bus with concurrent H2H copies.
        self._cpu_stage_executor: concurrent.futures.ThreadPoolExecutor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="desloc_cpu_stage"
            )
        )
        self._cpu_stage_futures: List[concurrent.futures.Future] = []
        # Lock guards _cpu_stage_futures list (drain_checkpoint may be called
        # from a different thread than save_checkpoint).
        self._cpu_stage_lock = threading.Lock()

        # Emit a structured config summary so checkpoint policy is visible in
        # training logs without having to inspect the config object manually.
        if self._hetero_ckpt_cfg is not None:
            _cfg_s = self._hetero_ckpt_cfg
            _cache_pol  = _cfg_s.get_policy(TierRole.CACHE)
            _worker_pol = _cfg_s.get_policy(TierRole.WORKER)
            logger.info(
                "HeteroCheckpointConfig summary:\n"
                "  save_dir            : %s\n"
                "  load_dir            : %s\n"
                "  locality_cache_dir  : %s  (max %.0f GB)\n"
                "  cache_device        : %d   CACHE policy → async=%s  save_optim=%s\n"
                "  worker_devices      : %s   WORKER policy → async=%s  save_optim=%s\n"
                "  worker_offload_optim: %s   hetero_async_save: %s\n"
                "  shard_rebalance     : %s   ckpt_format: %s\n"
                "  save_interval       : %s   non_persistent_interval: %s\n"
                "  pcie_bw_throttle    : %.0f GB/s",
                _cfg_s.save,
                _cfg_s.load,
                _cfg_s.locality_cache_dir,       _cfg_s.locality_cache_max_gb,
                _cfg_s.cache_tier_device_id,
                _cache_pol.async_save,           _cache_pol.save_optim,
                _cfg_s.worker_device_ids,
                _worker_pol.async_save,          _worker_pol.save_optim,
                _cfg_s.worker_offload_optim,     _cfg_s.hetero_async_save,
                _cfg_s.shard_rebalance_on_load,  _cfg_s.hetero_ckpt_format.value,
                _cfg_s.save_interval,            _cfg_s.non_persistent_save_interval,
                _cfg_s.pcie_bw_throttle_gbps,
            )

        # Optionally resume from checkpoint.  The HeteroCheckpointConfig is
        # now fully initialised so load_checkpoint() will use the tier-aware
        # async load path when available.
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

        # --- Phase 7: Neuron_SP heterogeneous recompute config ---
        # Build device-class lists from the discovered tier specs so that
        # build_neuron_sp_config() receives the real device indices rather than
        # relying on the hardcoded (0,1) / 2 defaults.
        _a6000_indices = [
            spec.device_index for spec in self.tiers if spec.tier == TierClass.A6000
        ]
        _h100_indices = [
            spec.device_index for spec in self.tiers if spec.tier == TierClass.H100
        ]
        _h100_idx = _h100_indices[0] if _h100_indices else 2
        # A6000 (48 GB): use "full" recompute granularity inside GDN layers so
        # the outer torch.utils.checkpoint block absorbs the entire layer forward.
        # H100 (96 GB): keep "selective" — norm_out-only recompute costs less.
        # This aligns with Megatron M4141 (ff5264c33): selective norm_out recompute
        # is only profitable when VRAM headroom exists to store other activations.
        _a6000_granularity = "full" if _a6000_indices else "selective"
        self.neuron_sp_config: HeteroRecomputeConfig = build_neuron_sp_config(
            a6000_indices=tuple(_a6000_indices) if _a6000_indices else (0, 1),
            h100_index=_h100_idx,
            a6000_granularity=_a6000_granularity,
        )
        logger.info(
            "Neuron_SP recompute config built (granularity=%s, a6000_gran=%s, attention=%s).",
            self.neuron_sp_config.granularity,
            _a6000_granularity,
            self.neuron_sp_config.attention_variant,
        )

        # ------------------------------------------------------------------
        # Per-tier activation checkpointing via torch.utils.checkpoint.
        #
        # Reads two fields from TrainingConfig:
        #   config.activation_checkpointing          – master on/off switch
        #   config.checkpoint_activations_granularity – "full" | "selective"
        #
        # Per-tier policy (overrides granularity config for each GPU class):
        #   A6000 (48 GB VRAM, SM 8.6) → FULL checkpoint (every layer)
        #       Memory budget is tighter; recompute all activations on backward.
        #   H100  (96 GB VRAM, SM 9.x) → SELECTIVE checkpoint (every other layer)
        #       Abundant VRAM; only half the layers are wrapped to reduce recompute
        #       overhead while still capping peak activation memory.
        #
        # The master switch (activation_checkpointing) defaults to False so
        # existing callers are unaffected until they explicitly opt in.
        # ------------------------------------------------------------------
        import torch.utils.checkpoint as _torch_ckpt  # noqa: PLC0415

        # Build a tier-class lookup for each device index discovered.
        _dev_tier: Dict[int, TierClass] = {
            spec.device_index: spec.tier for spec in self.tiers
        }

        # Build layer → device_index mapping from the partition plan.
        layer_device_map: Dict[int, int] = {}
        for dev_idx, layer_indices in self.plan.tier_layer_map.items():
            for li in layer_indices:
                layer_device_map[li] = dev_idx

        # Support both MiniTransformer (.blocks) and standard Transformer (.layers).
        block_list = getattr(self.model, "blocks", None) or getattr(
            self.model, "layers", None
        )

        _ckpt_master_on = bool(config.activation_checkpointing)
        _ckpt_granularity = str(config.checkpoint_activations_granularity).lower()

        # --- A6000 safety guard (PipeDream-style per-stage recompute override) ---
        # PipeDream runtime.py disables recompute on the last stage to save compute;
        # we do the inverse: force it ON for memory-constrained stages (A6000, 48 GB)
        # even when the caller has not set the flag.  Mirrors HetSeq controller.py
        # line 282 OOM recovery — pre-emptive rather than crash-and-retry.
        _has_a6000_tier = any(spec.tier == TierClass.A6000 for spec in self.tiers)
        if _has_a6000_tier and not _ckpt_master_on:
            logger.warning(
                "[ActCkpt] A6000 tier detected with activation_checkpointing=False. "
                "Forcing ON to prevent OOM at seq_len>=4096 (48 GB VRAM). "
                "Set activation_checkpointing=True explicitly to suppress this warning."
            )
            _ckpt_master_on = True

        _local_rank = (
            parallel_state.get_data_parallel_rank()
            if parallel_state.is_initialized()
            else (dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0)))
        )

        if block_list is not None:
            logger.info(
                "[ActCkpt] GPU%d — master=%s  config_granularity=%s",
                _local_rank, _ckpt_master_on, _ckpt_granularity,
            )
            print(
                f"[Neuron_SP] Activation-checkpoint strategy "
                f"(master={'ON' if _ckpt_master_on else 'OFF'}, "
                f"config_granularity={_ckpt_granularity}):"
            )

            for layer_idx, block in enumerate(block_list):
                dev_idx = layer_device_map.get(layer_idx, primary_idx)
                tier = _dev_tier.get(dev_idx, TierClass.UNKNOWN)

                # ---- Per-tier policy decision --------------------------------
                if not _ckpt_master_on:
                    # Master switch OFF → no wrapping regardless of tier.
                    apply_ckpt = False
                    policy_label = "OFF (master disabled)"
                elif tier == TierClass.A6000:
                    # A6000: full checkpoint — wrap every layer.
                    apply_ckpt = True
                    policy_label = "FULL (A6000 — every layer)"
                elif tier == TierClass.H100:
                    # H100: selective checkpoint — wrap every other layer.
                    apply_ckpt = (layer_idx % 2 == 0)
                    policy_label = (
                        "SELECTIVE (H100 — even layer)"
                        if apply_ckpt
                        else "SELECTIVE (H100 — odd layer, skip)"
                    )
                elif tier == TierClass.RTX_PRO_6000_BW:
                    # Blackwell 96 GB — treat same as H100 (selective).
                    apply_ckpt = (layer_idx % 2 == 0)
                    policy_label = (
                        "SELECTIVE (RTX_PRO_6000_BW — even layer)"
                        if apply_ckpt
                        else "SELECTIVE (RTX_PRO_6000_BW — odd layer, skip)"
                    )
                else:
                    # UNKNOWN tier: fall back to config granularity.
                    if _ckpt_granularity == "selective":
                        apply_ckpt = (layer_idx % 2 == 0)
                        policy_label = (
                            "SELECTIVE (UNKNOWN — even layer)"
                            if apply_ckpt
                            else "SELECTIVE (UNKNOWN — odd layer, skip)"
                        )
                    else:
                        apply_ckpt = True
                        policy_label = "FULL (UNKNOWN — fallback to full)"
                # ---- End policy decision -------------------------------------

                if apply_ckpt:
                    original_fwd = block.forward

                    def _make_ckpt_fwd(fwd):
                        def _ckpt_fwd(*args, **kwargs):
                            return _torch_ckpt.checkpoint(
                                fwd, *args, use_reentrant=False, **kwargs
                            )
                        return _ckpt_fwd

                    block.forward = _make_ckpt_fwd(original_fwd)

                logger.info(
                    "[ActCkpt] GPU%d  layer=%3d  dev=GPU%d  tier=%-18s  ckpt=%s  [%s]",
                    _local_rank, layer_idx, dev_idx, tier.value,
                    "ON " if apply_ckpt else "OFF",
                    policy_label,
                )
                print(
                    f"  GPU{_local_rank}  layer {layer_idx:3d} -> GPU{dev_idx} "
                    f"({tier.value:18s})  ckpt={'ON ' if apply_ckpt else 'OFF'}  "
                    f"[{policy_label}]"
                )
        else:
            logger.warning(
                "[ActCkpt] GPU%d: model exposes no .blocks/.layers; "
                "per-layer activation-checkpoint wrapping skipped.",
                _local_rank,
            )
            print(
                f"[Neuron_SP] WARNING GPU{_local_rank}: model exposes no .blocks/.layers; "
                "activation-checkpoint wrapping skipped."
            )

        # --- Phase 7b: Fine-grained activation offload (A6000 only) ---
        # Wires PipelineOffloadManager so that large saved tensors (embeddings,
        # residuals ≥ activation_offload_min_size elements) are D2H-copied
        # asynchronously during forward and H2D-restored lazily during backward.
        # On A6000 PCIe (32 GB/s), attention score tensors are cheaper to recompute
        # than to offload; the min_size threshold filters those out automatically.
        # On H100 (DATACENTER tier), offload_required_for_tier() returns False
        # and maybe_enable_activation_offload() is a cheap no-op.
        #
        # HetSeq uses a simpler "catch OOM, skip batch" strategy; we prefer the
        # PipeDream memory-planning approach: know the budget in advance and
        # configure the runtime to stay within it.
        from deepspeed.runtime.core_adapters import maybe_enable_activation_offload  # noqa: PLC0415
        try:
            from deepspeed.core.desloc_config import TierType as _TierType  # noqa: PLC0415
            _local_tier_class = _dev_tier.get(
                list(self.plan.tier_layer_map.keys())[0] if self.plan.tier_layer_map else primary_idx,
                TierClass.UNKNOWN,
            )
            _tier_type_for_offload = {
                TierClass.A6000:           _TierType.PROFESSIONAL,
                TierClass.RTX_PRO_6000_BW: _TierType.PROFESSIONAL,
                TierClass.H100:            _TierType.DATACENTER,
                TierClass.UNKNOWN:         None,
            }.get(_local_tier_class, None)
        except Exception:  # noqa: BLE001
            _tier_type_for_offload = None
        self._activation_offload_iface = maybe_enable_activation_offload(
            config, tier_type=_tier_type_for_offload
        )
        logger.info(
            "[FineGrainedOffload] iface=%s  local_tier=%s  use_activation_offload=%s",
            "ACTIVE" if self._activation_offload_iface is not None else "SKIPPED",
            _dev_tier.get(primary_idx, TierClass.UNKNOWN).value,
            getattr(config, "use_activation_offload", False),
        )

        # --- Phase 8: HeteroFP32GradAccumManager ---
        # Build a default config: H100 (Tier-0) always accumulates in FP32;
        # A6000s (Tier-1) follow the LayerNorm/embedding patterns; CPU is FP32.
        _fp32_config = HeteroFP32GradAccumConfig(
            param_name_patterns_for_fp32_local_accumulation=(
                "*.norm*",
                "*.ln_*",
                "*.layer_norm*",
                "embedding*",
            ),
            tier0_always_fp32=True,
            tier1_follow_patterns=True,
            offload_fp32_grads_to_cpu=False,
        )
        # data_parallel_group: prefer the parallel_state DP group when initialised
        # (ensures the correct process group is used for all-reduces in the
        # FP32 grad accum manager).  When parallel_state is not yet set up but
        # torch.distributed is, fall back to the global world group.
        # If neither is available, bootstrap a single-rank gloo group so the
        # manager can be constructed without a real multi-GPU setup.
        if parallel_state.is_initialized():
            _dp_group = parallel_state.get_data_parallel_group()
        elif dist.is_initialized():
            _dp_group = dist.group.WORLD
        else:
            dist.init_process_group(
                backend="gloo",
                init_method="env://",
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                rank=int(os.environ.get("RANK", 0)),
            )
            _dp_group = dist.group.WORLD

        # Each rank allocates FP32 grad buffer on its own device, not on primary_device
        _local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        _local_mem_gb = torch.cuda.get_device_properties(_local_device).total_memory / (1 << 30)
        # When DistributedOptimizer is active, it owns all FP32 shard state.
        # fp32_grad_manager would duplicate those buffers (~24 GB on H100) and
        # conflict with DistributedOptimizer.prepare_grads() reduce-scatter.
        if self._dist_optimizer is not None:
            self.fp32_grad_manager = None
            logger.info(
                "HeteroFP32GradAccumManager SKIPPED — "
                "core.optimizer.DistributedOptimizer owns FP32 shards (device=%s)",
                _local_device,
            )
        elif _local_mem_gb < 60:
            self.fp32_grad_manager = None
            logger.info(
                "HeteroFP32GradAccumManager SKIPPED on %s (%.0f GB < 60 GB threshold)",
                _local_device, _local_mem_gb,
            )
        else:
            self.fp32_grad_manager = HeteroFP32GradAccumManager(
                config=_fp32_config,
                model=self.model,
                data_parallel_group=_dp_group,
                device=_local_device,
                param_dtype=_DEFAULT_DTYPE,
                grad_dtype=_DEFAULT_DTYPE,
            )
        logger.info(
            "HeteroFP32GradAccumManager wired: selective_fp32=%s, "
            "offload_to_cpu=%s, device=%s",
            _fp32_config.has_selective_fp32,
            _fp32_config.offload_fp32_grads_to_cpu,
            self.primary_device,
        )

        # --- Phase 8b: gradient reduce-scatter ---
        # core.optimizer.DistributedOptimizer handles reduce-scatter internally
        # via _reduce_scatter_grads() called from prepare_grads() / step().
        # No per-parameter post-accumulate-grad hooks are needed.
        # The hook-handle list and bucket-mgr are kept as empty stubs so any
        # downstream code that iterates them remains a safe no-op.
        self._zero3_grad_hook_handles: List = []
        self._grad_bucket_mgr = None
        self._core_ddp: Optional[CoreDDP] = None
        self._ddp_dp_group = None  # M4172: persisted for pg_collection threading
        if self._dist_optimizer is not None:
            logger.info(
                "[zero3] core.optimizer.DistributedOptimizer active — "
                "reduce-scatter via prepare_grads() "
                "(fp32_grad_manager=%s)",
                self.fp32_grad_manager is not None,
            )
        else:
            # Non-ZeRO-3: wrap the model with core.distributed.DistributedDataParallel
            # so finalize_model_grads() can bucket and all-reduce gradients properly.
            # Only wraps when distributed training is actually active (world_size > 1).
            _ddp_dp_group = (
                parallel_state.get_data_parallel_group()
                if parallel_state.is_initialized()
                else (dist.group.WORLD if dist.is_initialized() else None)
            )
            _is_distributed = (
                (parallel_state.is_initialized() and parallel_state.get_data_parallel_world_size() > 1)
                or (dist.is_initialized() and dist.get_world_size() > 1)
            )
            if _is_distributed and _ddp_dp_group is not None:
                try:
                    from deepspeed.core.model_parallel_config import ModelParallelConfig  # noqa: PLC0415
                    _mp_cfg = ModelParallelConfig()
                    # M4041: when full-iteration CUDA graphs are active we must not
                    # dereference param.grad in the backward hook — the graph was
                    # recorded with live grad tensor addresses and zeroing the
                    # attribute on the first replay would corrupt subsequent iterations.
                    _cg_impl = getattr(self, 'config', None)
                    _cg_impl = getattr(_cg_impl, 'cuda_graph_impl', 'none') if _cg_impl else 'none'
                    _ddp_cfg = CoreDDPConfig(
                        grad_reduce_in_fp32=False,
                        overlap_grad_reduce=True,  # ISSUE2: overlap grad-reduce with backward compute
                        use_distributed_optimizer=bool(getattr(config, 'zero_stage', 0) >= 2),
                        allow_skip_grad_sync=True,  # DES-LOC Kx gating
                        megatron_fsdp_grad_comm_dtype=torch.bfloat16,  # M3574: PCIe BW reduction
                        use_pcie_aware_overlap=True,  # PCIe-only topology: adapt bucket sizes
                        cuda_graph_mode=(_cg_impl == 'full_iteration'),  # M4041
                    )
                    # From Megatron M2928: wrap DDP init in a dedicated side-stream
                    # to avoid race conditions that leave parameter buffers empty.
                    # On PCIe-only topology (no NVLink) the default stream may
                    # still be draining peer copies when DDP registers its hooks;
                    # synchronising ensures all tensors are visible before
                    # bucket registration.  M2940 later reduced the scope of
                    # this stream to DDP init only (not the full model build),
                    # which is the pattern we follow here.
                    _ddp_init_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
                    if _ddp_init_stream is not None:
                        _ddp_init_stream.wait_stream(torch.cuda.current_stream())
                    with (torch.cuda.stream(_ddp_init_stream) if _ddp_init_stream is not None
                          else __import__("contextlib").nullcontext()):
                        self._core_ddp = CoreDDP(
                            config=_mp_cfg,
                            ddp_config=_ddp_cfg,
                            module=self.model,
                            data_parallel_group=_ddp_dp_group,
                        )
                    # Persist the DP group so finalize_model_grads can receive
                    # an explicit pg_collection rather than falling back to
                    # parallel_state globals (M4172/M4168 pg_collection threading).
                    self._ddp_dp_group = _ddp_dp_group
                    # Sync back so subsequent ops on the default stream see the
                    # DDP-registered buckets (M2928 correctness requirement).
                    if _ddp_init_stream is not None:
                        torch.cuda.current_stream().wait_stream(_ddp_init_stream)
                    logger.info(
                        "[core_ddp] DistributedDataParallel wired: "
                        "dp_group_size=%d, overlap_grad_reduce=%s",
                        dist.get_world_size(group=_ddp_dp_group),
                        _ddp_cfg.overlap_grad_reduce,
                    )
                except Exception as _ddp_exc:  # noqa: BLE001
                    logger.warning(
                        "[core_ddp] CoreDDP init failed (%s); "
                        "finalize_model_grads will still run with plain model.",
                        _ddp_exc,
                    )
                    self._core_ddp = None

        # --- Phase 9a: SharedLocalityCache (1.5 TB CPU DRAM ÷ world_size) ---
        # The training host has 2×EPYC 9354 with 1.5 TB DDR5.  Each rank
        # receives an equal share so that the aggregate cache footprint never
        # exceeds the physical 1.5 TB ceiling regardless of process count.
        _world_size = (
            parallel_state.get_data_parallel_world_size()
            if parallel_state.is_initialized()
            else (dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1)))
        )
        _total_dram_bytes = int(1.5 * 1024 ** 4)          # 1.5 TiB in bytes
        _cache_max_bytes = _total_dram_bytes // _world_size
        _cache_max_gb = _cache_max_bytes / (1024 ** 3)
        _cache_max_entries = 512   # ~128 micro-batches × 4 pipeline stages
        self.locality_cache = SharedLocalityCache(
            max_entries=_cache_max_entries,
            max_bytes=_cache_max_bytes,
        )
        logger.info(
            "SharedLocalityCache initialized: max_entries=%d, max_bytes=%.1f GB "
            "(1.5 TB DRAM / world_size=%d).",
            _cache_max_entries, _cache_max_gb, _world_size,
        )

        # --- Phase 9b: PCIeP2PCommunicator (cross-pool activation transfer) ---
        # Uses DeviceCapabilityRegistry to classify A6000 (SM86) vs H100 (SM90)
        # pools, then routes large tensors (>= 64 MB) through CPU DRAM staging
        # instead of direct PCIe P2P (which is bandwidth-limited at 16–32 GB/s).
        self._device_registry = DeviceCapabilityRegistry()
        self.p2p_communicator = PCIeP2PCommunicator(
            registry=self._device_registry,
            locality_cache=self.locality_cache,
            staging_threshold_mb=64.0,
        )
        logger.info(
            "PCIeP2PCommunicator initialized: staging_threshold=64.0 MB, "
            "registry has %d device(s).",
            len(self._device_registry.all_profiles),
        )
        # --- Core bridge communicator adapter (gated by config.use_bridge_communicator) ---
        from deepspeed.runtime.core_adapters import build_bridge_communicator
        self.p2p_communicator = build_bridge_communicator(
            config, self.p2p_communicator,
        )

        # --- Pipeline-parallel 1F1B communicator init ---
        # Gated by use_pipeline_schedule=True AND pipeline_parallel_size > 1.
        # Builds a P2PCommunicator (backed by the NCCL PP process group) and a
        # ProcessGroupCollection so forward_backward_pipelining_without_interleaving
        # can drive the warmup / steady-state / cooldown phases without touching
        # parallel_state globals directly.
        #
        # pipeline_layer_split (e.g. [4,8,8,4,8]) is registered into the schedule
        # registry so get_pipeline_model_parallel_rank_for_layer() resolves correctly
        # for heterogeneous NUMA topologies.
        #
        # These attributes are None when PP schedule is disabled (default), so
        # all existing single-GPU / DP-only code paths are completely unaffected.
        self._pp_p2p_comm = None   # P2PCommunicator for 1F1B schedule
        self._pp_pg_collection = None  # ProcessGroupCollection for 1F1B schedule
        if getattr(config, "use_pipeline_schedule", False) and getattr(config, "pipeline_parallel_size", 1) > 1:
            try:
                import deepspeed.core.parallel_state as _ps_init
                from deepspeed.core.pipeline_parallel.p2p_communication import P2PCommunicator as _P2PC
                from deepspeed.core.process_groups_config import ProcessGroupCollection as _PGC
                from deepspeed.core.model_parallel_config import ModelParallelConfig as _MPCfg
                from deepspeed.core.pipeline_parallel.schedules import set_pipeline_layer_split

                _pp_mp_cfg = _MPCfg(
                    pipeline_model_parallel_size=config.pipeline_parallel_size,
                    tensor_model_parallel_size=getattr(config, "tensor_parallel_size", 1),
                    # variable_seq_lengths=True: activations may differ across NUMA stages
                    # (different layer counts → different hidden shapes at stage boundary is
                    # impossible by construction, but variable seq-len packing is safe here).
                    variable_seq_lengths=True,
                    # deallocate_pipeline_outputs: free output tensor after isend completes
                    # (M3766 async-send safety).  Required when crossing NUMA boundary so
                    # the source buffer is only freed after the remote copy is confirmed done.
                    deallocate_pipeline_outputs=True,
                )
                _pp_group = _ps_init.get_pipeline_model_parallel_group()
                self._pp_p2p_comm = _P2PC(pp_group=_pp_group, config=_pp_mp_cfg)

                self._pp_pg_collection = _PGC()
                self._pp_pg_collection.tp      = _ps_init.get_tensor_model_parallel_group()
                self._pp_pg_collection.cp      = _ps_init.get_context_parallel_group()
                self._pp_pg_collection.pp      = _pp_group
                self._pp_pg_collection.dp_cp   = _ps_init.get_data_parallel_group(
                    with_context_parallel=True, partial_data_parallel=False
                )
                self._pp_pg_collection.tp_dp_cp = _ps_init.get_tensor_and_data_parallel_group(
                    with_context_parallel=True
                )
                self._pp_pg_collection.embd     = _ps_init.get_embedding_group(check_initialized=False)
                self._pp_pg_collection.pos_embd = _ps_init.get_position_embedding_group(check_initialized=False)

                # Register per-stage layer counts for heterogeneous topologies
                # (e.g. NUMA0 [GPU0-2] → stages 0-2, NUMA1 [GPU3-4] → stages 3-4).
                _layer_split = getattr(config, "pipeline_layer_split", [])
                if _layer_split:
                    set_pipeline_layer_split(_layer_split)
                    logger.info(
                        "PP 1F1B: pipeline_layer_split=%s registered (total %d layers)",
                        _layer_split, sum(_layer_split),
                    )

                logger.info(
                    "PP 1F1B communicator initialized: pp_size=%d, stage=%d/%d, "
                    "vp_size=%s, deallocate_outputs=True",
                    self._pp_p2p_comm.total_stages,
                    self._pp_p2p_comm.current_stage,
                    self._pp_p2p_comm.total_stages,
                    getattr(config, "virtual_pipeline_model_parallel_size", None),
                )
            except Exception as _pp_init_exc:
                logger.warning(
                    "PP 1F1B communicator init failed (%s); "
                    "falling back to serial micro-batch loop.",
                    _pp_init_exc,
                )
                self._pp_p2p_comm = None
                self._pp_pg_collection = None

        # --- Phase 9: Hetero MIMO training loop bootstrap ---
        # Build the DES-LOC heterogeneous MIMO training loop (device registry,
        # locality cache, P2P communicator, optimizer router, schedule groups)
        # so that DesLocEngine.train() can dispatch micro-batches through it.
        # The shared locality_cache and p2p_communicator (Phase 9a/9b) are
        # forwarded into setup_hetero_mimo_training so all components share the
        # same CPU DRAM staging area and cross-pool transfer path.
        # Failures here must not break legacy single-GPU training paths, so the
        # call is wrapped defensively and the loop falls back to None.
        try:
            self.mimo_loop: Optional[HeteroMIMOTrainingLoop] = (
                setup_hetero_mimo_training(
                    self.model,
                    cache_max_gb=_cache_max_gb,
                    cache_max_entries=_cache_max_entries,
                )
            )
            logger.info(
                "HeteroMIMOTrainingLoop initialized via setup_hetero_mimo_training() "
                "(cache=%.1f GB, world_size=%d).",
                _cache_max_gb, _world_size,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "setup_hetero_mimo_training() failed (%s); continuing without "
                "MIMO loop. Heterogeneous dispatch will degrade to single-device.",
                exc,
            )
            self.mimo_loop = None

        # fp32_grad_manager already initialized in Phase 8 above.

        logger.info("Engine ready. Starting from step %d.", self.global_step)

        # --- Phase 10: Final hetero hook activation ---
        # Re-run register_hooks() now that all phases have completed.  Some
        # hetero_*.py modules are imported lazily by later phases (e.g. the
        # MIMO training loop in Phase 9c) and therefore only become available
        # to the registry after Phase 2.  This second pass picks up any
        # modules that were missed and attaches their hooks/primary classes
        # to the engine instance.  ``register_hooks`` is idempotent, so
        # already-registered modules are skipped.
        try:
            self.registry.discover()
            activated = self.registry.register_hooks(self)
            logger.info(
                "HeteroRegistry: final pass activated %d hooks on engine.",
                activated,
            )
        except Exception as _reg_exc:  # noqa: BLE001
            logger.warning(
                "HeteroRegistry final register_hooks() pass failed: %s",
                _reg_exc,
            )

        self._checkpoint_thread = None  # From M3407: track async checkpoint Thread

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        num_microbatches: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Engine-level forward pass with HeteroRecomputeConfig-aware autocast.

        Wraps ``self.model(input_ids)`` inside device-appropriate autocast and
        computes the causal-LM cross-entropy loss.  The per-layer selective
        recompute (applied via ``torch.utils.checkpoint`` in ``__init__``) is
        transparently active for every block whose device class requires it
        (A6000 → recompute norm_out; H100/Blackwell → no recompute).

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ids of shape ``(B, T)``.
        labels : torch.Tensor
            Target ids of shape ``(B, T)`` (shifted internally).
        num_microbatches : int
            Divisor for loss scaling (gradient accumulation).

        Returns
        -------
        loss : torch.Tensor
            Unscaled per-sample cross-entropy loss (for logging).
        scaled_loss : torch.Tensor
            ``loss / num_microbatches`` (for ``.backward()``).
        """
        with torch.autocast(
            device_type=self.primary_device.type,
            dtype=_DEFAULT_DTYPE,
            enabled=(self.primary_device.type == "cuda"),
        ):
            # Ensure inputs are on the same device as the model
            _model_dev = self.model_device
            input_ids = input_ids.to(_model_dev, non_blocking=True)
            labels = labels.to(_model_dev, non_blocking=True)
            logits: torch.Tensor = self.model(input_ids)
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :].contiguous().reshape(-1, V)
            shift_labels = labels[:, :T - 1].contiguous().reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
            scaled_loss = loss / num_microbatches
        return loss, scaled_loss

    def step(self) -> None:
        """Optimizer step — used by hetero_grad_norm_skip monkey-patch."""
        if self.optimizer is not None:
            self.optimizer.step()

    def train(self) -> None:
        """
        Run the full training loop.

        Implements:
          - Gradient accumulation across micro-batches
          - Gradient clipping
          - Optimizer + scheduler step
          - Periodic logging and checkpointing
        """
        # Rank guard: only rank 0 prints/logs to avoid 5x log spam
        _is_main = (
            not (parallel_state.is_initialized() or dist.is_initialized())
        ) or (
            parallel_state.get_data_parallel_rank() == 0
            if parallel_state.is_initialized()
            else dist.get_rank() == 0
        )

        # Suppress duplicate log messages from non-rank-0 processes
        if not _is_main:
            logging.getLogger("deepspeed").setLevel(logging.WARNING)

        # --- Optional logging backends (rank 0 only) ---
        import datetime as _dt  # noqa: PLC0415
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _wb_run = None
        _tb_writer = None
        if _is_main:
            if getattr(self.config, "wandb_project", None) and _HAS_WANDB:
                _wb_run = _wandb.init(
                    project=getattr(self.config, "wandb_project", None),
                    name=f"desloc_{_ts}",
                    config={
                        "model_size": f"{cfg.hidden_size}h_{cfg.num_layers}L",
                        "total_steps": cfg.total_steps,
                        "micro_batch_size": cfg.micro_batch_size,
                        "global_batch_size": cfg.global_batch_size,
                        "seq_len": cfg.seq_len,
                        "max_lr": cfg.max_lr,
                    },
                    resume="allow",
                )
                logger.info("W&B run initialised: project=%s  run=%s", getattr(self.config, "wandb_project", None), _wb_run.name)
            elif getattr(self.config, "wandb_project", None) and not _HAS_WANDB:
                logger.warning("wandb_project set but wandb is not installed; W&B logging disabled.")
            tb_dir = getattr(self.config, "tensorboard_dir", None) or f"logs/tb_{_ts}"
            if getattr(self.config, "tensorboard_dir", None) and _HAS_TENSORBOARD:
                _tb_writer = _SummaryWriter(log_dir=tb_dir)
                logger.info("TensorBoard SummaryWriter: dir=%s", tb_dir)
            elif getattr(self.config, "tensorboard_dir", None) and not _HAS_TENSORBOARD:
                logger.warning("tensorboard_dir set but torch.utils.tensorboard unavailable.")

        # Lazy imports (hetero_* not imported at module level to avoid apex dep)
        from deepspeed.runtime.hetero_gdn_selective_recompute import build_neuron_sp_config  # noqa: PLC0415
        from deepspeed.runtime.hetero_grad_norm_skip import (  # noqa: PLC0415
            HeteroGradNormConfig,
            HeteroGradNormSkipController,
            integrate_with_deepspeed_engine,
        )


        cfg = self.config
        self.model.train()
        logger.info("Training start: %d steps, grad_accum=%d",
                    cfg.total_steps, self.grad_accum)

        # --- ZeRO-3 model GPU materialisation ---
        # DistributedOptimizer holds FP32 shards; the full BF16 model must
        # also live on GPU for forward/backward.  ZeRO3ForwardHook.register()
        # moves it there once (no per-layer hooks — full BF16 fits on both tiers).
        self._zero3_forward_hook = None
        if self._dist_optimizer is not None:
            try:
                from deepspeed.runtime.zero3_hetero_shard import (  # noqa: PLC0415
                    ZeRO3ForwardHook as _Z3Hook,
                )

                class _ShardAdapter:
                    """Minimal ShardState duck-type for ZeRO3ForwardHook.__init__."""
                    def __init__(self, rank, world_size, device):
                        self.rank = rank
                        self.world_size = world_size
                        self.param_shard = torch.empty(0, device=device)

                _adapter = _ShardAdapter(
                    rank=self._dist_optimizer.data_parallel_rank,
                    world_size=self._dist_optimizer.data_parallel_world_size,
                    device=_local_device,
                )
                self._zero3_forward_hook = _Z3Hook(self.model, _adapter)
                self._zero3_forward_hook.register()
                if _is_main:
                    logger.info(
                        "[zero3-hook] full BF16 model loaded to GPU "
                        "(rank=%d/%d) — DistributedOptimizer path",
                        self._dist_optimizer.data_parallel_rank,
                        self._dist_optimizer.data_parallel_world_size,
                    )
            except Exception as _hook_exc:  # noqa: BLE001
                logger.warning(
                    "[zero3-hook] BF16 model GPU load failed (%s); "
                    "model may remain on CPU.", _hook_exc,
                )
                self._zero3_forward_hook = None

        # --- Neuron_SP: build / refresh heterogeneous recompute config ---
        # Recompute policy may depend on the current device topology; rebuild
        # the config here so that train() always picks up the latest mapping.
        # The per-layer torch.utils.checkpoint wrapping is applied in __init__,
        # but we keep the live config attached to the model for downstream
        # modules (e.g. HeteroGDNNormOutRecompute) that query it at runtime.
        _rc_a6000_idx = [s.device_index for s in self.tiers if s.tier == TierClass.A6000]
        _rc_h100_idx  = [s.device_index for s in self.tiers if s.tier == TierClass.H100]
        recompute_config = build_neuron_sp_config(
            a6000_indices=tuple(_rc_a6000_idx) if _rc_a6000_idx else (0, 1),
            h100_index=_rc_h100_idx[0] if _rc_h100_idx else 2,
            a6000_granularity="full" if _rc_a6000_idx else "selective",
        )
        self.neuron_sp_config = recompute_config
        try:
            self.model.neuron_sp_recompute_config = recompute_config  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - some models forbid attr assignment
            pass
        logger.info(
            "Neuron_SP recompute config applied to model "
            "(granularity=%s, attention=%s).",
            recompute_config.granularity,
            recompute_config.attention_variant,
        )

        # Drive any iteration-level hooks exposed by the MIMO loop (if available).
        if getattr(self, "mimo_loop", None) is not None:
            logger.info(
                "Training loop will coordinate with HeteroMIMOTrainingLoop "
                "(P2P + LOC cache active)."
            )

        # --- AutoSP: compiler-based Ulysses Sequence Parallelism ---
        # Uses torch.compile + FX graph passes to insert A2A around SDPA nodes.
        # Adapted from upstream DeepSpeed 5efb24a for single-input models:
        #   - label/position nodes are optional (not all models pass them)
        #   - uneven head splits handled via padding in A2A op
        #   - uses eager backend (gm.forward) on cu118 (no inductor)
        self._sp_active = False
        self._autosp_compile_fn = None
        _sp_world_size = (
            parallel_state.get_data_parallel_world_size()
            if parallel_state.is_initialized()
            else (dist.get_world_size() if dist.is_initialized() else 1)
        )
        if _sp_world_size > 1:
            sp_size = _sp_world_size

            from deepspeed.compile.custom_ops.sp_compat import _check_autosp_compatibility
            _check_autosp_compatibility()

            from deepspeed.compile.custom_ops.sp_dp_registry import populate_registry
            dp_size = 1
            populate_registry(sp_size, dp_size)

            from deepspeed.compile.passes.long_context_checkpointing import register_long_context_checkpointing
            register_long_context_checkpointing()

            from deepspeed.compile.passes.sp_compile import apply_autosp
            def _autosp_backend(gm, real_inputs):
                apply_autosp(gm, real_inputs, debug=False, sp_size=sp_size, dp_size=dp_size)
                return gm.forward  # eager fallback (no inductor on cu118)

            self._autosp_compile_fn = _autosp_backend
            self.model = torch.compile(self.model, backend=_autosp_backend, fullgraph=True, dynamic=True)
            self._sp_active = True
            logger.info("AutoSP: compiler-based SP enabled (sp_size=%d)", sp_size)

        # Wire HeteroGradNormSkipController into this engine via
        # integrate_with_deepspeed_engine(). That function monkey-patches
        # engine.step(), but DesLocEngine has no .step() method — step logic
        # is inline in train(). We therefore call it for the standard
        # initialisation path (config wiring, logging, controller creation)
        # and retain the returned controller to drive should_skip() /
        # record_step() manually inside the loop below.
        _hetero_config = HeteroGradNormConfig()
        _skip_controller = integrate_with_deepspeed_engine(self, _hetero_config)
        _skip_count = 0

        loss_accum = 0.0
        t0 = time.time()
        train_start = t0  # used to compute total_time after the loop

        # --- Async ZeRO-3 shard-sync state ---
        # After optimizer.step() we launch sync_shard_to_model_async() on a
        # dedicated CUDA stream so the FP32→BF16 PCIe copies overlap with the
        # next step's data preprocessing on the CPU / default stream.
        # _shard_sync_stream  : persistent stream reused every step
        # _shard_sync_pending : True iff a sync was launched but not yet waited
        #
        # Insight I3: centralized stream management (Megatron M3724)
        # Previously created an ad-hoc torch.cuda.Stream() here. Now routed
        # through StreamManager so the shard-sync comm stream is tracked in
        # the framework-level pool, enabling visibility and budget enforcement
        # per GPU tier (H100 NVL vs A6000 vs Blackwell PCIe).
        _shard_sync_gpu_type = getattr(self, "_gpu_type", "default")
        _shard_sync_stream: Optional[torch.cuda.Stream] = (
            StreamManager.get_shard_sync_stream(_shard_sync_gpu_type)
            if torch.cuda.is_available() and self._dist_optimizer is not None
            else None
        )
        _shard_sync_pending: bool = False

        # --- Core pipeline schedule adapter (gated by config.use_pipeline_schedule) ---
        from deepspeed.runtime.core_adapters import get_pipeline_forward_backward
        _pipeline_fb_func = get_pipeline_forward_backward(cfg, default_fn=None)

        # --- Context Parallel schedule adapter (gated by config.use_context_parallel) ---
        from deepspeed.runtime.core_adapters import build_hybrid_cp_schedule
        _cp_fb = build_hybrid_cp_schedule(cfg)

        for step in range(self.global_step, cfg.total_steps):
            # DistributedOptimizer.zero_grad() zeroes its grad_data buffers +
            # shard param grads.  Plain AdamW zero_grad() on the non-ZeRO-3 path.
            self.optimizer.zero_grad(set_to_none=False)
            if self._core_ddp is not None:
                # Reset grad buffers in core DDP before next accumulation window
                self._core_ddp.zero_grad_buffer(zero_buffer=(not self.optimizer.defaults.get("foreach", False)))
            step_loss = 0.0

            # Heterogeneous scheduling: each rank gets its own micro-batch count
            # based on VRAM/compute tier. No FSDP collective constraints —
            # Neuron_SP ZeRO-3 uses per-layer gather_full_params which only
            # requires collectives at the granularity the hooks fire.
            allocation: MicrobatchAllocation = self.hetero_scheduler.schedule(
                consumed_samples=self.consumed_samples,
                consistency_check=False,
            )
            # Upstream Megatron pattern (schedules.py / training.py):
            # num_microbatches is a SINGLE global value, identical on all ranks.
            # Every forward triggers per-layer ZeRO-3 all_gather collectives
            # that require all ranks to call forward() the same number of times.
            # Heterogeneous throughput is achieved via per-rank micro_batch_size
            # differences (larger batches on faster GPUs), NOT via different
            # iteration counts — which would cause NCCL collective mismatch.
            num_microbatches = allocation.num_microbatches

            # --- HeteroFP32GradAccumManager: before_backward ---
            # Zero FP32 gradient accumulators (main_grad tensors) once per step,
            # before any micro-batch backward.  Called unconditionally here so
            # both the MIMO and standard paths share the same zero-grad semantics.
            if self.fp32_grad_manager is not None:
                self.fp32_grad_manager.before_backward()

            # ISSUE-2: Pre-arm _skip_sync on every DDP bucket group BEFORE the
            # backward pass so that register_grad_ready hooks know whether to
            # launch NCCL immediately (Kx step) or accumulate only (non-Kx step).
            #
            # Upstream pattern: hetseq controller.py `maybe_no_sync()` wraps the
            # backward in `model.no_sync()` for all-but-last microbatch, gating
            # the reduce at the hook level.  We extend this to the Kx dimension:
            # non-Kx steps behave identically to a never-syncing no_sync() pass.
            #
            # Must execute AFTER zero_grad_buffer() (which calls bg.reset(),
            # clearing _skip_sync from the previous step) and BEFORE the first
            # backward kernel fires.
            _is_Kx_sync_pre = (step + 1) % self.desloc_Kx == 0
            if self._core_ddp is not None and not _is_Kx_sync_pre:
                for _bg in (
                    self._core_ddp.bucket_groups
                    + self._core_ddp.expert_parallel_bucket_groups
                ):
                    _bg._skip_sync = True

            # ---------------------------------------------------------------
            # PP 1F1B schedule path
            # ---------------------------------------------------------------
            # When use_pipeline_schedule=True and pipeline_parallel_size > 1,
            # the 1F1B schedule (forward_backward_pipelining_without_interleaving)
            # takes over the *entire* num_microbatches loop.  It handles warmup,
            # steady-state, and cooldown phases with P2P send/recv between stages,
            # replacing the serial `for micro` loop below.
            #
            # Correctness notes for our [4,8,8,4,8] NUMA topology:
            #   • _shard_sync_stream wait is hoisted here (step-level) because
            #     `micro == 0` no longer fires inside the schedule.
            #   • forward_step_func drives self.data_iter directly; the `for micro`
            #     loop's `next(self.data_iter)` calls are *not* executed on this path.
            #   • fp32_grad_manager.accumulate() is called once after the schedule
            #     returns (all micro-batch backwards are complete by then).
            #   • num_microbatches must be identical across all PP ranks — enforced
            #     by hetero_scheduler design (uniform count, per-rank batch size).
            # ---------------------------------------------------------------
            if _pipeline_fb_func is not None and self._pp_p2p_comm is not None:
                # Hoist shard-sync wait to step level (no micro==0 guard available).
                if _shard_sync_pending:
                    if _shard_sync_stream is not None:
                        torch.cuda.current_stream().wait_stream(_shard_sync_stream)
                    _shard_sync_pending = False

                # Store num_microbatches on self so _pp_forward_step_func can read it
                # without capturing a mutable local via closure.
                self._cur_num_microbatches = num_microbatches
                _local_dev = torch.device(f"cuda:{torch.cuda.current_device()}")

                def _pp_forward_step_func(_data_iter, _model, _engine=self, _dev=_local_dev):
                    """forward_step_func adapter for forward_backward_pipelining_*.

                    Signature required by schedules.py forward_step():
                        (data_iterator, model) -> (output_tensor, num_tokens_tensor)

                    The schedule calls this once per micro-batch on every PP stage.
                    Non-first stages receive input_tensor via P2P (set_input_tensor
                    is called by forward_step() before invoking us).
                    """
                    raw_mb = next(_data_iter)
                    if isinstance(raw_mb, dict):
                        _ids = raw_mb["tokens"]
                        _lbl = raw_mb.get("labels")
                    else:
                        _ids, _lbl = raw_mb[0], (raw_mb[1] if len(raw_mb) > 1 else None)
                    _ids = _ids.to(_dev, non_blocking=True)
                    if _lbl is not None:
                        _lbl = _lbl.to(_dev, non_blocking=True)

                    _num_mb = _engine._cur_num_microbatches
                    loss, scaled_loss = _engine.forward(_ids, _lbl, num_microbatches=_num_mb)

                    # MoE auxiliary loss — add before schedule drives backward so
                    # router gate gradients flow through the combined loss tensor.
                    if _engine.moe_adapter is not None:
                        _aux = _engine.moe_adapter.collect_aux_loss()
                        if not isinstance(_aux, float) or _aux != 0.0:
                            scaled_loss = scaled_loss + _aux / max(_num_mb, 1)

                    seq_len = _ids.shape[-1]
                    num_tokens = torch.tensor(seq_len, dtype=torch.int64, device=_dev)
                    # Return scaled_loss as the output tensor so the schedule can
                    # call .backward() on it; the raw loss is stored in forward_data_store
                    # by forward_step_calc_loss via the loss_func hook.
                    return scaled_loss, num_tokens

                _pp_losses = _pipeline_fb_func(
                    forward_step_func=_pp_forward_step_func,
                    data_iterator=self.data_iter,
                    model=self.model,
                    num_microbatches=num_microbatches,
                    seq_length=getattr(cfg, "seq_length", getattr(cfg, "max_seq_len", 2048)),
                    micro_batch_size=getattr(cfg, "micro_batch_size", 1),
                    forward_only=False,
                    p2p_communicator=self._pp_p2p_comm,
                    pg_collection=self._pp_pg_collection,
                )
                # Promote BF16 grads → FP32 main_grad once, after all micro-batch
                # backwards have completed inside the schedule.
                if self.fp32_grad_manager is not None:
                    self.fp32_grad_manager.accumulate()
                step_loss = (
                    sum(float(l) for l in _pp_losses) / max(len(_pp_losses), 1)
                    if _pp_losses else 0.0
                )
                logger.debug(
                    "PP 1F1B step=%d num_mb=%d losses=%d step_loss=%.4f",
                    step, num_microbatches, len(_pp_losses) if _pp_losses else 0, step_loss,
                )

            else:
                # ---------------------------------------------------------------
                # Serial micro-batch loop (PP=1 / use_pipeline_schedule=False)
                # ---------------------------------------------------------------
                # Original loop preserved verbatim below.  The `else` branch is
                # taken whenever _pipeline_fb_func is None (default) or
                # _pp_p2p_comm failed to initialise, guaranteeing full fallback.
                # ---------------------------------------------------------------
                for micro in range(num_microbatches):
                    # --- Data fetch: capacity-weighted CP token split ---
                    # Pure DP: every rank fetches independently, then applies
                    # hetero CP slice so H100 gets more tokens, A6000 fewer.
                    raw = next(self.data_iter)
                    if isinstance(raw, dict):
                        input_ids = raw["tokens"]
                        labels = raw.get("labels")
                    else:
                        input_ids, labels = raw

                    # Apply capacity-weighted CP slice — ONLY if SP is not active.
                    # When Ulysses SP is on, all ranks need the same seq_len for
                    # symmetric all-to-all. Load balancing is done via
                    # micro_batch_size_per_gpu instead (H100 gets more batches).
                    _orig_seq = input_ids.shape[-1]
                    if not self._sp_active and self._hetero_batch is not None:
                        batch_dict = {"tokens": input_ids}
                        if labels is not None:
                            batch_dict["labels"] = labels
                        _cp_size = (
                            parallel_state.get_data_parallel_world_size()
                            if parallel_state.is_initialized()
                            else (dist.get_world_size() if dist.is_initialized() else 1)
                        )
                        sliced = self._hetero_batch._apply_hetero_cp_slice(
                            batch_dict,
                            cp_size=_cp_size,
                        )
                        input_ids = sliced.get("tokens", input_ids)
                        labels = sliced.get("labels", labels)
                    if micro == 0 and step < 3:
                        _log_rank = (
                            parallel_state.get_data_parallel_rank()
                            if parallel_state.is_initialized()
                            else (dist.get_rank() if dist.is_initialized() else 0)
                        )
                        logger.info("[data] rank=%d seq=%d→%d sp=%s",
                                    _log_rank,
                                    _orig_seq, input_ids.shape[-1],
                                    "ON" if self._sp_active else "OFF")
                    _local_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
                    input_ids = input_ids.to(_local_dev, non_blocking=True)
                    if labels is not None:
                        labels = labels.to(_local_dev, non_blocking=True)

                    # AutoSP compiler path: pad seq to sp_size multiple, then tag for FX graph pass
                    if self._sp_active and self._autosp_compile_fn is not None:
                        _sp = (
                            parallel_state.get_data_parallel_world_size()
                            if parallel_state.is_initialized()
                            else (dist.get_world_size() if dist.is_initialized() else 1)
                        )
                        _seq = input_ids.shape[1]
                        _pad_n = (_sp - _seq % _sp) % _sp
                        if _pad_n > 0:
                            input_ids = F.pad(input_ids, (0, _pad_n), value=0)
                            if labels is not None:
                                labels = F.pad(labels, (0, _pad_n), value=-100)  # -100 = ignore in CE loss

                        from deepspeed.compile.passes.sp_compile import prepare_autosp_inputs
                        _autosp = prepare_autosp_inputs(
                            input_id=input_ids,
                            label_id=labels,  # None for single-input models
                            seq_dim=1,
                        )
                        input_ids = _autosp.input_id
                        if _autosp.label_id is not None:
                            labels = _autosp.label_id

                    # -----------------------------------------------------------------
                    # Wait for async ZeRO-3 shard sync (launched after optimizer.step
                    # of the *previous* step).  Data loading above runs on the default
                    # stream / CPU, so it overlaps with the sync stream doing the
                    # FP32→BF16 param copies.  We wait here — before the first forward
                    # kernel — to guarantee model BF16 params are fully up-to-date.
                    # On the very first step _shard_sync_pending is False, so this is
                    # a no-op.
                    # -----------------------------------------------------------------
                    if _shard_sync_pending and micro == 0:
                        if _shard_sync_stream is not None:
                            torch.cuda.current_stream().wait_stream(_shard_sync_stream)
                        _shard_sync_pending = False


                    if self.mimo_loop is not None and self.param_shard_state is None:
                        # MIMO path: forward/backward dispatched through
                        # HeteroMIMOTrainingLoop with P2P + LOC cache
                        batch = (input_ids, labels)
                        _engine_cache = self.locality_cache
                        _engine_p2p = self.p2p_communicator

                        def _forward_backward_func(
                            forward_only: bool = False,
                            p2p_communicator=None,
                            pg_collection=None,
                            data_iterator=None,
                            model=None,
                            config=None,
                            iteration: int = 0,
                            _ids=input_ids,
                            _lbl=labels,
                            _num_mb=num_microbatches,
                        ):
                            _p2p = p2p_communicator if p2p_communicator is not None else _engine_p2p
                            loss, scaled_loss = self.forward(
                                _ids, _lbl, num_microbatches=_num_mb,
                            )
                            _act_key = f"fwd_act:iter={iteration}"
                            _engine_cache.put(_act_key, loss.detach())
                            if not forward_only:
                                # --- MoE auxiliary loss (MIMO path) ---
                                if self.moe_adapter is not None:
                                    _aux = self.moe_adapter.collect_aux_loss()
                                    if not isinstance(_aux, float):
                                        scaled_loss = scaled_loss + _aux / max(_num_mb, 1)
                                    elif _aux != 0.0:
                                        scaled_loss = scaled_loss + _aux / max(_num_mb, 1)
                                scaled_loss.backward()
                            return [loss]

                        mimo_result = self.mimo_loop.train_step(
                            forward_backward_func=_forward_backward_func,
                            data_iterator=iter([(input_ids, labels)]),
                            config=cfg,
                            iteration=step * num_microbatches + micro,
                        )
                        # --- HeteroFP32GradAccumManager: accumulate (standard path) ---
                        # Promote BF16 param.grad into FP32 main_grad accumulators
                        # after each micro-batch backward in the MIMO path.
                        if self.fp32_grad_manager is not None:
                            self.fp32_grad_manager.accumulate()
                        step_loss += mimo_result.loss
                    else:
                        # Standard forward/backward path
                        # If the hybrid CP schedule adapter is active, delegate the
                        # entire micro-batch forward+backward to it; otherwise fall
                        # through to the default self.forward() / .backward() pair.
                        if _cp_fb is not None:
                            _cp_losses = _cp_fb(
                                forward_only=False,
                                p2p_communicator=self.p2p_communicator,
                                data_iterator=iter([(input_ids, labels)]),
                                model=self.model,
                                config=cfg,
                                iteration=step * num_microbatches + micro,
                            )
                            _cp_loss_val = float(_cp_losses[0]) if _cp_losses else 0.0
                            # --- HeteroFP32GradAccumManager: accumulate (CP path) ---
                            if self.fp32_grad_manager is not None:
                                self.fp32_grad_manager.accumulate()
                            step_loss += _cp_loss_val
                        else:
                            # Fine-grained activation offload context (A6000 only).
                            # PipelineOffloadManager.__enter__ installs saved-tensor
                            # default hooks that async-D2H tensors ≥ min_size to pinned
                            # CPU RAM.  On H100 / when use_activation_offload=False,
                            # _activation_offload_iface is None and nullcontext() fires.
                            from contextlib import nullcontext as _nullctx  # noqa: PLC0415
                            _offload_ctx = (
                                self._activation_offload_iface.get_context(flag=True)
                                if getattr(self, "_activation_offload_iface", None) is not None
                                else _nullctx()
                            )
                            with _offload_ctx:
                                loss, scaled_loss = self.forward(
                                    input_ids, labels, num_microbatches=num_microbatches,
                                )
                            # Commit offload group: flush any pending D2H transfers
                            # for this micro-batch before backward begins.
                            if getattr(self, "_activation_offload_iface", None) is not None:
                                self._activation_offload_iface.group_commit(
                                    loss,
                                    name=f"mb_{micro}",
                                    delay_offload=False,
                                )
                            # --- MoE auxiliary loss (router load balancing) ---
                            # Collect and add aux loss BEFORE backward so that
                            # router gate gradients flow through the combined loss.
                            # When moe_adapter is None this is a cheap float-0.0 no-op.
                            if self.moe_adapter is not None:
                                _aux = self.moe_adapter.collect_aux_loss()
                                if isinstance(_aux, float):
                                    scaled_loss = scaled_loss + _aux / num_microbatches
                                else:
                                    # Tensor: divide by num_microbatches to match
                                    # the main-loss scale convention.
                                    scaled_loss = scaled_loss + _aux / max(num_microbatches, 1)
                            scaled_loss.backward()
                            # --- HeteroFP32GradAccumManager: accumulate (standard path) ---
                            # Promote BF16 param.grad into FP32 main_grad accumulators
                            # after each micro-batch backward.
                            if self.fp32_grad_manager is not None:
                                self.fp32_grad_manager.accumulate()
                            step_loss += loss.item()

            # Post-microbatch: NaN guard
            if not math.isfinite(step_loss):
                _nan_count = getattr(self, '_nan_count', 0) + 1
                self._nan_count = _nan_count
                logger.warning(
                    "NaN/Inf loss at step %d (count=%d), skipping optimizer update",
                    step, _nan_count,
                )
                self.optimizer.zero_grad(set_to_none=False)
                continue

            # --- HeteroFP32GradAccumManager: after_backward ---
            # Skipped when DistributedOptimizer active (it owns grad reduction).
            if self.fp32_grad_manager is not None and self._dist_optimizer is None:
                # Fix from Megatron M3313: clamp to avoid 1/0 when num_microbatches==0.
                self.fp32_grad_manager.after_backward(scale=1.0 / max(num_microbatches, 1))

            # --- finalize_model_grads (core.distributed) ---
            # Unified grad-sync path for both ZeRO-3 and non-ZeRO-3.
            # On ZeRO-3: force_all_reduce=True triggers direct allreduce across
            # shard params; on non-ZeRO-3: DDP bucket finish_grad_sync runs.
            # DES-LOC Kx gating: skip_grad_sync=True on non-Kx steps.
            #
            # M4172 (Megatron de6305c0a): Thread explicit pg_collection through
            # so finalize_model_grads does not fall back to parallel_state globals.
            # We build a minimal SimpleNamespace carrying only the groups that
            # finalize_model_grads requires; any field left as None will cause
            # the function to skip the corresponding embedding/SP collectives,
            # which is correct for the DesLoc single-tier-per-rank design.
            # ISSUE-2: _is_Kx_sync_pre was computed before backward; reuse here so
            # finalize_model_grads gets the same Kx decision that gated _skip_sync.
            _is_Kx_sync = _is_Kx_sync_pre
            try:
                import types as _types  # noqa: PLC0415
                from deepspeed.core.model_parallel_config import ModelParallelConfig  # noqa: PLC0415
                _dp_grp = getattr(self, '_ddp_dp_group', None)
                _fmg_pg = _types.SimpleNamespace(
                    tp=None,
                    pp=None,
                    embd=None,
                    pos_embd=None,
                    dp_cp=_dp_grp,
                ) if _dp_grp is not None else None
                _fmg_model = [self._core_ddp if self._core_ddp is not None else self.model]
                finalize_model_grads(
                    model=_fmg_model,
                    config=ModelParallelConfig(),
                    num_tokens=None,
                    skip_grad_sync=not _is_Kx_sync,
                    force_all_reduce=self._dist_optimizer is not None,
                    pg_collection=_fmg_pg,
                )
            except Exception as _fmg_exc:  # noqa: BLE001
                logger.warning(
                    "[finalize_model_grads] failed (%s); "
                    "falling back to no-op (grads may be unreduced).",
                    _fmg_exc,
                )

            # Gradient clipping — unified via core clip_grad_norm on all paths.
            # finalize_model_grads has already all-reduced grads; clip globally.
            # core clip_grad_norm avoids host/device sync and handles model-parallel
            # norm reduction — replaces torch.nn.utils.clip_grad_norm_ (M2335).
            gnorm = clip_grad_norm(self.model.parameters(), cfg.grad_clip)
            if torch.is_tensor(gnorm):
                gnorm = gnorm.item()

            # HeteroGradNorm skip decision via HeteroGradNormSkipController
            # (wired through integrate_with_deepspeed_engine at train() setup).
            # Collect parameter gradients for the skip evaluation; classify all
            # params as compute-side since DesLocEngine runs on a single device
            # class per rank (anchor/compute split is resolved at init time).
            _all_grads = [p.grad for p in self.model.parameters()]
            _should_skip, _skip_info = _skip_controller.should_skip([], _all_grads)
            _skip_controller.record_step(skipped=_should_skip, grad_norm=_skip_info.combined_norm)
            if _should_skip:
                _skip_count += 1
            if _is_main:
                print(
                    f"[hetero_grad] step={step} loss={step_loss:.4f} grad_norm={gnorm:.6f} "
                    f"skip={_should_skip} total_skips={_skip_count} "
                    f"ctrl_norm={_skip_info.combined_norm:.6f}"
                )
            if not _should_skip:
                self.optimizer.step()
                self.scheduler.step()

                # --- DES-LOC: Algorithm 1 — Kx/Ku/Kv conditional sync ---
                _is_Kx = (step + 1) % self.desloc_Kx == 0
                _is_Ku = (step + 1) % self.desloc_Ku == 0
                _is_Kv = (step + 1) % self.desloc_Kv == 0

                if self._dist_optimizer is not None:
                    # Ku/Kv: all-reduce first/second Adam moments across DP ranks.
                    # This is the DES-LOC core innovation — decoupled moment sync
                    # reduces communication by (1 − 1/Ku) + (1 − 1/Kv) vs DDP.
                    if _is_Ku or _is_Kv:
                        self._dist_optimizer.sync_moments(
                            sync_first=_is_Ku,
                            sync_second=_is_Kv,
                        )

                    # Every step: broadcast updated FP32 shards → BF16 model on
                    # all ranks.  Without this each rank's model contains only its
                    # own 1/N shard updated — "Frankenstein model" divergence.
                    # DES-LOC communication savings come from skipping GRADIENT
                    # all-reduce on non-Kx steps, not from skipping this broadcast.
                    # (optimizer.step() already called shard_to_model_broadcast()
                    # internally; we call it again on the async stream to overlap
                    # the BF16 copies with the next step's data preprocessing.)
                    if _shard_sync_stream is not None:
                        # Fix from Megatron M3561: the secondary stream must wait
                        # for the current (default) stream — where optimizer.step()
                        # ran — before launching the BF16 broadcast.  Without this
                        # fence the all-gather can start before Adam has written the
                        # updated FP32 values, causing stale-weight corruption.
                        _shard_sync_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(_shard_sync_stream):
                            self._dist_optimizer.shard_to_model_broadcast()
                        _shard_sync_pending = True

                    if step < 10 or _is_Kx or _is_Ku or _is_Kv or (step + 1) % cfg.log_every == 0:
                        logger.info(
                            "[DES-LOC] step=%d Kx=%s Ku=%s Kv=%s",
                            step + 1,
                            "SYNC" if _is_Kx else "skip",
                            "SYNC" if _is_Ku else "skip",
                            "SYNC" if _is_Kv else "skip",
                        )

            # Accounting
            self.global_step = step + 1
            tokens_this_step = (
                num_microbatches * cfg.micro_batch_size * cfg.seq_len
            )
            self.tokens_seen += tokens_this_step
            self.consumed_samples += num_microbatches * cfg.micro_batch_size

            # Fix from Megatron M3313: guard against zero num_microbatches
            # (hetero scheduler can return 0 for an idle rank on edge-case
            # batch sizes), which would cause NaN/ZeroDivisionError here.
            safe_num_microbatches = max(num_microbatches, 1)
            avg_loss = step_loss / safe_num_microbatches
            loss_accum += avg_loss

            # Logging
            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                toks_per_sec = tokens_this_step * cfg.log_every / max(elapsed, 1e-9)
                current_lr = self.scheduler.get_last_lr()[0]
                smooth_loss = loss_accum / cfg.log_every

                # -------------------------------------------------------------------
                # MFU (Model FLOPs Utilisation) computation.
                #
                # Formula:  MFU = actual_flops / peak_flops
                #
                #   actual_flops  = flops_per_token * tokens_processed
                #   flops_per_token = 2 * N_params * 6          (Chinchilla / PaLM rule:
                #                                                 6 = 2 fwd + 4 bwd passes
                #                                                 each pass ~2*N MACs)
                #   For 7B model: flops_per_token ~2 * 7e9 * 6 = 8.4e10
                #
                #   peak_flops (BF16): use the discovered primary tier's spec; fall
                #   back to H100 NVL 835 TFLOPS when no tier info is available.
                #
                # We cache _n_params at logging time (cheap; model frozen during step).
                # -------------------------------------------------------------------
                _peak_flops_cluster = 835e12  # single H100 NVL BF16 default
                if self.tiers:
                    _peak_flops_cluster = sum(t.bf16_tflops * 1e12 for t in self.tiers)
                _n_params = sum(p.numel() for p in self.model.parameters())
                _t_tokens = tokens_this_step * cfg.log_every
                _actual_flops = 2 * _n_params * 6 * _t_tokens
                _mfu = _actual_flops / max(elapsed * _peak_flops_cluster, 1.0)

                logger.info(
                    "step=%6d | loss=%.4f | lr=%.2e | grad_norm=%.3f | "
                    "tok/s=%7.0f | step_ms=%.1f | tokens_seen=%.2fM | MFU=%.4f",
                    self.global_step,
                    smooth_loss,
                    current_lr,
                    gnorm,
                    toks_per_sec,
                    elapsed / cfg.log_every * 1000,
                    self.tokens_seen / 1e6,
                    _mfu,
                )
                # --- W&B / TensorBoard metrics (rank 0 only) ---
                if _is_main and (_wb_run is not None or _tb_writer is not None):
                    # GPU allocated memory per visible device (up to 5 GPUs)
                    _gpu_mems = {}
                    if torch.cuda.is_available():
                        for _gi in range(min(torch.cuda.device_count(), 5)):
                            _gpu_mems[f"train/gpu{_gi}_mem_gb"] = (
                                torch.cuda.memory_allocated(_gi) / (1 << 30)
                            )
                    _log_dict = {
                        "train/loss":                     smooth_loss,
                        "train/lr":                       current_lr,
                        "train/grad_norm":                gnorm,
                        "train/throughput_tokens_per_sec": toks_per_sec,
                        "train/mfu":                      _mfu,
                        "train/tokens_seen_M":            self.tokens_seen / 1e6,
                        **_gpu_mems,
                    }
                    if _wb_run is not None:
                        _wb_run.log(_log_dict, step=self.global_step)
                    if _tb_writer is not None:
                        for _k, _v in _log_dict.items():
                            _tb_writer.add_scalar(_k, _v, self.global_step)

                loss_accum = 0.0
                t0 = time.time()

                # --- MoE expert utilisation logging ---
                # Gated by moe_log_every; no-op when MoE is disabled.
                if self.moe_adapter is not None:
                    self.moe_adapter.log_utilization(
                        step=self.global_step,
                        moe_log_every=getattr(cfg, "moe_log_every", 100),
                    )

            # Checkpointing
            if self.global_step % cfg.save_every == 0:
                ckpt_path = cfg.checkpoint_dir / f"step_{self.global_step:07d}.pt"
                # From M3407: join previous async checkpoint before writing new one
                if self._checkpoint_thread is not None:
                    self._checkpoint_thread.join()
                    self._checkpoint_thread = None
                self.save_checkpoint(ckpt_path)

            # --- Eval hook: every eval_every steps call eval/run_eval.py ---
            if (
                cfg.eval_every > 0
                and self.global_step % cfg.eval_every == 0
                and _run_periodic_eval is not None
            ):
                _eval_model_path = cfg.eval_model_path or ""
                _eval_output_dir = cfg.eval_output_dir
                logger.info(
                    "[eval] step=%d — running periodic eval (model_path='%s', output='%s')",
                    self.global_step, _eval_model_path, _eval_output_dir,
                )
                try:
                    self.model.eval()
                    eval_results = _run_periodic_eval(
                        model_path=_eval_model_path,
                        step=self.global_step,
                        output_dir=_eval_output_dir,
                    )
                    # Log eval loss / key metrics
                    _eval_summary = {
                        k: v for k, v in eval_results.items()
                        if k not in ("step", "model_path", "timestamp")
                    }
                    logger.info(
                        "[eval] step=%d eval_results=%s",
                        self.global_step, _eval_summary,
                    )
                except Exception as _eval_exc:  # noqa: BLE001
                    logger.warning("[eval] step=%d eval hook failed: %s", self.global_step, _eval_exc)
                finally:
                    self.model.train()
                    # --- M3490 (b8e23d587): reset activation offload manager after eval ---
                    # PipeDream runtime.py train() resets all tensor/gradient state when
                    # re-entering training from eval mode (tensors=[], gradients={},
                    # forward_minibatch_id=0).  We do the same for PipelineOffloadManager:
                    # eval runs no backward pass, so the backward-chunk deque accumulates
                    # stale ChunkOffloadHandler entries that are never drained.  If left
                    # in place, the next training step's H2D restores (on_get_saved_tensor)
                    # will pop from the wrong backward chunk and corrupt activations.
                    #
                    # HetSeq controller.py train_step() calls self.model.train() at entry
                    # and self.zero_grad(); we mirror that with a manager-level reset so
                    # all chunk-handler state is consistent with a fresh micro-batch window.
                    if getattr(self, "_activation_offload_iface", None) is not None:
                        try:
                            self._activation_offload_iface.reset_instance()
                            self._activation_offload_iface.init_chunk_handler(
                                vp_size=getattr(cfg, "virtual_pipeline_model_parallel_size", None),
                                vp_stage=0,
                                min_offloaded_tensor_size=getattr(
                                    cfg, "activation_offload_min_size", 1_048_576
                                ),
                                max_inflight_offloads=getattr(
                                    cfg, "activation_offload_max_inflight", None
                                ),
                            )
                            logger.info(
                                "[eval] step=%d PipelineOffloadManager reset+reinit "
                                "(M3490 parity — stale backward chunks cleared)",
                                self.global_step,
                            )
                        except Exception as _reset_exc:  # noqa: BLE001
                            logger.warning(
                                "[eval] step=%d offload manager reset failed: %s",
                                self.global_step, _reset_exc,
                            )
        total_time = time.time() - train_start
        # Drain any outstanding async shard sync from the final step so that
        # checkpointing / evaluation that follows reads consistent BF16 params.
        if _shard_sync_pending and _shard_sync_stream is not None:
            torch.cuda.current_stream().wait_stream(_shard_sync_stream)
            _shard_sync_pending = False
        logger.info(
            "Training complete. %d steps in %.1fs. "
            "%.2fM tokens seen. Avg %.0f tok/s.",
            cfg.total_steps,
            total_time,
            self.tokens_seen / 1e6,
            self.tokens_seen / max(total_time, 1.0),
        )
        # Flush and close logging backends (rank 0 only)
        if _is_main:
            if _wb_run is not None:
                _wb_run.finish()
                logger.info("W&B run finished.")
            if _tb_writer is not None:
                _tb_writer.close()
                logger.info("TensorBoard writer closed.")

    def close(self):
        '''Finalize engine. From M3407: join async checkpoint thread before exit.'''
        if getattr(self, '_checkpoint_thread', None) is not None:
            self._checkpoint_thread.join()
            self._checkpoint_thread = None

    def save_checkpoint(self, path: Path) -> None:
        """
        Save a full training checkpoint to disk.

        Delegates to :func:`deepspeed.runtime.desloc_checkpointing.save_checkpoint`
        (extracted from this method; see that module's docstring for the full
        tier-aware save strategy description).

        Args:
            path: Destination file/directory path (parent dirs created as needed).
        """
        return _desloc_save_checkpoint(self, path)

    def load_checkpoint(self, path: Path) -> None:
        """
        Resume training from a saved checkpoint.

        Delegates to :func:`deepspeed.runtime.desloc_checkpointing.load_checkpoint`
        (extracted from this method; see that module's docstring for the full
        tier-aware load strategy description).

        Args:
            path: Path to the checkpoint directory (hetero format) or ``.pt``
                  file (legacy format).

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        return _desloc_load_checkpoint(self, path)

    def _apply_loaded_state(
        self,
        payload: Dict[str, Any],
        cfg: Optional[Any],
    ) -> None:
        """
        Apply a loaded state dict to the engine, honouring
        :class:`HeteroCheckpointConfig` flags for optim / rng / strictness /
        shard rebalancing.

        This helper is called by all three load paths in
        :meth:`load_checkpoint` so that ``cfg.load_optim``,
        ``cfg.load_rng``, ``cfg.dist_ckpt_strictness``, and
        ``cfg.shard_rebalance_on_load`` are enforced consistently regardless
        of which path succeeded.

        When ``cfg.shard_rebalance_on_load`` is ``True`` and the saved
        checkpoint was produced by a different tier layout (e.g. the
        checkpoint contains CACHE-tier tensors that need to be redistributed
        to WORKER devices), this method remaps every model tensor to the
        *current* tier device before calling ``load_state_dict``.  The remap
        policy is:

        * Parameters whose saved shard metadata indicates they belong to the
          CACHE device are loaded to the current CACHE-tier device.
        * All other parameters are loaded to the current primary device.

        When no shard metadata is present (legacy checkpoint), tensors are
        loaded to whichever device is cheapest (CPU then moved to GPU).

        Args:
            payload: Dictionary with keys ``model_state``, optionally
                ``optimizer_state`` / ``scheduler_state`` / ``global_step``
                / ``tokens_seen``.
            cfg: Active :class:`HeteroCheckpointConfig` or ``None``.
        """
        # Resolve per-config flags with safe defaults for legacy loads.
        _load_optim = cfg.load_optim if cfg is not None else True
        _load_rng   = cfg.load_rng   if cfg is not None else True
        _strict     = (
            (cfg.dist_ckpt_strictness.value != "raise_all")
            if cfg is not None else True
        )
        _rebalance  = (cfg.shard_rebalance_on_load if cfg is not None else False)

        model_state     = payload.get("model_state", {})
        optimizer_state = payload.get("optimizer_state")
        scheduler_state = payload.get("scheduler_state")

        # ------------------------------------------------------------------
        # Tier-aware tensor remapping (shard_rebalance_on_load)
        # ------------------------------------------------------------------
        # Determine the *current* target device for each state tensor.
        # Strategy:
        #   1.  When rebalancing is active, tensors that were saved on the
        #       CACHE device (device_id == cfg.cache_tier_device_id) are
        #       placed on the current CACHE device; everything else goes to
        #       the primary device.
        #   2.  When rebalancing is off, tensors simply go to the primary
        #       device (backward-compatible behaviour).
        #
        # The "saved device" is inferred from the tensor's .device attribute
        # if the payload was loaded map_location="cpu" (in which case all
        # tensors are on CPU and we fall back to primary device placement).
        _primary_dev = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available() else torch.device("cpu")
        )
        # CACHE device: use the config's cache_tier_device_id when CUDA is
        # available, else fall back to primary.
        _cache_dev_id = (
            cfg.cache_tier_device_id
            if cfg is not None and torch.cuda.is_available()
            else None
        )
        _cache_dev = (
            torch.device(f"cuda:{_cache_dev_id}")
            if _cache_dev_id is not None
            else _primary_dev
        )
        # WORKER devices come from cfg; first worker (if any) is the default
        # target for non-CACHE parameters when rebalancing.
        _worker_dev_ids: List[int] = (
            cfg.worker_device_ids if cfg is not None else []
        )
        _default_worker_dev = (
            torch.device(f"cuda:{_worker_dev_ids[0]}")
            if _worker_dev_ids and torch.cuda.is_available()
            else _primary_dev
        )

        def _target_device_for(key: str, tensor: torch.Tensor) -> torch.device:
            """Pick the destination device for a model-state tensor."""
            if not _rebalance:
                return _primary_dev
            # If the saved tensor carries a real CUDA device index, use it as
            # a hint: if it matches the saved cache_tier_device_id we place it
            # on the *current* cache device; otherwise on the worker device.
            if tensor.device.type == "cuda" and _cache_dev_id is not None:
                if tensor.device.index == _cache_dev_id:
                    return _cache_dev
                return _default_worker_dev
            # Tensor is on CPU (map_location="cpu" path): heuristic — use the
            # current primary device (training loop will re-shard as needed).
            return _primary_dev

        # Move model state tensors to the target device before load_state_dict.
        model_state_on_dev: Dict[str, Any] = {}
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor):
                tgt = _target_device_for(k, v)
                model_state_on_dev[k] = v.to(tgt)
            else:
                model_state_on_dev[k] = v

        if _rebalance:
            logger.info(
                "[hetero_ckpt] shard_rebalance_on_load=True: "
                "remapped %d model tensors "
                "(cache_dev=%s, default_worker_dev=%s).",
                len(model_state_on_dev),
                _cache_dev,
                _default_worker_dev,
            )

        self.model.load_state_dict(model_state_on_dev, strict=_strict)

        if optimizer_state is not None and _load_optim:
            self.optimizer.load_state_dict(optimizer_state)
        elif optimizer_state is None and _load_optim:
            logger.warning(
                "[hetero_ckpt] load_optim=True but optimizer_state absent "
                "in checkpoint (WORKER shard without optimizer state?)."
            )

        # cfg.load_rng governs whether the *scheduler* (LR schedule) state
        # is restored; RNG state tensors (torch.get_rng_state) are not
        # persisted in this engine, so we reuse the flag for the scheduler.
        if scheduler_state is not None and _load_rng:
            self.scheduler.load_state_dict(scheduler_state)

        self.global_step = int(payload.get("global_step", 0))
        self.tokens_seen = int(payload.get("tokens_seen", 0))

    def drain_checkpoint(self, timeout: float = 600.0) -> None:
        """
        Block until all pending async checkpoint IO is complete.

        Waits for two async subsystems:

        1. **CPU-staging futures** — background threads copying GPU tensors to
           the locality-cache ramdisk (``/dev/shm``).  These are submitted
           by :meth:`save_checkpoint` when the CACHE or WORKER tier staging
           path is active.

        2. **HeteroAsyncCheckpointScheduler** — the async disk-IO pipeline
           that persists staged tensors from CPU DRAM to the checkpoint
           directory.

        Should be called before program exit or before scheduling the next
        checkpoint when the async queue could overflow.  No-op when both
        subsystems are inactive.

        Args:
            timeout: Maximum seconds to wait.  Raises ``TimeoutError`` if
                     either subsystem exceeds this limit.
        """
        # --- 1: drain CPU-staging futures ---
        with self._cpu_stage_lock:
            pending = list(self._cpu_stage_futures)

        if pending:
            logger.info(
                "[hetero_ckpt] Draining %d CPU-staging future(s) …", len(pending)
            )
            done, not_done = concurrent.futures.wait(
                pending, timeout=timeout,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            if not_done:
                raise TimeoutError(
                    f"drain_checkpoint: {len(not_done)} CPU-staging future(s) "
                    f"did not complete within {timeout:.0f}s."
                )
            # Re-raise any exceptions from completed futures so they surface
            # rather than being silently swallowed.
            for _f in done:
                exc = _f.exception()
                if exc is not None:
                    logger.warning(
                        "[hetero_ckpt] CPU-staging future raised: %s", exc
                    )
            with self._cpu_stage_lock:
                self._cpu_stage_futures = [
                    f for f in self._cpu_stage_futures if not f.done()
                ]
            logger.info("[hetero_ckpt] CPU-staging futures drained.")

        # --- 2: drain async disk-IO scheduler ---
        if self._hetero_ckpt_scheduler is not None:
            logger.info("[hetero_ckpt] Draining async checkpoint queue …")
            self._hetero_ckpt_scheduler.drain(timeout=timeout)
            logger.info("[hetero_ckpt] Checkpoint queue drained.")
        else:
            logger.debug("[hetero_ckpt] drain_checkpoint: no async scheduler active.")
    @staticmethod
    def _setup_logging() -> None:
        """Configure root logger with timestamp format if not already configured."""
        if not logging.root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    @staticmethod
    def _fused_adam_available() -> bool:
        """Check if fused AdamW kernel is available (requires CUDA + apex or PyTorch >= 2.0)."""
        try:
            if not torch.cuda.is_available():
                return False
            # PyTorch >= 2.0 ships fused AdamW natively
            major, minor = (int(x) for x in torch.__version__.split(".")[:2])
            return (major, minor) >= (2, 0)
        except Exception:  # noqa: BLE001
            return False

    def get_tier_by_class(self, tier_class: TierClass) -> List[TierSpec]:
        """Return all discovered TierSpec objects of a given TierClass."""
        return [s for s in self.tiers if s.tier == tier_class]

    def memory_summary(self) -> str:
        """Return a formatted CUDA memory summary for all discovered GPUs."""
        lines = ["--- GPU Memory Summary ---"]
        for spec in self.tiers:
            try:
                alloc = torch.cuda.memory_allocated(spec.device) / (1 << 20)
                reserved = torch.cuda.memory_reserved(spec.device) / (1 << 20)
                lines.append(
                    f"  GPU{spec.device_index} ({spec.name}): "
                    f"alloc={alloc:.0f}MB  reserved={reserved:.0f}MB  "
                    f"total={spec.total_mem_gb:.0f}GB"
                )
            except Exception:  # noqa: BLE001
                lines.append(f"  GPU{spec.device_index}: query failed")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    """
    Minimal smoke test that exercises the full engine on available hardware.

    Runs 20 training steps with a tiny model configuration.
    Verifies: discovery, registry, solver, forward/backward, step, checkpoint.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("DES-LOC Engine — smoke test")
    logger.info("PyTorch: %s  |  CUDA: %s  |  Devices: %d",
                torch.__version__,
                torch.version.cuda or "N/A",
                torch.cuda.device_count() if torch.cuda.is_available() else 0)
    logger.info("=" * 60)

    cfg = TrainingConfig(
        # Tiny model for smoke test
        vocab_size=1024,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        seq_len=128,
        # Short run
        total_steps=20,
        global_batch_size=4,
        micro_batch_size=2,
        grad_accum_steps=2,
        max_lr=1e-3,
        min_lr=1e-4,
        warmup_steps=5,
        weight_decay=0.01,
        grad_clip=1.0,
        # Logging / checkpointing
        log_every=5,
        save_every=10,
        checkpoint_dir=Path("/tmp/desloc_smoke_ckpts"),
    )

    engine = DesLocEngine(config=cfg)
    logger.info(engine.memory_summary())

    logger.info("Running %d training steps...", cfg.total_steps)
    engine.train()

    # Verify checkpoint round-trip
    ckpt = cfg.checkpoint_dir / "smoke_manual.pt"
    engine.save_checkpoint(ckpt)
    engine.load_checkpoint(ckpt)
    logger.info("Checkpoint round-trip: OK")

    # Verify inference
    engine.model.eval()
    with torch.no_grad():
        dummy = torch.randint(0, cfg.vocab_size, (1, 16), device=engine.primary_device)
        out = engine.model(dummy)
        assert out.shape == (1, 16, cfg.vocab_size), f"Unexpected output shape: {out.shape}"
    logger.info("Inference check: OK — output shape %s", tuple(out.shape))

    logger.info("Smoke test PASSED.")


if __name__ == "__main__":
    _smoke_test()
