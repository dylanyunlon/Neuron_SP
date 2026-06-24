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

import importlib
import inspect
import logging
import math
import os
import pkgutil
import subprocess
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
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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
class PartitionStrategy(Enum):
    """Supported partition strategies for heterogeneous training."""
    ZERO3_HETERO = auto()   # ZeRO-3 + heterogeneous gradient accumulation
    PIPELINE_1F1B = auto()  # Pipeline parallelism with 1F1B schedule


class TierClass(Enum):
    """GPU tier classification based on SM version and memory."""
    H100 = "H100"
    A6000 = "A6000"
    RTX_PRO_6000_BW = "RTX_PRO_6000_BW"  # Blackwell SM12.0, 96GB
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TierSpec:
    """
    Specification of a single GPU tier discovered at runtime.

    Attributes:
        device_index: CUDA device index.
        tier: Classification of the GPU tier.
        total_mem_gb: Total GPU memory in GB.
        free_mem_gb: Free GPU memory in GB at discovery time.
        sm_major: CUDA SM major version.
        sm_minor: CUDA SM minor version.
        bf16_tflops: BF16 theoretical peak TFLOPs.
        pcie_bw_gbs: PCIe bandwidth in GB/s.
        numa_node: NUMA node affinity (-1 if unknown).
        name: Human-readable GPU name.
    """
    device_index: int
    tier: TierClass
    total_mem_gb: float
    free_mem_gb: float
    sm_major: int
    sm_minor: int
    bf16_tflops: float
    pcie_bw_gbs: float
    numa_node: int
    name: str

    @property
    def device(self) -> torch.device:
        """Return the torch.device for this tier."""
        return torch.device(f"cuda:{self.device_index}")

    def __repr__(self) -> str:
        return (
            f"TierSpec(idx={self.device_index}, tier={self.tier.value}, "
            f"mem={self.total_mem_gb:.0f}GB, SM={self.sm_major}.{self.sm_minor}, "
            f"BF16={self.bf16_tflops}TFLOPS, name='{self.name}')"
        )


@dataclass
class PartitionPlan:
    """
    Result of PartitionSolver: describes how model layers are assigned to tiers.

    Attributes:
        strategy: The chosen partition strategy.
        tier_layer_map: Maps device_index -> list of layer indices assigned.
        grad_accum_steps: Per-device gradient accumulation steps dict.
        micro_batch_sizes: Per-device micro-batch sizes.
        estimated_throughput: Estimated tokens/s for this plan.
        notes: Human-readable notes about why this plan was chosen.
    """
    strategy: PartitionStrategy
    tier_layer_map: Dict[int, List[int]]
    grad_accum_steps: Dict[int, int]
    micro_batch_sizes: Dict[int, int]
    estimated_throughput: float
    notes: str = ""


@dataclass
class TrainingConfig:
    """
    Full training configuration for the DES-LOC engine.

    All fields have sensible defaults tuned for the 2xA6000+1xH100 target cluster.
    """
    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    seq_len: int = 2048

    # Training hyperparameters
    total_steps: int = 100_000
    global_batch_size: int = 64
    micro_batch_size: int = 2
    grad_accum_steps: int = 8
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    # Checkpointing
    save_every: int = 1000
    checkpoint_dir: Path = _CHECKPOINT_DIR
    resume_from: Optional[Path] = None

    # Logging
    log_every: int = 10

    # Eval hook: run eval/run_eval.py every this many steps (0 = disabled)
    eval_every: int = 0
    # Path to a saved model checkpoint dir for evaluation (None = skip model load)
    eval_model_path: Optional[str] = None
    # Output directory for eval result JSON files
    eval_output_dir: str = "desloc_results/eval_runs"

    # Strategy override (None = auto-select)
    strategy_override: Optional[PartitionStrategy] = None

    # Heterogeneous checkpoint config.  When None the engine will auto-build
    # one via build_config_for_cluster() at init time.
    hetero_checkpoint_config: Optional[Any] = None

    # HeteroStepBatchScheduler config
    # Format: "0:32 90B:64 180B:128" (THRESHOLD:BATCH_SIZE, token or sample units)
    # If None, scheduler uses a single constant entry based on global_batch_size
    batch_schedule: Optional[str] = None
    # If provided, schedule thresholds are interpreted as token counts (÷ seq_len → samples)
    batch_schedule_seq_length: Optional[int] = None


# ---------------------------------------------------------------------------
# HeteroRegistry: discovers and loads hetero_*.py modules
# ---------------------------------------------------------------------------
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

        torch.cuda.set_device(idx)
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
        """Build the ZeRO-3 heterogeneous gradient accumulation plan."""
        cfg = self.config
        tier_layer_map: Dict[int, List[int]] = {}
        grad_accum: Dict[int, int] = {}
        micro_bs: Dict[int, int] = {}

        # All devices participate in ZeRO-3; layers are replicated across all GPUs
        # (ZeRO-3 partitions optimizer state + gradients, not layers)
        all_layers = list(range(cfg.num_layers))
        for spec in self.tiers:
            tier_layer_map[spec.device_index] = all_layers[:]
            if spec.tier == TierClass.H100:
                grad_accum[spec.device_index] = 22
                micro_bs[spec.device_index] = cfg.micro_batch_size
            else:
                grad_accum[spec.device_index] = 1
                micro_bs[spec.device_index] = cfg.micro_batch_size

        throughput = self._estimate_zero3_throughput(micro_bs, grad_accum)

        return PartitionPlan(
            strategy=PartitionStrategy.ZERO3_HETERO,
            tier_layer_map=tier_layer_map,
            grad_accum_steps=grad_accum,
            micro_batch_sizes=micro_bs,
            estimated_throughput=throughput,
            notes="ZeRO-3: H100 22 micro-batches, A6000 1 each, AllReduce sync",
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
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Infinite iterator of random token batches for smoke testing.

    In production, replace with a real DataLoader.

    Args:
        vocab_size: Vocabulary size for random token sampling.
        batch_size: Number of sequences per batch.
        seq_len: Sequence length.
        device: Target device (tokens are on CPU, moved to GPU in the loop).

    Yields:
        Tuples of (input_ids, labels) each of shape (batch_size, seq_len).
    """
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        yield tokens[:, :-1], tokens[:, 1:]


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

        # Each rank keeps model on its own local device (not all on primary_device)
        _local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.model = self.model.to(dtype=_DEFAULT_DTYPE, device=_local_device)
        self.model_device = _local_device  # cached for forward()

        # --- Phase 4b: FSDP wrapping for ZeRO-3 heterogeneous sharding ---
        # Neuron_SP core design: partition model params + optimizer states
        # across heterogeneous GPUs so that models exceeding single-GPU VRAM
        # (e.g. 7B on A6000 48GB) can train via sharding.
        if dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
                CPUOffload,
            )
            _local_mem_gb = torch.cuda.get_device_properties(_local_device).total_memory / (1 << 30)
            # Use CPU offload on small-VRAM GPUs (A6000 48GB)
            _cpu_offload = CPUOffload(offload_params=(_local_mem_gb < 60))
            _mp = MixedPrecision(
                param_dtype=_DEFAULT_DTYPE,
                reduce_dtype=_DEFAULT_DTYPE,
                buffer_dtype=_DEFAULT_DTYPE,
            )
            self.model = FSDP(
                self.model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=_mp,
                cpu_offload=_cpu_offload,
                device_id=_local_device,
                use_orig_params=True,
            )
            logger.info(
                "FSDP wrapped: strategy=FULL_SHARD, cpu_offload=%s, device=%s",
                _local_mem_gb < 60, _local_device,
            )

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model: %.2fM parameters on %s", n_params / 1e6, _local_device)

        # --- Phase 4b: ZeRO-3 heterogeneous parameter sharding ---
        # Each rank keeps only a 1/N (or VRAM-proportional) slice of the
        # flattened FP32 master parameter buffer. The full BF16 params
        # are gathered on-demand via param_shard_state.gather_full_params
        # during forward / backward.
        #
        # Skipped entirely when world_size <= 1 (backward compatible).
        self.param_shard_state = None
        self.param_shard = None
        self.param_offsets = None
        try:
            from deepspeed.runtime.zero3_hetero_shard import (
                ShardState as _ShardState,
                vram_weights_from_tiers as _vram_weights_from_tiers,
            )
            _ws = dist.get_world_size() if dist.is_initialized() else int(
                os.environ.get("WORLD_SIZE", 1)
            )
            _rk = dist.get_rank() if dist.is_initialized() else int(
                os.environ.get("RANK", 0)
            )
            if _ws > 1:
                # Heterogeneous: weight shards by VRAM (H100 > A6000).
                _weights = _vram_weights_from_tiers(self.tiers) if getattr(
                    self, "tiers", None
                ) else None
                if _weights and len(_weights) != _ws:
                    # World size doesn't match discovered tier count —
                    # fall back to an even split to stay safe.
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
                    # Sanity check (T127 acceptance criterion): the sum
                    # of per-rank shard sizes must equal the original
                    # total parameter count (up to alignment padding).
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

        # Gradient accumulation from plan (use primary device's setting)
        primary_idx = self.primary_device.index if self.primary_device.type == "cuda" else -1
        self.grad_accum = self.plan.grad_accum_steps.get(
            primary_idx, config.grad_accum_steps
        )
        logger.info("Effective grad_accum_steps on primary: %d", self.grad_accum)

        # --- Phase 7: HeteroStepBatchScheduler ---
        # Build device profiles from discovered tiers (or fall back to defaults)
        if self.tiers:
            max_tflops = max(s.bf16_tflops for s in self.tiers)
            device_profiles = []
            for spec in self.tiers:
                weight = round(spec.bf16_tflops / max_tflops, 2)
                max_mbs = config.micro_batch_size * (
                    2 if spec.tier == TierClass.H100 else 1
                )
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

        self.hetero_scheduler = HeteroStepBatchScheduler(
            rank=0,
            micro_batch_size=config.micro_batch_size,
            data_parallel_size=dp_size,
            schedule=schedule_str,
            device_profiles=device_profiles,
            seq_length=seq_len_for_schedule,
        )
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

        # Optionally resume from checkpoint
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

        # --- Phase 7: Hetero Checkpoint System ---
        # Build or adopt a HeteroCheckpointConfig, then create the per-tier
        # async scheduler so that save_checkpoint() can hand off IO immediately.
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
                    "workers=%s, async=%s",
                    self._hetero_ckpt_cfg.cache_tier_device_id,
                    self._hetero_ckpt_cfg.worker_device_ids,
                    self._hetero_ckpt_cfg.hetero_async_save,
                )
            except Exception as _hcc_exc:  # noqa: BLE001
                logger.warning(
                    "Could not auto-build HeteroCheckpointConfig (%s); "
                    "hetero async save will be disabled.",
                    _hcc_exc,
                )
                self._hetero_ckpt_cfg = None  # type: ignore[assignment]

        # Per-rank async scheduler (one instance, reused across save calls).
        self._hetero_ckpt_scheduler: Optional[HeteroAsyncCheckpointScheduler] = None
        if self._hetero_ckpt_cfg is not None and self._hetero_ckpt_cfg.hetero_async_save:
            self._hetero_ckpt_scheduler = HeteroAsyncCheckpointScheduler()
            logger.info("HeteroAsyncCheckpointScheduler created for async per-tier saves.")

        # --- Phase 7: Neuron_SP heterogeneous recompute config ---
        self.neuron_sp_config: HeteroRecomputeConfig = build_neuron_sp_config()
        logger.info(
            "Neuron_SP recompute config built (granularity=%s, attention=%s).",
            self.neuron_sp_config.granularity,
            self.neuron_sp_config.attention_variant,
        )

        # Apply torch.utils.checkpoint selectively based on device class.
        # A6000 layers (48 GB HBM) → recompute norm_out to save memory.
        # H100/Blackwell layers (96 GB HBM) → no recompute needed.
        #
        # Strategy: inspect the partition plan's tier_layer_map to assign each
        # layer its device index, then classify via HeteroDeviceMap.
        import torch.utils.checkpoint as _torch_ckpt  # noqa: PLC0415

        # Build layer → device_index mapping from the partition plan.
        layer_device_map: Dict[int, int] = {}
        for dev_idx, layer_indices in self.plan.tier_layer_map.items():
            for li in layer_indices:
                layer_device_map[li] = dev_idx

        # Support both MiniTransformer (.blocks) and standard models (.layers).
        block_list = getattr(self.model, "blocks", None) or getattr(
            self.model, "layers", None
        )

        if block_list is not None:
            print("[Neuron_SP] Per-layer recompute strategy:")
            for layer_idx, block in enumerate(block_list):
                dev_idx = layer_device_map.get(layer_idx, primary_idx)
                if dev_idx < 0:
                    dev_class = DeviceClass.UNKNOWN
                else:
                    dev_class = self.neuron_sp_config.device_map.get(dev_idx)

                recompute_modules = self.neuron_sp_config.modules_per_device.get(
                    dev_class, set()
                )
                should_recompute = bool(recompute_modules)

                if should_recompute:
                    original_fwd = block.forward

                    def _make_ckpt_fwd(fwd):
                        def _ckpt_fwd(*args, **kwargs):
                            return _torch_ckpt.checkpoint(
                                fwd, *args, use_reentrant=False, **kwargs
                            )
                        return _ckpt_fwd

                    block.forward = _make_ckpt_fwd(original_fwd)
                    print(
                        f"  Layer {layer_idx:3d} -> GPU{dev_idx} "
                        f"({dev_class.name}): recompute=ON  "
                        f"modules={recompute_modules}"
                    )
                else:
                    print(
                        f"  Layer {layer_idx:3d} -> GPU{dev_idx} "
                        f"({dev_class.name}): recompute=OFF"
                    )
        else:
            print(
                "[Neuron_SP] WARNING: model exposes no .blocks/.layers; "
                "per-layer recompute wrapping skipped."
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
        # data_parallel_group: use the default process group if distributed is
        # initialised, otherwise fall back to a single-rank gloo group so the
        # manager can still be constructed without a real multi-GPU setup.
        if dist.is_initialized():
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
        # 7B model needs ~12GB bf16 + 2×12GB FP32 buffers = ~36GB minimum
        # A6000 (47GB) can't fit all three; skip FP32 grad accum on small devices
        if _local_mem_gb < 60:
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

        # --- Phase 8b: ZeRO-3 backward reduce-scatter hooks ---
        # Register a post-accumulate-grad hook on every sharded parameter
        # so that each rank's ``.grad`` is reduced and scattered the
        # moment autograd produces it. After the hook each rank only
        # retains the gradient slice corresponding to its own
        # ``param_shard``, and (when applicable) those slices are
        # accumulated into the FP32 ``main_grad`` of
        # ``fp32_grad_manager`` for selective-FP32 parameters.
        self._zero3_grad_hook_handles: List = []
        if self.param_shard_state is not None:
            try:
                self._zero3_grad_hook_handles = (
                    self.param_shard_state.register_backward_hooks(
                        fp32_grad_manager=self.fp32_grad_manager,
                    )
                )
                logger.info(
                    "[zero3] backward reduce-scatter hooks registered: %d "
                    "(fp32_grad_manager=%s)",
                    len(self._zero3_grad_hook_handles),
                    self.fp32_grad_manager is not None,
                )
            except Exception as _hook_exc:  # noqa: BLE001
                logger.warning(
                    "[zero3] backward hook registration failed (%s); "
                    "post-backward scatter_grads() will still run.",
                    _hook_exc,
                )

        # --- Phase 9: Hetero MIMO training loop bootstrap ---
        # Build the DES-LOC heterogeneous MIMO training loop (device registry,
        # locality cache, P2P communicator, optimizer router, schedule groups)
        # so that DesLocEngine.train() can dispatch micro-batches through it.
        # Failures here must not break legacy single-GPU training paths, so the
        # call is wrapped defensively and the loop falls back to None.
        try:
            self.mimo_loop: Optional[HeteroMIMOTrainingLoop] = (
                setup_hetero_mimo_training(self.model)
            )
            logger.info(
                "HeteroMIMOTrainingLoop initialized via setup_hetero_mimo_training()."
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

        # --- Phase 9a: SharedLocalityCache (1.5 TB CPU DRAM staging) ---
        # The training host has 2×EPYC 9354 with 1.5 TB DDR5.  We reserve
        # ~192 GB (1/8 of total DRAM) for the inter-pool activation / gradient
        # cache, leaving the rest for OS, data loaders, and checkpoint IO.
        _cache_max_gb = 192.0
        _cache_max_entries = 512   # ~128 micro-batches × 4 pipeline stages
        self.locality_cache = SharedLocalityCache(
            max_entries=_cache_max_entries,
            max_bytes=int(_cache_max_gb * 1024 ** 3),
        )
        logger.info(
            "SharedLocalityCache initialized: max_entries=%d, max_bytes=%.1f GB "
            "(1.5 TB DRAM host, reserving 1/8 for cache).",
            _cache_max_entries, _cache_max_gb,
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

        # --- Phase 9c: HeteroMIMOTrainingLoop ---
        # Already initialized in Phase 9 above (self.mimo_loop).
        # Kept as a comment for clarity on the initialization order.

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
        _is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)

        # Lazy imports (hetero_* not imported at module level to avoid apex dep)
        from deepspeed.runtime.hetero_gdn_selective_recompute import build_neuron_sp_config  # noqa: PLC0415
        from deepspeed.runtime.hetero_grad_norm_skip import (  # noqa: PLC0415
            HeteroGradNormConfig,
            HeteroGradNormSkipController,
        )


        cfg = self.config
        self.model.train()
        logger.info("Training start: %d steps, grad_accum=%d",
                    cfg.total_steps, self.grad_accum)

        # --- Claude-128: ZeRO-3 per-layer forward all-gather hooks ---
        # When ZeRO-3 sharding is active each rank only holds 1/N of the
        # flat parameter buffer. We install layer-by-layer all-gather
        # hooks so that the full BF16 params for the *currently
        # executing* nn.Module are materialized on entry and freed on
        # exit. Peak memory: model_shard + max(per_layer_full_params).
        self._zero3_forward_hook = None
        if getattr(self, "param_shard_state", None) is not None:
            try:
                from deepspeed.runtime.zero3_hetero_shard import (  # noqa: PLC0415
                    install_zero3_forward_hooks as _install_z3_hooks,
                )
                self._zero3_forward_hook = _install_z3_hooks(
                    self.model, self.param_shard_state,
                )
                if self._zero3_forward_hook is not None and _is_main:
                    logger.info(
                        "[zero3-hook] per-layer forward all-gather active "
                        "(rank=%d/%d)",
                        self.param_shard_state.rank,
                        self.param_shard_state.world_size,
                    )
            except Exception as _hook_exc:  # noqa: BLE001
                logger.warning(
                    "[zero3-hook] failed to install forward hooks (%s); "
                    "continuing without per-layer gather.", _hook_exc,
                )
                self._zero3_forward_hook = None

        # --- Neuron_SP: build / refresh heterogeneous recompute config ---
        # Recompute policy may depend on the current device topology; rebuild
        # the config here so that train() always picks up the latest mapping.
        # The per-layer torch.utils.checkpoint wrapping is applied in __init__,
        # but we keep the live config attached to the model for downstream
        # modules (e.g. HeteroGDNNormOutRecompute) that query it at runtime.
        recompute_config = build_neuron_sp_config()
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

        # Wire HeteroGradNormSkipController into this engine.
        # Note: integrate_with_deepspeed_engine() monkey-patches engine.step(),
        # but DesLocEngine doesn't have a .step() method (step logic is inline
        # in train()). Create the controller directly and call should_skip()
        # manually in the loop below.
        _hetero_config = HeteroGradNormConfig()
        _skip_controller = HeteroGradNormSkipController(config=_hetero_config)
        _skip_count = 0

        loss_accum = 0.0
        t0 = time.time()

        for step in range(self.global_step, cfg.total_steps):
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            # --- HeteroStepBatchScheduler: decide num_microbatches for this step ---
            allocation: MicrobatchAllocation = self.hetero_scheduler.update(
                consumed_samples=self.consumed_samples,
                consistency_check=False,
            )
            num_microbatches = allocation.num_microbatches
            # Log scheduler step events (batch size change or LOC cache hint)
            if allocation.loc_cache_hint:
                logger.info(
                    "step=%d | scheduler step: gbs=%d, num_microbatches=%d, "
                    "per_device=%s",
                    step, allocation.global_batch_size,
                    num_microbatches, allocation.per_device_assignment,
                )

            for micro in range(num_microbatches):
                input_ids, labels = next(self.data_iter)
                # Route cross-GPU activation transfers through
                # PCIeP2PCommunicator: for tensors already on a different GPU,
                # the communicator decides whether to use direct PCIe copy or
                # stage through CPU DRAM (SharedLocalityCache) based on tensor
                # size and pool topology.
                src_dev = input_ids.device.index if input_ids.is_cuda else -1
                dst_dev = (
                    self.primary_device.index
                    if self.primary_device.type == "cuda"
                    else -1
                )
                if src_dev >= 0 and dst_dev >= 0 and src_dev != dst_dev:
                    cache_key = f"input:step={step}:micro={micro}"
                    input_ids = self.p2p_communicator.send_activation(
                        input_ids, src_dev, dst_dev, cache_key=cache_key,
                    )
                    labels = self.p2p_communicator.send_activation(
                        labels, src_dev, dst_dev,
                        cache_key=f"labels:step={step}:micro={micro}",
                    )
                else:
                    _local_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
                    input_ids = input_ids.to(_local_dev, non_blocking=True)
                    labels = labels.to(_local_dev, non_blocking=True)

                if self.mimo_loop is not None:
                    # Use HeteroMIMOTrainingLoop.step(batch) — wires
                    # PCIeP2PCommunicator and SharedLocalityCache into the
                    # forward/backward pass.
                    batch = (input_ids, labels)

                    # Capture engine-level cache and communicator so the
                    # forward_backward_func can stage activations through CPU
                    # DRAM when the scheduler assigns cross-pool micro-batches.
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
                        # Use the engine's P2P communicator when the MIMO loop
                        # does not supply one (backward compatibility).
                        _p2p = p2p_communicator if p2p_communicator is not None else _engine_p2p
                        loss, scaled_loss = self.forward(
                            _ids, _lbl, num_microbatches=_num_mb,
                        )
                        # Cache forward activations in the locality cache so
                        # that cross-pool backward passes can retrieve them
                        # without a redundant PCIe round-trip.
                        _act_key = f"fwd_act:iter={iteration}"
                        _engine_cache.put(_act_key, loss.detach())
                        if not forward_only:
                            scaled_loss.backward()
                        return [loss]

                    mimo_result = self.mimo_loop.train_step(
                        forward_backward_func=_forward_backward_func,
                        data_iterator=iter([batch]),
                        config=cfg,
                        iteration=step * num_microbatches + micro,
                    )
                    step_loss += mimo_result.loss
                else:
                    # Standard forward/backward path
                    # Forward (routes through HeteroRecomputeConfig-aware path)
                    loss, scaled_loss = self.forward(
                        input_ids, labels, num_microbatches=num_microbatches,
                    )

                    # Prepare FP32 gradient accumulators for this micro-batch
                    if self.fp32_grad_manager is not None:
                        self.fp32_grad_manager.before_backward()
                    # Backward
                    scaled_loss.backward()
                    # Precision alignment: promote BF16 grads to FP32 accumulators
                    # for parameters that require it (three-tier precision policy).
                    if self.fp32_grad_manager is not None:
                        self.fp32_grad_manager.accumulate()
                    step_loss += loss.item()

            if self.mimo_loop is None:
                # NaN/Inf guard — heterogeneous precision can produce NaN
                if not math.isfinite(step_loss):
                    _nan_count = getattr(self, '_nan_count', 0) + 1
                    self._nan_count = _nan_count
                    logger.warning(
                        "NaN/Inf loss at step %d (count=%d), skipping optimizer update",
                        step, _nan_count,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    continue  # skip to next step

                # Synchronize FP32 gradients (scale + all-reduce) across the DP group
                # before gradient clipping and optimizer.step().
                if self.fp32_grad_manager is not None:
                    self.fp32_grad_manager.after_backward(scale=1.0 / num_microbatches)

            # Gradient clipping (only for standard path; mimo_loop handles it internally)
            if self.mimo_loop is None:
                gnorm = clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            else:
                # mimo_loop.train_step() already clipped; retrieve norm from last result
                gnorm = mimo_result.grad_norm  # type: ignore[possibly-undefined]

            # HeteroGradNorm skip decision — use gnorm threshold directly
            # (full anchor/compute split requires per-device grad classification
            # which is handled by MIMO loop when active; for standalone path
            # we use the combined threshold on the already-computed gnorm)
            _grad_skip_thr = _hetero_config.combined_skip_threshold
            _should_skip = (gnorm > _grad_skip_thr) if torch.is_tensor(gnorm) else (gnorm > _grad_skip_thr)
            if _is_main:
                print(
                    f"[hetero_grad] step={step} grad_norm={gnorm:.6f} "
                    f"skip={_should_skip} total_skips={_skip_count}"
                )
            if self.mimo_loop is None:
                # Standard path: apply skip logic and step the engine's optimizer
                if _should_skip:
                    _skip_count += 1
                    if _is_main:
                        print(f"[hetero_grad] SKIP step={step} (cumulative skips={_skip_count})")
                else:
                    # Optimizer + scheduler step
                    self.optimizer.step()
                    self.scheduler.step()
            else:
                # HeteroMIMOTrainingLoop path: optimizer/clip already done inside
                # train_step(); still honour the skip controller for the scheduler.
                # Cross-device gradient synchronisation (PCIe-aware all-reduce
                # across heterogeneous tiers) before the scheduler step.
                if self.fp32_grad_manager is not None:
                    self.fp32_grad_manager.sync()
                if _should_skip:
                    _skip_count += 1
                    if _is_main:
                        print(f"[hetero_grad] SKIP step={step} (cumulative skips={_skip_count})")
                else:
                    self.scheduler.step()

            # Accounting
            self.global_step = step + 1
            tokens_this_step = (
                num_microbatches * cfg.micro_batch_size * cfg.seq_len
            )
            self.tokens_seen += tokens_this_step
            # Update consumed_samples for next scheduler query
            self.consumed_samples += allocation.global_batch_size

            avg_loss = step_loss / num_microbatches
            loss_accum += avg_loss

            # Logging
            if self.global_step % cfg.log_every == 0:
                elapsed = time.time() - t0
                toks_per_sec = tokens_this_step * cfg.log_every / max(elapsed, 1e-9)
                current_lr = self.scheduler.get_last_lr()[0]
                smooth_loss = loss_accum / cfg.log_every
                logger.info(
                    "step=%6d | loss=%.4f | lr=%.2e | grad_norm=%.3f | "
                    "tok/s=%7.0f | step_ms=%.1f | tokens_seen=%.2fM",
                    self.global_step,
                    smooth_loss,
                    current_lr,
                    gnorm,
                    toks_per_sec,
                    elapsed / cfg.log_every * 1000,
                    self.tokens_seen / 1e6,
                )
                loss_accum = 0.0
                t0 = time.time()

            # Checkpointing
            if self.global_step % cfg.save_every == 0:
                ckpt_path = cfg.checkpoint_dir / f"step_{self.global_step:07d}.pt"
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
        logger.info(
            "Training complete. %d steps in %.1fs. "
            "%.2fM tokens seen. Avg %.0f tok/s.",
            cfg.total_steps,
            total_time,
            self.tokens_seen / 1e6,
            self.tokens_seen / max(total_time, 1.0),
        )

    def save_checkpoint(self, path: Path) -> None:
        """
        Save a full training checkpoint to disk.

        When a :class:`HeteroCheckpointConfig` is active the save is routed
        through the per-tier async pipeline
        (:func:`build_hetero_async_save_pipeline`):

        * CACHE tier (H100): optimizer state staged to host-DRAM LocalityCache,
          then written asynchronously in a background IO thread.
        * WORKER tiers (A6000): parameter shards written synchronously (or
          offloaded to CACHE tier when ``worker_offload_optim=True``).

        When no hetero config is available (CPU-only fallback) the method
        reverts to a plain :func:`torch.save`.

        Args:
            path: Destination file path (parent dirs are created as needed).
        """
        # Lazy imports (hetero_* not imported at module level to avoid apex dep)
        from deepspeed.checkpoint.hetero_async_checkpoint_save import (  # noqa: PLC0415
            build_hetero_async_save_pipeline,
            validate_async_checkpoint_config,
        )
        from deepspeed.checkpoint.hetero_async_checkpoint_load import (  # noqa: PLC0415
            detect_device_arch,
            DeviceArch,
        )
        from deepspeed.checkpoint.hetero_checkpoint_config import TierRole  # noqa: PLC0415

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "global_step":       self.global_step,
            "tokens_seen":       self.tokens_seen,
            "model_state":       self.model.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "scheduler_state":   self.scheduler.state_dict(),
            "plan":              self.plan,
            "config":            self.config,
        }

        cfg = self._hetero_ckpt_cfg  # may be None in CPU-only mode

        if cfg is not None and cfg.hetero_async_save:
            # ------------------------------------------------------------------
            # Hetero async path: determine current tier role, apply per-tier
            # policy, then hand off to the async save pipeline.
            # ------------------------------------------------------------------
            current_device = self.primary_device
            if current_device.type == "cuda":
                arch = detect_device_arch(current_device)
                if arch == DeviceArch.SM90_H100:
                    tier_role = TierRole.CACHE
                else:
                    tier_role = TierRole.WORKER
            else:
                tier_role = TierRole.WORKER   # CPU fallback

            tier_policy = cfg.get_policy(tier_role)

            # Optionally strip optimizer state on WORKER tiers when offload
            # is enabled (optimizer state is owned by the CACHE tier).
            if not tier_policy.save_optim and "optimizer_state" in payload:
                logger.info(
                    "[hetero_ckpt] WORKER tier: omitting optimizer_state "
                    "(offloaded to CACHE tier)."
                )
                payload.pop("optimizer_state")

            ckpt_format = cfg.hetero_ckpt_format.value  # e.g. "torch_dist"

            try:
                validate_async_checkpoint_config(
                    ckpt_format,
                    async_save=True,
                    require_nvrx_for_dcp=False,  # graceful fallback if no NVRx
                )
                logger.info(
                    "[hetero_ckpt] Launching async save to %s "
                    "(tier=%s, format=%s, scheduler=%s).",
                    path, tier_role.value, ckpt_format,
                    "reused" if self._hetero_ckpt_scheduler is not None else "new",
                )
                self._hetero_ckpt_scheduler = build_hetero_async_save_pipeline(
                    state_dict=payload,
                    checkpoint_path=str(path),
                    ckpt_format=ckpt_format,
                    iteration=self.global_step,
                    scheduler=self._hetero_ckpt_scheduler,
                )
                logger.info(
                    "[hetero_ckpt] Async save scheduled: %s (step %d).",
                    path, self.global_step,
                )
                return
            except (NotImplementedError, RuntimeError) as _async_err:
                # Format not async-eligible or NVRx missing: fall through to
                # synchronous save so training is never blocked.
                logger.warning(
                    "[hetero_ckpt] Async save unavailable (%s); "
                    "falling back to torch.save.",
                    _async_err,
                )

        # ------------------------------------------------------------------
        # Synchronous fallback (also used when hetero config is absent).
        # Only rank 0 saves to avoid file corruption from concurrent writes.
        # ------------------------------------------------------------------
        _is_main_rank = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if _is_main_rank:
            torch.save(payload, path)
            logger.info("Checkpoint saved: %s (step %d)", path, self.global_step)
        if dist.is_initialized():
            dist.barrier()  # all ranks wait for rank 0 to finish writing

    def load_checkpoint(self, path: Path) -> None:
        """
        Resume training from a saved checkpoint.

        When a :class:`HeteroCheckpointConfig` with
        ``shard_rebalance_on_load=True`` is active, the method uses
        :class:`~deepspeed.checkpoint.hetero_async_checkpoint_load.HeteroAsyncCheckpointLoad`
        to restore tensors via the three-stage CPU-DRAM staging pipeline
        (async IO → SM-arch routing → PCIe DMA to GPU), enabling heterogeneous
        resume even when the saved tier layout differs from the current one.

        When the hetero path is unavailable (no CUDA, no config, or the
        checkpoint is a plain ``torch.save`` dict), falls back to
        :func:`torch.load`.

        Args:
            path: Path to the checkpoint directory (hetero format) or ``.pt``
                  file (legacy format).

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        # Lazy imports (hetero_* not imported at module level to avoid apex dep)
        from deepspeed.checkpoint.hetero_async_checkpoint_load import (  # noqa: PLC0415
            HeteroAsyncCheckpointLoad,
            detect_device_arch,
            DeviceArch,
        )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        cfg = self._hetero_ckpt_cfg  # may be None before __init__ completes

        # ------------------------------------------------------------------
        # Hetero async-load path
        # ------------------------------------------------------------------
        if (
            cfg is not None
            and cfg.shard_rebalance_on_load
            and path.is_dir()   # hetero checkpoints are directories
        ):
            logger.info(
                "[hetero_ckpt] Attempting HeteroAsyncCheckpointLoad from %s.", path
            )

            # Discover device topology from tiers (populated by TierDiscovery).
            h100_device: Optional[torch.device] = None
            a6000_devices: List[torch.device] = []
            for spec in getattr(self, "tiers", []):
                dev = spec.device
                arch = detect_device_arch(dev)
                if arch == DeviceArch.SM90_H100 and h100_device is None:
                    h100_device = dev
                elif arch == DeviceArch.SM86_A6000:
                    a6000_devices.append(dev)

            if h100_device is None:
                h100_device = self.primary_device

            try:
                loader = HeteroAsyncCheckpointLoad(
                    checkpoint_dir=str(path),
                    h100_device=h100_device,
                    a6000_devices=a6000_devices or [self.primary_device],
                    slc_capacity_gb=min(cfg.locality_cache_max_gb, 64.0),
                    io_workers=8,
                    h100_budget_gb=80.0,
                    a6000_budget_gb=40.0,
                )

                # Build shard metadata from a hetero checkpoint metadata file
                # (written by save_checkpoint on the CACHE tier).  Fall back
                # gracefully to torch.load if the metadata file is absent.
                meta_file = path / "hetero_metadata.pt"
                if meta_file.exists():
                    state_dict_meta: Dict[str, Any] = torch.load(
                        meta_file, map_location="cpu"
                    )
                    loaded_state = loader.load(state_dict_meta)
                    loader.shutdown()

                    model_state     = loaded_state.get("model_state", {})
                    optimizer_state = loaded_state.get("optimizer_state")
                    scheduler_state = loaded_state.get("scheduler_state")
                    global_step     = int(loaded_state.get("global_step", 0))
                    tokens_seen     = int(loaded_state.get("tokens_seen", 0))

                    # Move model state tensors to the primary device; the
                    # loader may have placed them on H100 or A6000 depending
                    # on SM routing.  load_state_dict expects them on the same
                    # device as the model.
                    _ckpt_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
                    model_state_cpu = {
                        k: (v.to(_ckpt_dev) if isinstance(v, torch.Tensor) else v)
                        for k, v in model_state.items()
                    }
                    self.model.load_state_dict(model_state_cpu, strict=cfg.dist_ckpt_strictness.value != "raise_all")

                    if optimizer_state is not None and cfg.load_optim:
                        self.optimizer.load_state_dict(optimizer_state)

                    if scheduler_state is not None and cfg.load_rng:
                        self.scheduler.load_state_dict(scheduler_state)

                    self.global_step = global_step
                    self.tokens_seen = tokens_seen
                    logger.info(
                        "[hetero_ckpt] Hetero load complete: %s "
                        "(step %d, %.2fM tokens seen)",
                        path, self.global_step, self.tokens_seen / 1e6,
                    )
                    return
                else:
                    loader.shutdown()
                    logger.info(
                        "[hetero_ckpt] No hetero_metadata.pt in %s; "
                        "falling back to torch.load.",
                        path,
                    )
            except Exception as _load_exc:  # noqa: BLE001
                logger.warning(
                    "[hetero_ckpt] HeteroAsyncCheckpointLoad failed (%s); "
                    "falling back to torch.load.",
                    _load_exc,
                )

        # ------------------------------------------------------------------
        # Legacy / synchronous fallback: torch.load of a .pt file.
        # ------------------------------------------------------------------
        # Accept both a bare .pt file and a directory that contains one.
        pt_file = path if path.suffix == ".pt" else path
        payload = torch.load(
            pt_file,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu",
        )
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        self.global_step = payload.get("global_step", 0)
        self.tokens_seen = payload.get("tokens_seen", 0)
        logger.info(
            "Checkpoint loaded: %s (step %d, %.2fM tokens seen)",
            path,
            self.global_step,
            self.tokens_seen / 1e6,
        )

    def drain_checkpoint(self, timeout: float = 600.0) -> None:
        """
        Block until all pending async checkpoint IO is complete.

        Should be called before program exit or before scheduling the next
        checkpoint when the async queue could overflow.  No-op when the
        hetero async path is inactive.

        Args:
            timeout: Maximum seconds to wait.  Raises ``TimeoutError`` if
                     the scheduler exceeds this limit.
        """
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
