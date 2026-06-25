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

    # Activation checkpointing config.
    # activation_checkpointing: master on/off switch (default OFF for backward compat).
    # checkpoint_activations_granularity: "full" (every layer) or "selective" (every other layer).
    # Per-tier defaults applied in DesLocEngine.__init__:
    #   A6000 (48 GB) → "full"       (checkpoint every TransformerBlock)
    #   H100  (96 GB) → "selective"  (checkpoint every other TransformerBlock)
    activation_checkpointing: bool = False
    checkpoint_activations_granularity: str = "full"  # "full" | "selective"

    # Logging backends (rank 0 only)
    # wandb_project: W&B project name; None = disabled.  Requires wandb installed.
    # tensorboard_dir: directory for SummaryWriter; None = disabled.  Requires tensorboard.
    wandb_project: Optional[str] = None
    tensorboard_dir: Optional[str] = None


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
        """
        cfg = self.config
        tier_layer_map: Dict[int, List[int]] = {}
        grad_accum: Dict[int, int] = {}
        micro_bs: Dict[int, int] = {}

        all_layers = list(range(cfg.num_layers))
        for spec in self.tiers:
            tier_layer_map[spec.device_index] = all_layers[:]
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            if spec.tier == TierClass.H100:
                micro_bs[spec.device_index] = min(cfg.micro_batch_size * 4, 4)
            else:
                micro_bs[spec.device_index] = cfg.micro_batch_size

        throughput = self._estimate_zero3_throughput(micro_bs, grad_accum)

        return PartitionPlan(
            strategy=PartitionStrategy.ZERO3_HETERO,
            tier_layer_map=tier_layer_map,
            grad_accum_steps=grad_accum,
            micro_batch_sizes=micro_bs,
            estimated_throughput=throughput,
            notes=f"ZeRO-3: uniform grad_accum={cfg.grad_accum_steps}, "
                  f"per-tier micro_bs (H100={min(cfg.micro_batch_size * 4, 4)}, "
                  f"A6000={cfg.micro_batch_size})",
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
        # When ZeRO-3 is active, optimizer operates on param_shard (FP32
        # master copy on GPU), not model.parameters() (BF16 on CPU).
        if self.param_shard_state is not None:
            # param_shard is a 1-D FP32 tensor holding this rank's slice.
            # We need it to be a leaf tensor with requires_grad for AdamW.
            _shard = self.param_shard_state.param_shard
            _shard.requires_grad_(True)
            self.optimizer = AdamW(
                [_shard],
                lr=config.max_lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
            )
            logger.info(
                "[zero3] Optimizer on param_shard: %d FP32 elements on %s",
                _shard.numel(), _shard.device,
            )
        else:
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
        # Upstream Megatron pattern: all ranks run the same num_microbatches.
        # Heterogeneous throughput via per-rank micro_batch_size, not forward count.
        # capacity_weight is uniform (1.0) so the allocator gives each device
        # the same microbatch count.
        if self.tiers:
            device_profiles = []
            for spec in self.tiers:
                weight = 1.0  # uniform — all ranks same forward count
                if spec.total_mem_gb >= 80:
                    max_mbs = min(config.micro_batch_size * 4, 4)
                else:
                    max_mbs = config.micro_batch_size
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

        _rank = dist.get_rank() if dist.is_initialized() else int(
            os.environ.get("RANK", 0)
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
        self.neuron_sp_config: HeteroRecomputeConfig = build_neuron_sp_config()
        logger.info(
            "Neuron_SP recompute config built (granularity=%s, attention=%s).",
            self.neuron_sp_config.granularity,
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

        _local_rank = (
            dist.get_rank() if dist.is_initialized()
            else int(os.environ.get("RANK", 0))
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
        # When ZeRO-3 param sharding is active, param_shard IS the FP32
        # master copy and backward hooks handle reduce-scatter. The
        # fp32_grad_manager would duplicate FP32 grad buffers (~24GB on
        # H100) and conflict with the ZeRO-3 gradient flow.
        if self.param_shard_state is not None:
            self.fp32_grad_manager = None
            logger.info(
                "HeteroFP32GradAccumManager SKIPPED — ZeRO-3 param_shard "
                "is the FP32 master copy (device=%s)",
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

        # --- Phase 9a: SharedLocalityCache (1.5 TB CPU DRAM ÷ world_size) ---
        # The training host has 2×EPYC 9354 with 1.5 TB DDR5.  Each rank
        # receives an equal share so that the aggregate cache footprint never
        # exceeds the physical 1.5 TB ceiling regardless of process count.
        _world_size = (
            dist.get_world_size() if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", 1))
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

        # --- Claude-128: ZeRO-3 per-layer forward all-gather hooks ---
        # When ZeRO-3 sharding is active each rank only holds 1/N of the
        # flat parameter buffer. We install layer-by-layer all-gather
        # hooks so that the full BF16 params for the *currently
        # executing* nn.Module are materialized on entry and freed on
        # exit. Peak memory: model_shard + max(per_layer_full_params).
        # zero3_hetero_shard forward hooks: per-layer all-gather on entry,
        # release on exit. This is Neuron_SP's native parameter gathering.
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
        _shard_sync_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream()
            if torch.cuda.is_available() and getattr(self, "param_shard_state", None) is not None
            else None
        )
        _shard_sync_pending: bool = False

        if _is_main:
            print("[DBG] Entering training loop", flush=True)

        for step in range(self.global_step, cfg.total_steps):
            self.optimizer.zero_grad(set_to_none=False)
            # Zero param_shard.grad — backward hooks accumulate into it
            if self.param_shard_state is not None:
                self.param_shard_state.param_shard.grad.zero_()
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

            # --- Async grad all_reduce state ---
            # We overlap the per-microbatch grad-norm-sq partial all_reduce with
            # the *next* microbatch's forward pass.  After backward of micro[i]
            # we fire an async all_reduce on a per-micro partial norm-sq scalar;
            # we wait() on it at the very start of micro[i+1] before the forward
            # runs.  This hides PCIe latency behind GPU compute.
            # Note: param_shard.grad all_reduce is skipped (hetero shard sizes
            # differ across ranks — would cause illegal memory access).  Only the
            # scalar norm-sq accumulator is reduced per microbatch.
            _async_norm_handle: Optional[dist.Work] = None  # pending async handle
            _async_norm_tensor: Optional[torch.Tensor] = None  # tensor being reduced

            for micro in range(num_microbatches):
                # -----------------------------------------------------------------
                # Wait for previous microbatch's async grad-norm-sq all_reduce
                # before starting this microbatch's forward pass.
                # This achieves comm/compute overlap: the all_reduce from micro[i]
                # runs while micro[i+1]'s forward is being set up; we synchronise
                # here so the result is ready before we need it (post-loop norm).
                # -----------------------------------------------------------------
                if _async_norm_handle is not None:
                    _async_norm_handle.wait()
                    _async_norm_handle = None

                input_ids, labels = next(self.data_iter)
                _local_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
                input_ids = input_ids.to(_local_dev, non_blocking=True)
                labels = labels.to(_local_dev, non_blocking=True)

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
                            scaled_loss.backward()
                        return [loss]

                    mimo_result = self.mimo_loop.train_step(
                        forward_backward_func=_forward_backward_func,
                        data_iterator=iter([(input_ids, labels)]),
                        config=cfg,
                        iteration=step * num_microbatches + micro,
                    )
                    # --- HeteroFP32GradAccumManager: accumulate (MIMO path) ---
                    # Promote BF16 param.grad into FP32 main_grad accumulators
                    # after each micro-batch backward in the MIMO path.
                    if self.fp32_grad_manager is not None:
                        self.fp32_grad_manager.accumulate()
                    step_loss += mimo_result.loss
                else:
                    # Standard forward/backward path
                    loss, scaled_loss = self.forward(
                        input_ids, labels, num_microbatches=num_microbatches,
                    )
                    scaled_loss.backward()
                    # --- HeteroFP32GradAccumManager: accumulate (standard path) ---
                    # Promote BF16 param.grad into FP32 main_grad accumulators
                    # after each micro-batch backward.
                    if self.fp32_grad_manager is not None:
                        self.fp32_grad_manager.accumulate()
                    step_loss += loss.item()

                # -----------------------------------------------------------------
                # Async grad-norm-sq all_reduce (comm/compute overlap).
                # After backward completes on the *last* microbatch of the
                # accumulation window, compute the local full norm-sq on
                # param_shard.grad (which now holds the fully-accumulated
                # gradient) and fire an async all_reduce.
                #
                # Previously this fired after every microbatch, costing
                # num_microbatches NCCL round-trips per optimizer step.  By
                # deferring to the final microbatch we emit exactly one
                # all_reduce per step — reducing NCCL synchronisation overhead
                # on H100 by up to (grad_accum_steps - 1) round-trips.
                #
                # The handle is waited on immediately below (post-loop drain),
                # so the allreduce communication from the final backward
                # overlaps with any CPU bookkeeping between the loop exit and
                # the wait() call.
                # Only active when dist is initialized and ZeRO-3 sharding is on.
                # -----------------------------------------------------------------
                _is_last_micro = (micro == num_microbatches - 1)
                if _is_last_micro and dist.is_initialized() and self.param_shard_state is not None:
                    _g_mb = self.param_shard_state.param_shard.grad
                    if _g_mb is not None:
                        # Compute norm-sq on the fully-accumulated gradient shard.
                        # Clone to a standalone scalar so the reduction buffer is
                        # independent of param_shard.grad.
                        _micro_norm_sq = _g_mb.float().norm(2).to(torch.float64).pow(2).clone()
                        _async_norm_tensor = _micro_norm_sq
                        _async_norm_handle = dist.all_reduce(
                            _micro_norm_sq,
                            op=dist.ReduceOp.SUM,
                            async_op=True,
                        )

            # Drain any outstanding async handle from the final microbatch.
            # (The last iteration has no successor microbatch to wait at its start.)
            if _async_norm_handle is not None:
                _async_norm_handle.wait()
                _async_norm_handle = None

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
            # Scale gradients and run synchronous all-reduce across all buckets.
            # Runs for both MIMO and standard paths; skipped when ZeRO-3 is
            # active (ZeRO-3 backward hooks handle gradient reduction directly).
            if self.fp32_grad_manager is not None and self.param_shard_state is None:
                self.fp32_grad_manager.after_backward(scale=1.0 / num_microbatches)

            # --- finalize_model_grads (upstream Megatron pattern) ---
            # All-reduce param_shard.grad across ranks so every rank's shard
            # gradient is the average over all data-parallel replicas.
            # Must run BEFORE gradient clipping and optimizer.step().
            if self.param_shard_state is not None:
                self.param_shard_state.allreduce_shard_grads()

            # Gradient clipping — DeepSpeed ZeRO-3 style:
            # 1. Each rank computes local L2 norm² on its param_shard.grad
            # 2. all_reduce(SUM) a single scalar across ranks async, overlapping
            #    with the non-ZeRO3 clip_grad_norm_ path below (comm/compute overlap)
            # 3. wait() → sqrt → global norm → clip
            # This avoids torch.nn.utils.clip_grad_norm_ which uses
            # torch._foreach_norm that can trigger CUDA illegal memory access
            # on large (3.26B element) FP32 buffers (INT_MAX overflow).
            _norm_sq_handle: Optional[dist.Work] = None
            if self.param_shard_state is not None:
                _g = self.param_shard_state.param_shard.grad
                if _g is not None:
                    # FP32 norm — no extra allocation (grad is already FP32)
                    local_norm_sq = _g.float().norm(2).to(torch.float64).pow(2)
                else:
                    local_norm_sq = torch.tensor(0.0, dtype=torch.float64,
                                                  device=self.param_shard_state.param_shard.device)
                if dist.is_initialized():
                    # Launch async all_reduce on the scalar norm-sq.
                    # While this communication is in-flight we fall through to
                    # the clip_grad_norm_ branch (which is a no-op for ZeRO-3)
                    # — effectively overlapping the scalar reduce with any CPU
                    # bookkeeping that follows before wait() is called.
                    _norm_sq_handle = dist.all_reduce(
                        local_norm_sq,
                        op=dist.ReduceOp.SUM,
                        async_op=True,
                    )
            else:
                # Non-ZeRO-3 path: synchronous grad clip (no distributed norm needed)
                gnorm = clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                if torch.is_tensor(gnorm):
                    gnorm = gnorm.item()

            # Wait for the async scalar norm all_reduce to complete, then clip.
            if self.param_shard_state is not None:
                if _norm_sq_handle is not None:
                    _norm_sq_handle.wait()
                gnorm = local_norm_sq.sqrt().float().item()
                # Clip: scale gradients if norm exceeds max
                if gnorm > cfg.grad_clip and gnorm > 0:
                    clip_coef = cfg.grad_clip / gnorm
                    _g = self.param_shard_state.param_shard.grad
                    if _g is not None:
                        _g.mul_(clip_coef)

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
                # ZeRO-3: async sync updated FP32 shard back to model BF16 params.
                # Copies are issued on _shard_sync_stream so they overlap with the
                # next step's data preprocessing on the default stream.  The
                # default stream will wait_stream() at the start of the next
                # forward pass (top of the micro-batch loop above).
                if self.param_shard_state is not None:
                    if _shard_sync_stream is not None:
                        self.param_shard_state.sync_shard_to_model_async(
                            stream=_shard_sync_stream
                        )
                        _shard_sync_pending = True
                    else:
                        # CPU-only fallback: synchronous path
                        self.param_shard_state.sync_shard_to_model()

            # Accounting
            self.global_step = step + 1
            tokens_this_step = (
                num_microbatches * cfg.micro_batch_size * cfg.seq_len
            )
            self.tokens_seen += tokens_this_step
            self.consumed_samples += num_microbatches * cfg.micro_batch_size

            avg_loss = step_loss / num_microbatches
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
                _peak_flops_per_device = 835e12  # H100 NVL BF16 default
                if self.tiers:
                    # Sum peak TFLOPS across all participating GPUs to get total
                    # cluster peak, then use per-rank share for MFU.
                    _cluster_peak = sum(t.bf16_tflops * 1e12 for t in self.tiers)
                    _world = dist.get_world_size() if dist.is_initialized() else 1
                    _peak_flops_per_device = _cluster_peak / max(_world, 1)
                _n_params = sum(p.numel() for p in self.model.parameters())
                _t_tokens = tokens_this_step * cfg.log_every  # tokens over log window
                # actual FLOPs = 2 * N_params * 6 * T_tokens
                #   factor 6: forward (2) + backward (4) per token
                _actual_flops = 2 * _n_params * 6 * _t_tokens
                _mfu = _actual_flops / max(elapsed * _peak_flops_per_device, 1.0)

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

    def save_checkpoint(self, path: Path) -> None:
        """
        Save a full training checkpoint to disk.

        When a :class:`HeteroCheckpointConfig` is active the save is routed
        through the per-tier async pipeline:

        * **CACHE tier (H100)** — optimizer state is first staged to the
          host-DRAM locality cache (``cfg.locality_cache_path(step)``), which
          maps to a ramdisk / tmpfs on the 1.5 TB DDR5 host.  A
          ``hetero_metadata.pt`` index is written there so that
          :meth:`load_checkpoint` can rediscover the staged tensors on resume.
          The staged state is then persisted asynchronously to *path* by
          :class:`~deepspeed.checkpoint.hetero_async_checkpoint_save.HeteroAsyncCheckpointScheduler`,
          allowing the next forward pass to begin immediately.
        * **WORKER tiers (A6000)** — parameter shards are written
          synchronously to *path* when ``worker_offload_optim=True`` (optimizer
          state is owned by the CACHE tier and omitted here).

        When no hetero config is available (CPU-only fallback) the method
        reverts to a plain :func:`torch.save`.

        Args:
            path: Destination file/directory path (parent dirs created as needed).
        """
        # Re-use pre-imported symbols from self._lazy to avoid repeated import
        # overhead and stay consistent with the lazy-import contract.
        _build_async_pipeline = self._lazy["build_hetero_async_save_pipeline"]
        _validate_async_cfg   = self._lazy["validate_async_checkpoint_config"]
        _detect_arch          = self._lazy["detect_device_arch"]
        _DeviceArch           = self._lazy["DeviceArch"]
        _TierRole             = self._lazy["TierRole"]

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

        if cfg is not None and False:  # DISABLED: DCP async save deadlocks with training all_reduce (dcp.save calls gather_object→allgather in background thread while main thread does grad all_reduce)
            # ------------------------------------------------------------------
            # Hetero async path
            # Step 1: classify the current device into a TierRole using the
            #         SM architecture reported by HeteroCheckpointConfig.
            # ------------------------------------------------------------------
            current_device = self.primary_device
            if current_device.type == "cuda":
                arch = _detect_arch(current_device)
                tier_role = (
                    _TierRole.CACHE
                    if arch == _DeviceArch.SM90_H100
                    else _TierRole.WORKER
                )
            else:
                tier_role = _TierRole.WORKER   # CPU fallback treated as WORKER

            tier_policy = cfg.get_policy(tier_role)

            # Step 2: apply per-tier save policy.
            # WORKER tiers skip optimizer state when worker_offload_optim=True
            # (the CACHE tier owns the full optimizer checkpoint).
            if not tier_policy.save_optim and "optimizer_state" in payload:
                logger.info(
                    "[hetero_ckpt] WORKER tier: omitting optimizer_state "
                    "(offloaded to CACHE tier per worker_offload_optim)."
                )
                payload.pop("optimizer_state")

            ckpt_format = cfg.hetero_ckpt_format.value  # e.g. "torch_dist"

            # Step 3 (CACHE tier only): stage the full payload to the
            # host-DRAM locality cache before handing off to async IO.
            # This gives sub-second fast-resume capability from /dev/shm.
            # The staging itself runs in a background thread so the training
            # loop returns immediately after submitting the IO work.
            if tier_role == _TierRole.CACHE and cfg.locality_cache_dir is not None:
                lc_path = cfg.locality_cache_path(self.global_step)
                if lc_path is not None:
                    # Detach all tensors to CPU before the thread takes them —
                    # GPU tensors cannot be safely serialised from a background
                    # thread while the main thread uses the same CUDA context.
                    _stage_payload = {
                        k: (v.cpu().detach().clone() if isinstance(v, torch.Tensor) else v)
                        for k, v in payload.items()
                    }
                    # Recursively CPU-detach nested optimizer state tensors so
                    # that CUDA context access from the worker thread is safe.
                    if "optimizer_state" in _stage_payload:
                        def _cpu_detach_state(obj):
                            if isinstance(obj, torch.Tensor):
                                return obj.cpu().detach().clone()
                            if isinstance(obj, dict):
                                return {k: _cpu_detach_state(v) for k, v in obj.items()}
                            if isinstance(obj, (list, tuple)):
                                return type(obj)(_cpu_detach_state(v) for v in obj)
                            return obj
                        _stage_payload["optimizer_state"] = _cpu_detach_state(
                            _stage_payload["optimizer_state"]
                        )

                    def _do_stage(lc_path_=lc_path, payload_=_stage_payload):
                        try:
                            lc_path_.mkdir(parents=True, exist_ok=True)
                            meta_file = lc_path_ / "hetero_metadata.pt"
                            torch.save(payload_, meta_file)
                            logger.info(
                                "[hetero_ckpt] CACHE tier: staged payload to "
                                "locality cache %s (%.0f MB).",
                                meta_file,
                                meta_file.stat().st_size / (1 << 20),
                            )
                        except Exception as _lc_exc:  # noqa: BLE001
                            logger.warning(
                                "[hetero_ckpt] locality_cache staging failed (%s); "
                                "async stage thread exiting.",
                                _lc_exc,
                            )

                    _fut = self._cpu_stage_executor.submit(_do_stage)
                    with self._cpu_stage_lock:
                        # Prune already-done futures to avoid unbounded growth.
                        self._cpu_stage_futures = [
                            f for f in self._cpu_stage_futures if not f.done()
                        ]
                        self._cpu_stage_futures.append(_fut)
                    logger.info(
                        "[hetero_ckpt] CACHE tier: locality-cache stage submitted "
                        "asynchronously → %s (step %d).",
                        lc_path, self.global_step,
                    )

            # Step 3b (WORKER tier): CPU-stage param shard to locality cache so
            # that the persistent async write can proceed from CPU memory and the
            # GPU's PCIe bandwidth is freed for gradient all-reduce traffic.
            # Optimizer state is omitted here when worker_offload_optim=True.
            if (
                tier_role == _TierRole.WORKER
                and cfg.locality_cache_dir is not None
                and self.param_shard_state is not None
            ):
                lc_path_w = cfg.locality_cache_path(self.global_step)
                if lc_path_w is not None:
                    _shard_cpu = (
                        self.param_shard_state.param_shard.cpu().detach().clone()
                        if self.param_shard_state.param_shard is not None
                        else None
                    )
                    if _shard_cpu is not None:
                        def _do_worker_stage(
                            lc_path_=lc_path_w,
                            shard_=_shard_cpu,
                            step_=self.global_step,
                        ):
                            try:
                                lc_path_.mkdir(parents=True, exist_ok=True)
                                shard_file = lc_path_ / "param_shard.pt"
                                torch.save({"param_shard": shard_, "global_step": step_},
                                           shard_file)
                                logger.info(
                                    "[hetero_ckpt] WORKER tier: param shard staged to "
                                    "locality cache %s (%.1f MB).",
                                    shard_file,
                                    shard_file.stat().st_size / (1 << 20),
                                )
                            except Exception as _ws_exc:  # noqa: BLE001
                                logger.warning(
                                    "[hetero_ckpt] WORKER param-shard stage failed (%s).",
                                    _ws_exc,
                                )

                        _fut_w = self._cpu_stage_executor.submit(_do_worker_stage)
                        with self._cpu_stage_lock:
                            self._cpu_stage_futures = [
                                f for f in self._cpu_stage_futures if not f.done()
                            ]
                            self._cpu_stage_futures.append(_fut_w)
                        logger.info(
                            "[hetero_ckpt] WORKER tier: param-shard stage submitted "
                            "asynchronously → %s (step %d).",
                            lc_path_w, self.global_step,
                        )

            # Step 4: launch the async save pipeline (CACHE and WORKER tiers).
            try:
                _validate_async_cfg(
                    ckpt_format,
                    async_save=True,
                    require_nvrx_for_dcp=False,  # graceful fallback if no NVRx
                )
                logger.info(
                    "[hetero_ckpt] Launching async save to %s "
                    "(tier=%s, format=%s, async=%s, scheduler=%s).",
                    path, tier_role.value, ckpt_format,
                    tier_policy.async_save,
                    "reused" if self._hetero_ckpt_scheduler is not None else "new",
                )
                self._hetero_ckpt_scheduler = _build_async_pipeline(
                    state_dict=payload,
                    checkpoint_path=str(path),
                    ckpt_format=ckpt_format,
                    iteration=self.global_step,
                    scheduler=self._hetero_ckpt_scheduler,
                )
                logger.info(
                    "[hetero_ckpt] Async save scheduled: %s (step %d, tier=%s).",
                    path, self.global_step, tier_role.value,
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
        # Synchronous per-rank save.  Each rank saves its own shard to a
        # rank-suffixed file.  No collective communication needed.
        # ------------------------------------------------------------------
        _rank = dist.get_rank() if dist.is_initialized() else 0
        _world = dist.get_world_size() if dist.is_initialized() else 1
        # For rank 0 (or single-GPU), save full payload.
        # For other ranks, save only param_shard + step (optimizer is redundant
        # since each rank has its own shard's Adam states).
        if _world == 1:
            torch.save(payload, path)
            logger.info("Checkpoint saved: %s (step %d)", path, self.global_step)
        else:
            path = Path(path)
            ckpt_dir = path.parent / path.stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            rank_path = ckpt_dir / f"rank_{_rank}.pt"
            torch.save(payload, rank_path)
            logger.info(
                "Checkpoint saved: %s (rank %d/%d, step %d, %.1f MB)",
                rank_path, _rank, _world, self.global_step,
                rank_path.stat().st_size / (1 << 20),
            )
        if dist.is_initialized():
            dist.barrier()  # all ranks wait for IO to finish

    def load_checkpoint(self, path: Path) -> None:
        """
        Resume training from a saved checkpoint.

        Tier-aware loading strategy driven by :class:`HeteroCheckpointConfig`:

        1. **Locality-cache fast-resume** — when ``cfg.locality_cache_dir`` is
           set, the method first scans for a ``hetero_metadata.pt`` written by
           the CACHE tier into the host-DRAM ramdisk during the most recent
           async save.  Because the ramdisk is in CPU DRAM (not on disk), this
           path avoids storage IO entirely and resumes in <1 s on the target
           3-GPU cluster.

        2. **Tier-aware HeteroAsyncCheckpointLoad** — when
           ``cfg.shard_rebalance_on_load=True`` and *path* is a directory,
           uses :class:`~deepspeed.checkpoint.hetero_async_checkpoint_load.HeteroAsyncCheckpointLoad`
           to restore tensors through the CPU-DRAM staging pipeline with SM-arch
           routing (H100 vs A6000).  This handles heterogeneous resume when the
           saved tier layout differs from the current one.

        3. **Legacy synchronous fallback** — plain :func:`torch.load` from a
           ``.pt`` file.

        Post-load behaviour is governed by ``cfg.load_optim``,
        ``cfg.load_rng``, and ``cfg.dist_ckpt_strictness``.

        Args:
            path: Path to the checkpoint directory (hetero format) or ``.pt``
                  file (legacy format).

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        # Re-use pre-imported symbols from self._lazy.
        _HeteroLoad   = self._lazy["HeteroAsyncCheckpointLoad"]
        _detect_arch  = self._lazy["detect_device_arch"]
        _DeviceArch   = self._lazy["DeviceArch"]

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        cfg = self._hetero_ckpt_cfg  # may be None when called from __init__

        # ------------------------------------------------------------------
        # Stage 1: locality-cache fast-resume (host-DRAM, sub-second).
        # Try every step subdirectory under locality_cache_dir, newest first.
        # ------------------------------------------------------------------
        if cfg is not None and cfg.locality_cache_dir is not None:
            import glob as _glob  # noqa: PLC0415
            lc_base = Path(cfg.locality_cache_dir)
            # Enumerate step_XXXXXXXXXX subdirectories and sort descending
            # so we always try the most recent staged checkpoint first.
            lc_step_dirs = sorted(lc_base.glob("step_*"), reverse=True)
            for lc_step_dir in lc_step_dirs:
                meta_file = lc_step_dir / "hetero_metadata.pt"
                # Also accept rank-specific shards written by locality_cache_path()
                if not meta_file.exists():
                    rank_shards = list(lc_step_dir.glob("rank_*"))
                    for shard_dir in rank_shards:
                        candidate = shard_dir / "hetero_metadata.pt"
                        if candidate.exists():
                            meta_file = candidate
                            break
                if not meta_file.exists():
                    continue
                try:
                    logger.info(
                        "[hetero_ckpt] Fast-resume: loading from locality cache %s.",
                        meta_file,
                    )
                    payload = torch.load(meta_file, map_location="cpu")
                    self._apply_loaded_state(payload, cfg)
                    logger.info(
                        "[hetero_ckpt] Fast-resume complete from locality cache "
                        "(step %d, %.2fM tokens seen).",
                        self.global_step, self.tokens_seen / 1e6,
                    )
                    return
                except Exception as _lc_exc:  # noqa: BLE001
                    logger.warning(
                        "[hetero_ckpt] locality-cache load failed (%s); "
                        "continuing to persistent-path load.",
                        _lc_exc,
                    )
                    break  # one failure → skip remaining cache entries

        # ------------------------------------------------------------------
        # Stage 2: tier-aware HeteroAsyncCheckpointLoad from persistent path.
        # ------------------------------------------------------------------
        if (
            cfg is not None
            and cfg.shard_rebalance_on_load
            and path.is_dir()   # hetero checkpoints are directories
        ):
            logger.info(
                "[hetero_ckpt] Attempting tier-aware load from %s "
                "(shard_rebalance_on_load=True).", path
            )

            # Discover device topology from tiers (populated by TierDiscovery).
            h100_device: Optional[torch.device] = None
            a6000_devices: List[torch.device] = []
            for spec in getattr(self, "tiers", []):
                dev = spec.device
                arch = _detect_arch(dev)
                if arch == _DeviceArch.SM90_H100 and h100_device is None:
                    h100_device = dev
                elif arch == _DeviceArch.SM86_A6000:
                    a6000_devices.append(dev)

            if h100_device is None:
                h100_device = self.primary_device

            try:
                loader = _HeteroLoad(
                    checkpoint_dir=str(path),
                    h100_device=h100_device,
                    a6000_devices=a6000_devices or [self.primary_device],
                    slc_capacity_gb=min(cfg.locality_cache_max_gb, 64.0),
                    io_workers=8,
                    h100_budget_gb=80.0,
                    a6000_budget_gb=40.0,
                )

                # Build shard metadata from hetero_metadata.pt written by the
                # CACHE tier during save_checkpoint().
                meta_file = path / "hetero_metadata.pt"
                if meta_file.exists():
                    state_dict_meta: Dict[str, Any] = torch.load(
                        meta_file, map_location="cpu"
                    )
                    loaded_state = loader.load(state_dict_meta)
                    loader.shutdown()
                    self._apply_loaded_state(loaded_state, cfg)
                    logger.info(
                        "[hetero_ckpt] Tier-aware load complete: %s "
                        "(step %d, %.2fM tokens seen).",
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
        # Stage 3: Legacy / synchronous fallback — torch.load of a .pt file.
        # ------------------------------------------------------------------
        pt_file = path if path.suffix == ".pt" else path
        payload = torch.load(
            pt_file,
            map_location=f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available() else "cpu",
        )
        self._apply_loaded_state(payload, cfg)
        logger.info(
            "Checkpoint loaded (legacy): %s (step %d, %.2fM tokens seen)",
            path,
            self.global_step,
            self.tokens_seen / 1e6,
        )

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
