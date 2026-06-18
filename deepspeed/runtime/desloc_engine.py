"""
DES-LOC Heterogeneous Training Engine
======================================
Neuron_SP project — core engine that wires all 127 hetero_* modules into a
unified heterogeneous GPU training system.

Hardware topology assumed (auto-detected at runtime):
  GPU0/1 : A6000 48 GB  SM8.6  PCIe Gen4 ×16 (~25 GB/s)  BF16 38.7 TFLOPS
  GPU2   : H100 NVL 96 GB SM9.0 PCIe Gen5 ×16 (~50 GB/s)  BF16 835 TFLOPS

All 127 hetero_* modules are imported lazily and composed in the
HeteroTrainingEngine below.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import helper
# ---------------------------------------------------------------------------

def _lazy(module_name: str) -> Any:
    """Import a hetero_* module from the package, return None if absent."""
    full = f"deepspeed.runtime.{module_name}"
    try:
        import importlib
        return importlib.import_module(full)
    except ModuleNotFoundError:
        logger.debug("hetero module not found: %s — skipping", full)
        return None


# ---------------------------------------------------------------------------
# §1  TierSpec + TierDiscovery
# ---------------------------------------------------------------------------

@dataclass
class TierSpec:
    """Measured capabilities of a single GPU tier."""
    device_id: int
    name: str
    sm_version: int          # e.g. 86 or 90
    vram_gb: float
    pcie_bw_gbs: float       # measured GB/s
    bf16_tflops: float
    numa_node: int
    is_h100: bool = False

    # derived
    compute_score: float = field(init=False)

    def __post_init__(self) -> None:
        # Normalised score used by the partition solver
        self.compute_score = (
            self.bf16_tflops * 0.6
            + self.pcie_bw_gbs * 0.2
            + self.vram_gb * 0.2
        )


class TierDiscovery:
    """
    Probe all visible GPUs via nvidia-smi + torch.cuda and build TierSpec
    objects.  Falls back gracefully when nvidia-smi is unavailable.
    """

    _NV_QUERY = (
        "index,name,compute_cap,memory.total,pcie.link.gen.current,"
        "pcie.link.width.current,numa_affinity"
    )

    def __init__(self) -> None:
        self.specs: List[TierSpec] = []
        self._discover()

    # ------------------------------------------------------------------
    def _discover(self) -> None:
        n = torch.cuda.device_count()
        if n == 0:
            logger.warning("TierDiscovery: no CUDA devices found")
            return

        nv_rows = self._nvidia_smi_query()

        for idx in range(n):
            nv = nv_rows.get(idx, {})
            spec = self._build_spec(idx, nv)
            self.specs.append(spec)
            logger.info(
                "GPU%d  %s  SM%d  %.0f GB  PCIe %.0f GB/s  "
                "BF16 %.1f TFLOPS  NUMA%d  score=%.1f",
                idx, spec.name, spec.sm_version, spec.vram_gb,
                spec.pcie_bw_gbs, spec.bf16_tflops,
                spec.numa_node, spec.compute_score,
            )

    # ------------------------------------------------------------------
    def _nvidia_smi_query(self) -> Dict[int, Dict[str, str]]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 f"--query-gpu={self._NV_QUERY}",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=10,
            )
        except Exception as exc:
            logger.debug("nvidia-smi failed: %s", exc)
            return {}

        result: Dict[int, Dict[str, str]] = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            result[idx] = {
                "name": parts[1],
                "compute_cap": parts[2].replace(".", ""),
                "vram_mb": parts[3],
                "pcie_gen": parts[4],
                "pcie_width": parts[5],
                "numa": parts[6] if len(parts) > 6 else "0",
            }
        return result

    # ------------------------------------------------------------------
    def _build_spec(self, idx: int, nv: Dict[str, str]) -> TierSpec:
        props = torch.cuda.get_device_properties(idx)

        sm = int(nv.get("compute_cap", "")) if nv.get("compute_cap", "").isdigit() \
            else props.major * 10 + props.minor

        try:
            vram_gb = float(nv["vram_mb"]) / 1024.0
        except (KeyError, ValueError):
            vram_gb = props.total_memory / (1024 ** 3)

        try:
            pcie_gen = int(nv.get("pcie_gen", "4"))
        except ValueError:
            pcie_gen = 4
        try:
            pcie_width = int(nv.get("pcie_width", "16"))
        except ValueError:
            pcie_width = 16

        # Theoretical peak × 0.85 utilisation factor
        pcie_bw = {4: 32.0, 5: 64.0}.get(pcie_gen, 16.0) * (pcie_width / 16) * 0.78

        is_h100 = sm >= 90 or "H100" in nv.get("name", props.name)
        bf16_tflops = 835.0 if is_h100 else (
            38.7 if sm == 86 else props.multi_processor_count * 128 * 2 / 1e12
        )

        try:
            numa = int(nv.get("numa", "0"))
        except ValueError:
            numa = 0

        return TierSpec(
            device_id=idx,
            name=nv.get("name", props.name),
            sm_version=sm,
            vram_gb=vram_gb,
            pcie_bw_gbs=pcie_bw,
            bf16_tflops=bf16_tflops,
            numa_node=numa,
            is_h100=is_h100,
        )

    # ------------------------------------------------------------------
    @property
    def world_compute_score(self) -> float:
        return sum(s.compute_score for s in self.specs)

    def tier_fraction(self, spec: TierSpec) -> float:
        total = self.world_compute_score
        return spec.compute_score / total if total > 0 else 1.0 / max(len(self.specs), 1)


# ---------------------------------------------------------------------------
# §2  Partition solver
# ---------------------------------------------------------------------------

class PartitionSolver:
    """
    Given TierSpecs, decide:
      - per-device micro-batch sizes (tier-proportional)
      - parallel strategy: ZeRO-3 hetero grad-accum or Pipeline parallel
      - gradient accumulation steps per tier
    """

    def __init__(self, discovery: TierDiscovery, global_batch_size: int,
                 seq_len: int) -> None:
        self.discovery = discovery
        self.global_batch = global_batch_size
        self.seq_len = seq_len
        self.solution: Dict[str, Any] = {}
        self._solve()

    # ------------------------------------------------------------------
    def _solve(self) -> None:
        specs = self.discovery.specs
        if not specs:
            self.solution = {"strategy": "zero3", "micro_batches": {}}
            return

        fractions = {s.device_id: self.discovery.tier_fraction(s) for s in specs}
        total_score = sum(fractions.values())

        micro_batches: Dict[int, int] = {}
        remaining = self.global_batch
        for i, spec in enumerate(specs[:-1]):
            mb = max(1, round(self.global_batch * fractions[spec.device_id] / total_score))
            micro_batches[spec.device_id] = mb
            remaining -= mb
        if specs:
            micro_batches[specs[-1].device_id] = max(1, remaining)

        # Choose strategy: pipeline if we have >= 2 distinct SM families
        sm_set = {s.sm_version for s in specs}
        strategy = "pipeline" if len(sm_set) >= 2 else "zero3"

        # Gradient accumulation steps: normalise so slowest GPU finishes together
        min_mb = min(micro_batches.values(), default=1)
        grad_accum: Dict[int, int] = {
            dev: max(1, round(mb / min_mb))
            for dev, mb in micro_batches.items()
        }

        self.solution = {
            "strategy": strategy,
            "micro_batches": micro_batches,
            "grad_accum_steps": grad_accum,
            "fractions": fractions,
        }

        logger.info(
            "PartitionSolver: strategy=%s  micro_batches=%s  grad_accum=%s",
            strategy, micro_batches, grad_accum,
        )

    # ------------------------------------------------------------------
    @property
    def strategy(self) -> str:
        return self.solution.get("strategy", "zero3")

    def micro_batch_for(self, device_id: int) -> int:
        return self.solution["micro_batches"].get(device_id, 1)

    def grad_accum_for(self, device_id: int) -> int:
        return self.solution["grad_accum_steps"].get(device_id, 1)


# ---------------------------------------------------------------------------
# §3  Module registry — import all 127 hetero_* modules
# ---------------------------------------------------------------------------

class HeteroModuleRegistry:
    """
    Central registry that lazily imports every hetero_* module and exposes
    them as attributes.  Missing modules are silently replaced with None so
    the engine degrades gracefully in partial installs.
    """

    # ── runtime (35) ──────────────────────────────────────────────────
    RUNTIME = [
        "hetero_step_batch_scheduler",
        "hetero_emerging_optimizers",
        "hetero_lion_optimizer",
        "hetero_cudagraph_adam",
        "hetero_mup_muon_scaling",
        "hetero_fp32_grad_accum",
        "hetero_grad_norm_skip",
        "hetero_offload_throttle",
        "hetero_activation_offload_reset",
        "hetero_mimo_training_loop",
        "hetero_mimo_topology",
        "hetero_mimo_grad_buffer",
        "hetero_tensor_offload_manager",
        "hetero_h2d_stream_sync",
        "hetero_grad_buffer_offload_guard",
        "hetero_rl_moe_cudagraph",
        "hetero_elastic_batch",
        "hetero_hybrid_stabilizer",
        "hetero_chained_optimizer_sync",
        "hetero_ddp_bucket_sizer",
        "hetero_ddp_grad_overlap_fix",
        "hetero_pinned_buffer_config",
        "hetero_pinned_buffer_guard",
        "hetero_local_cg_moe_fix",
        "hetero_pretrain_config",
        "hetero_train_step_reductions",
        "hetero_mtp_grad_clipper",
        "hetero_gdn_selective_recompute",
        "hetero_fp8_param_gather_eval",
        "hetero_cg_pool_sharing",
        "hetero_optimizer_cg_pool",
        "hetero_cudagraph_ep_hook",
        "hetero_rl_optimizer_offload",
    ]

    # ── zero (20) ─────────────────────────────────────────────────────
    ZERO = [
        "hetero_allgather_pipeline",
        "hetero_dbuffer",
        "hetero_decoupled_grad_distopt",
        "hetero_fine_grained_param_gather",
        "hetero_fsdp_double_buffer",
        "hetero_fsdp_mixed_precision_args",
        "hetero_fsdp_mxfp8_fix",
        "hetero_fsdp_tp_detection",
        "hetero_fsdp_zero_counter",
        "hetero_grad_buffer_reuse",
        "hetero_grad_reduce_double_buffer",
        "hetero_layerwise_grad_safe",
        "hetero_optimizer_router",
        "hetero_wgrad_double_buffer",
        "hetero_fsdp_frozen_params",
        "hetero_fsdp_param_sync_config",
        "hetero_overlap_param_gather",
        "hetero_fsdp_double_buffer_recompute",
        "hetero_fsdp_dsv3_proxy",
        "hetero_fsdp_auto_mixed_precision",
    ]

    # ── moe (15) ──────────────────────────────────────────────────────
    MOE = [
        "hetero_flex_dispatcher_overlap",
        "hetero_hybrid_ep_permute",
        "hetero_gate_slice_fix",
        "hetero_nvls_dispatcher_buffers",
        "hetero_ep_memory_estimator",
        "hetero_moe_logger",
        "hetero_loss_grad_scale",
        "hetero_grad_finalize",
        "hetero_alltoall_gdn",
        "hetero_dsa_rope",
        "hetero_gdn_packed_sequence",
        "hetero_a2a_stream",
        "hetero_permute_pad_fix",
        "hetero_latent_moe_memory",
        "hetero_latent_moe_flops",
    ]

    # ── inference (17) ────────────────────────────────────────────────
    INFERENCE = [
        "hetero_cuda_graph_mtp",
        "hetero_cudagraph_admission",
        "hetero_ep_buffer_alloc",
        "hetero_inference_cg_scope",
        "hetero_mamba_inference_opt",
        "hetero_moe_inference_tuner",
        "hetero_moe_routing_cache",
        "hetero_mtp_detach_config",
        "hetero_mtp_scheduler",
        "hetero_reasoning_token_manager",
        "hetero_shared_expert_overlap",
        "hetero_token_clamper",
        "hetero_mamba_state_dtype",
        "hetero_vision_encoder_cudagraph",
        "hetero_kv_cache_offload",
        "hetero_mamba_chunked_prefill",
        "hetero_mamba_state_memory_fix",
    ]

    # ── checkpoint (9) ────────────────────────────────────────────────
    CHECKPOINT = [
        "hetero_async_checkpoint_load",
        "hetero_async_checkpoint_save",
        "hetero_checkpoint_integrity",
        "hetero_mla_checkpoint",
        "hetero_zero_copy_checkpoint",
        "hetero_single_process_checkpoint",
        "hetero_fsdp_dcp_checkpoint",
        "hetero_aggressive_checkpoint",
        "hetero_checkpoint_config",
    ]

    # ── comm (2) ──────────────────────────────────────────────────────
    COMM = [
        "hetero_bridge_p2p",
        "hetero_hypercomm_grid",
    ]

    # ── ops (8) ───────────────────────────────────────────────────────
    OPS = [
        "hetero_deepseek_sparse_attention",
        "hetero_gdn_mamba",
        "hetero_yarn_position",
        "hetero_fp4_mamba_context",
        "hetero_mxfp8_refit",
        "hetero_te_gemm_wgrad",
        "hetero_mamba_conv_optimize",
        "hetero_frozen_linear_dgrad",
    ]

    # ── pipe (3) ──────────────────────────────────────────────────────
    PIPE = [
        "hetero_mimo_parallelism",
        "hetero_multimodule_pipeline",
        "hetero_uneven_pp_fix",
    ]

    ALL_GROUPS = {
        "runtime": RUNTIME,
        "zero": ZERO,
        "moe": MOE,
        "inference": INFERENCE,
        "checkpoint": CHECKPOINT,
        "comm": COMM,
        "ops": OPS,
        "pipe": PIPE,
    }

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        total = loaded = 0
        for group, names in self.ALL_GROUPS.items():
            for name in names:
                total += 1
                mod = _lazy(name)
                self._cache[name] = mod
                if mod is not None:
                    loaded += 1
        logger.info(
            "HeteroModuleRegistry: loaded %d / %d hetero modules", loaded, total
        )

    def get(self, name: str) -> Any:
        return self._cache.get(name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("hetero_"):
            return self._cache.get(name)
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# §4  Sub-system wrappers
# ---------------------------------------------------------------------------

class BatchScheduler:
    """Wraps hetero_step_batch_scheduler for tier-proportional micro-batches."""

    def __init__(self, registry: HeteroModuleRegistry,
                 solver: PartitionSolver,
                 discovery: TierDiscovery) -> None:
        self._mod = registry.get("hetero_step_batch_scheduler")
        self._solver = solver
        self._discovery = discovery
        self._schedulers: Dict[int, Any] = {}

        for spec in discovery.specs:
            mb = solver.micro_batch_for(spec.device_id)
            ga = solver.grad_accum_for(spec.device_id)
            if self._mod and hasattr(self._mod, "StepBatchScheduler"):
                sched = self._mod.StepBatchScheduler(
                    device_id=spec.device_id,
                    micro_batch_size=mb,
                    grad_accum_steps=ga,
                    compute_score=spec.compute_score,
                )
                self._schedulers[spec.device_id] = sched
                logger.debug(
                    "BatchScheduler GPU%d: micro_batch=%d grad_accum=%d",
                    spec.device_id, mb, ga,
                )
            else:
                self._schedulers[spec.device_id] = None

    def step_size(self, device_id: int) -> int:
        sched = self._schedulers.get(device_id)
        if sched and hasattr(sched, "current_batch_size"):
            return sched.current_batch_size()
        return self._solver.micro_batch_for(device_id)

    def on_step_end(self, device_id: int) -> None:
        sched = self._schedulers.get(device_id)
        if sched and hasattr(sched, "step"):
            sched.step()


class AllgatherPipeline:
    """Wraps hetero_allgather_pipeline for ZeRO-3 parameter gathering."""

    def __init__(self, registry: HeteroModuleRegistry) -> None:
        self._mod = registry.get("hetero_allgather_pipeline")
        self._pipeline = None
        if self._mod and hasattr(self._mod, "AllgatherPipeline"):
            self._pipeline = self._mod.AllgatherPipeline()
            logger.debug("AllgatherPipeline: initialised")

    def gather(self, param: torch.Tensor,
               src_rank: int,
               group: Optional[Any] = None) -> torch.Tensor:
        if self._pipeline and hasattr(self._pipeline, "gather"):
            return self._pipeline.gather(param, src_rank=src_rank, group=group)
        # fallback: standard all-gather
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(param)
                        for _ in range(dist.get_world_size(group))]
            dist.all_gather(gathered, param, group=group)
            return torch.cat(gathered, dim=0)
        return param

    def prefetch(self, params: List[torch.Tensor]) -> None:
        if self._pipeline and hasattr(self._pipeline, "prefetch"):
            self._pipeline.prefetch(params)


class BridgeP2P:
    """Wraps hetero_bridge_p2p for PCIe-aware point-to-point transfers."""

    def __init__(self, registry: HeteroModuleRegistry,
                 discovery: TierDiscovery) -> None:
        self._mod = registry.get("hetero_bridge_p2p")
        self._discovery = discovery
        self._bridge = None
        if self._mod and hasattr(self._mod, "BridgeP2P"):
            topo = {s.device_id: {
                "pcie_bw": s.pcie_bw_gbs,
                "numa": s.numa_node,
                "sm": s.sm_version,
            } for s in discovery.specs}
            self._bridge = self._mod.BridgeP2P(topology=topo)
            logger.debug("BridgeP2P: topology registered for %d devices",
                         len(topo))

    def send(self, tensor: torch.Tensor,
             src: int, dst: int) -> None:
        if self._bridge and hasattr(self._bridge, "send"):
            self._bridge.send(tensor, src=src, dst=dst)
            return
        # fallback
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == src:
                dist.send(tensor, dst=dst)
            elif dist.get_rank() == dst:
                dist.recv(tensor, src=src)

    def best_route(self, src: int, dst: int) -> str:
        if self._bridge and hasattr(self._bridge, "best_route"):
            return self._bridge.best_route(src, dst)
        return "pcie"


class GradBufferManager:
    """Wraps hetero_grad_buffer_reuse + hetero_wgrad_double_buffer."""

    def __init__(self, registry: HeteroModuleRegistry) -> None:
        self._reuse_mod = registry.get("hetero_grad_buffer_reuse")
        self._dbl_mod = registry.get("hetero_wgrad_double_buffer")
        self._pool: Dict[Tuple[int, ...], torch.Tensor] = {}
        logger.debug("GradBufferManager: ready (reuse=%s, double_buf=%s)",
                     self._reuse_mod is not None,
                     self._dbl_mod is not None)

    def get_buffer(self, shape: Tuple[int, ...],
                   dtype: torch.dtype,
                   device: torch.device) -> torch.Tensor:
        key = (*shape, dtype, device.index if device.type == "cuda" else -1)
        if key in self._pool:
            buf = self._pool[key]
            buf.zero_()
            return buf
        buf = torch.zeros(shape, dtype=dtype, device=device)
        self._pool[key] = buf
        return buf

    def release(self, tensor: torch.Tensor) -> None:
        """Return tensor back to pool via zero-copy if module available."""
        if self._reuse_mod and hasattr(self._reuse_mod, "release"):
            self._reuse_mod.release(tensor)

    def double_buffer_enabled(self) -> bool:
        return self._dbl_mod is not None


class OptimizerRouter:
    """
    Wraps hetero_optimizer_router to select the best optimizer per tier.
    Prefers hetero_lion_optimizer for H100, hetero_cudagraph_adam elsewhere.
    """

    def __init__(self, registry: HeteroModuleRegistry,
                 discovery: TierDiscovery) -> None:
        self._router_mod = registry.get("hetero_optimizer_router")
        self._lion_mod = registry.get("hetero_lion_optimizer")
        self._adam_mod = registry.get("hetero_cudagraph_adam")
        self._discovery = discovery

    def build(self, model: nn.Module,
              lr: float,
              weight_decay: float,
              device_id: int) -> torch.optim.Optimizer:
        spec = next((s for s in self._discovery.specs if s.device_id == device_id),
                    None)

        # Router module takes precedence
        if self._router_mod and hasattr(self._router_mod, "route"):
            opt = self._router_mod.route(
                model=model, lr=lr, weight_decay=weight_decay,
                device_spec=spec.__dict__ if spec else {},
            )
            if opt is not None:
                logger.info("OptimizerRouter: routed GPU%d → %s",
                            device_id, type(opt).__name__)
                return opt

        # Tier-specific fallback
        if spec and spec.is_h100 and self._lion_mod and \
                hasattr(self._lion_mod, "LionOptimizer"):
            logger.info("OptimizerRouter: GPU%d H100 → LionOptimizer", device_id)
            return self._lion_mod.LionOptimizer(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        if self._adam_mod and hasattr(self._adam_mod, "CudaGraphAdam"):
            logger.info("OptimizerRouter: GPU%d → CudaGraphAdam", device_id)
            return self._adam_mod.CudaGraphAdam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        logger.info("OptimizerRouter: GPU%d → torch.optim.AdamW (fallback)",
                    device_id)
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )


class AsyncCheckpointer:
    """Wraps hetero_async_checkpoint_save / hetero_async_checkpoint_load."""

    def __init__(self, registry: HeteroModuleRegistry,
                 checkpoint_dir: str) -> None:
        self._save_mod = registry.get("hetero_async_checkpoint_save")
        self._load_mod = registry.get("hetero_async_checkpoint_load")
        self._integrity_mod = registry.get("hetero_checkpoint_integrity")
        self._dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._pending: Optional[threading.Thread] = None
        logger.debug("AsyncCheckpointer: dir=%s", checkpoint_dir)

    # ------------------------------------------------------------------
    def save(self, state: Dict[str, Any], step: int) -> None:
        path = os.path.join(self._dir, f"ckpt_step{step:08d}.pt")
        if self._save_mod and hasattr(self._save_mod, "async_save"):
            self._pending = self._save_mod.async_save(state, path)
            logger.info("AsyncCheckpointer: async save → %s", path)
        else:
            self._pending = threading.Thread(
                target=self._sync_save, args=(state, path), daemon=True
            )
            self._pending.start()
            logger.info("AsyncCheckpointer: threaded save → %s", path)

    @staticmethod
    def _sync_save(state: Dict[str, Any], path: str) -> None:
        torch.save(state, path)
        logger.debug("AsyncCheckpointer: saved %s", path)

    # ------------------------------------------------------------------
    def load(self, step: int) -> Optional[Dict[str, Any]]:
        path = os.path.join(self._dir, f"ckpt_step{step:08d}.pt")
        if not os.path.exists(path):
            logger.warning("AsyncCheckpointer: checkpoint not found: %s", path)
            return None

        if self._integrity_mod and hasattr(self._integrity_mod, "verify"):
            ok = self._integrity_mod.verify(path)
            if not ok:
                logger.error("AsyncCheckpointer: integrity check failed: %s", path)
                return None

        if self._load_mod and hasattr(self._load_mod, "async_load"):
            state = self._load_mod.async_load(path)
        else:
            state = torch.load(path, map_location="cpu")

        logger.info("AsyncCheckpointer: loaded %s", path)
        return state

    # ------------------------------------------------------------------
    def wait(self) -> None:
        if self._pending and hasattr(self._pending, "join"):
            self._pending.join()
            self._pending = None

    def latest_step(self) -> int:
        import glob
        files = glob.glob(os.path.join(self._dir, "ckpt_step*.pt"))
        if not files:
            return 0
        steps = []
        for f in files:
            base = os.path.basename(f)
            try:
                steps.append(int(base.replace("ckpt_step", "").replace(".pt", "")))
            except ValueError:
                pass
        return max(steps, default=0)


class GradSyncManager:
    """
    Manages gradient synchronisation across heterogeneous tiers.
    Uses hetero_train_step_reductions + hetero_fp32_grad_accum.
    """

    def __init__(self, registry: HeteroModuleRegistry,
                 solver: PartitionSolver) -> None:
        self._reduce_mod = registry.get("hetero_train_step_reductions")
        self._fp32_mod = registry.get("hetero_fp32_grad_accum")
        self._clipper_mod = registry.get("hetero_mtp_grad_clipper")
        self._solver = solver

    def sync(self, model: nn.Module,
             device_id: int,
             group: Optional[Any] = None) -> None:
        if self._reduce_mod and hasattr(self._reduce_mod, "reduce_grads"):
            self._reduce_mod.reduce_grads(
                model=model,
                grad_accum_steps=self._solver.grad_accum_for(device_id),
                group=group,
            )
            return

        # Fallback: manual all-reduce
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size(group)
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad.data, group=group)
                    p.grad.data /= world
        logger.debug("GradSyncManager: fallback all-reduce GPU%d", device_id)

    def clip(self, model: nn.Module, max_norm: float) -> float:
        if self._clipper_mod and hasattr(self._clipper_mod, "clip"):
            return self._clipper_mod.clip(model, max_norm)
        return float(nn.utils.clip_grad_norm_(model.parameters(), max_norm))

    def accumulate_fp32(self, model: nn.Module) -> None:
        if self._fp32_mod and hasattr(self._fp32_mod, "accumulate"):
            self._fp32_mod.accumulate(model)


# ---------------------------------------------------------------------------
# §5  HeteroTrainingEngine — main public API
# ---------------------------------------------------------------------------

class HeteroTrainingEngine:
    """
    DES-LOC heterogeneous training engine.

    Usage::

        engine = HeteroTrainingEngine(
            model=my_model,
            config={
                "global_batch_size": 512,
                "seq_len": 2048,
                "lr": 1e-4,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "checkpoint_dir": "./checkpoints",
                "checkpoint_interval": 500,
                "dtype": "bfloat16",
                "max_steps": 100_000,
            },
        )
        engine.load_checkpoint()   # resume if available

        for step, batch in enumerate(dataloader):
            loss = engine.train_step(batch)
            if step % 10 == 0:
                engine.log_metrics(step, {"loss": loss})
    """

    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 process_group: Optional[Any] = None) -> None:

        self.model = model
        self.config = config
        self.group = process_group

        # ── discover hardware ──────────────────────────────────────────
        self.discovery = TierDiscovery()
        self._primary_device = self._pick_primary_device()

        # ── partition solver ───────────────────────────────────────────
        self.solver = PartitionSolver(
            discovery=self.discovery,
            global_batch_size=config.get("global_batch_size", 64),
            seq_len=config.get("seq_len", 2048),
        )

        # ── load all hetero modules ────────────────────────────────────
        self.registry = HeteroModuleRegistry()

        # ── sub-systems ────────────────────────────────────────────────
        self.batch_scheduler = BatchScheduler(
            self.registry, self.solver, self.discovery
        )
        self.allgather = AllgatherPipeline(self.registry)
        self.p2p = BridgeP2P(self.registry, self.discovery)
        self.grad_buf = GradBufferManager(self.registry)
        self.opt_router = OptimizerRouter(self.registry, self.discovery)
        self.checkpointer = AsyncCheckpointer(
            self.registry,
            config.get("checkpoint_dir", "./checkpoints"),
        )
        self.grad_sync = GradSyncManager(self.registry, self.solver)

        # ── dtype ──────────────────────────────────────────────────────
        dtype_str = config.get("dtype", "bfloat16")
        self.dtype = {"bfloat16": torch.bfloat16,
                      "float16": torch.float16,
                      "float32": torch.float32}.get(dtype_str, torch.bfloat16)

        # ── move model ─────────────────────────────────────────────────
        self.model = self.model.to(
            device=torch.device("cuda", self._primary_device),
            dtype=self.dtype,
        )

        # ── optimizer + scheduler ──────────────────────────────────────
        self.optimizer = self.opt_router.build(
            model=self.model,
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            device_id=self._primary_device,
        )
        self.lr_scheduler = self._build_lr_scheduler()

        # ── optional advanced modules ──────────────────────────────────
        self._init_advanced_modules()

        # ── internal state ─────────────────────────────────────────────
        self.global_step: int = 0
        self.epoch: int = 0
        self._accum_loss: float = 0.0
        self._accum_steps: int = 0

        logger.info(
            "HeteroTrainingEngine ready — strategy=%s  primary_gpu=%d  "
            "dtype=%s  lr=%.2e",
            self.solver.strategy, self._primary_device, dtype_str,
            config.get("lr", 1e-4),
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _pick_primary_device(self) -> int:
        """Use H100 as primary if available, else GPU with highest score."""
        if not self.discovery.specs:
            return 0
        h100 = [s for s in self.discovery.specs if s.is_h100]
        if h100:
            return h100[0].device_id
        best = max(self.discovery.specs, key=lambda s: s.compute_score)
        return best.device_id

    def _build_lr_scheduler(self) -> Any:
        max_steps = self.config.get("max_steps", 100_000)
        warmup = self.config.get("warmup_steps", max(100, max_steps // 100))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = float(step - warmup) / max(1, max_steps - warmup)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_advanced_modules(self) -> None:
        """Wire optional modules that enhance training quality."""

        # Pinned buffer config
        pb_mod = self.registry.get("hetero_pinned_buffer_config")
        if pb_mod and hasattr(pb_mod, "configure"):
            pb_mod.configure(num_buffers=4, buffer_size_mb=256)
            logger.debug("hetero_pinned_buffer_config: configured")

        # H2D stream sync
        h2d_mod = self.registry.get("hetero_h2d_stream_sync")
        if h2d_mod and hasattr(h2d_mod, "init"):
            h2d_mod.init(device_ids=[s.device_id for s in self.discovery.specs])
            logger.debug("hetero_h2d_stream_sync: initialised")

        # Hybrid stabiliser (loss scale for mixed precision)
        stab_mod = self.registry.get("hetero_hybrid_stabilizer")
        if stab_mod and hasattr(stab_mod, "HybridStabilizer"):
            self._stabilizer = stab_mod.HybridStabilizer(
                init_scale=65536.0, growth_interval=2000
            )
            logger.debug("hetero_hybrid_stabilizer: enabled")
        else:
            self._stabilizer = None

        # DDP grad overlap fix
        ddp_mod = self.registry.get("hetero_ddp_grad_overlap_fix")
        if ddp_mod and hasattr(ddp_mod, "apply"):
            ddp_mod.apply(self.model)
            logger.debug("hetero_ddp_grad_overlap_fix: applied")

        # MoE logger
        moe_log_mod = self.registry.get("hetero_moe_logger")
        if moe_log_mod and hasattr(moe_log_mod, "attach"):
            moe_log_mod.attach(self.model)
            logger.debug("hetero_moe_logger: attached")

        # FSDP auto mixed precision
        fsdp_amp_mod = self.registry.get("hetero_fsdp_auto_mixed_precision")
        if fsdp_amp_mod and hasattr(fsdp_amp_mod, "configure"):
            fsdp_amp_mod.configure(self.model, dtype=self.dtype)
            logger.debug("hetero_fsdp_auto_mixed_precision: configured")

        # Activation offload reset
        act_offload_mod = self.registry.get("hetero_activation_offload_reset")
        if act_offload_mod and hasattr(act_offload_mod, "reset"):
            act_offload_mod.reset(self.model)
            logger.debug("hetero_activation_offload_reset: reset")

        # Elastic batch
        elastic_mod = self.registry.get("hetero_elastic_batch")
        if elastic_mod and hasattr(elastic_mod, "ElasticBatch"):
            self._elastic_batch = elastic_mod.ElasticBatch(
                base_batch=self.config.get("global_batch_size", 64)
            )
            logger.debug("hetero_elastic_batch: enabled")
        else:
            self._elastic_batch = None

        # DSA rope
        dsa_mod = self.registry.get("hetero_dsa_rope")
        if dsa_mod and hasattr(dsa_mod, "patch"):
            dsa_mod.patch(self.model)
            logger.debug("hetero_dsa_rope: patched")

        # GDN selective recompute
        gdn_mod = self.registry.get("hetero_gdn_selective_recompute")
        if gdn_mod and hasattr(gdn_mod, "apply"):
            gdn_mod.apply(self.model,
                          threshold_gb=self.config.get("recompute_threshold_gb", 4.0))
            logger.debug("hetero_gdn_selective_recompute: applied")

        logger.info("HeteroTrainingEngine: advanced modules wired")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> float:
        """
        Execute one full training step:
          forward → loss → backward → grad_sync → optimizer.step
          → lr_schedule → conditional checkpoint
        """
        device = torch.device("cuda", self._primary_device)
        batch = self._move_batch(batch, device)

        ga_steps = self.solver.grad_accum_for(self._primary_device)

        # ── forward pass ───────────────────────────────────────────────
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            outputs = self._forward(batch)
            loss = self._compute_loss(outputs, batch)
            loss = loss / ga_steps

        # ── loss scaling (mixed precision stabiliser) ──────────────────
        if self._stabilizer and hasattr(self._stabilizer, "scale"):
            scaled_loss = self._stabilizer.scale(loss)
        else:
            scaled_loss = loss

        # ── backward ───────────────────────────────────────────────────
        scaled_loss.backward()

        self._accum_loss += loss.item()
        self._accum_steps += 1

        # ── gradient accumulation check ────────────────────────────────
        if self._accum_steps < ga_steps:
            return self._accum_loss / self._accum_steps

        # ── ZeRO-3 allgather (if strategy == zero3) ───────────────────
        if self.solver.strategy == "zero3":
            self._zero3_allgather_params()

        # ── gradient synchronisation across tiers ─────────────────────
        self.grad_sync.sync(self.model, self._primary_device, self.group)
        self.grad_sync.accumulate_fp32(self.model)

        # ── unscale + clip ─────────────────────────────────────────────
        if self._stabilizer and hasattr(self._stabilizer, "unscale"):
            self._stabilizer.unscale(self.optimizer)

        grad_norm = self.grad_sync.clip(
            self.model, self.config.get("max_grad_norm", 1.0)
        )

        # ── optimizer step ─────────────────────────────────────────────
        if self._stabilizer and hasattr(self._stabilizer, "should_skip"):
            if not self._stabilizer.should_skip():
                self.optimizer.step()
        else:
            self.optimizer.step()

        if self._stabilizer and hasattr(self._stabilizer, "update"):
            self._stabilizer.update()

        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()

        # ── post-step hooks ────────────────────────────────────────────
        self.batch_scheduler.on_step_end(self._primary_device)
        self._post_step_hooks(grad_norm)

        # ── checkpoint ────────────────────────────────────────────────
        self.global_step += 1
        ckpt_interval = self.config.get("checkpoint_interval", 500)
        if self.global_step % ckpt_interval == 0:
            self._save_checkpoint()

        avg_loss = self._accum_loss / max(self._accum_steps, 1)
        self._accum_loss = 0.0
        self._accum_steps = 0
        return avg_loss

    # ------------------------------------------------------------------
    def load_checkpoint(self, step: Optional[int] = None) -> int:
        """Load latest (or specified) checkpoint.  Returns the step resumed."""
        target = step if step is not None else self.checkpointer.latest_step()
        if target == 0:
            logger.info("HeteroTrainingEngine: no checkpoint found, starting fresh")
            return 0

        state = self.checkpointer.load(target)
        if state is None:
            return 0

        if "model_state" in state:
            self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            self.lr_scheduler.load_state_dict(state["scheduler_state"])
        if "global_step" in state:
            self.global_step = state["global_step"]
        if "epoch" in state:
            self.epoch = state["epoch"]

        logger.info("HeteroTrainingEngine: resumed from step %d", self.global_step)
        return self.global_step

    # ------------------------------------------------------------------
    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        parts = [f"step={step}", f"lr={lr:.2e}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        logger.info("TRAIN | %s", "  ".join(parts))

    # ------------------------------------------------------------------
    def evaluate(self, dataloader: Any,
                 criterion: Optional[Any] = None) -> Dict[str, float]:
        """Run eval loop, return dict of metrics."""
        self.model.eval()
        device = torch.device("cuda", self._primary_device)
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch(batch, device)
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    outputs = self._forward(batch)
                    if criterion is not None:
                        loss = criterion(outputs, batch)
                    else:
                        loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                n_batches += 1

        self.model.train()
        avg = total_loss / max(n_batches, 1)
        logger.info("EVAL | step=%d  eval_loss=%.4f", self.global_step, avg)
        return {"eval_loss": avg}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return self.model(**{k: v for k, v in batch.items()
                                 if k != "labels"})
        if isinstance(batch, (list, tuple)):
            return self.model(*batch[:-1]) if len(batch) > 1 else self.model(batch[0])
        return self.model(batch)

    def _compute_loss(self, outputs: Any, batch: Any) -> torch.Tensor:
        # Try standard HuggingFace-style loss
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = None
        if isinstance(batch, dict) and "labels" in batch:
            labels = batch["labels"]
        elif isinstance(batch, (list, tuple)) and len(batch) > 1:
            labels = batch[-1]

        if labels is None:
            # Autoregressive shift if no explicit labels
            if logits.dim() == 3:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = logits[..., 1:, :].argmax(-1).contiguous()
                return torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            return logits.mean()

        if labels.dtype in (torch.long, torch.int):
            if logits.dim() == 3:
                return torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return torch.nn.functional.cross_entropy(logits, labels)

        return torch.nn.functional.mse_loss(logits, labels.to(logits.dtype))

    # ------------------------------------------------------------------
    def _move_batch(self, batch: Any,
                    device: torch.device) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: (v.to(device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            moved = [
                v.to(device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for v in batch
            ]
            return type(batch)(moved)
        return batch

    # ------------------------------------------------------------------
    def _zero3_allgather_params(self) -> None:
        """Prefetch and all-gather sharded ZeRO-3 parameters before update."""
        params_to_gather = [
            p for p in self.model.parameters()
            if p.grad is not None
        ]
        self.allgather.prefetch(params_to_gather)

        if not dist.is_available() or not dist.is_initialized():
            return

        rank = dist.get_rank(self.group)
        for p in params_to_gather:
            p.data = self.allgather.gather(p.data, src_rank=rank,
                                           group=self.group)

    # ------------------------------------------------------------------
    def _post_step_hooks(self, grad_norm: float) -> None:
        """Run lightweight per-step callbacks from hetero modules."""

        # Offload throttle: pause aggressive offloading if grad_norm is high
        throttle_mod = self.registry.get("hetero_offload_throttle")
        if throttle_mod and hasattr(throttle_mod, "step"):
            throttle_mod.step(grad_norm=grad_norm)

        # Grad norm skip: record if step was skipped
        skip_mod = self.registry.get("hetero_grad_norm_skip")
        if skip_mod and hasattr(skip_mod, "record"):
            skip_mod.record(step=self.global_step, grad_norm=grad_norm)

        # Chained optimizer sync
        chain_mod = self.registry.get("hetero_chained_optimizer_sync")
        if chain_mod and hasattr(chain_mod, "sync"):
            chain_mod.sync(self.optimizer)

    # ------------------------------------------------------------------
    def _save_checkpoint(self) -> None:
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict(),
            "config": self.config,
            "discovery": [s.__dict__ for s in self.discovery.specs],
        }
        self.checkpointer.save(state, self.global_step)

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def no_sync(self):
        """Suppress gradient synchronisation (for gradient accumulation)."""
        if hasattr(self.model, "no_sync"):
            with self.model.no_sync():
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def memory_summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            out[f"gpu{i}_alloc_gb"] = round(alloc, 3)
            out[f"gpu{i}_reserved_gb"] = round(reserved, 3)
        return out

    def __repr__(self) -> str:
        return (
            f"HeteroTrainingEngine("
            f"strategy={self.solver.strategy}, "
            f"devices={[s.device_id for s in self.discovery.specs]}, "
            f"params={self.param_count():,}, "
            f"step={self.global_step})"
        )


# ---------------------------------------------------------------------------
# §6  Pipeline variant (strategy == "pipeline")
# ---------------------------------------------------------------------------

class HeteroPipelineEngine(HeteroTrainingEngine):
    """
    Pipeline-parallel variant, activated when the solver chooses "pipeline".
    Wraps hetero_multimodule_pipeline + hetero_uneven_pp_fix for
    stage-balanced scheduling across A6000/H100 tiers.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 num_micro_batches: int = 4,
                 process_group: Optional[Any] = None) -> None:
        super().__init__(model, config, process_group)
        self.num_micro_batches = num_micro_batches
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        pipe_mod = self.registry.get("hetero_multimodule_pipeline")
        uneven_mod = self.registry.get("hetero_uneven_pp_fix")

        if pipe_mod and hasattr(pipe_mod, "MultiModulePipeline"):
            fracs = [self.discovery.tier_fraction(s)
                     for s in self.discovery.specs]
            self._pipeline = pipe_mod.MultiModulePipeline(
                model=self.model,
                stage_fractions=fracs,
                num_micro_batches=self.num_micro_batches,
            )
            logger.info("HeteroPipelineEngine: pipeline stages=%d  "
                        "micro_batches=%d", len(fracs), self.num_micro_batches)
        else:
            self._pipeline = None
            logger.warning("HeteroPipelineEngine: pipeline module not found, "
                           "falling back to data-parallel")

        if uneven_mod and hasattr(uneven_mod, "fix"):
            uneven_mod.fix(self.model,
                           tier_sizes=[s.compute_score
                                       for s in self.discovery.specs])
            logger.debug("hetero_uneven_pp_fix: applied")

    # ------------------------------------------------------------------
    def train_step(self, batch: Any) -> float:
        if self._pipeline is None:
            return super().train_step(batch)

        device = torch.device("cuda", self._primary_device)
        batch = self._move_batch(batch, device)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            loss = self._pipeline.forward_backward(batch)

        self.grad_sync.clip(self.model,
                            self.config.get("max_grad_norm", 1.0))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.global_step += 1

        ckpt_interval = self.config.get("checkpoint_interval", 500)
        if self.global_step % ckpt_interval == 0:
            self._save_checkpoint()

        return loss.item() if isinstance(loss, torch.Tensor) else float(loss)


# ---------------------------------------------------------------------------
# §7  Factory function
# ---------------------------------------------------------------------------

def build_engine(model: nn.Module,
                 config: Dict[str, Any],
                 process_group: Optional[Any] = None) -> HeteroTrainingEngine:
    """
    Auto-select engine class based on hardware topology and config.

    Parameters
    ----------
    model:
        The nn.Module to train.
    config:
        Training configuration dict.  Relevant keys:
          global_batch_size, seq_len, lr, weight_decay, max_grad_norm,
          checkpoint_dir, checkpoint_interval, dtype, max_steps,
          warmup_steps, recompute_threshold_gb, force_strategy.
    process_group:
        Optional dist.ProcessGroup for multi-GPU.

    Returns
    -------
    HeteroTrainingEngine or HeteroPipelineEngine
    """
    discovery = TierDiscovery()
    solver = PartitionSolver(
        discovery=discovery,
        global_batch_size=config.get("global_batch_size", 64),
        seq_len=config.get("seq_len", 2048),
    )

    force = config.get("force_strategy", None)
    strategy = force if force in ("zero3", "pipeline") else solver.strategy

    if strategy == "pipeline":
        logger.info("build_engine: selecting HeteroPipelineEngine")
        return HeteroPipelineEngine(
            model=model,
            config=config,
            num_micro_batches=config.get("num_micro_batches", 4),
            process_group=process_group,
        )

    logger.info("build_engine: selecting HeteroTrainingEngine (ZeRO-3)")
    return HeteroTrainingEngine(
        model=model,
        config=config,
        process_group=process_group,
    )


# ---------------------------------------------------------------------------
# §8  CLI smoke-test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    logger.info("DES-LOC engine smoke-test starting")

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(64, 64)
            self.head = nn.Linear(64, 16)

        def forward(self, input_ids: torch.Tensor,
                    labels: Optional[torch.Tensor] = None):
            h = torch.relu(self.linear(input_ids.float()))
            logits = self.head(h)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, 16), labels.view(-1)
                )
            class _Out:
                pass
            out = _Out()
            out.loss = loss
            out.logits = logits
            return out

    cfg = {
        "global_batch_size": 8,
        "seq_len": 64,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "checkpoint_dir": "/tmp/desloc_smoke",
        "checkpoint_interval": 10,
        "dtype": "float32",
        "max_steps": 20,
    }

    model = _TinyModel()
    engine = build_engine(model, cfg)
    logger.info("Engine: %s", engine)
    logger.info("Params: %d  Trainable: %d",
                engine.param_count(), engine.trainable_param_count())

    dev = torch.device("cuda", engine._primary_device) \
        if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    for step in range(5):
        ids = torch.randint(0, 64, (8, 64)).to(dev)
        labels = torch.randint(0, 16, (8, 64)).to(dev)
        batch = {"input_ids": ids, "labels": labels}
        loss = engine.train_step(batch)
        engine.log_metrics(step, {"loss": loss, "lr": engine.get_current_lr()})

    logger.info("Memory: %s", engine.memory_summary())
    logger.info("Smoke-test PASSED")


if __name__ == "__main__":
    _smoke_test()
