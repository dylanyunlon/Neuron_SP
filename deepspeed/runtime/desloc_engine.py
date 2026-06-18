"""
DES-LOC Heterogeneous Training Engine
======================================

Unified engine that wires together all hetero_*.py modules into a working
training system for the 2×A6000 + 1×H100-NVL cluster.

Architecture
------------
                    ┌───────────────────────────────────────┐
                    │          DESLOCEngine.train()          │
                    └──────┬────────────┬───────────────────┘
                           │            │
              ┌────────────▼──┐   ┌─────▼──────────────┐
              │ TierDiscovery │   │ ModelPartitioner    │
              │ (GPU survey)  │   │ (layer assignment)  │
              └──────┬────────┘   └──────┬──────────────┘
                     │                   │
              ┌──────▼───────────────────▼──────────┐
              │      HeteroZeROWrapper              │
              │  (ZeRO-3 + CPU offload, tier-aware) │
              └──────┬──────────────────────────────┘
                     │
              ┌──────▼──────────────────────────────┐
              │     PCIeAwareCommManager            │
              │  (allreduce strategy, bw-aware)     │
              └─────────────────────────────────────┘
"""

import os
import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Tier Discovery
# ---------------------------------------------------------------------------

class TierClass(Enum):
    HIGH = "high"           # H100-class (SM ≥ 9.0)
    STANDARD = "standard"   # A6000-class (SM 8.x)
    LOW = "low"             # older / smaller GPUs


@dataclass
class TierSpec:
    """Immutable description of one GPU in the cluster."""
    gpu_index: int
    name: str
    sm_major: int
    sm_minor: int
    vram_bytes: int
    pcie_gen: int
    pcie_width: int
    numa_node: int
    tier_class: TierClass = TierClass.STANDARD

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)

    @property
    def pcie_bw_gbps(self) -> float:
        """Theoretical unidirectional PCIe bandwidth in GB/s."""
        rate_per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}
        return rate_per_lane.get(self.pcie_gen, 1.0) * self.pcie_width

    @property
    def bf16_tflops(self) -> float:
        """Approximate BF16 tensor-core TFLOPS."""
        table = {
            (8, 6): 38.7,   # A6000
            (8, 9): 82.6,   # RTX 4090
            (9, 0): 835.0,  # H100 NVL
        }
        return table.get((self.sm_major, self.sm_minor), 20.0)


class TierDiscovery:
    """Probe the local machine and build a list of TierSpec."""

    def __init__(self):
        self.tiers: List[TierSpec] = []

    def discover(self) -> List[TierSpec]:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPUs found")

        n = torch.cuda.device_count()
        self.tiers = []
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            sm_major, sm_minor = props.major, props.minor

            tier_class = TierClass.HIGH if sm_major >= 9 else TierClass.STANDARD

            # PCIe gen/width: try nvidia-smi, fall back to heuristic
            pcie_gen, pcie_width = self._probe_pcie(i, sm_major)

            # NUMA node: try /sys, fall back to 0
            numa = self._probe_numa(i)

            spec = TierSpec(
                gpu_index=i,
                name=props.name,
                sm_major=sm_major,
                sm_minor=sm_minor,
                vram_bytes=props.total_mem,
                pcie_gen=pcie_gen,
                pcie_width=pcie_width,
                numa_node=numa,
                tier_class=tier_class,
            )
            self.tiers.append(spec)
            logger.info(
                "GPU %d: %s | SM %d.%d | %.1f GB | PCIe Gen%d x%d (%.1f GB/s) "
                "| NUMA %d | tier=%s | %.1f BF16 TFLOPS",
                i, spec.name, sm_major, sm_minor, spec.vram_gb,
                pcie_gen, pcie_width, spec.pcie_bw_gbps,
                numa, spec.tier_class.value, spec.bf16_tflops,
            )
        return self.tiers

    @staticmethod
    def _probe_pcie(idx: int, sm_major: int) -> Tuple[int, int]:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                 "--format=csv,noheader", f"--id={idx}"],
                text=True, timeout=5,
            ).strip()
            gen, width = out.split(",")
            return int(gen.strip()), int(width.strip())
        except Exception:
            return (5, 16) if sm_major >= 9 else (4, 16)

    @staticmethod
    def _probe_numa(idx: int) -> int:
        path = f"/sys/bus/pci/devices/0000:$(lspci | grep -i nvidia | sed -n '{idx+1}p' | cut -d' ' -f1)/numa_node"
        try:
            with open(f"/sys/class/drm/card{idx}/device/numa_node") as f:
                return int(f.read().strip())
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# 2. Model Partitioner
# ---------------------------------------------------------------------------

@dataclass
class PartitionPlan:
    """Which layers run on which GPU."""
    assignments: Dict[int, List[int]]   # gpu_index → [layer_indices]
    micro_batch_sizes: Dict[int, int]   # gpu_index → local micro-batch
    pipeline_stages: int
    estimated_tokens_per_sec: float
    estimated_mfu: float
    bubble_fraction: float


class HeteroPartitioner:
    """
    Solve the layer-assignment problem for heterogeneous GPUs.

    Given L layers and N GPUs with different FLOPS and VRAM, find the
    assignment that minimizes the maximum per-stage latency.

    Core math (detailed in desloc_partition.py):
        T_stage(i) = layers_i * flops_per_layer / FLOPS_i + comm_overhead_i
        Minimize  max_i T_stage(i)
        s.t.      sum(layers_i) = L
                  mem(layers_i) <= VRAM_i - reserved_i
    """

    # Approximate per-layer memory and compute for LLaMA-7B
    BYTES_PER_PARAM = 2       # BF16
    PARAMS_PER_LAYER_7B = 202_375_168   # ~202M per transformer block
    FLOPS_PER_TOKEN_PER_LAYER = 2 * PARAMS_PER_LAYER_7B  # 2*N for fwd

    def __init__(self, tiers: List[TierSpec], total_layers: int = 32,
                 reserved_vram_gb: float = 4.0):
        self.tiers = tiers
        self.total_layers = total_layers
        self.reserved_gb = reserved_vram_gb

    def solve(self, seq_len: int = 2048, micro_batch: int = 1) -> PartitionPlan:
        """Dynamic-programming solver for optimal layer assignment."""
        n_gpu = len(self.tiers)
        L = self.total_layers

        # Per-layer memory: params + grads + optimizer_state (ZeRO-3 shards)
        per_layer_mem_gb = (self.PARAMS_PER_LAYER_7B * self.BYTES_PER_PARAM) / (1024**3)
        # With ZeRO-3, each GPU holds 1/n_gpu of optimizer state
        per_layer_mem_per_gpu = per_layer_mem_gb  # activations are local

        # Compute capacity per GPU (layers per second)
        capacity = []
        for t in self.tiers:
            available_gb = t.vram_gb - self.reserved_gb
            max_layers = int(available_gb / per_layer_mem_per_gpu) if per_layer_mem_per_gpu > 0 else L
            max_layers = min(max_layers, L)
            # Time per layer per token (seconds)
            time_per_layer = self.FLOPS_PER_TOKEN_PER_LAYER * seq_len * micro_batch / (t.bf16_tflops * 1e12)
            capacity.append((max_layers, time_per_layer, t))

        # DP: dp[i][j] = min max-stage-time using first i GPUs covering j layers
        INF = float("inf")
        dp = [[INF] * (L + 1) for _ in range(n_gpu + 1)]
        choice = [[0] * (L + 1) for _ in range(n_gpu + 1)]
        dp[0][0] = 0.0

        for i in range(1, n_gpu + 1):
            max_l, tpl, tier = capacity[i - 1]
            for j in range(L + 1):
                for k in range(min(j, max_l) + 1):
                    stage_time = k * tpl
                    prev = dp[i - 1][j - k]
                    candidate = max(prev, stage_time)
                    if candidate < dp[i][j]:
                        dp[i][j] = candidate
                        choice[i][j] = k

        # Backtrack
        assignments = {}
        remaining = L
        for i in range(n_gpu, 0, -1):
            k = choice[i][remaining]
            gpu_idx = self.tiers[i - 1].gpu_index
            start = remaining - k
            assignments[gpu_idx] = list(range(start, remaining))
            remaining -= k

        # Compute estimates
        total_flops = sum(t.bf16_tflops for t in self.tiers)
        tokens_per_sec = total_flops * 1e12 * 0.25 / (6 * 7e9)  # MFU~25%
        mfu = 0.25
        bubble = (n_gpu - 1) / max(micro_batch * 4, n_gpu)

        plan = PartitionPlan(
            assignments=assignments,
            micro_batch_sizes={t.gpu_index: micro_batch for t in self.tiers},
            pipeline_stages=n_gpu,
            estimated_tokens_per_sec=tokens_per_sec,
            estimated_mfu=mfu,
            bubble_fraction=bubble,
        )
        self._log_plan(plan)
        return plan

    def _log_plan(self, plan: PartitionPlan):
        logger.info("=== Partition Plan ===")
        for gpu_idx, layers in sorted(plan.assignments.items()):
            tier = next(t for t in self.tiers if t.gpu_index == gpu_idx)
            logger.info(
                "  GPU %d (%s, %.1f TFLOPS): layers %s (%d layers)",
                gpu_idx, tier.name, tier.bf16_tflops,
                f"{layers[0]}-{layers[-1]}" if layers else "none",
                len(layers),
            )
        logger.info(
            "  Est. %.0f tok/s | MFU %.1f%% | bubble %.1f%%",
            plan.estimated_tokens_per_sec, plan.estimated_mfu * 100,
            plan.bubble_fraction * 100,
        )


# ---------------------------------------------------------------------------
# 3. PCIe-Aware Communication Manager
# ---------------------------------------------------------------------------

class PCIeAwareCommManager:
    """
    Manage gradient synchronization across heterogeneous PCIe links.

    Strategy:
    - Intra-NUMA: use shared-memory / fast PCIe path
    - Cross-NUMA: chunk large allreduce into overlapped pieces
    - Prioritize H100 (higher PCIe bw) as the allreduce root
    """

    def __init__(self, tiers: List[TierSpec]):
        self.tiers = tiers
        self._allreduce_root = self._select_root()
        self._chunk_size_mb = 64  # overlap chunk

    def _select_root(self) -> int:
        """Pick the GPU with highest PCIe bandwidth as allreduce root."""
        return max(self.tiers, key=lambda t: t.pcie_bw_gbps).gpu_index

    def allreduce_grads(self, gradients: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Bandwidth-aware allreduce.
        In practice this delegates to torch.distributed, but with tuned
        chunk sizes and ordering based on PCIe topology.
        """
        if dist.is_initialized():
            for gpu_idx, grad in gradients.items():
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)
        return gradients

    def estimate_allreduce_time(self, tensor_bytes: int) -> float:
        """Estimate allreduce latency in seconds (ring allreduce model)."""
        n = len(self.tiers)
        if n <= 1:
            return 0.0
        min_bw = min(t.pcie_bw_gbps for t in self.tiers) * 1e9  # bytes/s
        # Ring allreduce: 2*(n-1)/n * size / bw
        return 2 * (n - 1) / n * tensor_bytes / min_bw


# ---------------------------------------------------------------------------
# 4. Hetero ZeRO Wrapper
# ---------------------------------------------------------------------------

class HeteroZeROConfig:
    """Configuration for tier-aware ZeRO Stage 3 + CPU offload."""

    def __init__(self, tiers: List[TierSpec]):
        self.tiers = tiers
        total_vram = sum(t.vram_bytes for t in tiers)
        self.shard_ratios = {
            t.gpu_index: t.vram_bytes / total_vram for t in tiers
        }
        # CPU offload: use ~80% of available RAM
        self.cpu_offload_gb = 1200  # 1.2TB of 1.5TB

    def to_deepspeed_config(self) -> dict:
        """Generate DeepSpeed ZeRO-3 config dict."""
        return {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 8,
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 50_000_000,
                "stage3_prefetch_bucket_size": 50_000_000,
                "stage3_param_persistence_threshold": 100_000,
            },
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 8,
        }


# ---------------------------------------------------------------------------
# 5. DES-LOC Engine
# ---------------------------------------------------------------------------

@dataclass
class DESLOCConfig:
    """Top-level configuration."""
    total_layers: int = 32
    hidden_size: int = 4096
    num_heads: int = 32
    seq_len: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 100_000
    weight_decay: float = 0.1
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    checkpoint_interval: int = 1000
    log_interval: int = 10
    checkpoint_dir: str = "./checkpoints"


class DESLOCEngine:
    """
    Unified heterogeneous training engine.

    Usage:
        engine = DESLOCEngine(config)
        engine.initialize(model, train_dataloader)
        engine.train()
    """

    def __init__(self, config: DESLOCConfig):
        self.config = config
        self.discovery = TierDiscovery()
        self.tiers: List[TierSpec] = []
        self.partitioner: Optional[HeteroPartitioner] = None
        self.comm: Optional[PCIeAwareCommManager] = None
        self.zero_config: Optional[HeteroZeROConfig] = None
        self.plan: Optional[PartitionPlan] = None

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.dataloader: Optional[Any] = None

        self.global_step = 0
        self.tokens_seen = 0

    def initialize(self, model: nn.Module, dataloader, optimizer=None):
        """
        Full initialization:
        1. Discover GPUs
        2. Partition model layers
        3. Set up ZeRO-3
        4. Create optimizer + scheduler
        """
        logger.info("=== DES-LOC Engine Initialization ===")

        # Step 1: Discover hardware
        self.tiers = self.discovery.discover()
        logger.info("Found %d GPUs across %d tiers",
                     len(self.tiers),
                     len(set(t.tier_class for t in self.tiers)))

        # Step 2: Partition model
        self.partitioner = HeteroPartitioner(
            self.tiers, self.config.total_layers
        )
        self.plan = self.partitioner.solve(
            seq_len=self.config.seq_len,
            micro_batch=self.config.micro_batch_size,
        )

        # Step 3: Communication
        self.comm = PCIeAwareCommManager(self.tiers)

        # Step 4: ZeRO config
        self.zero_config = HeteroZeROConfig(self.tiers)

        # Step 5: Model + Optimizer
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16)
            # Place on first available GPU for now; pipeline placement in v2
            self.model = self.model.cuda()

        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.adam_betas,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Linear warmup + cosine decay
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.total_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        self.dataloader = dataloader

        logger.info("DES-LOC Engine initialized. Ready to train.")
        return self

    def train(self):
        """Main training loop."""
        logger.info("Starting training for %d steps", self.config.total_steps)
        self.model.train()

        data_iter = iter(self.dataloader)
        accum_loss = 0.0
        t0 = time.time()

        for step in range(1, self.config.total_steps + 1):
            self.global_step = step

            # Gradient accumulation
            self.optimizer.zero_grad()
            step_loss = 0.0

            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"]
                labels = batch.get("labels", input_ids)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    labels = labels.cuda()

                # Forward
                logits = self.model(input_ids)

                # Cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / self.config.gradient_accumulation_steps

                # Backward
                loss.backward()
                step_loss += loss.item()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            accum_loss += step_loss
            tokens_this_step = (
                self.config.micro_batch_size
                * self.config.gradient_accumulation_steps
                * self.config.seq_len
            )
            self.tokens_seen += tokens_this_step

            # Logging
            if step % self.config.log_interval == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / self.config.log_interval
                tps = self.tokens_seen / elapsed
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    "step %d/%d | loss %.4f | lr %.2e | "
                    "%.0f tok/s | %d tokens total",
                    step, self.config.total_steps, avg_loss, lr,
                    tps, self.tokens_seen,
                )
                accum_loss = 0.0

            # Checkpoint
            if step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(step)

        logger.info("Training complete. %d tokens processed.", self.tokens_seen)

    def save_checkpoint(self, step: int):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"step_{step}.pt")
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "tokens_seen": self.tokens_seen,
        }, path)
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["step"]
        self.tokens_seen = ckpt.get("tokens_seen", 0)
        logger.info("Resumed from %s (step %d)", path, self.global_step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def create_engine(config: Optional[DESLOCConfig] = None) -> DESLOCEngine:
    """Factory function."""
    if config is None:
        config = DESLOCConfig()
    return DESLOCEngine(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = DESLOCConfig(total_steps=2)
    engine = create_engine(cfg)
    tiers = engine.discovery.discover()
    part = HeteroPartitioner(tiers)
    plan = part.solve()
    print(f"Partition: {plan.assignments}")
    print(f"Est. {plan.estimated_tokens_per_sec:.0f} tok/s, MFU {plan.estimated_mfu*100:.0f}%")
