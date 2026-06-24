# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
run_pretrain.py — Neuron_SP pretraining entry point

Direct entry point for ags1: python run_pretrain.py

Supports:
  --config      configs/7b_commitpack.yaml   (YAML config file, optional)
  --model-size  7b / 1b / 70m               (default: 7b; override YAML)
  --data-path   /path/to/data               (default: synthetic; override YAML)
  --steps       N                           (default: 100; override YAML)
  --batch-size  N                           (default: 2; override YAML)
  --seq-len     N                           (default: 2048; override YAML)
  --log-every   N                           (default: 10; override YAML)
  --fsdp                                    (wrap model with FSDP instead of DDP)

Config priority: explicit CLI flags > YAML file > built-in defaults.

Import discipline:
  - NO `import deepspeed` (avoids apex/op_builder init chain)
  - Direct submodule imports only: deepspeed.runtime.desloc_engine
  - torch / torch.nn / torch.distributed used directly
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

try:
    import yaml as _yaml  # PyYAML
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# ---------------------------------------------------------------------------
# Logging setup (before any import that might log)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pretrain")

# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def _load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a nested dict.

    Requires PyYAML (``pip install pyyaml``).  Raises RuntimeError when the
    file exists but PyYAML is unavailable so the user gets a clear error
    instead of a silent no-op.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not _HAS_YAML:
        raise RuntimeError(
            f"PyYAML is required to read '{config_path}'.  "
            "Install it with:  pip install pyyaml"
        )
    with config_path.open("r") as fh:
        cfg = _yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    logger.info("Loaded config from %s", config_path)
    return cfg


def _apply_yaml_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """
    Merge YAML config values into *args*, giving CLI flags priority.

    Only the keys that map to existing argparse attributes are applied; unknown
    YAML sections (e.g. ``parallelism``, ``nccl``) are silently ignored here
    but remain accessible via the returned ``args.yaml_cfg`` attribute for
    downstream consumers such as DesLocEngine.

    Mapping (YAML path → argparse attribute):
        model.size           → model_size
        model.vocab_size     → (stored in yaml_cfg only; overrides _MODEL_CONFIGS at runtime)
        training.steps       → steps
        training.micro_batch_size → batch_size
        training.seq_len     / model.seq_len → seq_len
        logging.log_every    → log_every
        data.path            → data_path   (CLI --data-path takes priority)
        desloc.enabled       → use_desloc
    """
    # Attach the raw cfg for optional downstream inspection
    args.yaml_cfg = cfg

    # Helper: read a dotted path from nested dict, return default if missing
    def _get(section: str, key: str, default: Any = None) -> Any:
        return cfg.get(section, {}).get(key, default)

    # model.size / --model-size
    if _get("model", "size") is not None:
        if "--model-size" not in sys.argv and "-model-size" not in sys.argv:
            args.model_size = _get("model", "size")

    # training.steps / --steps
    if _get("training", "steps") is not None:
        if "--steps" not in sys.argv:
            args.steps = int(_get("training", "steps"))

    # training.micro_batch_size / --batch-size
    if _get("training", "micro_batch_size") is not None:
        if "--batch-size" not in sys.argv:
            args.batch_size = int(_get("training", "micro_batch_size"))

    # model.seq_len or training.seq_len / --seq-len
    seq_len_val = _get("model", "seq_len") or _get("training", "seq_len")
    if seq_len_val is not None:
        if "--seq-len" not in sys.argv:
            args.seq_len = int(seq_len_val)

    # logging.log_every / --log-every
    if _get("logging", "log_every") is not None:
        if "--log-every" not in sys.argv:
            args.log_every = int(_get("logging", "log_every"))

    # data.path / --data-path
    if _get("data", "path") is not None:
        if "--data-path" not in sys.argv:
            args.data_path = _get("data", "path")

    # desloc.enabled / --use-desloc
    if _get("desloc", "enabled") is not None:
        if "--use-desloc" not in sys.argv:
            args.use_desloc = bool(_get("desloc", "enabled"))

    return args


# ---------------------------------------------------------------------------
# Targeted import — bypass deepspeed.__init__ entirely
# ---------------------------------------------------------------------------
# We stub the deepspeed package so Python never executes
# deepspeed/__init__.py (which pulls in apex, op_builder, triton, etc.).
import importlib.util
import types as _types

def _stub_deepspeed():
    """Create minimal package stubs so submodule import works."""
    for name in ("deepspeed", "deepspeed.runtime"):
        if name not in sys.modules:
            stub = _types.ModuleType(name)
            stub.__path__ = [name.replace(".", "/")]
            stub.__package__ = name
            sys.modules[name] = stub

_stub_deepspeed()

try:
    _spec = importlib.util.spec_from_file_location(
        "deepspeed.runtime.desloc_engine",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepspeed", "runtime", "desloc_engine.py"),
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["deepspeed.runtime.desloc_engine"] = _mod
    _spec.loader.exec_module(_mod)
    DesLocEngine = _mod.DesLocEngine
    TrainingConfig = _mod.TrainingConfig
    logger.info("DesLocEngine / TrainingConfig imported (apex-free path)")
    _HAS_DESLOC = True
except Exception as _desloc_import_err:
    logger.warning(
        "Could not import DesLocEngine (%s); falling back to standalone training loop.",
        _desloc_import_err,
    )
    _HAS_DESLOC = False

# ---------------------------------------------------------------------------
# Model size presets (Llama-style)
# ---------------------------------------------------------------------------
_MODEL_CONFIGS: Dict[str, Dict] = {
    "70m": dict(hidden_size=512,  num_layers=8,  num_heads=8,  vocab_size=32000),
    "1b":  dict(hidden_size=2048, num_layers=16, num_heads=16, vocab_size=32000),
    "3b":  dict(hidden_size=3200, num_layers=26, num_heads=32, vocab_size=32000),
    "7b":  dict(hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32000),
}

# ---------------------------------------------------------------------------
# Standalone Llama-style model (used when DesLocEngine is unavailable)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """Root-mean-square layer norm, no bias, BF16-friendly."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class _SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network (Llama2 variant)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        intermediate = int(hidden * 8 / 3)
        intermediate = (intermediate + 63) // 64 * 64
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up   = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _CausalAttn(nn.Module):
    """Multi-head causal self-attention (sdpa / FlashAttention path)."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().reshape(B, T, C))


class _TransformerBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(hidden)
        self.attn  = _CausalAttn(hidden, n_heads)
        self.norm2 = _RMSNorm(hidden)
        self.mlp   = _SwiGLUMLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaModel(nn.Module):
    """
    7B / 1B / 70M Llama-style causal LM.

    hidden_size=4096, layers=32, heads=32, vocab=32000 for 7B.
    Weight-ties embedding ↔ lm_head (standard LLaMA practice).
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [_TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.norm    = _RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tie

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embedding(input_ids) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def synthetic_iter(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Infinite synthetic token iterator for smoke-testing without real data."""
    while True:
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        yield tokens[:, :-1].to(device), tokens[:, 1:].to(device)


def real_data_iter(
    data_path: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load a raw binary token file (uint16 / int32 flat array) and yield batches.

    Expected format: flat array of token ids, as produced by Megatron-style
    preprocess_data.py.  Falls back to synthetic data if loading fails.
    """
    import numpy as np  # noqa: PLC0415

    path = Path(data_path)
    if not path.exists():
        logger.warning("data-path '%s' not found; using synthetic data.", data_path)
        yield from synthetic_iter(32000, batch_size, seq_len, device)
        return

    # .npy files: memory-mapped int32 arrays (output of data/prepare_commits.py --task npy)
    if path.suffix == ".npy":
        try:
            tokens = np.load(str(path), mmap_mode="r").astype(np.int64)
            logger.info(
                "Loaded %s as memory-mapped .npy: %d tokens (int32→int64)", path.name, len(tokens)
            )
        except Exception as _npy_exc:
            logger.warning("Failed to mmap '%s' (%s); using synthetic data.", data_path, _npy_exc)
            yield from synthetic_iter(32000, batch_size, seq_len, device)
            return
    else:
        # Try common binary formats
        tokens = None
        for dtype in (np.uint16, np.int32):
            try:
                tokens = np.fromfile(str(path), dtype=dtype)
                logger.info(
                    "Loaded %s: %d tokens (dtype=%s)", path.name, len(tokens), dtype.__name__
                )
                break
            except Exception:
                tokens = None

        if tokens is None or len(tokens) < seq_len + 1:
            logger.warning("Could not read '%s'; using synthetic data.", data_path)
            yield from synthetic_iter(32000, batch_size, seq_len, device)
            return

    chunk = seq_len + 1
    n_chunks = len(tokens) // chunk
    token_t = torch.tensor(tokens[:n_chunks * chunk].astype("int64")).reshape(n_chunks, chunk)
    idx = 0
    while True:
        end = idx + batch_size
        if end > n_chunks:
            idx = 0
            end = batch_size
        batch = token_t[idx:end].to(device)
        yield batch[:, :-1], batch[:, 1:]
        idx = end


# ---------------------------------------------------------------------------
# LR schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def build_cosine_schedule(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def _gpu_mem_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A"
    alloc_gb = torch.cuda.memory_allocated(device) / (1 << 30)
    resv_gb  = torch.cuda.memory_reserved(device)  / (1 << 30)
    return f"{alloc_gb:.2f}/{resv_gb:.2f} GB"


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _init_distributed() -> Tuple[int, int, bool]:
    """
    Initialise torch.distributed when launched via torchrun / torch.multiprocessing.

    Returns (rank, world_size, is_distributed).
    Works transparently for single-process runs as well.
    """
    # torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
    rank       = int(os.environ.get("RANK",       "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    is_dist = world_size > 1

    if is_dist:
        if not dist.is_initialized():
            import datetime as _dt
            dist.init_process_group(
                backend="nccl", init_method="env://",
                timeout=_dt.timedelta(minutes=30),
            )
        torch.cuda.set_device(local_rank)
        logger.info(
            "torch.distributed initialised: rank=%d / world_size=%d  local_rank=%d",
            rank, world_size, local_rank,
        )
    else:
        logger.info("Single-process run (no torch.distributed).")

    return rank, world_size, is_dist


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Standalone training loop (fallback when DesLocEngine unavailable)
# ---------------------------------------------------------------------------

def run_standalone(args: argparse.Namespace) -> None:
    """Standalone PyTorch training loop — supports torchrun multi-GPU via DDP or FSDP."""
    cfg = _MODEL_CONFIGS[args.model_size]

    # ------------------------------------------------------------------ setup
    rank, world_size, is_dist = _init_distributed()
    is_main = (rank == 0)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if is_main:
        logger.info("=== Standalone training (DDP/FSDP-aware, no DesLocEngine) ===")
        logger.info(
            "model-size=%s  device=%s  dtype=%s  world_size=%d  fsdp=%s",
            args.model_size, device, dtype, world_size, getattr(args, "fsdp", False),
        )

    # ------------------------------------------------------------------ model
    model = LlamaModel(
        vocab_size  = cfg["vocab_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        num_heads   = cfg["num_heads"],
        seq_len     = args.seq_len,
    ).to(dtype=dtype, device=device)

    if getattr(args, "gradient_checkpointing", False):
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            if is_main:
                logger.info("Gradient checkpointing enabled")
        else:
            # Fallback: wrap each layer manually
            model._gradient_checkpointing = True
            if is_main:
                logger.info("Gradient checkpointing flag set (manual)")

    if is_main:
        logger.info("Model: %.2fM parameters", model.num_parameters / 1e6)

    # Wrap with FSDP or DDP when running in a distributed context.
    # FSDP is required for heterogeneous GPU clusters (different VRAM per GPU)
    # because DDP requires all ranks to hold a full model replica — impossible
    # when VRAM differs. FSDP shards parameters, gradients, and (with
    # cpu_offload) optimizer states across ranks, letting each GPU contribute
    # its own capacity to the collective pool.
    use_fsdp = is_dist and getattr(args, "fsdp", False)
    if use_fsdp:
        # auto_wrap_policy: shard at _TransformerBlock boundaries so each block
        # is an independent FSDP unit.  This gives fine-grained overlap of
        # all-gather / computation and limits peak memory per unit.
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={_TransformerBlock},
        )
        # FULL_SHARD = ZeRO-3 equivalent: shards params + grads + optimizer states.
        # cpu_offload=True moves optimizer states (and optionally grads) to host
        # DRAM, which is critical when 5 heterogeneous GPUs have mismatched VRAM.
        fsdp_cpu_offload = CPUOffload(offload_params=True)
        # Mixed precision: bf16 for forward/backward, fp32 for reduction.
        fsdp_mp = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=dtype,
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=fsdp_cpu_offload,
            mixed_precision=fsdp_mp if device.type == "cuda" else None,
            auto_wrap_policy=auto_wrap,
            device_id=local_rank if device.type == "cuda" else None,
        )
        if is_main:
            logger.info(
                "Model wrapped with FSDP (FULL_SHARD / ZeRO-3, cpu_offload=True, "
                "auto_wrap=_TransformerBlock).  Heterogeneous-GPU safe."
            )
    elif is_dist:
        from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: PLC0415
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            logger.info("Model wrapped with DistributedDataParallel (DDP).")

    # ---------------------------------------------------------------- optim
    # For FSDP: call model.parameters() directly — FSDP manages the sharded
    # view of params for the optimizer on this rank.
    # For DDP: use model.module to reach the underlying nn.Module.
    # For single-GPU: model IS the raw module.
    raw_model = model.module if (is_dist and not use_fsdp) else model
    optimizer = AdamW(
        model.parameters() if use_fsdp else raw_model.parameters(),
        lr           = 3e-4,
        betas        = (0.9, 0.95),
        eps          = 1e-8,
        weight_decay = 0.1,
    )
    scheduler = build_cosine_schedule(optimizer, warmup_steps=10, total_steps=args.steps)

    # ------------------------------------------------------------------ data
    # Each rank uses a different random seed so data batches differ across GPUs,
    # simulating independent data-parallel shards on synthetic data.
    if getattr(args, "data_mode", "single") == "blend":
        # ---- blend mode: read data.sources from YAML and build blended loader ----
        yaml_cfg = getattr(args, "yaml_cfg", {})
        sources = yaml_cfg.get("data", {}).get("sources", [])
        if not sources:
            raise ValueError(
                "--data-mode blend requires 'data.sources' list in the YAML config, "
                "but none was found. Example:\n"
                "  data:\n"
                "    sources:\n"
                "      - {path: data/stack_v2.bin, weight: 0.5}\n"
                "      - {path: data/commitpack.bin, weight: 0.5}\n"
            )
        if is_main:
            logger.info(
                "blend mode: loading %d source(s) from data.sources", len(sources)
            )
        from data.blend_datasets import build_blended_dataloader  # noqa: PLC0415
        _blend_loader = build_blended_dataloader(
            sources=sources,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        _blend_iter = iter(_blend_loader)

        def _blend_data_iter():
            nonlocal _blend_iter
            while True:
                try:
                    x, y = next(_blend_iter)
                except StopIteration:
                    _blend_iter = iter(_blend_loader)
                    x, y = next(_blend_iter)
                yield x.to(device), y.to(device)

        data = _blend_data_iter()
    elif args.data_path:
        # Use MmapTokenDataset via build_dataloader for memory-mapped int32 token loading
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from data.commit_loader import build_dataloader as _build_dataloader  # noqa: PLC0415
        _mmap_loader = _build_dataloader(
            mode="mmap",
            data_path=args.data_path,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_workers=2,
        )
        if is_main:
            logger.info(
                "mmap data: %s  |  %d samples  |  batch=%d  seq_len=%d",
                args.data_path, len(_mmap_loader.dataset), args.batch_size, args.seq_len,
            )
        _mmap_iter_ref = [iter(_mmap_loader)]

        def _mmap_data_iter():
            while True:
                try:
                    x, y = next(_mmap_iter_ref[0])
                except StopIteration:
                    _mmap_iter_ref[0] = iter(_mmap_loader)
                    x, y = next(_mmap_iter_ref[0])
                yield x.to(device), y.to(device)

        data = _mmap_data_iter()
    else:
        # Offset the RNG per rank so each worker draws different tokens
        torch.manual_seed(42 + rank)
        data = synthetic_iter(cfg["vocab_size"], args.batch_size, args.seq_len, device)

    # ----------------------------------------------------------------- resume
    start_step = 1
    if getattr(args, "resume_from", None) and os.path.isfile(args.resume_from):
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume_from)
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        if not use_fsdp:
            raw_model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        if is_main:
            logger.info("Resumed at step %d (loss=%.4f, tokens_seen=%s)",
                        start_step, ckpt.get("loss", 0), ckpt.get("tokens_seen", "?"))

    # ----------------------------------------------------------------- train
    model.train()
    t0     = time.time()
    losses: list = []

    # ------------------------------------------------- wandb / tensorboard init
    _wb_run = None
    _tb_writer = None
    if is_main:
        if getattr(args, "wandb_project", None) and _HAS_WANDB:
            _wb_run = _wandb.init(
                project=args.wandb_project,
                config=vars(args),
                resume="allow",
            )
            logger.info("W&B run initialised: project=%s  run=%s", args.wandb_project, _wb_run.name)
        elif getattr(args, "wandb_project", None) and not _HAS_WANDB:
            logger.warning("--wandb-project set but wandb is not installed; skipping W&B logging.")

        if getattr(args, "tensorboard_dir", None) and _HAS_TENSORBOARD:
            _tb_writer = _SummaryWriter(log_dir=args.tensorboard_dir)
            logger.info("TensorBoard SummaryWriter initialised: dir=%s", args.tensorboard_dir)
        elif getattr(args, "tensorboard_dir", None) and not _HAS_TENSORBOARD:
            logger.warning(
                "--tensorboard-dir set but torch.utils.tensorboard is not available; "
                "skipping TensorBoard logging."
            )

    if is_main:
        print(
            f"\n{'step':>6}  {'loss':>9}  {'lr':>10}  {'tok/s':>9}  {'GPU mem':>15}"
        )
        print("-" * 60)

    for step in range(start_step, args.steps + 1):
        input_ids, labels = next(data)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().reshape(-1, V),
                labels[:, :T - 1].contiguous().reshape(-1),
            )

        loss.backward()
        # For FSDP: use model.clip_grad_norm_ which handles the sharded reduce
        # correctly across ranks.  For DDP/single-GPU: use the standard utility.
        if use_fsdp:
            model.clip_grad_norm_(1.0)
        else:
            clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Average loss across ranks so rank-0 logs the true global mean
        loss_val = loss.detach()
        if is_dist:
            dist.all_reduce(loss_val, op=dist.ReduceOp.AVG)
        losses.append(loss_val.item())

        # --- Checkpoint save ---
        save_every = getattr(args, "save_every", 500)
        ckpt_dir = getattr(args, "checkpoint_dir", "checkpoints")
        if is_main and save_every > 0 and step % save_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            ckpt = {
                "step": step,
                "model_state_dict": (raw_model.state_dict() if not use_fsdp
                                     else model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "tokens_seen": step * args.batch_size * args.seq_len * world_size,
                "loss": loss_val.item(),
                "args": vars(args),
            }
            torch.save(ckpt, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

        if is_main and step % args.log_every == 0:
            elapsed   = time.time() - t0
            avg_loss  = sum(losses[-args.log_every:]) / args.log_every
            toks_step = args.batch_size * args.seq_len * world_size
            tok_s     = toks_step * args.log_every / max(elapsed, 1e-9)
            cur_lr    = scheduler.get_last_lr()[0]
            mem_str   = _gpu_mem_str(device)
            step_time = elapsed / args.log_every

            # Compute grad norm (after clip_grad_norm_ has already clipped; re-use last value)
            grad_norm = sum(
                p.grad.detach().float().norm() ** 2
                for p in raw_model.parameters()
                if p.grad is not None
            ) ** 0.5

            # GPU memory in GB (allocated)
            gpu_mem_gb = (
                torch.cuda.memory_allocated(device) / (1 << 30)
                if device.type == "cuda" else 0.0
            )

            print(
                f"{step:>6}  {avg_loss:>9.4f}  {cur_lr:>10.2e}  {tok_s:>9.0f}  {mem_str:>15}"
            )

            # ---- W&B logging ----
            if _wb_run is not None:
                _wb_run.log(
                    {
                        "train/loss":      avg_loss,
                        "train/lr":        cur_lr,
                        "train/grad_norm": grad_norm,
                        "train/tok_per_s": tok_s,
                        "train/gpu_mem_gb": gpu_mem_gb,
                        "train/step_time_s": step_time,
                    },
                    step=step,
                )

            # ---- TensorBoard logging ----
            if _tb_writer is not None:
                _tb_writer.add_scalar("train/loss",        avg_loss,    step)
                _tb_writer.add_scalar("train/lr",          cur_lr,      step)
                _tb_writer.add_scalar("train/grad_norm",   grad_norm,   step)
                _tb_writer.add_scalar("train/tok_per_s",   tok_s,       step)
                _tb_writer.add_scalar("train/gpu_mem_gb",  gpu_mem_gb,  step)
                _tb_writer.add_scalar("train/step_time_s", step_time,   step)

            t0 = time.time()

    # Barrier before final stats so all ranks finish together
    if is_dist:
        dist.barrier()

    if is_main:
        final_loss = sum(losses[-min(10, len(losses)):]) / min(10, len(losses))
        initial_10 = sum(losses[:min(10, len(losses))]) / min(10, len(losses))
        loss_drop  = initial_10 - final_loss
        logger.info(
            "Training complete: initial_loss=%.4f  final_loss=%.4f  drop=%.4f (%.1f%%)",
            initial_10, final_loss, loss_drop, 100.0 * loss_drop / max(initial_10, 1e-9),
        )

        if args.steps >= 10:
            assert final_loss < initial_10 + 0.5, (
                f"Loss did not decrease: initial={initial_10:.4f}, final={final_loss:.4f}"
            )
            logger.info("✅  Loss decreased — training loop verified.")

    # ------------------------------------------- cleanup loggers (rank 0 only)
    if is_main:
        if _wb_run is not None:
            _wb_run.finish()
            logger.info("W&B run finished.")
        if _tb_writer is not None:
            _tb_writer.close()
            logger.info("TensorBoard writer closed.")

    _cleanup_distributed()


# ---------------------------------------------------------------------------
# DesLocEngine training path
# ---------------------------------------------------------------------------

def run_desloc(args: argparse.Namespace) -> None:
    """Run training via the DES-LOC heterogeneous engine."""
    # CRITICAL: set_device MUST come before init_process_group(backend="nccl").
    # NCCL binds to the current CUDA device at init time. If all ranks default
    # to cuda:0, NCCL sees "Duplicate GPU" and crashes.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Distributed init (torchrun sets env vars but we must call init ourselves)
    import torch.distributed as _dist
    if not _dist.is_initialized() and "RANK" in os.environ:
        import datetime as _dt
        _dist.init_process_group(
            backend="nccl", init_method="env://",
            timeout=_dt.timedelta(minutes=30),
        )

    cfg = _MODEL_CONFIGS[args.model_size]

    # Compute world size from torchrun env; fallback to 1 for single-GPU
    _world = int(os.environ.get("WORLD_SIZE", 1))
    _grad_accum = getattr(args, "grad_accum_steps", 8)

    tc = TrainingConfig(
        vocab_size      = cfg["vocab_size"],
        hidden_size     = cfg["hidden_size"],
        num_layers      = cfg["num_layers"],
        num_heads       = cfg["num_heads"],
        seq_len         = args.seq_len,
        total_steps     = args.steps,
        micro_batch_size= args.batch_size,
        global_batch_size= args.batch_size * _world * _grad_accum,
        grad_accum_steps= _grad_accum,
        warmup_steps    = min(2000, args.steps // 10),
        log_every       = args.log_every,
        save_every      = getattr(args, "save_every", 1000),
        eval_every      = 0,
        wandb_project   = getattr(args, "wandb_desloc_project", None),
        tensorboard_dir = getattr(args, "tensorboard_desloc_dir", None),
    )

    # Build the LlamaModel and pass it in (DesLocEngine wraps it)
    # Model stays on CPU here — DesLocEngine/FSDP handles device placement.
    # Moving to GPU first then FSDP flatten causes OOM on A6000 (47GB).
    dtype = torch.bfloat16
    model = LlamaModel(
        vocab_size  = cfg["vocab_size"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        num_heads   = cfg["num_heads"],
        seq_len     = args.seq_len,
    ).to(dtype=dtype)

    if getattr(args, "gradient_checkpointing", False) and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    logger.info(
        "=== DesLocEngine path ===  model-size=%s  params=%.2fM  device=%s",
        args.model_size, model.num_parameters / 1e6, device,
    )

    data_iter = None
    if getattr(args, "data_mode", "single") == "blend":
        yaml_cfg = getattr(args, "yaml_cfg", {})
        sources = yaml_cfg.get("data", {}).get("sources", [])
        if not sources:
            raise ValueError(
                "--data-mode blend requires 'data.sources' list in the YAML config."
            )
        logger.info(
            "blend mode (desloc): loading %d source(s) from data.sources", len(sources)
        )
        from data.blend_datasets import build_blended_dataloader  # noqa: PLC0415
        _blend_loader = build_blended_dataloader(
            sources=sources,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        _blend_iter_ref = [iter(_blend_loader)]

        def _blend_desloc_iter():
            while True:
                try:
                    x, y = next(_blend_iter_ref[0])
                except StopIteration:
                    _blend_iter_ref[0] = iter(_blend_loader)
                    x, y = next(_blend_iter_ref[0])
                yield x.to(device), y.to(device)

        data_iter = _blend_desloc_iter()
    elif args.data_path:
        data_iter = real_data_iter(args.data_path, args.batch_size, args.seq_len, device)

    engine = DesLocEngine(config=tc, model=model, data_iter=data_iter)
    engine.train()

    logger.info("DesLocEngine training complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neuron_SP pretraining entry point (ags1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help="Path to a YAML config file (e.g. configs/7b_commitpack.yaml). "
             "Values in the file act as defaults; explicit CLI flags override them.",
    )
    p.add_argument(
        "--model-size",
        choices=list(_MODEL_CONFIGS.keys()),
        default="7b",
        help="Model size preset: 7b (hidden=4096, layers=32, heads=32), "
             "1b (hidden=2048, layers=16, heads=16), "
             "70m (hidden=512, layers=8, heads=8).",
    )
    p.add_argument(
        "--data-path",
        default=None,
        metavar="PATH",
        help="Path to a flat binary token file of int32 token ids (e.g. produced "
             "by Megatron-style preprocess_data.py).  When specified, uses "
             "numpy.memmap (MmapTokenDataset) for zero-copy memory-mapped loading "
             "instead of synthetic torch.randint data.  If omitted, synthetic "
             "random tokens are used.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Micro-batch size (sequences per step).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log interval in steps.",
    )
    p.add_argument(
        "--use-desloc",
        action="store_true",
        default=False,
        help="Use DesLocEngine instead of standalone loop (requires full cluster).",
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (required for 7B on 48GB GPUs).",
    )
    p.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help=(
            "Wrap model with FullyShardedDataParallel (FSDP) instead of DDP. "
            "Use for heterogeneous-GPU clusters where GPUs have different VRAM. "
            "Applies FULL_SHARD (ZeRO-3 equivalent) with CPU offload for "
            "optimizer states (requires ample host DRAM, e.g. 1.5 TB)."
        ),
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        metavar="PROJECT",
        help="Weights & Biases project name. If set, metrics are logged to W&B "
             "(requires `pip install wandb`). Disabled when omitted.",
    )
    p.add_argument(
        "--tensorboard-dir",
        default=None,
        metavar="DIR",
        help="Directory for TensorBoard SummaryWriter logs. If set, metrics are "
             "written to this directory (requires `pip install tensorboard`). "
             "Disabled when omitted.",
    )
    p.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from.",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (0 to disable).",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints.",
    )
    p.add_argument(
        "--wandb",
        default=None,
        metavar="PROJECT",
        dest="wandb_desloc_project",
        help=(
            "W&B project name for DesLocEngine training metrics. "
            "If set, logs train/loss, lr, grad_norm, throughput, MFU, GPU mem to W&B. "
            "Distinct from --wandb-project which controls the standalone loop."
        ),
    )
    p.add_argument(
        "--tensorboard",
        default=None,
        metavar="DIR",
        dest="tensorboard_desloc_dir",
        help=(
            "Directory for DesLocEngine TensorBoard logs. "
            "If set, writes train/* scalars every log_every steps (rank 0 only)."
        ),
    )
    p.add_argument(
        "--data-mode",
        choices=["single", "blend"],
        default="single",
        help=(
            "Data loading mode. 'single' (default) reads a flat binary token file "
            "via --data-path. 'blend' reads a list of weighted sources from the "
            "YAML config's data.sources field and calls "
            "data.blend_datasets.build_blended_dataloader()."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # YAML config: load file first, then let explicit CLI flags override
    # ------------------------------------------------------------------
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        args = _apply_yaml_config(args, yaml_cfg)
    else:
        args.yaml_cfg = {}

    # Only rank 0 prints the startup banner (avoids duplicate output under torchrun)
    _rank  = int(os.environ.get("RANK",       "0"))
    _world = int(os.environ.get("WORLD_SIZE", "1"))
    if _rank == 0:
        logger.info("=" * 60)
        logger.info("Neuron_SP run_pretrain.py")
        logger.info("  config     : %s", args.config or "(none — CLI only)")
        logger.info("  model-size : %s  (%s)", args.model_size, _MODEL_CONFIGS[args.model_size])
        logger.info("  steps      : %d", args.steps)
        logger.info("  batch-size : %d", args.batch_size)
        logger.info("  seq-len    : %d", args.seq_len)
        logger.info("  data-path  : %s", args.data_path or "(synthetic)")
        logger.info("  data-mode  : %s", getattr(args, "data_mode", "single"))
        logger.info("  use-desloc : %s", args.use_desloc)
        logger.info("  fsdp       : %s", getattr(args, "fsdp", False))
        logger.info("  world-size : %d", _world)
        logger.info("  wandb      : %s", args.wandb_project or "(disabled)")
        logger.info("  tensorboard: %s", args.tensorboard_dir or "(disabled)")
        logger.info("  torch      : %s", torch.__version__)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(
                    "  GPU %d: %s  %.0f GB  SM%d.%d",
                    i, props.name, props.total_memory / (1 << 30),
                    props.major, props.minor,
                )
        else:
            logger.info("  No CUDA GPUs found — running on CPU")
        logger.info("=" * 60)

    if args.use_desloc and _HAS_DESLOC:
        run_desloc(args)
    else:
        if args.use_desloc and not _HAS_DESLOC:
            logger.warning(
                "--use-desloc requested but DesLocEngine unavailable; "
                "using standalone loop."
            )
        run_standalone(args)


if __name__ == "__main__":
    main()
