"""
Neuron_SP Project — 7B LLaMA Pretraining Entry Script
Heterogeneous cluster: 2× A6000-48GB + 1× H100-NVL-96GB, no NVLink
Orchestrated by DeepSpeed ZeRO-3 + optional pipeline parallel
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ensure repo root is on the path regardless of CWD
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import deepspeed
from deepspeed.utils import logger as ds_logger

from deepspeed.models.llama_7b import LLaMAConfig, build_llama_7b
from deepspeed.data.pretrain_dataloader import create_pretrain_dataloader, get_tokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Neuron_SP 7B LLaMA pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data ---
    p.add_argument("--data_paths", nargs="+", required=True,
                   help="Glob pattern(s) or explicit file paths (.jsonl/.parquet/.txt)")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--num_workers",  type=int, default=4,
                   help="DataLoader worker processes per rank")

    # --- model ---
    p.add_argument("--vocab_size",         type=int, default=32000)
    p.add_argument("--hidden_size",        type=int, default=4096)
    p.add_argument("--num_hidden_layers",  type=int, default=32)
    p.add_argument("--num_attention_heads",type=int, default=32)
    p.add_argument("--num_kv_heads",       type=int, default=32,
                   help="GQA key/value heads. Set <num_attention_heads for GQA.")
    p.add_argument("--intermediate_size",  type=int, default=11008)
    p.add_argument("--rope_theta",         type=float, default=10000.0)

    # --- optimiser ---
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--min_lr",      type=float, default=3e-5,
                   help="Final LR after cosine decay")
    p.add_argument("--weight_decay",type=float, default=0.1)
    p.add_argument("--beta1",       type=float, default=0.9)
    p.add_argument("--beta2",       type=float, default=0.95)
    p.add_argument("--grad_clip",   type=float, default=1.0)

    # --- schedule ---
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--total_steps",  type=int, default=250_000)

    # --- batch ---
    p.add_argument("--micro_batch",  type=int, default=2,
                   help="Per-GPU micro-batch size")
    p.add_argument("--grad_accum",   type=int, default=8,
                   help="Gradient accumulation steps")

    # --- checkpointing ---
    p.add_argument("--checkpoint_dir",   type=str, default="checkpoints/llama7b")
    p.add_argument("--save_every",       type=int, default=1000,
                   help="Save checkpoint every N steps")
    p.add_argument("--resume_from",      type=str, default=None,
                   help="Path to checkpoint directory to resume from")
    p.add_argument("--keep_last_n",      type=int, default=3,
                   help="Keep only N most recent checkpoints")

    # --- logging ---
    p.add_argument("--log_every",     type=int,  default=10)
    p.add_argument("--wandb_project", type=str,  default=None,
                   help="Weights & Biases project name (optional)")
    p.add_argument("--wandb_run_name",type=str,  default=None)
    p.add_argument("--wandb_tags",    nargs="*", default=[])

    # --- DeepSpeed ---
    p.add_argument("--zero_stage",    type=int, default=3,
                   help="ZeRO optimisation stage (1/2/3)")
    p.add_argument("--bf16",          action="store_true", default=True,
                   help="Use BF16 mixed precision (recommended for H100)")
    p.add_argument("--fp16",          action="store_true", default=False)
    p.add_argument("--offload_param", action="store_true", default=False,
                   help="ZeRO-3: offload parameters to CPU RAM")
    p.add_argument("--offload_optim", action="store_true", default=False,
                   help="ZeRO-3: offload optimiser states to CPU RAM")

    # --- misc ---
    p.add_argument("--seed", type=int, default=42)

    # DeepSpeed launcher injects --local_rank
    p.add_argument("--local_rank", type=int, default=-1)

    return p.parse_args()


# ---------------------------------------------------------------------------
# DeepSpeed config builder
# ---------------------------------------------------------------------------

def build_ds_config(args: argparse.Namespace) -> dict:
    """Construct DeepSpeed config dict programmatically."""

    assert not (args.bf16 and args.fp16), "Cannot use both --bf16 and --fp16"

    zero_cfg: dict = {
        "stage": args.zero_stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    }

    if args.offload_param and args.zero_stage == 3:
        zero_cfg["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
        }

    if args.offload_optim:
        zero_cfg["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    cfg = {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "gradient_clipping": args.grad_clip,
        "steps_per_print": args.log_every,
        "wall_clock_breakdown": False,
        "zero_optimization": zero_cfg,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [args.beta1, args.beta2],
                "eps": 1e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineAnnealing",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "total_num_steps": args.total_steps,
                "decay_rate": args.min_lr / args.lr,
            },
        },
        "bf16": {"enabled": args.bf16},
        "fp16": {"enabled": args.fp16, "loss_scale": 0, "loss_scale_window": 1000},
    }
    return cfg


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str):
    if is_main():
        print(f"[Neuron_SP] {msg}", flush=True)


def log_startup_banner(args: argparse.Namespace, model: torch.nn.Module,
                        num_data_files: int, world_size: int):
    total_params   = sum(p.numel() for p in model.parameters())
    trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff_batch      = args.micro_batch * args.grad_accum * world_size
    tokens_per_step = eff_batch * args.max_seq_len
    total_tokens   = tokens_per_step * args.total_steps

    # Rough time estimate: 6*N flops per token, H100 ≈ 1.98e15 FLOP/s BF16 (SXM)
    flops_per_step = 6 * total_params * tokens_per_step
    # Assume ~30% MFU on heterogeneous cluster
    mfu            = 0.30
    gpu_flops      = 1.98e15 * world_size * mfu
    eta_hours      = (flops_per_step * args.total_steps) / gpu_flops / 3600

    log("=" * 70)
    log("  Neuron_SP — 7B LLaMA Pretraining")
    log("=" * 70)
    log(f"  World size          : {world_size} GPUs")
    log(f"  Total parameters    : {total_params:,}  ({total_params/1e9:.3f} B)")
    log(f"  Trainable params    : {trainable:,}")
    log(f"  Data files found    : {num_data_files}")
    log(f"  Max seq len         : {args.max_seq_len}")
    log(f"  Micro batch / GPU   : {args.micro_batch}")
    log(f"  Grad accum steps    : {args.grad_accum}")
    log(f"  Effective batch     : {eff_batch} sequences  ({eff_batch * args.max_seq_len:,} tokens)")
    log(f"  Total steps         : {args.total_steps:,}")
    log(f"  Total tokens        : {total_tokens / 1e9:.1f} B")
    log(f"  Warmup steps        : {args.warmup_steps}")
    log(f"  Learning rate       : {args.lr}  → min {args.min_lr}")
    log(f"  ZeRO stage          : {args.zero_stage}")
    log(f"  Precision           : {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
    log(f"  Offload params/optim: {args.offload_param}/{args.offload_optim}")
    log(f"  Checkpoint dir      : {args.checkpoint_dir}")
    log(f"  ETA (rough)         : ~{eta_hours:.1f} hours at {mfu*100:.0f}% MFU")
    log("=" * 70)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(engine: deepspeed.DeepSpeedEngine, step: int,
                    checkpoint_dir: str, keep_last_n: int):
    tag = f"step_{step:08d}"
    engine.save_checkpoint(checkpoint_dir, tag=tag)
    log(f"Checkpoint saved: {checkpoint_dir}/{tag}")

    # Prune old checkpoints
    if is_main():
        tags = sorted(
            [d.name for d in Path(checkpoint_dir).iterdir()
             if d.is_dir() and d.name.startswith("step_")],
            reverse=True,
        )
        for old_tag in tags[keep_last_n:]:
            import shutil
            shutil.rmtree(Path(checkpoint_dir) / old_tag, ignore_errors=True)
            log(f"Pruned old checkpoint: {old_tag}")


# ---------------------------------------------------------------------------
# W&B initialisation
# ---------------------------------------------------------------------------

def maybe_init_wandb(args: argparse.Namespace, model: torch.nn.Module):
    if not args.wandb_project or not is_main():
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=vars(args),
            resume="allow",
        )
        wandb.watch(model, log="gradients", log_freq=500)
        log(f"W&B run: {run.url}")
        return run
    except ImportError:
        log("wandb not installed — skipping W&B logging.")
        return None


def log_metrics(run, step: int, metrics: dict):
    if run is None or not is_main():
        return
    run.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)

    # ---- Resolve data files ----
    all_files = []
    for pattern in args.data_paths:
        matched = sorted(glob.glob(pattern, recursive=True))
        all_files.extend(matched if matched else [pattern])
    if not all_files:
        raise FileNotFoundError(f"No data files found for patterns: {args.data_paths}")

    # ---- Build model ----
    cfg = LLaMAConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_len,
        rope_theta=args.rope_theta,
    )
    model = build_llama_7b(cfg)

    # ---- Build tokenizer & dataloader ----
    tokenizer = get_tokenizer()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", max(args.local_rank, 0)))

    dataloader = create_pretrain_dataloader(
        data_paths=all_files,
        max_seq_len=args.max_seq_len,
        micro_batch_size=args.micro_batch,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
        shuffle_files=True,
        seed=args.seed + local_rank,   # different seed per rank
        infinite=True,
    )

    # ---- DeepSpeed initialisation ----
    ds_config = build_ds_config(args)
    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.resume_from:
        _, client_sd = engine.load_checkpoint(args.resume_from)
        start_step = client_sd.get("step", 0) if client_sd else 0
        log(f"Resumed from checkpoint: {args.resume_from}  (step {start_step})")

    # ---- Startup banner ----
    log_startup_banner(args, model, len(all_files), world_size)

    # ---- W&B ----
    wandb_run = maybe_init_wandb(args, model)

    # ---- Ensure checkpoint directory exists ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Training loop ----
    data_iter  = iter(dataloader)
    step       = start_step
    t0         = time.perf_counter()
    loss_accum = 0.0
    tokens_seen = 0

    log(f"Starting training from step {step} …")

    while step < args.total_steps:
        # ---- Fetch batch ----
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(engine.device)
        labels    = input_ids.clone()

        # ---- Forward + backward ----
        loss, _ = engine(input_ids, labels=labels)
        engine.backward(loss)
        engine.step()

        loss_val    = loss.item()
        loss_accum += loss_val
        tokens_seen += input_ids.numel() * world_size
        step        += 1

        # ---- Logging ----
        if step % args.log_every == 0:
            elapsed      = time.perf_counter() - t0
            avg_loss     = loss_accum / args.log_every
            tokens_per_s = (args.log_every * args.micro_batch *
                            args.max_seq_len * world_size) / elapsed
            current_lr   = optimizer.param_groups[0]["lr"] if optimizer else args.lr

            log(
                f"step={step:7d}/{args.total_steps}  "
                f"loss={avg_loss:.4f}  "
                f"lr={current_lr:.2e}  "
                f"tok/s={tokens_per_s:,.0f}  "
                f"tokens={tokens_seen/1e9:.3f}B  "
                f"elapsed={elapsed/3600:.2f}h"
            )

            log_metrics(wandb_run, step, {
                "train/loss":       avg_loss,
                "train/lr":         current_lr,
                "train/tokens_per_s": tokens_per_s,
                "train/tokens_seen_B": tokens_seen / 1e9,
                "train/step":       step,
            })

            loss_accum = 0.0
            t0         = time.perf_counter()

        # ---- Checkpointing ----
        if step % args.save_every == 0:
            save_checkpoint(engine, step, args.checkpoint_dir, args.keep_last_n)

    # ---- Final checkpoint ----
    save_checkpoint(engine, step, args.checkpoint_dir, args.keep_last_n)
    log("Training complete.")

    if wandb_run and is_main():
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
