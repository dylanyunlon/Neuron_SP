#!/usr/bin/env python3
"""
DES-LOC 7B Pretraining Script
==============================
Entry point for pretraining LLaMA-7B on a heterogeneous 2×A6000 + 1×H100 cluster.

Usage:
    python deepspeed/scripts/pretrain_7b.py \\
        --data_paths /data/slimpajama/*.jsonl \\
        --total_steps 100000 --lr 3e-4

    # Resume:
    python deepspeed/scripts/pretrain_7b.py \\
        --data_paths /data/*.jsonl --resume_from ./checkpoints/step_5000.pt
"""

import argparse
import glob
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from deepspeed.runtime.desloc_engine import DESLOCEngine, DESLOCConfig
from deepspeed.models.llama_7b import LLaMA7B, LLaMA7BConfig
from deepspeed.data.pretrain_dataloader import create_pretrain_dataloader, SimpleTokenizer

logger = logging.getLogger("pretrain_7b")


def parse_args():
    p = argparse.ArgumentParser(description="DES-LOC 7B Pretraining")
    # Data
    p.add_argument("--data_paths", nargs="+", required=True,
                    help="Training data paths (glob OK). JSONL/Parquet/TXT.")
    p.add_argument("--text_key", default="text")
    p.add_argument("--max_seq_len", type=int, default=2048)
    # Model
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--hidden_size", type=int, default=4096)
    p.add_argument("--num_layers", type=int, default=32)
    p.add_argument("--num_heads", type=int, default=32)
    p.add_argument("--intermediate_size", type=int, default=11008)
    # Training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--total_steps", type=int, default=100000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Checkpoint
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--checkpoint_interval", type=int, default=1000)
    p.add_argument("--resume_from", default=None)
    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run", default=None)
    # Workers
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def expand_glob(patterns):
    """Expand glob patterns, return sorted file list."""
    paths = []
    for pat in patterns:
        expanded = sorted(glob.glob(pat))
        if expanded:
            paths.extend(expanded)
        elif os.path.isfile(pat):
            paths.append(pat)
        else:
            logger.warning("Not found: %s", pat)
    return paths


def estimate_training_time(total_steps, tokens_per_step, tokens_per_sec=5800):
    """Rough estimate based on partition solver prediction."""
    total_tokens = total_steps * tokens_per_step
    hours = total_tokens / tokens_per_sec / 3600
    if hours < 24:
        return f"{hours:.1f} hours"
    return f"{hours/24:.1f} days"


def setup_wandb(args):
    if args.wandb_project is None:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"7b_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )
        logger.info("W&B: %s", run.url)
        return run
    except ImportError:
        logger.warning("wandb not installed, skipping")
        return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Data
    data_paths = expand_glob(args.data_paths)
    if not data_paths:
        logger.error("No data files found. Check --data_paths.")
        sys.exit(1)

    # W&B
    wandb_run = setup_wandb(args)

    # Model
    model_config = LLaMA7BConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
    )

    logger.info("Building LLaMA-7B...")
    model = LLaMA7B(model_config)
    n_params = model.count_parameters()

    # Data pipeline
    tokenizer = SimpleTokenizer()
    dataloader = create_pretrain_dataloader(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # Engine
    engine_config = DESLOCConfig(
        total_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        seq_len=args.max_seq_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        weight_decay=args.weight_decay,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    engine = DESLOCEngine(engine_config)
    engine.initialize(model, dataloader)

    if args.resume_from:
        engine.load_checkpoint(args.resume_from)

    # Print summary
    tokens_per_step = (args.micro_batch_size * args.gradient_accumulation_steps
                       * args.max_seq_len)
    est_time = estimate_training_time(args.total_steps, tokens_per_step)

    logger.info("=" * 60)
    logger.info("DES-LOC 7B Pretraining")
    logger.info("=" * 60)
    logger.info("  Parameters:     %s (%.2fB)", f"{n_params:,}", n_params / 1e9)
    logger.info("  Data files:     %d", len(data_paths))
    logger.info("  Seq length:     %d", args.max_seq_len)
    logger.info("  Micro batch:    %d", args.micro_batch_size)
    logger.info("  Grad accum:     %d", args.gradient_accumulation_steps)
    logger.info("  Eff. batch:     %d tokens/step", tokens_per_step)
    logger.info("  LR:             %.1e (warmup %d)", args.lr, args.warmup_steps)
    logger.info("  Total steps:    %d", args.total_steps)
    logger.info("  Total tokens:   %.1fB", args.total_steps * tokens_per_step / 1e9)
    logger.info("  Est. time:      %s", est_time)
    logger.info("=" * 60)

    engine.train()

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
