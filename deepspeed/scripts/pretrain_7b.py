#!/usr/bin/env python3
"""
DES-LOC 7B Pretraining Script
==============================

Entry point for pretraining LLaMA-7B on a heterogeneous 2×A6000 + 1×H100 cluster.

Usage:
    python deepspeed/scripts/pretrain_7b.py \
        --data_paths /data/slimpajama/*.jsonl \
        --total_steps 100000 \
        --lr 3e-4 \
        --checkpoint_dir ./checkpoints/7b_run1

    # Resume from checkpoint:
    python deepspeed/scripts/pretrain_7b.py \
        --data_paths /data/slimpajama/*.jsonl \
        --resume_from ./checkpoints/7b_run1/step_5000.pt
"""

import argparse
import logging
import os
import sys
import glob
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deepspeed.runtime.desloc_engine import DESLOCEngine, DESLOCConfig
from deepspeed.models.llama_7b import LLaMA7B, LLaMA7BConfig
from deepspeed.data.pretrain_dataloader import (
    create_pretrain_dataloader,
    SimpleTokenizer,
)

logger = logging.getLogger("pretrain_7b")


def parse_args():
    p = argparse.ArgumentParser(description="DES-LOC 7B Pretraining")

    # Data
    p.add_argument("--data_paths", nargs="+", required=True,
                    help="Paths to training data (jsonl/parquet/txt). Glob patterns OK.")
    p.add_argument("--text_key", default="text", help="JSON key for text field")
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
    p.add_argument("--resume_from", default=None, help="Path to checkpoint .pt file")

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--wandb_project", default=None, help="W&B project name")
    p.add_argument("--wandb_run", default=None, help="W&B run name")

    # Workers
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


def expand_glob_paths(patterns):
    """Expand glob patterns in data paths."""
    paths = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(expanded)
        elif os.path.exists(pattern):
            paths.append(pattern)
        else:
            logger.warning("Path not found: %s", pattern)
    return paths


def setup_wandb(args):
    """Initialize Weights & Biases if requested."""
    if args.wandb_project is None:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"7b_{time.strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
        )
        logger.info("W&B initialized: %s", run.url)
        return run
    except ImportError:
        logger.warning("wandb not installed; skipping")
        return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # Expand data paths
    data_paths = expand_glob_paths(args.data_paths)
    if not data_paths:
        logger.error("No training data found. Check --data_paths.")
        sys.exit(1)
    logger.info("Training data: %d files", len(data_paths))

    # W&B
    wandb_run = setup_wandb(args)

    # Model config
    model_config = LLaMA7BConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
    )

    # Build model
    logger.info("Building LLaMA-7B model...")
    model = LLaMA7B(model_config)
    n_params = model.count_parameters()
    logger.info("Model: %s parameters (%.2fB)", f"{n_params:,}", n_params / 1e9)

    # Build dataloader
    tokenizer = SimpleTokenizer()
    dataloader = create_pretrain_dataloader(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
    )

    # Engine config
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

    # Create and initialize engine
    engine = DESLOCEngine(engine_config)
    engine.initialize(model, dataloader)

    # Resume if requested
    if args.resume_from:
        engine.load_checkpoint(args.resume_from)

    # Train
    logger.info("=" * 60)
    logger.info("Starting 7B pretraining")
    logger.info("  Data files: %d", len(data_paths))
    logger.info("  Parameters: %.2fB", n_params / 1e9)
    logger.info("  Seq length: %d", args.max_seq_len)
    logger.info("  Micro batch: %d", args.micro_batch_size)
    logger.info("  Grad accum: %d", args.gradient_accumulation_steps)
    logger.info("  Effective batch: %d tokens",
                args.micro_batch_size * args.gradient_accumulation_steps * args.max_seq_len)
    logger.info("  LR: %.1e (warmup %d steps)", args.lr, args.warmup_steps)
    logger.info("  Total steps: %d", args.total_steps)
    logger.info("=" * 60)

    engine.train()

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("Done.")


if __name__ == "__main__":
    main()
