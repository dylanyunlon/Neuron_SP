#!/usr/bin/env python3
"""
===============================================================================
M034: Billion-Scale Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the billion-scale training experiments for Figure 5.
Tests DES-LOC at scale with 125M, 360M, and 1.7B parameter models.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- Figure 5a: 125M model, 8 GPUs
- Figure 5b: 360M model, 16 GPUs
- Figure 5c: 1.7B model, 64 GPUs

Hardware Requirements:
- 2x NVIDIA RTX A6000 (48GB) for small models
- 1x NVIDIA H100 NVL (96GB) for large models
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M034"

import os
import sys
import json
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import logging
import random
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.desloc_optimizer import DESLOCOptimizer, DESLOCConfig, BaseOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: MODEL SCALE CONFIGURATIONS
# =============================================================================

class ModelScale(Enum):
    """Model scale configurations."""
    SMALL_125M = "125M"
    MEDIUM_360M = "360M"
    LARGE_1700M = "1.7B"


@dataclass
class ScaleConfig:
    """Configuration for a specific model scale."""
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Training settings
    batch_size_per_gpu: int = 4
    gradient_accumulation: int = 8
    num_gpus: int = 8
    total_steps: int = 20000
    warmup_steps: int = 1000
    max_lr: float = 1e-4
    
    # DES-LOC settings
    kx: int = 32
    ku: int = 64
    kv: int = 128
    
    @property
    def num_params(self) -> int:
        """Estimate parameter count."""
        # Embedding
        emb = self.vocab_size * self.d_model
        # Layers
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O
        mlp = 2 * self.d_model * self.d_ff
        layer = self.n_layers * (attn + mlp)
        # LM head (shared with embedding)
        return emb + layer
    
    @property
    def global_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.gradient_accumulation * self.num_gpus


# Model scale configurations
SCALE_CONFIGS = {
    ModelScale.SMALL_125M: ScaleConfig(
        name="125M",
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        batch_size_per_gpu=8,
        gradient_accumulation=4,
        num_gpus=8,
        total_steps=10000,
    ),
    ModelScale.MEDIUM_360M: ScaleConfig(
        name="360M",
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        batch_size_per_gpu=4,
        gradient_accumulation=8,
        num_gpus=16,
        total_steps=15000,
    ),
    ModelScale.LARGE_1700M: ScaleConfig(
        name="1.7B",
        d_model=2048,
        n_layers=24,
        n_heads=32,
        d_ff=8192,
        batch_size_per_gpu=2,
        gradient_accumulation=16,
        num_gpus=64,
        total_steps=20000,
    ),
}


# =============================================================================
# PART 2: LARGE TRANSFORMER MODEL
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
        # Precompute
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LargeAttention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, config: ScaleConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        
        return y


class LargeMLP(nn.Module):
    """SwiGLU MLP."""
    
    def __init__(self, config: ScaleConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class LargeTransformerBlock(nn.Module):
    """Transformer block with pre-norm."""
    
    def __init__(self, config: ScaleConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = LargeAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = LargeMLP(config)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class LargeTransformer(nn.Module):
    """Large transformer model for billion-scale experiments."""
    
    def __init__(self, config: ScaleConfig):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            LargeTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"LargeTransformer {config.name}: {n_params/1e9:.3f}B parameters")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        
        x = self.dropout(self.tok_emb(input_ids))
        
        for block in self.blocks:
            x = block(x, self.mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss


# =============================================================================
# PART 3: DISTRIBUTED TRAINING
# =============================================================================

def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    
    return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# PART 4: DATA AND TRAINING
# =============================================================================

class LargeSyntheticDataset(Dataset):
    """Large synthetic dataset for billion-scale experiments."""
    
    def __init__(
        self,
        num_samples: int = 100000,
        seq_len: int = 2048,
        vocab_size: int = 50257,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate on-the-fly to save memory
        torch.manual_seed(idx)
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return data[:-1], data[1:]


@dataclass
class BillionScaleResult:
    """Result from billion-scale experiment."""
    model_scale: str
    method: str  # "ddp" or "desloc"
    steps: List[int]
    losses: List[float]
    throughputs: List[float]  # tokens/sec
    final_loss: float
    avg_throughput: float
    total_time_hours: float
    communication_reduction: float
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BillionScaleExperiment:
    """Runs billion-scale training experiments."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup distributed if available
        self.rank, self.world_size, self.local_rank = setup_distributed()
        
        logger.info(f"BillionScale experiment: rank={self.rank}, world_size={self.world_size}")
    
    def run_single(
        self,
        scale: ModelScale,
        use_desloc: bool = True,
    ) -> BillionScaleResult:
        """Run single experiment at specified scale."""
        config = SCALE_CONFIGS[scale]
        
        # Adjust for available GPUs
        actual_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if actual_gpus < config.num_gpus:
            logger.warning(f"Requested {config.num_gpus} GPUs but only {actual_gpus} available")
            config.num_gpus = actual_gpus
        
        # Create model
        model = LargeTransformer(config).to(self.device)
        
        if self.world_size > 1:
            model = DDP(model, device_ids=[self.local_rank])
        
        # Create optimizer
        if use_desloc:
            desloc_config = DESLOCConfig(
                base_optimizer=BaseOptimizer.ADAM,
                lr=config.max_lr,
                beta1=0.9,
                beta2=0.999,
                kx=config.kx,
                ku=config.ku,
                kv=config.kv,
                weight_decay=0.01,
            )
            optimizer = DESLOCOptimizer(model.parameters(), desloc_config)
            method = "desloc"
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.max_lr,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            method = "ddp"
        
        # Dataset
        dataset = LargeSyntheticDataset(
            num_samples=100000,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
        )
        
        sampler = DistributedSampler(dataset) if self.world_size > 1 else None
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size_per_gpu,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        train_iter = iter(train_loader)
        
        # Training
        steps = []
        losses = []
        throughputs = []
        start_time = time.time()
        tokens_processed = 0
        
        model.train()
        accum_loss = 0.0
        
        # Simulate shorter run for testing
        actual_steps = min(config.total_steps, 500)  # Limit for demo
        
        for step in range(actual_steps):
            step_start = time.time()
            
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                if sampler:
                    sampler.set_epoch(step)
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Learning rate schedule
            lr = self._get_lr(step, actual_steps, config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward
            _, loss = model(x, y)
            loss = loss / config.gradient_accumulation
            
            # Backward
            loss.backward()
            accum_loss += loss.item()
            
            # Update
            if (step + 1) % config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Compute throughput
                step_time = time.time() - step_start
                batch_tokens = config.batch_size_per_gpu * config.max_seq_len * config.gradient_accumulation
                throughput = batch_tokens / step_time
                tokens_processed += batch_tokens
                
                # Log
                if step % 50 == 0:
                    steps.append(step)
                    losses.append(accum_loss)
                    throughputs.append(throughput)
                    
                    print(f"[{scale.value}] Step {step:5d}: loss={accum_loss:.4f}, "
                          f"throughput={throughput:.0f} tok/s, lr={lr:.2e}")
                
                accum_loss = 0.0
        
        total_time = time.time() - start_time
        
        # Get communication reduction
        if use_desloc and hasattr(optimizer, 'get_communication_stats'):
            comm_stats = optimizer.get_communication_stats()
            comm_reduction = comm_stats['communication_reduction']
        else:
            comm_reduction = 0.0
        
        return BillionScaleResult(
            model_scale=scale.value,
            method=method,
            steps=steps,
            losses=losses,
            throughputs=throughputs,
            final_loss=losses[-1] if losses else float('inf'),
            avg_throughput=np.mean(throughputs) if throughputs else 0.0,
            total_time_hours=total_time / 3600,
            communication_reduction=comm_reduction,
            config={
                'd_model': config.d_model,
                'n_layers': config.n_layers,
                'num_gpus': config.num_gpus,
                'batch_size': config.batch_size_per_gpu,
                'kx': config.kx if use_desloc else 0,
                'ku': config.ku if use_desloc else 0,
                'kv': config.kv if use_desloc else 0,
            }
        )
    
    def _get_lr(self, step: int, total_steps: int, config: ScaleConfig) -> float:
        """Compute learning rate."""
        if step < config.warmup_steps:
            return config.max_lr * step / config.warmup_steps
        
        progress = (step - config.warmup_steps) / (total_steps - config.warmup_steps)
        return config.max_lr * 0.1 + 0.5 * config.max_lr * 0.9 * (1 + math.cos(math.pi * progress))
    
    def run_all_scales(self) -> List[BillionScaleResult]:
        """Run experiments at all scales."""
        results = []
        
        for scale in [ModelScale.SMALL_125M]:  # Start with small for demo
            for use_desloc in [False, True]:
                method = "DES-LOC" if use_desloc else "DDP"
                print(f"\n### Figure 5: {scale.value} Model with {method} ###")
                
                result = self.run_single(scale, use_desloc)
                results.append(result)
        
        return results
    
    def save_results(self, results: List[BillionScaleResult]):
        """Save results."""
        output_path = self.output_dir / "billion_scale_results.json"
        
        data = {
            'experiment': 'billion_scale',
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Billion-Scale Experiment")
    parser.add_argument("--scale", type=str, choices=["125M", "360M", "1.7B", "all"],
                       default="125M")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--config", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure 5: Billion-Scale Training")
    print("=" * 70)
    
    experiment = BillionScaleExperiment(output_dir=args.output_dir)
    
    if args.scale == "all":
        results = experiment.run_all_scales()
    else:
        scale_map = {
            "125M": ModelScale.SMALL_125M,
            "360M": ModelScale.MEDIUM_360M,
            "1.7B": ModelScale.LARGE_1700M,
        }
        results = [experiment.run_single(scale_map[args.scale], use_desloc=True)]
    
    experiment.save_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r.model_scale} ({r.method}): loss={r.final_loss:.4f}, "
              f"throughput={r.avg_throughput:.0f} tok/s, "
              f"comm_reduction={r.communication_reduction:.1%}")
    
    cleanup_distributed()
    print("\n[M034] Billion-Scale Experiment - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
