#!/usr/bin/env python3
"""
===============================================================================
M032: Momentum Ablation Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the momentum (β₁) ablation study for Figure 2.
Studies the effect of different momentum coefficients on DES-LOC convergence.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- Model: 125M parameter GPT-2 style
- β₁ values: 0.8, 0.9, 0.95, 0.99
- Fixed: Kx=32, Ku=64, Kv=128
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M032"

import os
import sys
import json
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Iterator
from datetime import datetime
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
# PART 1: MODEL DEFINITION (GPT-2 STYLE)
# =============================================================================

@dataclass
class GPTConfig:
    """Configuration for GPT model."""
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    
    @property
    def num_params(self) -> int:
        """Estimate number of parameters."""
        # Embedding + Layers + LM Head
        emb = self.vocab_size * self.n_embd
        layer = self.n_layer * (4 * self.n_embd ** 2 + 8 * self.n_embd ** 2)
        lm_head = self.n_embd * self.vocab_size
        return emb + layer + lm_head


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"GPT model initialized with {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss


# =============================================================================
# PART 2: SYNTHETIC DATA
# =============================================================================

class SyntheticTextDataset(Dataset):
    """Synthetic dataset for language modeling."""
    
    def __init__(
        self, 
        num_samples: int = 10000,
        block_size: int = 1024,
        vocab_size: int = 50257,
    ):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Pre-generate data
        self.data = torch.randint(0, vocab_size, (num_samples, block_size + 1))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y


# =============================================================================
# PART 3: TRAINING UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 8
    total_steps: int = 5000
    eval_interval: int = 100
    log_interval: int = 10
    warmup_steps: int = 512
    max_lr: float = 1e-4
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 42


def get_lr(step: int, config: TrainingConfig) -> float:
    """Compute learning rate with warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.max_lr * step / config.warmup_steps
    
    decay_ratio = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


# =============================================================================
# PART 4: MOMENTUM ABLATION EXPERIMENT
# =============================================================================

@dataclass
class AblationResult:
    """Results from a single ablation run."""
    beta1: float
    steps: List[int]
    losses: List[float]
    final_loss: float
    psi_factor: float
    runtime_seconds: float
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MomentumAblationExperiment:
    """Runs momentum ablation experiments."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        self.train_dataset = SyntheticTextDataset(num_samples=10000, block_size=256)
        
        logger.info(f"MomentumAblation on {self.device}")
    
    def run_single(
        self,
        beta1: float,
        training_config: TrainingConfig,
        kx: int = 32,
        ku: int = 64,
        kv: int = 128,
    ) -> AblationResult:
        """Run single ablation with specific β₁."""
        set_seed(training_config.seed)
        
        # Create model
        model_config = GPTConfig(
            vocab_size=1000,
            block_size=256,
            n_layer=6,
            n_head=8,
            n_embd=512,
            dropout=0.1,
        )
        model = GPT(model_config).to(self.device)
        
        # Create optimizer with DES-LOC
        desloc_config = DESLOCConfig(
            base_optimizer=BaseOptimizer.ADAM,
            lr=training_config.max_lr,
            beta1=beta1,
            beta2=0.999,
            kx=kx,
            ku=ku,
            kv=kv,
            weight_decay=0.01,
        )
        
        optimizer = DESLOCOptimizer(model.parameters(), desloc_config)
        
        # Data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_iter = iter(train_loader)
        
        # Training loop
        steps = []
        losses = []
        start_time = time.time()
        
        model.train()
        for step in range(training_config.total_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Update learning rate
            lr = get_lr(step, training_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            _, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
            
            # Update
            optimizer.step()
            
            # Logging
            if step % training_config.log_interval == 0:
                steps.append(step)
                losses.append(loss.item())
                print(f"[Step {step:5d}] Loss: {loss.item():.6f}, LR: {lr:.2e}, β₁: {beta1}")
        
        runtime = time.time() - start_time
        
        return AblationResult(
            beta1=beta1,
            steps=steps,
            losses=losses,
            final_loss=losses[-1] if losses else float('inf'),
            psi_factor=optimizer.psi_factor,
            runtime_seconds=runtime,
            config={
                'beta1': beta1,
                'kx': kx,
                'ku': ku,
                'kv': kv,
                'total_steps': training_config.total_steps,
            }
        )
    
    def run_all(
        self,
        beta1_values: List[float] = [0.8, 0.9, 0.95, 0.99],
        training_config: TrainingConfig = None,
    ) -> List[AblationResult]:
        """Run all ablation configurations."""
        if training_config is None:
            training_config = TrainingConfig(total_steps=2000, log_interval=50)
        
        results = []
        
        for beta1 in beta1_values:
            logger.info(f"Running ablation with β₁={beta1}...")
            print(f"\n### Momentum Ablation: β₁ = {beta1} ###")
            
            result = self.run_single(beta1, training_config)
            results.append(result)
            
            print(f"Final loss: {result.final_loss:.6f}, ψ-factor: {result.psi_factor:.4f}")
        
        return results
    
    def save_results(self, results: List[AblationResult]):
        """Save results to file."""
        output_path = self.output_dir / "momentum_ablation_results.json"
        
        data = {
            'experiment': 'momentum_ablation',
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Save log file
        log_path = self.output_dir / "momentum_ablation.log"
        with open(log_path, 'w') as f:
            f.write("### Figure 2: Momentum (β₁) Ablation Study ###\n")
            f.write(f"### Timestamp: {datetime.now().isoformat()} ###\n\n")
            
            for result in results:
                f.write(f"### β₁ = {result.beta1} ###\n")
                f.write(f"ψ-factor: {result.psi_factor:.4f}\n")
                for step, loss in zip(result.steps, result.losses):
                    f.write(f"[Step {step:5d}] Loss: {loss:.6f}\n")
                f.write(f"Final Loss: {result.final_loss:.6f}\n")
                f.write(f"Runtime: {result.runtime_seconds:.2f}s\n\n")
        
        logger.info(f"Saved log to {log_path}")


# =============================================================================
# PART 5: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Momentum Ablation Experiment")
    parser.add_argument("--beta1-values", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("NeurIPS 2026 - DES-LOC Benchmark")
    print("Figure 2: Momentum (β₁) Ablation Study")
    print("=" * 70)
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        log_interval=50,
    )
    
    experiment = MomentumAblationExperiment(output_dir=args.output_dir)
    results = experiment.run_all(
        beta1_values=args.beta1_values,
        training_config=training_config,
    )
    experiment.save_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"β₁={r.beta1:.2f}: final_loss={r.final_loss:.6f}, ψ={r.psi_factor:.4f}")
    
    print("\n[M032] Momentum Ablation - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
