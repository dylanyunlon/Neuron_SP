#!/usr/bin/env python3
"""
===============================================================================
M033: Sync Period Ablation Experiment Script
===============================================================================
NeurIPS 2026 Submission - DES-LOC Benchmark Framework

This module implements the synchronization period ablation study for Figure 3.
Studies the effect of Kx (parameter sync), Ku (first momentum sync), and 
Kv (second momentum sync) on DES-LOC convergence.

Author: Claude (M026-M050)
Date: April 2026
License: MIT

Experiment Configuration:
- Figure 3a: Kx ablation (16, 32, 64, 128)
- Figure 3b: Ku ablation (32, 64, 128, 256)
- Figure 3c: Kv ablation (64, 128, 256, 512)
===============================================================================
"""

__version__ = "1.0.0"
__milestone__ = "M033"

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
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import random
import numpy as np
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.desloc_optimizer import DESLOCOptimizer, DESLOCConfig, BaseOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PART 1: ENUMS AND CONFIGURATION
# =============================================================================

class AblationType(Enum):
    """Type of ablation study."""
    KX = "kx"     # Parameter sync period
    KU = "ku"     # First momentum sync period
    KV = "kv"     # Second momentum sync period


@dataclass
class SyncAblationConfig:
    """Configuration for sync period ablation."""
    ablation_type: AblationType
    base_kx: int = 32
    base_ku: int = 64
    base_kv: int = 128
    
    # Values to sweep
    kx_values: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    ku_values: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kv_values: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Training settings
    total_steps: int = 3000
    batch_size: int = 8
    max_lr: float = 1e-4
    warmup_steps: int = 256
    
    def get_sweep_values(self) -> List[int]:
        """Get values for the current ablation type."""
        if self.ablation_type == AblationType.KX:
            return self.kx_values
        elif self.ablation_type == AblationType.KU:
            return self.ku_values
        elif self.ablation_type == AblationType.KV:
            return self.kv_values
        return []
    
    def get_config_for_value(self, value: int) -> Tuple[int, int, int]:
        """Get (kx, ku, kv) for a specific sweep value."""
        if self.ablation_type == AblationType.KX:
            return (value, self.base_ku, self.base_kv)
        elif self.ablation_type == AblationType.KU:
            return (self.base_kx, value, self.base_kv)
        elif self.ablation_type == AblationType.KV:
            return (self.base_kx, self.base_ku, value)
        return (self.base_kx, self.base_ku, self.base_kv)


# =============================================================================
# PART 2: SIMPLIFIED MODEL
# =============================================================================

class SimplifiedTransformer(nn.Module):
    """
    Simplified transformer for ablation studies.
    Uses ~25M parameters for faster iteration.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        d_ff: int = 1536,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SimplifiedTransformer: {n_params/1e6:.2f}M parameters")
    
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
        
        # Embeddings
        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer
        x = self.transformer(x, mask=self.causal_mask[:T, :T])
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss


# =============================================================================
# PART 3: DATA GENERATION
# =============================================================================

class SyntheticLMDataset(Dataset):
    """Synthetic language modeling dataset."""
    
    def __init__(
        self,
        num_samples: int = 50000,
        seq_len: int = 256,
        vocab_size: int = 10000,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate data with some structure
        torch.manual_seed(seed)
        
        # Create data with n-gram patterns
        self.data = torch.zeros(num_samples, seq_len + 1, dtype=torch.long)
        
        for i in range(num_samples):
            # Start with random tokens
            seq = torch.randint(0, vocab_size, (seq_len + 1,))
            
            # Add some repeated patterns
            pattern_len = random.randint(2, 5)
            pattern_start = random.randint(0, seq_len - pattern_len * 3)
            pattern = seq[pattern_start:pattern_start + pattern_len]
            
            for rep in range(2):
                start = pattern_start + (rep + 1) * pattern_len
                if start + pattern_len <= seq_len + 1:
                    seq[start:start + pattern_len] = pattern
            
            self.data[i] = seq
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return seq[:-1], seq[1:]


# =============================================================================
# PART 4: TRAINING UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr_schedule(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 0.0) -> float:
    """Cosine learning rate with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# PART 5: ABLATION EXPERIMENT RUNNER
# =============================================================================

@dataclass
class SyncAblationResult:
    """Result from a single sync period experiment."""
    ablation_type: str
    ablation_value: int
    kx: int
    ku: int
    kv: int
    steps: List[int]
    losses: List[float]
    final_loss: float
    communication_reduction: float
    psi_factor: float
    runtime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyncAblationExperiment:
    """Runs sync period ablation experiments."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "./outputs",
    ):
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset
        self.train_dataset = SyntheticLMDataset(num_samples=50000, seq_len=256)
        
        logger.info(f"SyncAblation experiment on {self.device}")
    
    def run_single(
        self,
        kx: int,
        ku: int,
        kv: int,
        config: SyncAblationConfig,
        seed: int = 42,
    ) -> SyncAblationResult:
        """Run single experiment with specific K values."""
        set_seed(seed)
        
        # Create model
        model = SimplifiedTransformer(
            vocab_size=10000,
            d_model=384,
            n_heads=6,
            n_layers=6,
        ).to(self.device)
        
        # Create optimizer
        desloc_config = DESLOCConfig(
            base_optimizer=BaseOptimizer.ADAM,
            lr=config.max_lr,
            beta1=0.9,
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
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        train_iter = iter(train_loader)
        
        # Training
        steps = []
        losses = []
        start_time = time.time()
        
        model.train()
        for step in range(config.total_steps):
            # Get batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Learning rate schedule
            lr = cosine_lr_schedule(
                step, config.total_steps, config.warmup_steps,
                config.max_lr, config.max_lr * 0.1
            )
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward
            _, loss = model(x, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Log every 50 steps
            if step % 50 == 0:
                steps.append(step)
                losses.append(loss.item())
                print(f"[Step {step:5d}] Loss: {loss.item():.6f}, LR: {lr:.2e}, "
                      f"Kx={kx}, Ku={ku}, Kv={kv}")
        
        runtime = time.time() - start_time
        
        # Get communication stats
        comm_stats = optimizer.get_communication_stats()
        
        # Determine ablation type and value
        if kx != config.base_kx:
            abl_type = "kx"
            abl_value = kx
        elif ku != config.base_ku:
            abl_type = "ku"
            abl_value = ku
        else:
            abl_type = "kv"
            abl_value = kv
        
        return SyncAblationResult(
            ablation_type=abl_type,
            ablation_value=abl_value,
            kx=kx,
            ku=ku,
            kv=kv,
            steps=steps,
            losses=losses,
            final_loss=losses[-1] if losses else float('inf'),
            communication_reduction=comm_stats['communication_reduction'],
            psi_factor=optimizer.psi_factor,
            runtime_seconds=runtime,
        )
    
    def run_ablation(
        self,
        ablation_type: AblationType,
        config: SyncAblationConfig = None,
    ) -> List[SyncAblationResult]:
        """Run ablation for a specific K parameter."""
        if config is None:
            config = SyncAblationConfig(ablation_type=ablation_type, total_steps=2000)
        else:
            config.ablation_type = ablation_type
        
        sweep_values = config.get_sweep_values()
        results = []
        
        print(f"\n### Figure 3{chr(ord('a') + list(AblationType).index(ablation_type))}: "
              f"{ablation_type.value.upper()} Ablation ###")
        
        for value in sweep_values:
            kx, ku, kv = config.get_config_for_value(value)
            logger.info(f"Running {ablation_type.value}={value} (Kx={kx}, Ku={ku}, Kv={kv})")
            
            result = self.run_single(kx, ku, kv, config)
            results.append(result)
            
            print(f"  {ablation_type.value}={value}: loss={result.final_loss:.6f}, "
                  f"comm_reduction={result.communication_reduction:.1%}")
        
        return results
    
    def run_all_ablations(
        self,
        config: SyncAblationConfig = None,
    ) -> Dict[str, List[SyncAblationResult]]:
        """Run all three ablation studies."""
        if config is None:
            config = SyncAblationConfig(
                ablation_type=AblationType.KX,
                total_steps=2000,
                batch_size=8,
            )
        
        all_results = {}
        
        for abl_type in [AblationType.KX, AblationType.KU, AblationType.KV]:
            results = self.run_ablation(abl_type, config)
            all_results[abl_type.value] = results
        
        return all_results
    
    def save_results(self, all_results: Dict[str, List[SyncAblationResult]]):
        """Save all results to files."""
        output_path = self.output_dir / "sync_ablation_results.json"
        
        data = {
            'experiment': 'sync_period_ablation',
            'timestamp': datetime.now().isoformat(),
            'results': {
                abl_type: [r.to_dict() for r in results]
                for abl_type, results in all_results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
        
        # Save log file
        log_path = self.output_dir / "sync_ablation.log"
        with open(log_path, 'w') as f:
            f.write("### Figure 3: Synchronization Period Ablation ###\n")
            f.write(f"### Timestamp: {datetime.now().isoformat()} ###\n\n")
            
            for abl_type, results in all_results.items():
                f.write(f"=== {abl_type.upper()} Ablation ===\n\n")
                
                for result in results:
                    f.write(f"### {abl_type}={result.ablation_value} "
                           f"(Kx={result.kx}, Ku={result.ku}, Kv={result.kv}) ###\n")
                    f.write(f"ψ-factor: {result.psi_factor:.4f}\n")
                    f.write(f"Communication Reduction: {result.communication_reduction:.1%}\n")
                    
                    for step, loss in zip(result.steps, result.losses):
                        f.write(f"[Step {step:5d}] Loss: {loss:.6f}\n")
                    
                    f.write(f"Final Loss: {result.final_loss:.6f}\n")
                    f.write(f"Runtime: {result.runtime_seconds:.2f}s\n\n")
        
        logger.info(f"Saved log to {log_path}")


# =============================================================================
# PART 6: MAIN ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sync Period Ablation Experiment")
    parser.add_argument("--ablation-type", type=str, choices=["kx", "ku", "kv", "all"],
                       default="all", help="Which ablation to run")
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
    print("Figure 3: Synchronization Period (Kx, Ku, Kv) Ablation")
    print("=" * 70)
    
    config = SyncAblationConfig(
        ablation_type=AblationType.KX,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
    )
    
    experiment = SyncAblationExperiment(output_dir=args.output_dir)
    
    if args.ablation_type == "all":
        all_results = experiment.run_all_ablations(config)
    else:
        abl_type = AblationType(args.ablation_type)
        results = experiment.run_ablation(abl_type, config)
        all_results = {args.ablation_type: results}
    
    experiment.save_results(all_results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for abl_type, results in all_results.items():
        print(f"\n{abl_type.upper()} Ablation:")
        for r in results:
            print(f"  {abl_type}={r.ablation_value}: loss={r.final_loss:.6f}, "
                  f"comm_reduction={r.communication_reduction:.1%}")
    
    print("\n[M033] Sync Period Ablation - COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
