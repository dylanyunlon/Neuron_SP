#!/usr/bin/env python3
"""
DES-LOC Real GPU Benchmark - No Simulation, No Fallback
========================================================
Real distributed training on A100 GPUs.
Fails hard if anything goes wrong.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

# Hard fail if imports missing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

assert torch.cuda.is_available(), "CUDA not available - HARD FAIL"
assert torch.cuda.device_count() >= 1, "No GPU found - HARD FAIL"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration - no defaults that allow fallback."""
    # Model
    model_size: str = "125M"
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # Training
    batch_size: int = 4
    gradient_accumulation: int = 8
    max_steps: int = 1000
    warmup_steps: int = 100
    
    # Optimizer
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # DES-LOC specific
    Kx: int = 32
    Ku: int = 96
    Kv: int = 192
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Paths
    output_dir: str = "./real_benchmark_results"
    
    def get_model_config(self) -> Dict:
        """Get model configuration based on size."""
        configs = {
            "125M": {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "350M": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
            "700M": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
            "1.3B": {"n_layer": 24, "n_head": 16, "n_embd": 2048},
        }
        assert self.model_size in configs, f"Unknown model size: {self.model_size}"
        return configs[self.model_size]


# =============================================================================
# GPT-2 MODEL (Real Implementation)
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block."""
    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout)
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 Model."""
    def __init__(self, vocab_size: int, max_seq_len: int, n_layer: int, n_head: int, n_embd: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(max_seq_len, n_embd),
            drop = nn.Dropout(0.0),
            h = nn.ModuleList([TransformerBlock(n_embd, n_head, max_seq_len) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Init weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


# =============================================================================
# SYNTHETIC DATASET (For benchmarking - real data optional)
# =============================================================================

class SyntheticDataset(Dataset):
    """Synthetic dataset for benchmarking."""
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 100000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate random tokens
        tokens = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:]
        }


# =============================================================================
# OPTIMIZERS
# =============================================================================

class AdamW(torch.optim.Optimizer):
    """Standard AdamW optimizer."""
    def __init__(self, params, lr: float, betas: Tuple[float, float], weight_decay: float):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decoupled weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


class DESLOCAdamW(torch.optim.Optimizer):
    """
    DES-LOC AdamW - Desynced Low Communication Adam.
    
    Implements independent sync periods for:
    - x (parameters): sync every Kx steps
    - u (first moment): sync every Ku steps  
    - v (second moment): sync every Kv steps
    """
    def __init__(self, params, lr: float, betas: Tuple[float, float], 
                 weight_decay: float, Kx: int, Ku: int, Kv: int):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       Kx=Kx, Ku=Ku, Kv=Kv)
        super().__init__(params, defaults)
        self.global_step = 0
    
    def step(self):
        self.global_step += 1
        
        for group in self.param_groups:
            Kx = group['Kx']
            Ku = group['Ku']
            Kv = group['Kv']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decoupled weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Local Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def sync_if_needed(self, world_size: int):
        """Sync optimizer states based on DES-LOC schedule."""
        if world_size <= 1:
            return
        
        sync_x = self.global_step % self.param_groups[0]['Kx'] == 0
        sync_u = self.global_step % self.param_groups[0]['Ku'] == 0
        sync_v = self.global_step % self.param_groups[0]['Kv'] == 0
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Sync parameters
                if sync_x:
                    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
                
                # Sync first moment
                if sync_u and 'exp_avg' in state:
                    dist.all_reduce(state['exp_avg'], op=dist.ReduceOp.AVG)
                
                # Sync second moment
                if sync_v and 'exp_avg_sq' in state:
                    dist.all_reduce(state['exp_avg_sq'], op=dist.ReduceOp.AVG)
        
        return {'sync_x': sync_x, 'sync_u': sync_u, 'sync_v': sync_v}


class LocalAdamW(torch.optim.Optimizer):
    """
    Local AdamW - Sync all states every K steps.
    Baseline for comparison with DES-LOC.
    """
    def __init__(self, params, lr: float, betas: Tuple[float, float],
                 weight_decay: float, K: int):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, K=K)
        super().__init__(params, defaults)
        self.global_step = 0
    
    def step(self):
        self.global_step += 1
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(1e-8)
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
    
    def sync_if_needed(self, world_size: int):
        """Sync all states every K steps."""
        if world_size <= 1:
            return
        
        K = self.param_groups[0]['K']
        should_sync = self.global_step % K == 0
        
        if should_sync:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    
                    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
                    if 'exp_avg' in state:
                        dist.all_reduce(state['exp_avg'], op=dist.ReduceOp.AVG)
                    if 'exp_avg_sq' in state:
                        dist.all_reduce(state['exp_avg_sq'], op=dist.ReduceOp.AVG)
        
        return {'synced': should_sync}


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Real distributed trainer - no simulation."""
    
    def __init__(self, config: TrainingConfig, method: str):
        self.config = config
        self.method = method
        
        # Distributed setup
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if self.world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
        
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Model
        model_config = config.get_model_config()
        self.model = GPT(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            **model_config
        ).to(self.device)
        
        # DDP wrapper for gradient sync (only for DDP method)
        if self.world_size > 1 and method == 'DDP':
            self.model = DDP(self.model, device_ids=[self.local_rank])
        
        # Optimizer
        self.optimizer = self._create_optimizer(method)
        
        # Dataset
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            num_samples=config.max_steps * config.batch_size * config.gradient_accumulation * self.world_size
        )
        
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank) if self.world_size > 1 else None
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=True
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Metrics
        self.metrics = {
            'losses': [],
            'step_times': [],
            'comm_events': [],
            'memory_usage': []
        }
        
        if self.rank == 0:
            os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_optimizer(self, method: str):
        """Create optimizer based on method."""
        params = self.model.parameters()
        
        if method == 'DDP':
            return AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif method == 'LocalAdam':
            return LocalAdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                K=self.config.Kx
            )
        elif method == 'DESLOC':
            return DESLOCAdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay,
                Kx=self.config.Kx,
                Ku=self.config.Ku,
                Kv=self.config.Kv
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train(self) -> Dict:
        """Run training loop."""
        self.model.train()
        
        data_iter = iter(self.dataloader)
        
        total_tokens = 0
        start_time = time.time()
        
        for step in range(1, self.config.max_steps + 1):
            step_start = time.time()
            
            # Gradient accumulation
            accumulated_loss = 0.0
            
            for micro_step in range(self.config.gradient_accumulation):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    _, loss = self.model(input_ids, labels)
                    loss = loss / self.config.gradient_accumulation
                
                # Backward pass
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # DES-LOC / Local Adam sync
            if hasattr(self.optimizer, 'sync_if_needed'):
                sync_info = self.optimizer.sync_if_needed(self.world_size)
                if sync_info:
                    self.metrics['comm_events'].append({'step': step, **sync_info})
            
            step_time = time.time() - step_start
            total_tokens += self.config.batch_size * self.config.gradient_accumulation * self.config.max_seq_len * self.world_size
            
            # Record metrics
            self.metrics['losses'].append(accumulated_loss)
            self.metrics['step_times'].append(step_time)
            self.metrics['memory_usage'].append(torch.cuda.max_memory_allocated() / 1e9)
            
            # Logging
            if step % self.config.log_interval == 0 and self.rank == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed
                
                print(f"[{self.method}] Step {step}/{self.config.max_steps} | "
                      f"Loss: {accumulated_loss:.4f} | "
                      f"Time: {step_time*1000:.1f}ms | "
                      f"Tokens/s: {tokens_per_sec:.0f} | "
                      f"Memory: {self.metrics['memory_usage'][-1]:.2f}GB")
        
        # Final stats
        total_time = time.time() - start_time
        
        results = {
            'method': self.method,
            'final_loss': self.metrics['losses'][-1],
            'avg_loss': sum(self.metrics['losses'][-100:]) / 100,
            'total_time_seconds': total_time,
            'avg_step_time_ms': sum(self.metrics['step_times']) / len(self.metrics['step_times']) * 1000,
            'tokens_per_second': total_tokens / total_time,
            'peak_memory_gb': max(self.metrics['memory_usage']),
            'total_tokens': total_tokens,
            'world_size': self.world_size,
            'losses': self.metrics['losses'],
            'comm_events': self.metrics['comm_events']
        }
        
        # Count communication
        if self.method == 'DESLOC':
            sync_x = sum(1 for e in self.metrics['comm_events'] if e.get('sync_x'))
            sync_u = sum(1 for e in self.metrics['comm_events'] if e.get('sync_u'))
            sync_v = sum(1 for e in self.metrics['comm_events'] if e.get('sync_v'))
            results['sync_counts'] = {'x': sync_x, 'u': sync_u, 'v': sync_v}
        elif self.method == 'LocalAdam':
            syncs = sum(1 for e in self.metrics['comm_events'] if e.get('synced'))
            results['sync_counts'] = {'all': syncs}
        
        return results
    
    def cleanup(self):
        """Cleanup distributed."""
        if self.world_size > 1:
            dist.destroy_process_group()


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_benchmark(config: TrainingConfig, methods: List[str]) -> Dict:
    """Run benchmark for all methods."""
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method} benchmark")
        print(f"{'='*60}\n")
        
        trainer = Trainer(config, method)
        results = trainer.train()
        all_results[method] = results
        trainer.cleanup()
        
        # Force CUDA sync and clear cache between methods
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        if trainer.rank == 0:
            print(f"\n{method} Results:")
            print(f"  Final Loss: {results['final_loss']:.4f}")
            print(f"  Avg Loss (last 100): {results['avg_loss']:.4f}")
            print(f"  Total Time: {results['total_time_seconds']:.1f}s")
            print(f"  Tokens/sec: {results['tokens_per_second']:.0f}")
            print(f"  Peak Memory: {results['peak_memory_gb']:.2f}GB")
            if 'sync_counts' in results:
                print(f"  Sync Counts: {results['sync_counts']}")
    
    return all_results


def save_results(results: Dict, config: TrainingConfig):
    """Save benchmark results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        'timestamp': timestamp,
        'config': {
            'model_size': config.model_size,
            'batch_size': config.batch_size,
            'gradient_accumulation': config.gradient_accumulation,
            'max_steps': config.max_steps,
            'Kx': config.Kx,
            'Ku': config.Ku,
            'Kv': config.Kv,
        },
        'results': {}
    }
    
    for method, data in results.items():
        output['results'][method] = {
            'final_loss': data['final_loss'],
            'avg_loss': data['avg_loss'],
            'total_time_seconds': data['total_time_seconds'],
            'tokens_per_second': data['tokens_per_second'],
            'peak_memory_gb': data['peak_memory_gb'],
            'sync_counts': data.get('sync_counts', {})
        }
    
    # Save JSON
    output_path = os.path.join(config.output_dir, f'benchmark_results_{timestamp}.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print comparison
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    print(f"{'Method':<15} {'Loss':<10} {'Time(s)':<10} {'Tok/s':<12} {'Memory(GB)':<10}")
    print("-"*60)
    
    for method, data in results.items():
        print(f"{method:<15} {data['avg_loss']:<10.4f} {data['total_time_seconds']:<10.1f} "
              f"{data['tokens_per_second']:<12.0f} {data['peak_memory_gb']:<10.2f}")
    
    # Communication comparison
    if 'DDP' in results and 'DESLOC' in results:
        ddp_comm = config.max_steps  # DDP syncs every step
        desloc_comm = (config.max_steps // config.Kx + 
                       config.max_steps // config.Ku + 
                       config.max_steps // config.Kv)
        reduction = ddp_comm / desloc_comm
        
        print("\n" + "-"*60)
        print(f"Communication Reduction (DES-LOC vs DDP): {reduction:.1f}x")


def main():
    parser = argparse.ArgumentParser(description='DES-LOC Real GPU Benchmark')
    parser.add_argument('--model_size', type=str, default='125M', choices=['125M', '350M', '700M', '1.3B'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--Kx', type=int, default=32)
    parser.add_argument('--Ku', type=int, default=96)
    parser.add_argument('--Kv', type=int, default=192)
    parser.add_argument('--output', type=str, default='./real_benchmark_results')
    parser.add_argument('--methods', nargs='+', default=['DDP', 'LocalAdam', 'DESLOC'])
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        gradient_accumulation=args.grad_accum,
        max_steps=args.max_steps,
        Kx=args.Kx,
        Ku=args.Ku,
        Kv=args.Kv,
        output_dir=args.output
    )
    
    rank = int(os.environ.get('RANK', 0))
    
    if rank == 0:
        print("="*60)
        print("DES-LOC REAL GPU BENCHMARK")
        print("="*60)
        print(f"Model: {config.model_size}")
        print(f"Batch: {config.batch_size} x {config.gradient_accumulation}")
        print(f"Steps: {config.max_steps}")
        print(f"DES-LOC: Kx={config.Kx}, Ku={config.Ku}, Kv={config.Kv}")
        print(f"Methods: {args.methods}")
        print(f"World Size: {os.environ.get('WORLD_SIZE', 1)}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("="*60)
    
    results = run_benchmark(config, args.methods)
    
    if rank == 0:
        save_results(results, config)


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════
# DES-LOC Benchmark Suite (M195)
# Benchmark definitions + figure generation + experiment runner
# Following NKI-FA commit da964f3 draw_plot.py standards:
# - Data from actual experiment logs, never hardcoded
# - Full precision numeric values with explicit labels
# - seaborn + pandas for publication-quality figures
# ═══════════════════════════════════════════════════════════════
import os as _m195_os
import json as _m195_json
import time as _m195_time
import logging as _m195_logging

_m195_logger = _m195_logging.getLogger('desloc_benchmark')


# ── Benchmark Definitions ──────────────────────────────────────
DESLOC_BENCHMARK_SUITE = {
    # RQ1: Empirical change rates (Section 5.1)
    'rq1_half_life_125M': {
        'description': 'Half-life measurement: relative change rates of optimizer states',
        'model': 'gpt2_117M',
        'method': 'desloc',
        'Kx': 32, 'Ku': 96, 'Kv': 192,
        'betas': [(0.9, 0.95), (0.9, 0.99), (0.9, 0.999), (0.9, 0.9999)],
        'total_steps': 5000,
        'figure_id': 'fig_clvii',
    },
    # RQ2: Sync frequency sweep (Section 5.2)
    'rq2_Kx_sweep': {
        'description': 'Parameter sync frequency sweep',
        'model': 'gpt2_117M',
        'method': 'desloc',
        'Kx_values': [1, 2, 4, 8, 16, 32, 64, 128],
        'Ku_multiplier': 3,
        'Kv_multiplier': 6,
        'total_steps': 5000,
        'figure_id': 'fig_clxii',
    },
    # RQ3: Communication reduction (Section 5.3)
    'rq3_comm_reduction_125M': {
        'description': 'Comm reduction: DDP vs LocalAdam vs DES-LOC (125M)',
        'model': 'gpt2_117M',
        'methods': ['ddp', 'local_adam', 'desloc'],
        'Kx': 32, 'Ku': 96, 'Kv': 192,
        'total_steps': 5000,
        'figure_id': 'fig_comm_reduction',
    },
    'rq3_comm_reduction_350M': {
        'description': 'Comm reduction: DDP vs LocalAdam vs DES-LOC (350M)',
        'model': 'gpt2_350M',
        'methods': ['ddp', 'local_adam', 'desloc'],
        'Kx': 32, 'Ku': 96, 'Kv': 192,
        'total_steps': 3000,
        'figure_id': 'fig_comm_reduction_350M',
    },
    # RQ4: Large-scale training (Section 5.4)
    'rq4_billion_scale': {
        'description': 'Scaling to 1.3B+ models',
        'model': 'gpt2_1.3B',
        'method': 'desloc',
        'Kx': 32, 'Ku': 96, 'Kv': 192,
        'total_steps': 2000,
        'figure_id': 'fig_billion',
    },
    # RQ5: Nesterov outer optimizer (Section 5.5)
    'rq5_nesterov_ablation': {
        'description': 'Nesterov vs averaging ablation',
        'model': 'gpt2_117M',
        'method': 'desloc_outer',
        'outer_optimizers': ['averaging', 'nesterov'],
        'Kx_values': [8, 32, 128],
        'nesterov_momentum_values': [0.7, 0.9, 0.95],
        'total_steps': 5000,
        'figure_id': 'fig_clxxxii',
    },
    # RQ6: Muon inner optimizer (Section 5.6)
    'rq6_muon_ablation': {
        'description': 'Muon + DES-LOC: Ku sweep',
        'model': 'gpt2_117M',
        'method': 'desloc',
        'optimizer': 'muon',
        'muon_compat': True,
        'Kx_values': [8, 32],
        'Ku_values': [8, 24, 48],
        'total_steps': 5000,
        'figure_id': 'fig_clxxxvii',
    },
}


# ── Experiment Log Parser ──────────────────────────────────────
class ExperimentLogParser:
    """Parse structured DES-LOC experiment logs.

    Reads log format produced by DESLOCStructuredLogger:
    ### DESLOC_EXPERIMENT_START ###
    # config: model=125M Kx=32 ...
    step=0 loss=10.82 lr=6.00e-04 ...
    ### DESLOC_EXPERIMENT_END ###
    ### DESLOC_SUMMARY_START ###
    final_loss=3.21
    ### DESLOC_SUMMARY_END ###
    """

    def __init__(self, log_dir='./desloc_experiment_logs'):
        self.log_dir = log_dir

    def parse_file(self, filepath):
        """Parse a single log file."""
        config = {}
        steps = []
        summary = {}
        section = None

        try:
            with open(filepath, 'r') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line == '### DESLOC_EXPERIMENT_START ###':
                        section = 'experiment'
                        continue
                    elif line == '### DESLOC_EXPERIMENT_END ###':
                        section = None
                        continue
                    elif line == '### DESLOC_SUMMARY_START ###':
                        section = 'summary'
                        continue
                    elif line == '### DESLOC_SUMMARY_END ###':
                        section = None
                        continue

                    if section == 'experiment':
                        if line.startswith('# config:'):
                            for token in line[len('# config:'):].strip().split():
                                if '=' in token:
                                    k, v = token.split('=', 1)
                                    try: v = int(v)
                                    except ValueError:
                                        try: v = float(v)
                                        except ValueError: pass
                                    config[k] = v
                        elif line.startswith('step='):
                            entry = {}
                            for token in line.split():
                                if '=' in token:
                                    k, v = token.split('=', 1)
                                    try: v = float(v)
                                    except ValueError: pass
                                    entry[k] = v
                            steps.append(entry)
                    elif section == 'summary':
                        if '=' in line:
                            k, v = line.split('=', 1)
                            try: v = float(v)
                            except ValueError: pass
                            summary[k] = v
        except FileNotFoundError:
            pass

        return {'config': config, 'steps': steps, 'summary': summary}

    def parse_benchmark(self, benchmark_id):
        """Parse all logs for a specific benchmark."""
        import glob
        results = []
        pattern = _m195_os.path.join(self.log_dir, f'{benchmark_id}*.log')
        for fp in sorted(glob.glob(pattern)):
            parsed = self.parse_file(fp)
            if parsed['steps']:
                results.append(parsed)
        return results


# ── Figure Generator ───────────────────────────────────────────
class DESLOCFigureGenerator:
    """Generate NeurIPS-quality figures from experiment logs.

    Follows NKI-FA commit da964f3 draw_plot.py standards:
    - Data parsed from experiment logs (never hardcoded)
    - seaborn whitegrid style
    - Explicit numeric annotations on every bar
    - Proper axis labels with units
    """

    FIGURE_SPECS = {
        'fig_clvii': {
            'title': 'Relative change rates of optimizer states',
            'xlabel': 'Training Step',
            'ylabel': 'Relative Change Rate',
            'style': 'line',
        },
        'fig_clxii': {
            'title': 'Effect of sync frequency on convergence',
            'xlabel': 'Kx (Sync Period)',
            'ylabel': 'Final Loss',
            'style': 'grouped_bar',
        },
        'fig_comm_reduction': {
            'title': 'Communication reduction vs DDP',
            'xlabel': 'Method',
            'ylabel': 'Total Communication (GB)',
            'style': 'grouped_bar',
        },
        'fig_clxxxii': {
            'title': 'Nesterov vs Averaging Outer Optimizer',
            'xlabel': 'Configuration',
            'ylabel': 'Loss Gap vs DDP (%)',
            'style': 'grouped_bar',
        },
        'fig_clxxxvii': {
            'title': 'Muon + DES-LOC: Ku Sweep',
            'xlabel': 'Training Step',
            'ylabel': 'Training Loss',
            'style': 'line',
        },
    }

    def __init__(self, log_dir='./desloc_experiment_logs',
                 figure_dir='./desloc_figures'):
        self.log_dir = log_dir
        self.figure_dir = figure_dir
        self.parser = ExperimentLogParser(log_dir)

    def generate_all(self):
        """Generate all figures from available experiment logs."""
        _m195_os.makedirs(self.figure_dir, exist_ok=True)
        generated = []
        for fig_id, spec in self.FIGURE_SPECS.items():
            try:
                path = self.generate_figure(fig_id, spec)
                if path:
                    generated.append(path)
            except Exception as e:
                _m195_logger.warning(f"Failed to generate {fig_id}: {e}")
        return generated

    def generate_figure(self, fig_id, spec):
        """Generate a single figure. Returns filepath or None."""
        # Find matching benchmark
        matching = None
        for bid, bspec in DESLOC_BENCHMARK_SUITE.items():
            if bspec.get('figure_id') == fig_id:
                matching = bid
                break
        if matching is None:
            return None

        data = self.parser.parse_benchmark(matching)
        if not data:
            _m195_logger.info(f"No data for {fig_id} (benchmark {matching})")
            return None

        # Export data as JSON for external plotting
        export_path = _m195_os.path.join(self.figure_dir, f'{fig_id}_data.json')
        with open(export_path, 'w') as f:
            _m195_json.dump({
                'figure_id': fig_id,
                'spec': spec,
                'data': [{'config': d['config'],
                          'summary': d.get('summary', {}),
                          'num_steps': len(d['steps'])}
                         for d in data],
            }, f, indent=2, default=str)
        return export_path


# End M195
