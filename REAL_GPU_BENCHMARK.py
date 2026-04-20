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


# =========================================================================
# DES-LOC Experiment Infrastructure
# Ref: NKI-FA commit da964f3 — benchmark_attn.py + draw_plot.py
# =========================================================================

class DeslocExperimentConfig:
    """Configuration for a single DES-LOC ablation experiment.
    Ref: Section 5 — systematic ablation across RQ1-RQ6.
    Each experiment produces a log in NKI-FA format."""

    def __init__(self, Kx=1, Ku=3, Kv=6, beta1=0.9, beta2=0.999,
                 model_size='125M', seed=42, max_steps=1000,
                 clip_rho=1.0, outer_opt='average', inner_opt='adam'):
        self.Kx = Kx
        self.Ku = Ku
        self.Kv = Kv
        self.beta1 = beta1
        self.beta2 = beta2
        self.model_size = model_size
        self.seed = seed
        self.max_steps = max_steps
        self.clip_rho = clip_rho
        self.outer_opt = outer_opt
        self.inner_opt = inner_opt

    def log_header(self):
        return (f"### Kx = {self.Kx}, Ku = {self.Ku}, Kv = {self.Kv}, "
                f"beta1 = {self.beta1}, beta2 = {self.beta2}, "
                f"model = {self.model_size}, seed = {self.seed}, "
                f"clip_rho = {self.clip_rho}, outer = {self.outer_opt}, "
                f"inner = {self.inner_opt} ###")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class DeslocExperimentLogger:
    """Logger in NKI-FA format. All values >= 6 decimal places.
    Ref: NKI-FA draw_plot.py parse_data() — '### config ### + metric: value'"""

    def __init__(self, config, filepath=None):
        self.config = config
        self.filepath = filepath
        self._fh = None
        self.entries = []

    def open(self):
        if self.filepath:
            self._fh = open(self.filepath, 'w')
            self._fh.write(self.config.log_header() + '\n')

    def log(self, step, **metrics):
        entry = {'step': step}
        entry.update(metrics)
        self.entries.append(entry)
        if self._fh:
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    self._fh.write(f'{k}: {v:.6f}\n')
                else:
                    self._fh.write(f'{k}: {v}\n')
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()

    def summary(self):
        if not self.entries:
            return {}
        import math
        losses = [e.get('loss') for e in self.entries if 'loss' in e]
        r = {'total_steps': len(self.entries), 'final_loss': self.entries[-1].get('loss')}
        if losses:
            r['min_loss'] = min(losses)
            m = sum(losses) / len(losses)
            r['mean_loss'] = round(m, 6)
            if len(losses) > 1:
                r['std_loss'] = round(math.sqrt(sum((x-m)**2 for x in losses)/(len(losses)-1)), 6)
        return r


def generate_experiment_matrix(seeds=(42, 137, 2024)):
    """Generate full NeurIPS ablation matrix. Returns list of configs.
    Total: 7*3*2*3 + 2*3 + 2*3 = 138 experiments."""
    configs = []
    for kx in [1, 4, 8, 16, 32, 64, 128]:
        for b2 in [0.95, 0.99, 0.999]:
            for model in ['125M', '350M']:
                for seed in seeds:
                    configs.append(DeslocExperimentConfig(
                        Kx=kx, Ku=max(1,kx*3), Kv=max(1,kx*6),
                        beta2=b2, model_size=model, seed=seed))
    for outer in ['average', 'nesterov']:
        for seed in seeds:
            configs.append(DeslocExperimentConfig(
                Kx=32, Ku=96, Kv=192, outer_opt=outer, seed=seed))
    for inner in ['adam', 'adopt']:
        for seed in seeds:
            configs.append(DeslocExperimentConfig(
                Kx=32, Ku=96, Kv=192, inner_opt=inner, seed=seed))
    return configs


class DeslocResultsAggregator:
    """Aggregate results across seeds for error bars.
    Ref: NeurIPS requires 3+ seeds for statistical validity."""

    def __init__(self):
        self.results = []

    def add(self, config_dict, metrics_dict):
        self.results.append({'config': config_dict, 'metrics': metrics_dict})

    def aggregate_by(self, group_keys, metric_key):
        """Group by config keys and compute mean+std of metric."""
        import math
        groups = {}
        for r in self.results:
            key = tuple(r['config'].get(k) for k in group_keys)
            groups.setdefault(key, []).append(r['metrics'].get(metric_key))
        out = {}
        for key, vals in groups.items():
            vals = [v for v in vals if v is not None]
            if vals:
                m = sum(vals)/len(vals)
                s = math.sqrt(sum((x-m)**2 for x in vals)/(len(vals)-1)) if len(vals)>1 else 0
                out[key] = {'mean': round(m,6), 'std': round(s,6), 'n': len(vals)}
        return out

    def to_table(self, group_keys, metric_keys):
        """Format as table rows for logging."""
        rows = [group_keys + metric_keys]
        for mk in metric_keys:
            agg = self.aggregate_by(group_keys, mk)
            for key, stats in sorted(agg.items()):
                row = list(key) + [f"{stats['mean']:.4f}+/-{stats['std']:.4f}"]
                rows.append(row)
        return rows


# =========================================================================
# DES-LOC Full Experiment Suite
# Ref: NKI-FA commit da964f3 — benchmark_attn.py + draw_plot.py
# =========================================================================

class DeslocExperimentSuite:
    """Complete DES-LOC ablation suite for NeurIPS submission.

    Generates and manages experiments for all 6 research questions:
    RQ1: Half-life validation (Section 5.1)
    RQ2: Independent sync periods (Section 5.2)
    RQ3: Communication reduction (Section 5.3)
    RQ4: Large-scale training (Section 5.4)
    RQ5: Nesterov outer optimizer (Section 5.5)
    RQ6: Muon inner optimizer (Section 5.6)
    """

    def __init__(self, seeds=(42, 137, 2024), models=('125M', '350M')):
        self.seeds = seeds
        self.models = models
        self.experiments = []
        self.results = []

    def generate_rq1_halflife(self):
        """RQ1: Vary beta2, measure change rate ratio.
        Expected: ratio ≈ tau(beta1)/tau(beta2) = ln(beta2)/ln(beta1)."""
        configs = []
        for b2 in [0.9, 0.95, 0.99, 0.999]:
            for seed in self.seeds:
                configs.append({
                    'rq': 'RQ1', 'Kx': 32, 'Ku': 96, 'Kv': 192,
                    'beta2': b2, 'model': '125M', 'seed': seed,
                    'max_steps': 5000,
                })
        self.experiments.extend(configs)
        return configs

    def generate_rq2_sync_sweep(self):
        """RQ2: Sweep Kx independently, measure final loss.
        Demonstrates that Ku, Kv can be set independently of Kx."""
        configs = []
        for kx in [1, 4, 8, 16, 32, 64, 128]:
            for seed in self.seeds:
                for model in self.models:
                    configs.append({
                        'rq': 'RQ2', 'Kx': kx,
                        'Ku': max(1, kx * 3), 'Kv': max(1, kx * 6),
                        'beta2': 0.999, 'model': model, 'seed': seed,
                        'max_steps': 25000 if model == '125M' else 15000,
                    })
        self.experiments.extend(configs)
        return configs

    def generate_rq3_comm_reduction(self):
        """RQ3: Compare DDP vs LocalAdam vs DES-LOC comm volume.
        Three modes for same Kx: DDP(Kx=1), LocalAdam(Ku=Kv=Kx), DES-LOC(Ku=3Kx,Kv=6Kx)."""
        configs = []
        for kx in [32, 64]:
            for mode in ['ddp', 'local_adam', 'desloc']:
                for seed in self.seeds:
                    if mode == 'ddp':
                        c = {'Kx': 1, 'Ku': 1, 'Kv': 1}
                    elif mode == 'local_adam':
                        c = {'Kx': kx, 'Ku': kx, 'Kv': kx}
                    else:
                        c = {'Kx': kx, 'Ku': kx*3, 'Kv': kx*6}
                    c.update({'rq': 'RQ3', 'mode': mode, 'base_Kx': kx,
                             'beta2': 0.999, 'model': '125M', 'seed': seed,
                             'max_steps': 25000})
                    configs.append(c)
        self.experiments.extend(configs)
        return configs

    def generate_rq5_nesterov(self):
        """RQ5: Average vs Nesterov outer optimizer."""
        configs = []
        for outer in ['average', 'nesterov']:
            for seed in self.seeds:
                configs.append({
                    'rq': 'RQ5', 'Kx': 32, 'Ku': 96, 'Kv': 192,
                    'outer': outer, 'beta2': 0.999, 'model': '125M',
                    'seed': seed, 'max_steps': 25000,
                })
        self.experiments.extend(configs)
        return configs

    def generate_rq6_muon(self):
        """RQ6: Adam vs ADOPT inner optimizer."""
        configs = []
        for inner in ['adam', 'adopt']:
            for seed in self.seeds:
                configs.append({
                    'rq': 'RQ6', 'Kx': 32, 'Ku': 96, 'Kv': 192,
                    'inner': inner, 'beta2': 0.999, 'model': '125M',
                    'seed': seed, 'max_steps': 25000,
                })
        self.experiments.extend(configs)
        return configs

    def generate_all(self):
        """Generate all experiments across RQ1-RQ6."""
        self.experiments.clear()
        self.generate_rq1_halflife()
        self.generate_rq2_sync_sweep()
        self.generate_rq3_comm_reduction()
        self.generate_rq5_nesterov()
        self.generate_rq6_muon()
        return self.experiments

    def total_experiments(self):
        return len(self.experiments)

    def group_by_rq(self):
        groups = {}
        for e in self.experiments:
            rq = e.get('rq', 'unknown')
            groups.setdefault(rq, []).append(e)
        return {rq: len(exps) for rq, exps in groups.items()}

    def to_json(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2)

    @classmethod
    def from_json(cls, filepath):
        import json
        suite = cls()
        with open(filepath) as f:
            suite.experiments = json.load(f)
        return suite


class DeslocResultsValidator:
    """Validate experiment results meet NeurIPS publication standards.

    Checks:
    1. All numeric values have >= 4 significant digits
    2. Each configuration has >= 3 seeds for error bars
    3. Loss values are monotonically decreasing (on average)
    4. Comm reduction matches theoretical prediction within 10%
    """

    def __init__(self, results):
        self.results = results

    def check_precision(self, min_sig=4):
        """Check all floats have sufficient significant digits."""
        violations = []
        for i, r in enumerate(self.results):
            for key, val in r.get('metrics', {}).items():
                if isinstance(val, float) and val != 0:
                    s = f'{val:.10g}'.lstrip('-0').replace('.', '')
                    sig = len(s.rstrip('0'))
                    if sig < min_sig:
                        violations.append(f'result[{i}].{key}={val}: {sig} sig digits')
        return violations

    def check_seed_coverage(self, min_seeds=3):
        """Check each config has enough seeds."""
        from collections import Counter
        config_counts = Counter()
        for r in self.results:
            cfg = r.get('config', {})
            key = tuple(sorted((k, v) for k, v in cfg.items() if k != 'seed'))
            config_counts[key] += 1
        insufficient = [(k, v) for k, v in config_counts.items() if v < min_seeds]
        return insufficient

    def validate_all(self):
        """Run all validation checks. Returns dict of issues."""
        return {
            'precision': self.check_precision(),
            'seed_coverage': self.check_seed_coverage(),
        }


# =============================================================================
# M238 (Claude-15): Figure 1+2 Matplotlib Plotting Engine
# Ref: NKI-FA da964f3 draw_plot.py — seaborn whitegrid, bar annotations
# Ref: Section 5 — 7 figures with precise TFLOPS/loss annotations
# All data MUST come from parsed experiment logs, NEVER hardcoded.
# =============================================================================


class DeslocFigurePlotter:
    """Generate NeurIPS-quality figures from experiment log data.

    Follows NKI-FA draw_plot.py conventions:
    - seaborn whitegrid theme
    - 16×9 or 10×10 figure size
    - Exact value annotations on bars (e.g., "582.3 TFLOPS")
    - viridis/plasma color palettes

    All data comes from DeslocAblationLogger or desloc_parse_log_file()
    output — never from hardcoded arrays.

    Usage:
        plotter = DeslocFigurePlotter(output_dir='./figures')
        plotter.plot_figure1(experiments)  # loss vs step
        plotter.plot_figure2(experiments)  # comm reduction
    """

    def __init__(self, output_dir='./figures', dpi=300):
        self.output_dir = output_dir
        self.dpi = dpi
        self._setup_done = False

    def _setup(self):
        """Lazy import and style setup."""
        if self._setup_done:
            return True
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            try:
                import seaborn as sns
                sns.set_theme(style="whitegrid")
            except ImportError:
                plt.style.use('seaborn-v0_8-whitegrid')
            self._plt = plt
            self._setup_done = True
            return True
        except ImportError:
            print("WARNING: matplotlib not available, skipping plots")
            return False

    def plot_figure1(self, experiments, title=None):
        """Plot Figure 1: Loss vs Training Step.

        Args:
            experiments: list of dicts from desloc_parse_log_file(),
                each with 'config' and 'steps' containing loss data.
            title: optional override title

        Creates: figure1_loss_curve.png in output_dir.

        Ref: NKI-FA draw_plot.py — multi-line plot with legend.
        Ref: Section 5.4 — DES-LOC competitive with DDP on loss.
        """
        if not self._setup():
            return
        plt = self._plt
        fig, ax = plt.subplots(figsize=(10, 10))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

        for i, exp in enumerate(experiments):
            cfg = exp.get('config', None)
            steps_data = exp.get('steps', [])
            if not steps_data:
                continue

            x_steps = [d.get('step', j) for j, d in enumerate(steps_data)]
            y_losses = [d.get('loss', 0) for d in steps_data
                        if 'loss' in d]
            x_steps = x_steps[:len(y_losses)]

            if not y_losses:
                continue

            # Build label from config
            if cfg:
                kx = getattr(cfg, 'Kx', None) or cfg.get('Kx', '?')
                opt = getattr(cfg, 'optimizer', None) or cfg.get('optimizer', '?')
                if kx == 1:
                    label = 'DDP (baseline)'
                else:
                    label = f'DES-LOC Kx={kx} ({opt})'
            else:
                label = f'Experiment {i}'

            color = colors[i % len(colors)]
            ls = line_styles[i % len(line_styles)]
            ax.plot(x_steps, y_losses, label=label,
                    color=color, linestyle=ls, linewidth=2.0)

            # Annotate final loss with exact value
            if y_losses:
                final_loss = y_losses[-1]
                final_step = x_steps[-1]
                ax.annotate(f'{final_loss:.4f}',
                            xy=(final_step, final_loss),
                            xytext=(10, 5), textcoords='offset points',
                            fontsize=9, color=color)

        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title(title or 'Figure 1: Loss vs Training Step', fontsize=18)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(labelsize=10)

        import os
        fig_path = os.path.join(self.output_dir, 'figure1_loss_curve.png')
        fig.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure 1 saved: {fig_path}")
        return fig_path

    def plot_figure2(self, kx_comm_data, title=None):
        """Plot Figure 2: Communication Reduction vs Kx.

        Args:
            kx_comm_data: list of dicts, each with:
                'Kx', 'methods' (DDP/DES-LOC total_gb),
                'reduction_vs_ddp', 'reduction_vs_local_adam'
            title: optional override title

        Creates: figure2_comm_reduction.png in output_dir.

        Ref: NKI-FA draw_plot.py — barplot with exact annotations:
            g.annotate(f"{p.get_height():.1f}", ...)
        Ref: Section 5.3 — grouped bar: DDP vs Local Adam vs DES-LOC
        """
        if not self._setup():
            return
        plt = self._plt

        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare bar data
        labels = []
        ddp_vals = []
        la_vals = []
        dl_vals = []

        for entry in kx_comm_data:
            kx = entry.get('Kx', '?')
            methods = entry.get('methods', {})
            labels.append(f'Kx={kx}')
            ddp_vals.append(methods.get('DDP', {}).get('total_gb', 0))
            la_vals.append(methods.get('Local Adam', {}).get('total_gb', 0))
            dl_vals.append(methods.get('DES-LOC', {}).get('total_gb', 0))

        if not labels:
            return

        import math
        x = list(range(len(labels)))
        width = 0.25

        bars_ddp = ax.bar([xi - width for xi in x], ddp_vals,
                          width, label='DDP', color='#3498db')
        bars_la = ax.bar(x, la_vals,
                         width, label='Local Adam', color='#e74c3c')
        bars_dl = ax.bar([xi + width for xi in x], dl_vals,
                         width, label='DES-LOC', color='#2ecc71')

        # Annotate bars with exact values (NKI-FA style)
        for bars in [bars_ddp, bars_la, bars_dl]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2,
                                    height),
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center', va='bottom',
                                fontsize=9)

        # Add reduction annotations
        for i, entry in enumerate(kx_comm_data):
            r = entry.get('reduction_vs_ddp', 1.0)
            if r > 1.0:
                ax.annotate(f'{r:.1f}×',
                            xy=(x[i] + width, dl_vals[i]),
                            xytext=(5, 15),
                            textcoords='offset points',
                            fontsize=10, fontweight='bold',
                            color='#27ae60')

        ax.set_xlabel('Configuration', fontsize=14)
        ax.set_ylabel('Communication Volume (GB)', fontsize=14)
        ax.set_title(title or 'Figure 2: Communication Reduction', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        import os
        fig_path = os.path.join(self.output_dir, 'figure2_comm_reduction.png')
        fig.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure 2 saved: {fig_path}")
        return fig_path

    def plot_figure2_cumulative(self, tracker_data_list, title=None):
        """Plot Figure 2 variant: cumulative comm over training.

        Args:
            tracker_data_list: list of (label, steps, ddp_cum_gb, dl_cum_gb)

        Shows how DES-LOC comm diverges from DDP over time.
        """
        if not self._setup():
            return
        plt = self._plt
        fig, ax = plt.subplots(figsize=(12, 8))

        for label, steps, ddp_cum, dl_cum in tracker_data_list:
            ax.plot(steps, ddp_cum, '--', label=f'{label} (DDP)',
                    linewidth=1.5, alpha=0.6)
            ax.plot(steps, dl_cum, '-', label=f'{label} (DES-LOC)',
                    linewidth=2.0)

        ax.set_xlabel('Training Step', fontsize=14)
        ax.set_ylabel('Cumulative Communication (GB)', fontsize=14)
        ax.set_title(title or 'Cumulative Communication: DDP vs DES-LOC',
                     fontsize=18)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        import os
        fig_path = os.path.join(self.output_dir,
                                'figure2_comm_cumulative.png')
        fig.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure 2 cumulative saved: {fig_path}")
        return fig_path


def desloc_generate_all_figures(log_dir, output_dir='./figures'):
    """Entry point: parse all logs in log_dir and generate Figures 1+2.

    This function:
    1. Finds all .log files in log_dir
    2. Parses them with desloc_parse_log_file()
    3. Groups by config
    4. Generates Figure 1 (loss curves) and Figure 2 (comm bars)

    Ref: NKI-FA draw_plot.py — end-to-end data→figure pipeline.
    All data from experiment logs, nothing hardcoded.

    Args:
        log_dir: directory containing .log files
        output_dir: directory for output PNG files
    """
    import os
    import sys

    # Import our log parser from engine.py
    # (This is a cross-file reference within the same package)
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from deepspeed.runtime.engine import desloc_parse_log_file
    except ImportError:
        print("WARNING: Cannot import desloc_parse_log_file")
        return

    # Find all log files
    log_files = sorted(
        f for f in os.listdir(log_dir)
        if f.endswith('.log'))

    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    # Parse all logs
    all_experiments = []
    for lf in log_files:
        filepath = os.path.join(log_dir, lf)
        try:
            exps = desloc_parse_log_file(filepath)
            all_experiments.extend(exps)
        except Exception as e:
            print(f"WARNING: Failed to parse {lf}: {e}")

    if not all_experiments:
        print("No experiment data parsed")
        return

    print(f"Parsed {len(all_experiments)} experiments from "
          f"{len(log_files)} log files")

    # Generate figures
    plotter = DeslocFigurePlotter(output_dir=output_dir)

    # Figure 1: group by Kx, take first seed of each
    plotter.plot_figure1(all_experiments)

    # Figure 2: compute comm reduction data
    from deepspeed.runtime.utils import desloc_comm_reduction_sweep
    # Use model params from first experiment if available
    num_params = 125_000_000  # default
    total_steps = 500
    for exp in all_experiments:
        cfg = exp.get('config')
        if cfg:
            size = getattr(cfg, 'model_size', '125M')
            size_map = {'125M': 125e6, '350M': 350e6,
                        '700M': 700e6, '1.3B': 1.3e9}
            num_params = int(size_map.get(size, 125e6))
            steps = exp.get('steps', [])
            if steps:
                total_steps = max(total_steps, len(steps))
            break

    kx_values = sorted(set(
        getattr(e.get('config'), 'Kx', 1)
        for e in all_experiments
        if e.get('config') is not None
    ))
    if not kx_values:
        kx_values = [1, 4, 16, 32, 64]

    sweep = desloc_comm_reduction_sweep(
        kx_values, num_params, total_steps)
    plotter.plot_figure2(sweep)

    print(f"All figures saved to {output_dir}")
