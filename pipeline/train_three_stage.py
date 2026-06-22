# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
train_three_stage.py — DES-LOC 三阶段预训练编排器

Stage 1: pretrain_base   — Stack v2 完整代码 → 建立代码理解
Stage 2: continue_commit — CommitPack diff 序列 → 学增量修改
Stage 3: instruct_tune   — CommitPackFT → 指令对齐

每个 stage 结束时保存 checkpoint, 下个 stage 从 checkpoint 恢复后
重新初始化 optimizer (lr schedule 从头开始).

Usage:
    # 完整三阶段
    python -m pipeline.train_three_stage --stages 1,2,3

    # 只跑 Stage 2+3 (已有 base checkpoint)
    python -m pipeline.train_three_stage --stages 2,3 --resume-from checkpoints/stage1

    # Smoke test (70M 模型, 各阶段 100 步)
    python -m pipeline.train_three_stage --smoke-test
"""

import os
import sys
import argparse
import time
import json
import math
from typing import Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.unified_tokenizer import build_megatron_tokenizer, get_tokenizer
from pipeline.engine_bridge import DESLOCEngine, build_ds_config, detect_gpu_tiers, compute_shard_ratios


# ── 模型构建 ──

def build_model(
    vocab_size: int,
    hidden_size: int = 4096,
    num_layers: int = 32,
    num_heads: int = 32,
    seq_len: int = 2048,
    position_encoding: str = "rotary",
) -> nn.Module:
    """构建 GPT 模型. 先尝试 Megatron, fallback 到 HF transformers."""
    print(f"[model] building: hidden={hidden_size}, layers={num_layers}, "
          f"heads={num_heads}, vocab={vocab_size}, pos={position_encoding}")

    try:
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=seq_len,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        model = GPT2LMHeadModel(config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] GPT2LMHeadModel: {n_params/1e6:.0f}M params")
        return model
    except ImportError:
        pass

    # Minimal GPT fallback
    from torch.nn import TransformerDecoderLayer, TransformerDecoder
    decoder_layer = TransformerDecoderLayer(
        d_model=hidden_size, nhead=num_heads,
        dim_feedforward=hidden_size * 4, batch_first=True,
    )
    decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)
    embedding = nn.Embedding(vocab_size, hidden_size)
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    class MinimalGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = embedding
            self.decoder = decoder
            self.lm_head = lm_head

        def forward(self, input_ids, labels=None, **kw):
            h = self.embed(input_ids)
            mask = nn.Transformer.generate_square_subsequent_mask(h.size(1)).to(h.device)
            h = self.decoder(h, h, tgt_mask=mask)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    labels[:, 1:].contiguous().view(-1),
                )
            return type("Out", (), {"loss": loss, "logits": logits})()

    model = MinimalGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MinimalGPT fallback: {n_params/1e6:.0f}M params")
    return model


# ── 数据加载 ──

def load_stage1_data(tokenizer, seq_len, batch_size, data_path=None):
    """Stage 1: Stack v2 完整代码文件."""
    from torch.utils.data import DataLoader, IterableDataset

    class StackV2Dataset(IterableDataset):
        def __init__(self, tok, sl, path):
            self.tok = tok
            self.sl = sl
            self.path = path

        def __iter__(self):
            # 优先用已转换的 Megatron indexed 格式
            idx_path = os.path.join(self.path or "", "stack_v2_text_document.bin")
            if os.path.exists(idx_path):
                from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
                ds = MMapIndexedDataset(idx_path.replace(".bin", ""))
                for i in range(len(ds)):
                    tokens = torch.tensor(ds[i][:self.sl], dtype=torch.long)
                    if len(tokens) < self.sl:
                        tokens = torch.cat([tokens, torch.zeros(self.sl - len(tokens), dtype=torch.long)])
                    yield {"input_ids": tokens, "labels": tokens}
            else:
                # Fallback: 流式加载 HuggingFace
                try:
                    from datasets import load_dataset
                    ds = load_dataset("bigcode/the-stack-v2", streaming=True, split="train")
                    buffer = []
                    for item in ds:
                        content = item.get("content", "")
                        ids = self.tok.encode(content, add_special_tokens=False)
                        buffer.extend(ids)
                        while len(buffer) >= self.sl:
                            chunk = torch.tensor(buffer[:self.sl], dtype=torch.long)
                            buffer = buffer[self.sl:]
                            yield {"input_ids": chunk, "labels": chunk}
                except Exception as e:
                    print(f"[stage1] Stack v2 unavailable ({e}), using CommitPackFT code as fallback")
                    yield from self._fallback_code_data()

        def _fallback_code_data(self):
            """没有 Stack v2 时用 CommitPackFT 的 old_contents+new_contents 作为代码语料."""
            import glob
            files = glob.glob("datasets/bigcode/commitpackft/*.jsonl")
            for f in files:
                import json as _json
                for line in open(f):
                    d = _json.loads(line)
                    text = d.get("old_contents", "") + "\n" + d.get("new_contents", "")
                    ids = self.tok.encode(text, add_special_tokens=False)
                    if len(ids) >= 32:
                        ids = ids[:self.sl]
                        if len(ids) < self.sl:
                            ids += [self.tok.eos_token_id] * (self.sl - len(ids))
                        yield {"input_ids": torch.tensor(ids), "labels": torch.tensor(ids)}

    ds = StackV2Dataset(tokenizer, seq_len, data_path)
    return DataLoader(ds, batch_size=batch_size)


def load_stage2_data(tokenizer, seq_len, batch_size, data_path=None):
    """Stage 2: CommitPack diff 序列 (streaming)."""
    from torch.utils.data import DataLoader, IterableDataset

    class CommitDataset(IterableDataset):
        def __init__(self, tok, sl):
            self.tok = tok
            self.sl = sl

        def __iter__(self):
            import json as _json
            # 用已下载的 CommitPack 样本
            sample_file = "datasets/bigcode/commitpack/python_sample_10k.jsonl"
            if not os.path.exists(sample_file):
                # Fallback: 用 CommitPackFT
                sample_file = "datasets/bigcode/commitpackft/python.jsonl"

            for line in open(sample_file):
                d = _json.loads(line)
                old = d.get("old_contents", "")
                new = d.get("new_contents", "")
                msg = d.get("subject", d.get("message", ""))

                if hasattr(self.tok, "encode_commit"):
                    ids = self.tok.encode_commit(old, new, msg)
                else:
                    text = f"<|diff_start|><|old|>{old}<|new|>{new}<|commit_msg|>{msg}<|diff_end|>"
                    ids = self.tok.encode(text, add_special_tokens=False)

                ids = ids[:self.sl]
                if len(ids) < 32:
                    continue
                if len(ids) < self.sl:
                    ids += [self.tok.eos_token_id] * (self.sl - len(ids))
                t = torch.tensor(ids, dtype=torch.long)
                yield {"input_ids": t, "labels": t}

    ds = CommitDataset(tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size)


def load_stage3_data(tokenizer, seq_len, batch_size, data_path=None):
    """Stage 3: CommitPackFT instruction tuning."""
    from torch.utils.data import DataLoader, IterableDataset

    class InstructDataset(IterableDataset):
        def __init__(self, tok, sl):
            self.tok = tok
            self.sl = sl

        def __iter__(self):
            import json as _json, glob
            files = sorted(glob.glob("datasets/bigcode/commitpackft/*.jsonl"))
            for f in files:
                for line in open(f):
                    d = _json.loads(line)
                    # Instruction format: [INST] given diff, what's the commit message? [/INST]
                    old = d.get("old_contents", "")[:1000]
                    new = d.get("new_contents", "")[:1000]
                    msg = d.get("subject", "")
                    prompt = f"<|diff_start|><|old|>{old}<|new|>{new}<|diff_end|>"
                    response = f"<|commit_msg|>{msg}"
                    full = prompt + response
                    ids = self.tok.encode(full, add_special_tokens=False)[:self.sl]
                    if len(ids) < 32:
                        continue
                    if len(ids) < self.sl:
                        ids += [self.tok.eos_token_id] * (self.sl - len(ids))
                    t = torch.tensor(ids, dtype=torch.long)
                    yield {"input_ids": t, "labels": t}

    ds = InstructDataset(tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size)


# ── 单阶段训练循环 ──

def train_one_stage(
    engine: DESLOCEngine,
    dataloader,
    max_steps: int,
    stage_name: str,
    log_interval: int = 10,
    save_interval: int = 1000,
    save_path: str = "checkpoints",
):
    """一个阶段的训练循环."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}: max_steps={max_steps}")
    print(f"{'='*60}\n")

    os.makedirs(save_path, exist_ok=True)
    engine.step_count = 0
    total_loss = 0.0
    t0 = time.time()

    data_iter = iter(dataloader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss = engine.train_step(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        total_loss += loss

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - t0
            tokens_per_sec = (step * batch["input_ids"].numel()) / elapsed
            lr = engine.current_lr()
            print(f"  [{stage_name}] step={step}/{max_steps}  "
                  f"loss={avg_loss:.4f}  lr={lr:.2e}  "
                  f"tok/s={tokens_per_sec:.0f}  "
                  f"elapsed={elapsed:.0f}s")
            total_loss = 0.0

        if step % save_interval == 0:
            engine.save_checkpoint(save_path, tag=f"{stage_name}_step{step}")

    # 阶段结束保存
    engine.save_checkpoint(save_path, tag=f"{stage_name}_final")
    print(f"\n  [{stage_name}] done. {max_steps} steps, {time.time()-t0:.0f}s total\n")


# ── 三阶段编排 ──

def run_three_stage(args):
    """主入口."""
    print("[pipeline] DES-LOC Three-Stage Pretraining")
    print(f"  stages: {args.stages}")
    print(f"  model: hidden={args.hidden_size}, layers={args.num_layers}, heads={args.num_heads}")
    print(f"  seq_len={args.seq_len}, batch_size={args.batch_size}")

    # Tokenizer
    tok = build_megatron_tokenizer()
    hf_tok = tok.tokenizer  # underlying HF tokenizer for data loading

    # Model
    model = build_model(
        vocab_size=tok.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        position_encoding=args.position_encoding,
    )

    stages_to_run = [int(s) for s in args.stages.split(",")]

    for stage_num in stages_to_run:
        # 每个阶段重新构建 engine (新的 optimizer + lr schedule)
        if stage_num == 1:
            max_steps = args.stage1_steps
            ds_config = build_ds_config(
                train_batch_size=args.batch_size * args.gradient_accumulation,
                micro_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.stage1_lr,
                warmup_steps=args.warmup_steps,
                total_steps=max_steps,
            )
            dataloader = load_stage1_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            stage_name = "stage1_base_code"

        elif stage_num == 2:
            max_steps = args.stage2_steps
            ds_config = build_ds_config(
                train_batch_size=args.batch_size * args.gradient_accumulation,
                micro_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.stage2_lr,
                warmup_steps=args.warmup_steps // 2,
                total_steps=max_steps,
            )
            dataloader = load_stage2_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            stage_name = "stage2_commit_cpt"

        elif stage_num == 3:
            max_steps = args.stage3_steps
            ds_config = build_ds_config(
                train_batch_size=args.batch_size * args.gradient_accumulation,
                micro_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                learning_rate=args.stage3_lr,
                warmup_steps=args.warmup_steps // 4,
                total_steps=max_steps,
            )
            dataloader = load_stage3_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            stage_name = "stage3_instruct"
        else:
            raise ValueError(f"Unknown stage: {stage_num}")

        engine = DESLOCEngine(model, ds_config)
        engine.init()

        # 从上一阶段的 checkpoint 恢复 (模型权重, 但不恢复 optimizer)
        if args.resume_from and stage_num == stages_to_run[0]:
            engine.load_checkpoint(args.resume_from)
        elif stage_num > 1:
            prev_tag = f"stage{stage_num-1}_{'base_code' if stage_num==2 else 'commit_cpt'}_final"
            ckpt_path = os.path.join(args.checkpoint_dir, prev_tag)
            if os.path.exists(os.path.join(args.checkpoint_dir, prev_tag)):
                engine.load_checkpoint(args.checkpoint_dir, tag=prev_tag)

        train_one_stage(
            engine=engine,
            dataloader=dataloader,
            max_steps=max_steps,
            stage_name=stage_name,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            save_path=args.checkpoint_dir,
        )

        # 取出模型权重给下一个 stage 用
        model = engine.engine.module

    print("\n[pipeline] All stages complete.")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="DES-LOC Three-Stage Pretraining")

    # Model
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--position-encoding", choices=["rotary", "alibi"], default="rotary")

    # Training
    parser.add_argument("--stages", default="1,2,3", help="comma-separated stage numbers")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2000)

    # Per-stage steps and LR
    parser.add_argument("--stage1-steps", type=int, default=100000)
    parser.add_argument("--stage1-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-steps", type=int, default=50000)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-steps", type=int, default=5000)
    parser.add_argument("--stage3-lr", type=float, default=5e-5)

    # IO
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)

    # Quick test
    parser.add_argument("--smoke-test", action="store_true",
                        help="70M model, 100 steps per stage")

    args = parser.parse_args()

    if args.smoke_test:
        args.hidden_size = 512
        args.num_layers = 8
        args.num_heads = 8
        args.stage1_steps = 100
        args.stage2_steps = 100
        args.stage3_steps = 50
        args.log_interval = 10
        args.save_interval = 50
        print("[pipeline] SMOKE TEST mode: 70M model, 100 steps/stage")

    run_three_stage(args)


if __name__ == "__main__":
    main()
