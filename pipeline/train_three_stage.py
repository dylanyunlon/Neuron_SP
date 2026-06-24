# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
train_three_stage.py — DES-LOC 三阶段预训练编排器

Stage 1: pretrain_base   — Stack v2 完整代码 → 建立代码理解
Stage 2: continue_commit — CommitPack diff 序列 → 学增量修改
Stage 3: instruct_tune   — CommitPackFT → 指令对齐

每个 stage 结束时保存 checkpoint, 下个 stage 从 checkpoint 恢复后
重新初始化 HeteroMIMOTrainingLoop (lr schedule 从头开始).

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
from pipeline.engine_bridge import (
    build_ds_config,
    detect_gpu_tiers,
    compute_shard_ratios,
    DESLOCEngine,
    build_neuron_sp_runtime,
    teardown_hetero_runtime,
)
from deepspeed.runtime.hetero_mimo_training_loop import (
    setup_hetero_mimo_training,
    HeteroMIMOTrainingLoop,
    PCIeP2PCommunicator,
    PerModuleOptimizerConfig,
    SharedLocalityCache,
)
from deepspeed.runtime.hetero_gdn_selective_recompute import build_neuron_sp_config
from megatron.core.datasets.commit_dataset import build_commit_datasets
from datasets.bigcode.commit_packing import (
    CommitSequencePacker,
    HeteroBatchSampler,
    compute_packing_stats,
)


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
    """Stage 2: CommitPack diff sequences — commit-boundary-aware packing.

    Loading strategy (in order):
    1. If ``data_path`` (or the default location) contains .jsonl files,
       stream the raw commit samples through :class:`CommitSequencePacker`
       so no training sequence ever crosses a commit boundary.
    2. Otherwise fall back to ``build_commit_datasets()`` from
       ``megatron.core.datasets.commit_dataset`` (Megatron indexed format).

    In both cases, short commits (≤ 256 tokens) are merged into the same
    packed sequence; commits longer than ``seq_len`` are split with a
    sliding window.  The resulting :class:`PackedDataset` is wrapped with
    :class:`HeteroBatchSampler` so sequences are distributed to GPUs
    proportionally to their available VRAM.
    """
    import glob as _glob
    from torch.utils.data import DataLoader, Dataset

    resolved_path = data_path or "datasets/bigcode/commitpack"

    # ── 1. Check for .jsonl files — prefer CommitSequencePacker path ───────
    jsonl_files = sorted(_glob.glob(os.path.join(resolved_path, "*.jsonl")))

    if jsonl_files:
        print(f"[stage2] found {len(jsonl_files)} .jsonl file(s) in {resolved_path}; "
              f"packing with CommitSequencePacker (seq_len={seq_len}) ...")

        # Stream all commit samples from the .jsonl files.
        def _iter_samples():
            for filepath in jsonl_files:
                with open(filepath) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue

        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
        packer = CommitSequencePacker(
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_id,
        )
        packed = packer.pack_dataset(_iter_samples())

        stats = compute_packing_stats(packed)
        print(f"[stage2] packing stats: {stats}")
        if not stats.get("meets_5pct_target", True):
            print(
                f"[stage2] WARNING: padding_ratio={stats['padding_ratio']:.3%} "
                f"exceeds 5% target. Check commit length distribution."
            )

        # ── Map-style dataset over PackedSequence objects ─────────────────
        class PackedDataset(Dataset):
            def __init__(self, sequences):
                self.sequences = sequences

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx):
                tokens = torch.tensor(self.sequences[idx].tokens, dtype=torch.long)
                return {"input_ids": tokens, "labels": tokens}

        ds = PackedDataset(packed)

        # ── HeteroBatchSampler: distribute by GPU VRAM ratio ──────────────
        gpu_tiers = detect_gpu_tiers()  # {rank: vram_gb}
        hetero_sampler = HeteroBatchSampler(
            sequences=packed,
            gpu_mem_map=gpu_tiers if gpu_tiers else {0: 96, 1: 49},
            base_batch=batch_size,
            verbose=True,
        )

        def _collate(items):
            input_ids = torch.stack([i["input_ids"] for i in items])
            return {"input_ids": input_ids, "labels": input_ids}

        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=None,
            collate_fn=_collate,
            batch_sampler=_HeteroBatchSamplerAdapter(hetero_sampler, ds),
        )

    # ── 2. Fallback: Megatron indexed CommitDataset ────────────────────────
    print(f"[stage2] no .jsonl files found; building CommitDatasets from "
          f"{resolved_path} (seq_length={seq_len}) ...")
    train_ds, valid_ds, test_ds = build_commit_datasets(
        data_path=resolved_path,
        seq_length=seq_len,
        tokenizer=tokenizer,
    )
    print(f"[stage2] CommitDataset sizes -- "
          f"train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")

    # -- Collate: CommitDataset returns 'tokens'/'labels'/'loss_mask'/
    #    'position_ids'; remap to the 'input_ids'/'labels' convention used
    #    by _make_forward_backward_func. --
    def _collate(items):
        input_ids = torch.stack([item["tokens"] for item in items])
        labels    = torch.stack([item["labels"]  for item in items])
        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(train_ds, batch_size=batch_size,
                      shuffle=True, collate_fn=_collate, drop_last=True)


class _HeteroBatchSamplerAdapter:
    """Bridge HeteroBatchSampler (yields {rank: [PackedSequence]}) to a flat
    index-based batch_sampler that PyTorch DataLoader expects (yields [int])."""

    def __init__(self, hetero_sampler, dataset):
        self._sampler = hetero_sampler
        self._dataset = dataset
        # Build a reverse index: PackedSequence id → dataset index
        self._seq_to_idx = {id(seq): i for i, seq in enumerate(dataset.sequences)}

    def __iter__(self):
        for batch_per_rank in self._sampler:
            indices = []
            for seqs in batch_per_rank.values():
                for seq in seqs:
                    idx = self._seq_to_idx.get(id(seq))
                    if idx is not None:
                        indices.append(idx)
            if indices:
                yield indices

    def __len__(self):
        return len(self._sampler)


def load_stage3_data(tokenizer, seq_len, batch_size, data_path=None):
    """Stage 3: CommitPackFT instruction tuning — uses CommitSequencePacker + HeteroBatchSampler."""
    import json as _json, glob
    from torch.utils.data import DataLoader, Dataset
    # CommitSequencePacker, HeteroBatchSampler, compute_packing_stats are
    # imported at module level from datasets.bigcode.commit_packing.

    # ── 1. 收集指令格式 commit 样本 ─────────────────────────────────────
    files = sorted(glob.glob("datasets/bigcode/commitpackft/*.jsonl"))
    raw_samples = []
    for f in files:
        for line in open(f):
            d = _json.loads(line)
            old = d.get("old_contents", "")[:1000]
            new = d.get("new_contents", "")[:1000]
            msg = d.get("subject", "")
            # Instruction format wrapped as single text for packer
            text = (
                f"<commit_before>\n{old}\n"
                f"<commit_msg>\n{msg}\n"
                f"<commit_after>\n{new}"
            )
            raw_samples.append({"text": text})

    # ── 2. CommitSequencePacker: 替代手动 tokenize+pad ──────────────────
    pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
    packer = CommitSequencePacker(tokenizer=tokenizer, seq_len=seq_len, pad_token_id=pad_id)
    packed = packer.pack_dataset(iter(raw_samples))

    # ── 3. 验证 padding ratio < 5% ──────────────────────────────────────
    stats = compute_packing_stats(packed)
    print(f"[stage3] packing stats: {stats}")
    if not stats.get("meets_5pct_target", True):
        print(
            f"[stage3] WARNING: padding_ratio={stats['padding_ratio']:.3%} "
            f"exceeds 5% target. Check commit length distribution."
        )

    # ── 4. 封装成 Dataset ────────────────────────────────────────────────
    class PackedDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            tokens = torch.tensor(self.sequences[idx].tokens, dtype=torch.long)
            return {"input_ids": tokens, "labels": tokens}

    ds = PackedDataset(packed)

    # ── 5. HeteroBatchSampler: 按 GPU 显存比例分配 batch ─────────────────
    gpu_tiers = detect_gpu_tiers()  # {rank: vram_gb}
    hetero_sampler = HeteroBatchSampler(
        sequences=packed,
        gpu_mem_map=gpu_tiers if gpu_tiers else {0: 96, 1: 49},
        base_batch=batch_size,
        verbose=True,
    )

    def _collate(items):
        input_ids = torch.stack([i["input_ids"] for i in items])
        return {"input_ids": input_ids, "labels": input_ids}

    return DataLoader(ds, batch_size=batch_size, sampler=None,
                      collate_fn=_collate,
                      batch_sampler=_HeteroBatchSamplerAdapter(hetero_sampler, ds))


# ── 单阶段训练循环 ──

def _make_forward_backward_func(model: nn.Module):
    """返回适配 HeteroMIMOTrainingLoop.train_step() 的 forward_backward_func.

    HeteroMIMOTrainingLoop.train_step() 期望的签名:
        forward_backward_func(
            forward_only, p2p_communicator, pg_collection,
            data_iterator, model, config, iteration
        ) -> List[Tensor]

    这里用闭包把实际的模型前向 + loss 计算封装进去.
    """
    def forward_backward_func(
        *,
        forward_only: bool,
        p2p_communicator,
        pg_collection,
        data_iterator,
        model: nn.Module,
        config,
        iteration: int,
        **kwargs,
    ):
        try:
            batch = next(data_iterator)
        except StopIteration:
            return []

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Route cross-GPU transfers through PCIeP2PCommunicator when
        # available — large tensors are staged through CPU DRAM
        # (SharedLocalityCache) instead of direct PCIe P2P copies.
        try:
            device = next(model.parameters()).device
            src_dev = input_ids.device.index if input_ids.is_cuda else -1
            dst_dev = device.index if device.type == "cuda" else -1
            if (
                p2p_communicator is not None
                and src_dev >= 0
                and dst_dev >= 0
                and src_dev != dst_dev
            ):
                input_ids = p2p_communicator.send_activation(
                    input_ids, src_dev, dst_dev,
                    cache_key=f"fwd_input:iter={iteration}",
                )
                labels = p2p_communicator.send_activation(
                    labels, src_dev, dst_dev,
                    cache_key=f"fwd_labels:iter={iteration}",
                )
            else:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
        except StopIteration:
            pass

        if forward_only:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            if loss is not None:
                loss.backward()

        return [loss] if loss is not None else []

    return forward_backward_func


def _save_model_checkpoint(model: nn.Module, save_path: str, tag: str) -> None:
    """保存模型权重到 checkpoint 文件 (torch.save)."""
    os.makedirs(save_path, exist_ok=True)
    ckpt_file = os.path.join(save_path, f"{tag}.pt")
    # unwrap potential DeepSpeed / DDP wrapper
    unwrapped = getattr(model, "module", model)
    torch.save(unwrapped.state_dict(), ckpt_file)
    print(f"[checkpoint] model weights saved: {ckpt_file}")


def _load_model_checkpoint(model: nn.Module, save_path: str, tag: str) -> None:
    """从 checkpoint 文件恢复模型权重."""
    ckpt_file = os.path.join(save_path, f"{tag}.pt")
    if not os.path.exists(ckpt_file):
        print(f"[checkpoint] WARNING: {ckpt_file} not found, skipping load")
        return
    state_dict = torch.load(ckpt_file, map_location="cpu")
    unwrapped = getattr(model, "module", model)
    unwrapped.load_state_dict(state_dict)
    print(f"[checkpoint] model weights loaded: {ckpt_file}")


def train_one_stage(
    loop: HeteroMIMOTrainingLoop,
    model: nn.Module,
    dataloader,
    max_steps: int,
    stage_name: str,
    log_interval: int = 10,
    save_interval: int = 1000,
    save_path: str = "checkpoints",
):
    """一个阶段的训练循环, 基于 HeteroMIMOTrainingLoop."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}: max_steps={max_steps}")
    print(f"{'='*60}\n")

    # Log SharedLocalityCache and PCIeP2PCommunicator status from the MIMO
    # loop so operators can verify the 1.5 TB DRAM staging path is active.
    _cache = loop._cache
    _p2p = loop._p2p
    print(f"  [MIMO] SharedLocalityCache: max_entries={_cache._max_entries}, "
          f"max_bytes={_cache._max_bytes / (1024**3):.1f} GB")
    print(f"  [MIMO] PCIeP2PCommunicator: staging_threshold="
          f"{_p2p._threshold_bytes / (1024**2):.1f} MB")

    os.makedirs(save_path, exist_ok=True)

    # 构造 forward_backward_func (闭包, 包含模型引用)
    forward_backward_func = _make_forward_backward_func(model)

    # 简单 config 对象 (lr schedule 占位)
    class _Config:
        pass
    config = _Config()

    total_loss = 0.0
    t0 = time.time()
    data_iter = iter(dataloader)

    for step in range(1, max_steps + 1):
        # 用 iteration_scope 管理 cache 淘汰
        with loop.iteration_scope(step - 1):
            result = loop.train_step(
                forward_backward_func=forward_backward_func,
                data_iterator=data_iter,
                config=config,
                iteration=step - 1,
            )

        loss_val = result.loss if not math.isnan(result.loss) else 0.0
        total_loss += loss_val

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - t0
            print(f"  [{stage_name}] step={step}/{max_steps}  "
                  f"loss={avg_loss:.4f}  grad_norm={result.grad_norm:.4f}  "
                  f"cross_pool_transfers={result.cross_pool_transfers}  "
                  f"elapsed={elapsed:.0f}s")
            total_loss = 0.0

        if step % save_interval == 0:
            _save_model_checkpoint(model, save_path, tag=f"{stage_name}_step{step}")

    # 阶段结束保存
    _save_model_checkpoint(model, save_path, tag=f"{stage_name}_final")
    print(f"\n  [{stage_name}] done. {max_steps} steps, {time.time()-t0:.0f}s total\n")


# ── Stage 3 评估: BLEU / ROUGE-L / Perplexity ──

def _compute_bleu(references, hypotheses):
    """Corpus-level BLEU-4 with +1 smoothing (no nltk dependency)."""
    from collections import Counter

    def _ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    total_bp_r = 0
    total_bp_c = 0
    clipped_counts = [0] * 4
    total_counts = [0] * 4

    for ref, hyp in zip(references, hypotheses):
        ref_tok = ref.split()
        hyp_tok = hyp.split()
        total_bp_r += len(ref_tok)
        total_bp_c += len(hyp_tok)
        for n in range(1, 5):
            ref_ngrams = Counter(_ngrams(ref_tok, n))
            hyp_ngrams = Counter(_ngrams(hyp_tok, n))
            for ng, cnt in hyp_ngrams.items():
                clipped_counts[n - 1] += min(cnt, ref_ngrams.get(ng, 0))
            total_counts[n - 1] += max(len(hyp_tok) - n + 1, 0)

    # +1 smoothing to avoid zero n-gram counts
    log_bleu = 0.0
    for n in range(4):
        precision = (clipped_counts[n] + 1.0) / (total_counts[n] + 1.0)
        log_bleu += math.log(precision) / 4.0

    # Brevity penalty
    if total_bp_c == 0:
        return 0.0
    bp = min(1.0, math.exp(1.0 - total_bp_r / total_bp_c))
    return bp * math.exp(log_bleu)


def _compute_rouge_l(references, hypotheses):
    """Corpus-level ROUGE-L F1 via longest common subsequence."""

    def _lcs_length(x, y):
        m, n = len(x), len(y)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr
        return prev[n]

    total_p = 0.0
    total_r = 0.0
    count = 0
    for ref, hyp in zip(references, hypotheses):
        ref_tok = ref.split()
        hyp_tok = hyp.split()
        if not ref_tok or not hyp_tok:
            continue
        lcs = _lcs_length(ref_tok, hyp_tok)
        p = lcs / len(hyp_tok)
        r = lcs / len(ref_tok)
        total_p += p
        total_r += r
        count += 1

    if count == 0:
        return 0.0
    avg_p = total_p / count
    avg_r = total_r / count
    if avg_p + avg_r == 0:
        return 0.0
    return 2.0 * avg_p * avg_r / (avg_p + avg_r)


def run_stage3_eval(model, tokenizer, dataloader, seq_len, output_dir="experiments/eval_results"):
    """Evaluate Stage 3 model: commit message → diff BLEU/ROUGE-L + perplexity.

    Runs the model in eval mode over the Stage 3 dataloader, collecting:
      1. Per-token cross-entropy → perplexity
      2. Greedy-decoded commit messages compared to ground truth → BLEU-4 / ROUGE-L

    Results are written as JSON to ``output_dir/stage3_eval.json``.
    """
    print("\n[eval] running Stage 3 evaluation (BLEU / ROUGE-L / perplexity) ...")
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    # Retrieve commit boundary token IDs for segmenting sequences
    commit_ids = tokenizer.commit_token_ids()
    msg_id = commit_ids["commit_msg"]
    diff_start_id = commit_ids["diff_start"]
    diff_end_id = commit_ids["diff_end"]

    total_loss = 0.0
    total_tokens = 0
    references = []
    hypotheses = []
    max_eval_batches = 200

    data_iter = iter(dataloader)
    with torch.no_grad():
        for batch_idx in range(max_eval_batches):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)

            # Accumulate cross-entropy loss for perplexity
            if outputs.loss is not None:
                n_tokens = (labels != -100).sum().item()
                if n_tokens == 0:
                    n_tokens = labels.numel()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

            # Extract reference commit messages and greedy-decode predictions.
            # Each sequence has the format:
            #   ... <|commit_msg|> message_tokens <|diff_end|> ...
            # We extract the ground-truth message tokens and compare against
            # the model's argmax predictions at those positions.
            logits = outputs.logits  # (B, T, V)
            pred_ids = logits.argmax(dim=-1)  # (B, T)

            for b in range(input_ids.size(0)):
                seq = input_ids[b].tolist()
                pred = pred_ids[b].tolist()

                # Find <|commit_msg|> and <|diff_end|> boundaries
                try:
                    msg_pos = seq.index(msg_id)
                except ValueError:
                    continue
                # Find the next diff_end after commit_msg
                end_pos = None
                for p in range(msg_pos + 1, len(seq)):
                    if seq[p] == diff_end_id:
                        end_pos = p
                        break
                if end_pos is None or end_pos <= msg_pos + 1:
                    continue

                ref_tokens = seq[msg_pos + 1:end_pos]
                hyp_tokens = pred[msg_pos:end_pos - 1]

                ref_text = tokenizer.detokenize(ref_tokens)
                hyp_text = tokenizer.detokenize(hyp_tokens)
                if ref_text.strip():
                    references.append(ref_text.strip())
                    hypotheses.append(hyp_text.strip())

    # Compute metrics
    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    bleu = _compute_bleu(references, hypotheses) if references else 0.0
    rouge_l = _compute_rouge_l(references, hypotheses) if references else 0.0

    results = {
        "stage": "stage3_instruct",
        "perplexity": round(perplexity, 4),
        "bleu4": round(bleu, 6),
        "rouge_l_f1": round(rouge_l, 6),
        "eval_samples": len(references),
        "eval_tokens": total_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = os.path.join(output_dir, "stage3_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] perplexity={perplexity:.4f}  BLEU-4={bleu:.6f}  "
          f"ROUGE-L={rouge_l:.6f}  samples={len(references)}")
    print(f"[eval] results written to {out_path}")

    model.train()
    return results


# ── 三阶段编排 ──

def run_three_stage(args):
    """主入口."""
    print("[pipeline] DES-LOC Three-Stage Pretraining (HeteroMIMOTrainingLoop)")
    print(f"  stages: {args.stages}")
    print(f"  model: hidden={args.hidden_size}, layers={args.num_layers}, heads={args.num_heads}")
    print(f"  seq_len={args.seq_len}, batch_size={args.batch_size}")

    # Tokenizer
    tok = build_megatron_tokenizer()
    hf_tok = tok.tokenizer  # underlying HF tokenizer for data loading

    # Model — 在三个 stage 间共享, 通过 checkpoint 传递权重
    model = build_model(
        vocab_size=tok.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        position_encoding=args.position_encoding,
    )

    # DeepSpeed engine — wraps model for heterogeneous ZeRO sharding
    ds_config = build_ds_config(
        hetero_shard_ratio=compute_shard_ratios(detect_gpu_tiers()) if torch.cuda.is_available() else None,
        train_batch_size=args.batch_size * args.gradient_accumulation,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    engine = DESLOCEngine(model, ds_config=ds_config)
    engine.init()

    stages_to_run = [int(s) for s in args.stages.split(",")]

    # Track the previous stage's hetero runtime so we can tear it down before
    # building a new one (fresh optimizer state + lr schedule + clean cache).
    prev_loop: Optional[HeteroMIMOTrainingLoop] = None

    for stage_num in stages_to_run:
        # 每个 stage 重新调用 setup_hetero_mimo_training → 新的 optimizer + lr schedule
        if stage_num == 1:
            max_steps = args.stage1_steps
            dataloader = load_stage1_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            stage_name = "stage1_base_code"

        elif stage_num == 2:
            max_steps = args.stage2_steps
            dataloader = load_stage2_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            stage_name = "stage2_commit_cpt"

        elif stage_num == 3:
            max_steps = args.stage3_steps
            dataloader = load_stage3_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            stage_name = "stage3_instruct"

        else:
            raise ValueError(f"Unknown stage: {stage_num}")

        # 从上一阶段 checkpoint 恢复模型权重 (optimizer 不恢复, 从头 warm-up)
        if args.resume_from and stage_num == stages_to_run[0]:
            _load_model_checkpoint(model, args.resume_from, tag="model_weights")
        elif stage_num > 1:
            prev_name = "stage1_base_code" if stage_num == 2 else "stage2_commit_cpt"
            prev_tag = f"{prev_name}_final"
            _load_model_checkpoint(model, args.checkpoint_dir, tag=prev_tag)

        # Tear down the previous stage's hetero runtime (clears locality cache,
        # drops optimizer state, frees CUDA mem) before re-initialising.
        if prev_loop is not None:
            print(f"[pipeline] tearing down hetero runtime before {stage_name} ...")
            teardown_hetero_runtime(prev_loop)
            prev_loop = None

        # Each stage builds a fresh hetero runtime: recompute policy via
        # build_neuron_sp_config() and training loop via setup_hetero_mimo_training().
        # PerModuleOptimizerConfig carries the stage-specific LR so the router
        # initialises each device-pool's optimizer with the correct learning rate
        # rather than the global default (1e-4).
        stage_lr_map = {1: args.stage1_lr, 2: args.stage2_lr, 3: args.stage3_lr}
        print(f"[pipeline] initialising hetero runtime for {stage_name} "
              f"(lr={stage_lr_map[stage_num]}) ...")
        loop, recompute_config = build_neuron_sp_runtime(
            model=engine.engine,
            lr=stage_lr_map[stage_num],
        )
        print(f"[pipeline] recompute policy: granularity={recompute_config.granularity}, "
              f"modules_per_device={ {str(k): v for k, v in recompute_config.modules_per_device.items()} }")

        train_one_stage(
            loop=loop,
            model=model,
            dataloader=dataloader,
            max_steps=max_steps,
            stage_name=stage_name,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            save_path=args.checkpoint_dir,
        )

        # Run commit completion eval after Stage 3 (instruct tuning)
        if stage_num == 3:
            eval_dl = load_stage3_data(hf_tok, args.seq_len, args.batch_size, args.data_path)
            run_stage3_eval(model, tok, eval_dl, args.seq_len)

        # Remember this stage's loop so the next iteration can tear it down.
        prev_loop = loop

    # Final teardown after the last stage.
    if prev_loop is not None:
        teardown_hetero_runtime(prev_loop)
        prev_loop = None

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
