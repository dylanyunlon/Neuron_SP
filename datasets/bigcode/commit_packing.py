# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
commit_packing.py — StarCoder Commit 数据 Sequence Packing 优化 (M731-M745)

Implements:
  1. Commit-diff-aware packing (no cross-commit boundary splits)
     - Short commits (≤256 tokens) → merged into same sequence
     - Long commits (>2048 tokens) → sliding window
  2. Heterogeneous batch sampler (H100 96GB / A6000 49GB)
  3. Diagnostic prints: per-rank token count + padding ratio

Usage:
    from datasets.bigcode.commit_packing import CommitSequencePacker, HeteroBatchSampler
    from datasets.bigcode.load_commits import load_commit_dataset

    packer = CommitSequencePacker(tokenizer, seq_len=2048)
    packed  = packer.pack_dataset(load_commit_dataset("commitpackft", lang="python"))
    sampler = HeteroBatchSampler(packed, gpu_mem_map={0: 96, 1: 49})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Token-length helper (works with any HF tokenizer *or* a plain word-split
# approximation so the module is importable without transformers installed)
# ---------------------------------------------------------------------------

def _token_len(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return max(1, len(text.split()))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(ids)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORT_COMMIT_THRESHOLD = 256   # tokens — merge freely
LONG_COMMIT_THRESHOLD  = 2048  # tokens — apply sliding window
SLIDING_WINDOW_STRIDE  = 1024  # tokens — overlap for long commits


# ---------------------------------------------------------------------------
# Core packing logic
# ---------------------------------------------------------------------------

@dataclass
class PackedSequence:
    """One packed training sequence composed of ≥1 commits."""
    tokens: List[int] = field(default_factory=list)
    commit_ids: List[int] = field(default_factory=list)   # source commit indices
    num_commits: int = 0
    seq_len: int = 0  # target length (for padding-ratio accounting)

    @property
    def length(self) -> int:
        return len(self.tokens)

    @property
    def padding(self) -> int:
        return max(0, self.seq_len - self.length)

    @property
    def padding_ratio(self) -> float:
        if self.seq_len == 0:
            return 0.0
        return self.padding / self.seq_len


class CommitSequencePacker:
    """
    Packs commit samples into fixed-length sequences without crossing commit
    boundaries.  Delegates actual tokenization to the provided tokenizer.

    Args:
        tokenizer  : HF tokenizer (or None for word-split approximation)
        seq_len    : target sequence length in tokens (default 2048)
        pad_token_id: id used to pad sequences (default 0)
    """

    def __init__(
        self,
        tokenizer=None,
        seq_len: int = 2048,
        pad_token_id: int = 0,
    ):
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.pad_token_id = pad_token_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pack_dataset(
        self,
        samples: Iterator[Dict],
        max_sequences: Optional[int] = None,
    ) -> List[PackedSequence]:
        """
        Consume an iterator of commit samples and return a list of
        PackedSequence objects ready for training.
        """
        packed: List[PackedSequence] = []
        current = self._new_seq()

        for idx, sample in enumerate(samples):
            if max_sequences and len(packed) >= max_sequences:
                break

            commit_tokens = self._encode_commit(sample)
            n = len(commit_tokens)

            if n <= SHORT_COMMIT_THRESHOLD:
                # Short commit: try to merge into current sequence
                if current.length + n <= self.seq_len:
                    self._append(current, commit_tokens, idx)
                else:
                    packed.append(self._finalize(current))
                    current = self._new_seq()
                    self._append(current, commit_tokens, idx)

            elif n <= self.seq_len:
                # Medium commit: fits in one seq but may not merge
                if current.length + n <= self.seq_len:
                    self._append(current, commit_tokens, idx)
                else:
                    if current.num_commits > 0:
                        packed.append(self._finalize(current))
                    current = self._new_seq()
                    self._append(current, commit_tokens, idx)

            else:
                # Long commit (> seq_len): sliding window
                if current.num_commits > 0:
                    packed.append(self._finalize(current))
                    current = self._new_seq()

                for window in self._sliding_window(commit_tokens, idx):
                    packed.append(window)

        # Flush remainder
        if current.num_commits > 0:
            packed.append(self._finalize(current))

        return packed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_seq(self) -> PackedSequence:
        return PackedSequence(seq_len=self.seq_len)

    def _encode_commit(self, sample: Dict) -> List[int]:
        text = sample.get("text") or sample.get("_text", "")
        if not text:
            old = sample.get("old_contents", "")
            msg = sample.get("subject", sample.get("message", ""))
            new = sample.get("new_contents", "")
            text = (
                f"<commit_before>\n{old}\n"
                f"<commit_msg>\n{msg}\n"
                f"<commit_after>\n{new}"
            )
        if self.tokenizer is not None:
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]
        # Fallback: character-level chunking approximation (4 chars ≈ 1 token)
        return list(range(max(1, len(text) // 4)))

    @staticmethod
    def _append(seq: PackedSequence, tokens: List[int], commit_idx: int) -> None:
        seq.tokens.extend(tokens)
        seq.commit_ids.append(commit_idx)
        seq.num_commits += 1

    def _finalize(self, seq: PackedSequence) -> PackedSequence:
        # Pad to seq_len
        pad_len = self.seq_len - len(seq.tokens)
        if pad_len > 0:
            seq.tokens.extend([self.pad_token_id] * pad_len)
        return seq

    def _sliding_window(
        self, tokens: List[int], commit_idx: int
    ) -> Iterator[PackedSequence]:
        start = 0
        while start < len(tokens):
            chunk = tokens[start : start + self.seq_len]
            seq = self._new_seq()
            seq.tokens = list(chunk)
            seq.commit_ids = [commit_idx]
            seq.num_commits = 1
            yield self._finalize(seq)
            if start + self.seq_len >= len(tokens):
                break
            start += SLIDING_WINDOW_STRIDE


# ---------------------------------------------------------------------------
# Heterogeneous batch sampler
# ---------------------------------------------------------------------------

class HeteroBatchSampler:
    """
    Distributes packed sequences across GPUs proportionally to their VRAM.

    H100 (96 GB) gets ~2× as many sequences per micro-batch as A6000 (49 GB).

    Args:
        sequences   : list of PackedSequence objects
        gpu_mem_map : {rank: vram_gb}, e.g. {0: 96, 1: 49}
        base_batch  : micro-batch size for the smallest GPU
        verbose     : if True, print per-rank diagnostics

    Example::
        sampler = HeteroBatchSampler(packed, gpu_mem_map={0: 96, 1: 49})
        for batch_per_rank in sampler:
            # batch_per_rank[rank] → list[PackedSequence]
            ...
    """

    def __init__(
        self,
        sequences: List[PackedSequence],
        gpu_mem_map: Optional[Dict[int, int]] = None,
        base_batch: int = 1,
        verbose: bool = True,
    ):
        if gpu_mem_map is None:
            gpu_mem_map = {0: 96, 1: 49}

        self.sequences   = sequences
        self.gpu_mem_map = gpu_mem_map
        self.base_batch  = base_batch
        self.verbose     = verbose

        min_mem = min(gpu_mem_map.values())
        # ratio[rank] = how many base_batches this rank gets
        self.ratios: Dict[int, int] = {
            rank: max(1, round(mem / min_mem))
            for rank, mem in gpu_mem_map.items()
        }
        self.total_per_step = sum(self.ratios.values()) * base_batch

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[int, List[PackedSequence]]]:
        pool = list(self.sequences)
        step = 0
        idx  = 0

        while idx < len(pool):
            batch: Dict[int, List[PackedSequence]] = {}
            for rank in sorted(self.ratios):
                n = self.ratios[rank] * self.base_batch
                batch[rank] = pool[idx : idx + n]
                idx += n

            # Drop incomplete final step
            if any(len(v) == 0 for v in batch.values()):
                break

            if self.verbose:
                self._print_diagnostics(step, batch)

            yield batch
            step += 1

    def __len__(self) -> int:
        return len(self.sequences) // self.total_per_step

    # ------------------------------------------------------------------

    def _print_diagnostics(
        self, step: int, batch: Dict[int, List[PackedSequence]]
    ) -> None:
        print(f"\n[HeteroBatchSampler] step={step}")
        for rank, seqs in sorted(batch.items()):
            if not seqs:
                continue
            real_tokens   = sum(s.length - s.padding for s in seqs)
            total_tokens  = sum(s.seq_len for s in seqs)
            pad_tokens    = sum(s.padding for s in seqs)
            pad_ratio     = pad_tokens / max(1, total_tokens)
            mem_gb        = self.gpu_mem_map.get(rank, "?")
            print(
                f"  rank={rank} mem={mem_gb}GB  "
                f"seqs={len(seqs)}  "
                f"real_tokens={real_tokens}  "
                f"padding_tokens={pad_tokens}  "
                f"pad_ratio={pad_ratio:.3f}"
            )


# ---------------------------------------------------------------------------
# Efficiency statistics helper
# ---------------------------------------------------------------------------

def compute_packing_stats(packed: List[PackedSequence]) -> Dict:
    """Return a dict summarising packing efficiency across a list of sequences."""
    if not packed:
        return {}

    total_slots   = sum(s.seq_len for s in packed)
    total_real    = sum(s.length - s.padding for s in packed)
    total_pad     = sum(s.padding for s in packed)
    padding_ratio = total_pad / max(1, total_slots)
    commits_total = sum(s.num_commits for s in packed)

    lengths = [s.length - s.padding for s in packed]
    avg_real  = sum(lengths) / len(lengths)
    min_real  = min(lengths)
    max_real  = max(lengths)

    return {
        "num_sequences"   : len(packed),
        "total_real_tokens": total_real,
        "total_pad_tokens" : total_pad,
        "total_slots"      : total_slots,
        "padding_ratio"    : round(padding_ratio, 5),
        "meets_5pct_target": padding_ratio < 0.05,
        "commits_packed"   : commits_total,
        "avg_commits_per_seq": round(commits_total / len(packed), 2),
        "real_token_stats" : {
            "mean": round(avg_real, 1),
            "min" : min_real,
            "max" : max_real,
        },
    }
