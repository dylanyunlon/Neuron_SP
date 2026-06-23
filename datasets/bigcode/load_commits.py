# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
BigCode commit dataset loader for DES-LOC pretraining.

Supports GLM-130B-style mixed training:
  - 70% span corruption (BERT-style): mask spans from diff, predict commit message
  - 30% causal LM  (GPT-style):  before → after (natural causal structure)

Usage:
    # On ags1, first run: bash pull_all_datasets.sh
    # Then in training:
    from datasets.bigcode.load_commits import load_commit_dataset, MixedCommitDataset

    # Plain iterator (original API, unchanged):
    ds = load_commit_dataset("commitpackft", lang="python")
    for sample in ds:
        print(sample["text"])          # tokenizer-ready string

    # Mixed-training iterator (GLM-130B style):
    mixed_ds = MixedCommitDataset("commitpackft", lang="python", alpha=0.7)
    for sample in mixed_ds:
        print(sample["mode"])          # "span_corruption" | "causal_lm"
        print(sample["text"])          # tokenizer-ready string
        print(sample["loss_weight"])   # alpha (span) or 1-alpha (causal)

Smoke-test (CommitPackFT Python subset):
    python load_commits.py commitpackft python --mixed --samples 500
"""

import json
import os
import glob
import random
import re
from pathlib import Path
from typing import Iterator, Dict, Optional, List

# ---------------------------------------------------------------------------
# Formatting templates
# ---------------------------------------------------------------------------

COMMIT_FORMAT = (
    "<commit_before>\n{old_contents}\n"
    "<commit_msg>\n{message}\n"
    "<commit_after>\n{new_contents}"
)

# Causal-LM template: diff encodes the causal signal, message is the target.
CAUSAL_FORMAT = (
    "<diff>\n{diff}\n"
    "<commit_msg>\n{message}"
)

# Span-corruption template: [MASK] tokens replace diff spans; model predicts
# the masked content conditioned on the surrounding context + message.
SPAN_CORRUPT_FORMAT = (
    "<corrupt_diff>\n{masked_diff}\n"
    "<commit_msg>\n{message}\n"
    "<targets>\n{targets}"
)

DATASETS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Each entry declares how a dataset should be loaded from HuggingFace Hub.
# ``streaming=True`` is mandatory for multi-TB datasets (commitpack) to avoid
# downloading the full corpus before iteration begins — the iterator fetches
# shards on demand, keeping memory footprint bounded to a single shard at a
# time instead of the entire 4 TB.
DATASET_REGISTRY: Dict[str, Dict] = {
    "commitpackft": {
        "hf_id": "bigcode/commitpackft",
        "streaming": False,    # 2 GB — full download is practical
        "split": "train",
        "config_field": "lang",  # load_dataset 2nd positional arg
        "local_subdir": "commitpackft",
        "description": "CommitPackFT: 2 GB GPT-4-filtered high-quality commits",
    },
    "starcoderdata_commits": {
        "hf_id": "bigcode/starcoderdata",
        "streaming": True,     # 32 GB subset — stream to avoid OOM
        "split": "train",
        "hf_kwargs": {"data_dir": "git-commits"},
        "local_subdir": "starcoderdata_commits",
        "description": "StarCoder git-commits: ~32 GB subset of starcoderdata",
    },
    "commitpack": {
        "hf_id": "bigcode/commitpack",
        # 4 TB corpus: streaming is the only viable access pattern.
        # Loading without streaming=True would attempt to materialise all
        # ~1.45 billion commits (≈4 TB) before yielding any sample, which
        # exhausts disk on any practical machine.
        "streaming": True,
        "split": "train",
        "config_field": "lang",  # one language config per load_dataset call
        "local_subdir": "commitpack",
        "description": "CommitPack: 4 TB full GHArchive Git commits (streaming only)",
    },
}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Each entry maps a logical name → HuggingFace load kwargs + local directory.
# The registry drives both pull_all_datasets.sh (for documentation) and the
# streaming loaders below, keeping config in one authoritative place.
#
# Fields per entry:
#   hf_repo      : HuggingFace dataset repo id
#   hf_kwargs    : extra kwargs forwarded to load_dataset() (e.g. data_dir)
#   local_dir    : subdirectory under DATASETS_DIR for cached JSONL files
#   size_hint    : human-readable size for operator reference
#   splits       : list of HF splits actually used (subset of the full dataset)
#   description  : one-line purpose summary
DATASET_REGISTRY: Dict[str, Dict] = {
    # CommitPackFT — GPT-4-filtered high-quality instruction commits (~2 GB)
    "commitpackft": {
        "hf_repo": "bigcode/commitpackft",
        "hf_kwargs": {},
        "local_dir": "commitpackft",
        "size_hint": "~2 GB",
        "splits": ["train"],
        "description": "GPT-4 curated instruction-style commits; primary SFT corpus",
    },
    # CommitPack — full unfiltered commit archive (~4 TB)
    "commitpack": {
        "hf_repo": "bigcode/commitpack",
        "hf_kwargs": {},
        "local_dir": "commitpack",
        "size_hint": "~4 TB",
        "splits": ["train"],
        "description": "Full GHArchive-sourced commits; per-language streaming only",
    },
    # StarCoder commits — raw git-commits split (~32 GB)
    # Single-file commits from Google BigQuery public GitHub data.
    # 80 % of samples use a ±32-line context window; 20 % retain full file.
    "starcoderdata": {
        "hf_repo": "bigcode/starcoderdata",
        "hf_kwargs": {"data_dir": "git-commits"},
        "local_dir": "starcoderdata_commits",
        "size_hint": "~32 GB",
        "splits": ["train"],
        "description": "StarCoder raw commit subset; baseline pretraining corpus",
    },
    # StarCoder commits — near-dedup filtered git-commits-cleaned split (~64 GB)
    # Applies MinHash-LSH near-deduplication + exact-dedup on top of git-commits;
    # higher token density and lower repetition → preferred for DES-LOC pretraining.
    "starcoderdata_cleaned": {
        "hf_repo": "bigcode/starcoderdata",
        "hf_kwargs": {"data_dir": "git-commits-cleaned"},
        "local_dir": "starcoderdata_commits",
        "size_hint": "~64 GB",
        "splits": ["train"],
        "description": "StarCoder deduped commit subset (git-commits-cleaned); recommended for pretraining",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_diff(old: str, new: str) -> str:
    """Minimal unified-diff-like representation from old/new contents."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff_lines: List[str] = []
    for line in old_lines:
        diff_lines.append(f"- {line}")
    for line in new_lines:
        diff_lines.append(f"+ {line}")
    return "\n".join(diff_lines)


def _span_corrupt(text: str,
                  mask_ratio: float = 0.15,
                  mean_span_len: int = 3,
                  mask_token: str = "[MASK]",
                  seed: Optional[int] = None) -> Dict[str, str]:
    """
    Apply span-corruption masking (T5 / GLM style) to *text*.

    Returns:
        {
          "masked": text with contiguous spans replaced by [MASK],
          "targets": space-separated masked spans in order,
        }
    """
    rng = random.Random(seed)
    tokens = text.split()
    n = len(tokens)
    if n == 0:
        return {"masked": text, "targets": ""}

    num_to_mask = max(1, int(n * mask_ratio))
    masked = tokens[:]
    targets: List[str] = []
    i = 0
    masked_count = 0

    while i < n and masked_count < num_to_mask:
        # Geometric span length
        span_len = min(
            max(1, int(rng.expovariate(1.0 / mean_span_len))),
            n - i,
            num_to_mask - masked_count,
        )
        span = " ".join(tokens[i: i + span_len])
        targets.append(span)
        masked[i] = mask_token
        for j in range(i + 1, i + span_len):
            masked[j] = ""          # will be stripped when joining
        masked_count += span_len
        # Skip a few tokens before masking again
        skip = rng.randint(1, max(2, mean_span_len * 2))
        i += span_len + skip

    # Collapse consecutive empty strings left by multi-token spans
    masked_str = re.sub(r"( +)", " ", " ".join(t for t in masked if t)).strip()
    return {
        "masked": masked_str,
        "targets": " | ".join(targets),
    }


# ---------------------------------------------------------------------------
# Core formatting
# ---------------------------------------------------------------------------

def _format_commit(sample: Dict) -> str:
    """Original format (unchanged public API)."""
    old = sample.get("old_contents", "")
    new = sample.get("new_contents", "")
    msg = sample.get("subject", sample.get("message", ""))
    return COMMIT_FORMAT.format(old_contents=old, message=msg, new_contents=new)


def _format_causal(sample: Dict) -> str:
    """Causal-LM format: diff → commit message."""
    old = sample.get("old_contents", "")
    new = sample.get("new_contents", "")
    msg = sample.get("subject", sample.get("message", ""))
    diff = _build_diff(old, new)
    return CAUSAL_FORMAT.format(diff=diff, message=msg)


def _format_span_corrupt(sample: Dict,
                          mask_ratio: float = 0.15,
                          mean_span_len: int = 3) -> str:
    """Span-corruption format: masked diff + message → masked spans."""
    old = sample.get("old_contents", "")
    new = sample.get("new_contents", "")
    msg = sample.get("subject", sample.get("message", ""))
    diff = _build_diff(old, new)
    result = _span_corrupt(diff, mask_ratio=mask_ratio, mean_span_len=mean_span_len)
    return SPAN_CORRUPT_FORMAT.format(
        masked_diff=result["masked"],
        message=msg,
        targets=result["targets"],
    )


def load_from_hub(
    dataset_name: str = "commitpack",
    lang: Optional[str] = "python",
    max_samples: Optional[int] = None,
) -> Iterator[Dict[str, str]]:
    """
    Stream commit samples directly from HuggingFace Hub using DATASET_REGISTRY.

    For ``commitpack`` (4 TB) the registry enforces ``streaming=True`` so that
    shards are fetched lazily — one Arrow record batch at a time — rather than
    materialising the entire corpus on disk first.  This keeps resident memory
    proportional to a single shard (typically ~128 MB) regardless of the
    total dataset size.

    Args:
        dataset_name: registry key, e.g. ``"commitpack"``, ``"commitpackft"``.
        lang: language config passed as the second positional argument to
              ``load_dataset`` when the registry entry has ``config_field``.
        max_samples: optional hard cap on yielded samples.

    Yields:
        dict with ``"text"`` (tokenizer-ready), ``"_source"``, and all
        original HuggingFace dataset fields.

    Raises:
        KeyError:  ``dataset_name`` not in ``DATASET_REGISTRY``.
        ImportError: ``datasets`` package not installed.
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError as exc:
        raise ImportError("pip install datasets  # required for load_from_hub") from exc

    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    entry = DATASET_REGISTRY[dataset_name]
    hf_id = entry["hf_id"]
    streaming = entry.get("streaming", False)
    split = entry.get("split", "train")
    extra_kwargs = dict(entry.get("hf_kwargs", {}))

    # Language-scoped config (commitpack / commitpackft pass lang as config)
    positional_config = lang if (lang and entry.get("config_field")) else None

    if positional_config:
        ds = hf_load(hf_id, positional_config, split=split, streaming=streaming, **extra_kwargs)
    else:
        ds = hf_load(hf_id, split=split, streaming=streaming, **extra_kwargs)

    count = 0
    for sample in ds:
        if max_samples and count >= max_samples:
            return
        sample["text"] = _format_commit(sample)
        sample["_source"] = dataset_name
        yield sample
        count += 1


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_commit_dataset(
    dataset_name: str = "commitpackft",
    lang: Optional[str] = "python",
    max_samples: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> Iterator[Dict[str, str]]:
    """
    Load commit samples from local JSONL files (original API, unchanged).

    Args:
        dataset_name: one of "commitpackft", "starcoderdata_commits", "commitpack"
        lang: language filter (e.g. "python"), None for all
        max_samples: cap on number of samples returned
        data_dir: override base directory

    Yields:
        dict with "text" (formatted for tokenizer) and original fields
    """
    base = Path(data_dir) if data_dir else DATASETS_DIR / dataset_name

    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Run 'bash pull_all_datasets.sh' on ags1 first."
        )

    # Find JSONL files
    if lang:
        patterns = [f"{lang}.jsonl", f"{lang}_sample_*.jsonl", f"sample_*.jsonl"]
    else:
        patterns = ["*.jsonl"]

    files = []
    for pat in patterns:
        files.extend(glob.glob(str(base / pat)))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No JSONL files found in {base} for lang={lang}")

    count = 0
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                if max_samples and count >= max_samples:
                    return
                sample = json.loads(line)
                sample["text"] = _format_commit(sample)
                sample["_source"] = dataset_name
                yield sample
                count += 1


class MixedCommitDataset:
    """
    GLM-130B-style mixed pretraining dataset over commit data.

    Mixing strategy
    ---------------
    alpha (default 0.7) fraction of samples → span corruption (BERT-style).
    (1 - alpha)         fraction of samples → causal LM              (GPT-style).

    Each yielded sample has:
        "text"         : tokenizer-ready string
        "mode"         : "span_corruption" | "causal_lm"
        "loss_weight"  : alpha for span samples, (1-alpha) for causal samples
        "_source"      : dataset name
        … plus original commit fields

    Gradient weighting note
    -----------------------
    In all-reduce / mixed-precision training the caller should scale each
    sample's loss by sample["loss_weight"] before accumulating gradients, i.e.:

        loss = alpha * span_loss + (1 - alpha) * causal_loss

    This matches the heterogeneous-gradient weighting described in GLM-130B §3.
    """

    def __init__(
        self,
        dataset_name: str = "commitpackft",
        lang: Optional[str] = "python",
        alpha: float = 0.7,
        mask_ratio: float = 0.15,
        mean_span_len: int = 3,
        max_samples: Optional[int] = None,
        data_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            alpha:          fraction of samples routed to span corruption.
            mask_ratio:     token-level masking probability for span corruption.
            mean_span_len:  expected span length (geometric distribution).
            seed:           RNG seed for reproducible mixing.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.dataset_name = dataset_name
        self.lang = lang
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.mean_span_len = mean_span_len
        self.max_samples = max_samples
        self.data_dir = data_dir
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[Dict]:
        count = 0
        for sample in load_commit_dataset(
            dataset_name=self.dataset_name,
            lang=self.lang,
            max_samples=self.max_samples,
            data_dir=self.data_dir,
        ):
            count += 1
            if self._rng.random() < self.alpha:
                # --- span corruption (BERT-style) ---
                sample["text"] = _format_span_corrupt(
                    sample,
                    mask_ratio=self.mask_ratio,
                    mean_span_len=self.mean_span_len,
                )
                sample["mode"] = "span_corruption"
                sample["loss_weight"] = self.alpha
            else:
                # --- causal LM (GPT-style) ---
                sample["text"] = _format_causal(sample)
                sample["mode"] = "causal_lm"
                sample["loss_weight"] = 1.0 - self.alpha
            yield sample


# ---------------------------------------------------------------------------
# Dataset stats helper (unchanged)
# ---------------------------------------------------------------------------

def get_dataset_stats(dataset_name: str = "commitpackft") -> Dict:
    """Get basic stats about a downloaded dataset."""
    base = DATASETS_DIR / dataset_name
    if not base.exists():
        return {"status": "not_downloaded", "path": str(base)}

    jsonl_files = list(base.glob("*.jsonl"))
    total_size = sum(f.stat().st_size for f in jsonl_files)
    total_lines = 0
    for f in jsonl_files:
        with open(f) as fh:
            total_lines += sum(1 for _ in fh)

    return {
        "status": "ready",
        "path": str(base),
        "files": len(jsonl_files),
        "total_samples": total_lines,
        "total_size_mb": round(total_size / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Smoke-test CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="CommitPackFT smoke test")
    parser.add_argument("dataset", nargs="?", default="commitpackft")
    parser.add_argument("lang",    nargs="?", default="python")
    parser.add_argument("--mixed",   action="store_true",
                        help="Use MixedCommitDataset (GLM-130B style)")
    parser.add_argument("--alpha",   type=float, default=0.7)
    parser.add_argument("--samples", type=int,   default=3)
    args = parser.parse_args()

    if args.mixed:
        print(f"=== Mixed training smoke test: {args.dataset} / {args.lang} ===")
        print(f"    alpha={args.alpha}  ({args.alpha*100:.0f}% span_corruption, "
              f"{(1-args.alpha)*100:.0f}% causal_lm)\n")

        counts = {"span_corruption": 0, "causal_lm": 0}
        ds = MixedCommitDataset(
            args.dataset, lang=args.lang,
            alpha=args.alpha, max_samples=args.samples,
        )
        for i, s in enumerate(ds):
            counts[s["mode"]] += 1
            print(f"--- Sample {i}  mode={s['mode']}  weight={s['loss_weight']:.2f} ---")
            print(s["text"][:400])
            print()

        print(f"Totals: {counts}")
    else:
        print(f"Stats for {args.dataset}:", get_dataset_stats(args.dataset))
        print("\nFirst 3 samples:")
        for i, s in enumerate(load_commit_dataset(args.dataset, max_samples=3)):
            print(f"\n--- Sample {i} ---")
            print(s["text"][:300])
