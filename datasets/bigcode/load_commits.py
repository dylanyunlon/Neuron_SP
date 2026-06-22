# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
BigCode commit dataset loader for DES-LOC pretraining.

Usage:
    # On ags1, first run: bash pull_all_datasets.sh
    # Then in training:
    from datasets.bigcode.load_commits import load_commit_dataset

    ds = load_commit_dataset("commitpackft", lang="python")
    for sample in ds:
        print(sample["text"])  # tokenizer-ready format
"""

import json
import os
import glob
from pathlib import Path
from typing import Iterator, Dict, Optional

COMMIT_FORMAT = (
    "<commit_before>\n{old_contents}\n"
    "<commit_msg>\n{message}\n"
    "<commit_after>\n{new_contents}"
)

DATASETS_DIR = Path(__file__).parent


def _format_commit(sample: Dict) -> str:
    """Format a commit sample into CODE_BEFORE / MSG / CODE_AFTER structure."""
    old = sample.get("old_contents", "")
    new = sample.get("new_contents", "")
    msg = sample.get("subject", sample.get("message", ""))
    return COMMIT_FORMAT.format(
        old_contents=old,
        message=msg,
        new_contents=new,
    )


def load_commit_dataset(
    dataset_name: str = "commitpackft",
    lang: Optional[str] = "python",
    max_samples: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> Iterator[Dict[str, str]]:
    """
    Load commit samples from local JSONL files.

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


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "commitpackft"
    print(f"Stats for {name}:", get_dataset_stats(name))
    print("\nFirst 3 samples:")
    for i, s in enumerate(load_commit_dataset(name, max_samples=3)):
        print(f"\n--- Sample {i} ---")
        print(s["text"][:300])
