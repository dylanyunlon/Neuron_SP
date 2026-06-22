#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
validate_commit_packing.py — M731-M745 验证脚本

验证 CommitPackFT Python subset 的 packing 效率，目标: padding ratio < 5%
结果写入 desloc_results/phase6/packing_efficiency.json

用法:
    python datasets/bigcode/validate_commit_packing.py
    python datasets/bigcode/validate_commit_packing.py --max-samples 5000
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.bigcode.commit_packing import (
    CommitSequencePacker,
    HeteroBatchSampler,
    compute_packing_stats,
)
from datasets.bigcode.load_commits import load_commit_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dataset", default="commitpackft")
    parser.add_argument("--lang", default="python")
    parser.add_argument(
        "--output",
        default=str(ROOT / "desloc_results" / "phase6" / "packing_efficiency.json"),
    )
    args = parser.parse_args()

    print(f"[validate] dataset={args.dataset}  lang={args.lang}  "
          f"seq_len={args.seq_len}  max_samples={args.max_samples}")

    # --- Pack ---
    packer = CommitSequencePacker(tokenizer=None, seq_len=args.seq_len)
    samples = load_commit_dataset(
        args.dataset, lang=args.lang, max_samples=args.max_samples
    )
    packed = packer.pack_dataset(samples)

    stats = compute_packing_stats(packed)
    print("\n=== Packing statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # --- Hetero batch sampler diagnostic (first 3 steps) ---
    print("\n=== HeteroBatchSampler diagnostics (first 3 steps) ===")
    sampler = HeteroBatchSampler(
        packed,
        gpu_mem_map={0: 96, 1: 49},   # H100 / A6000
        base_batch=1,
        verbose=True,
    )
    for step_idx, _batch in enumerate(sampler):
        if step_idx >= 3:
            break

    # --- Write results ---
    output = {
        "task": "M731-M745",
        "session": "Claude-20",
        "dataset": args.dataset,
        "lang": args.lang,
        "seq_len": args.seq_len,
        "max_samples": args.max_samples,
        **stats,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[validate] Results written → {out_path}")

    # Exit non-zero if padding target not met
    if not stats.get("meets_5pct_target", False):
        print(f"[WARN] padding_ratio={stats['padding_ratio']:.3%} exceeds 5% target")
        sys.exit(1)
    print("[OK] padding ratio < 5% ✓")


if __name__ == "__main__":
    main()
