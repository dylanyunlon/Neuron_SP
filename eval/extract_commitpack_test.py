#!/usr/bin/env python3
"""
extract_commitpack_test.py — pull N test samples from CommitPackFT and
write them to eval/data/commitpack_test_1000.jsonl for offline eval.

Usage:
    python eval/extract_commitpack_test.py --n 1000 --out eval/data/commitpack_test_1000.jsonl
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", default="eval/data/commitpack_test_1000.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package required.  pip install datasets")
        raise SystemExit(1)

    print(f"Loading CommitPackFT (streaming, n={args.n})…")
    ds = load_dataset("bigcode/commitpackft", split="test", streaming=True)

    count = 0
    with open(args.out, "w") as fout:
        for item in ds:
            row = {
                "diff": item.get("diff", ""),
                "message": item.get("message", ""),
                "lang": item.get("lang", ""),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.n:
                break

    print(f"Saved {count} samples → {args.out}")


if __name__ == "__main__":
    main()
