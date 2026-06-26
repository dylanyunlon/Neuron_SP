#!/usr/bin/env python3
"""Train Neuron-SP BPE tokenizer on REAL CommitPack data (not synthetic).

Uses our self-developed tokenizer/core/ (pure Python, zero external tokenizer deps).
Only external dep: `datasets` for streaming CommitPack from HuggingFace.

Usage:
    pip install datasets
    python tokenizer/train_on_real_data.py --num-samples 200000 --vocab-size 32000

Output:
    tokenizer/neuron_sp_32k/vocab.json
    tokenizer/neuron_sp_32k/merges.txt
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


COMMIT_SPECIAL_TOKENS = [
    "<pad>",
    "<|endoftext|>",
    "<|diff_start|>",
    "<|diff_end|>",
    "<|old|>",
    "<|new|>",
    "<|commit_msg|>",
    "<|file_path|>",
    "<|lang|>",
]


def stream_commitpack_texts(num_samples: int, languages=("python", "javascript")):
    """Stream real commit texts from bigcode/commitpack."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("pip install datasets")
        sys.exit(1)

    texts = []
    for lang in languages:
        log.info("Streaming bigcode/commitpack/%s ...", lang)
        try:
            ds = load_dataset(
                "bigcode/commitpack",
                data_dir=lang,
                split="train",
                streaming=True,
            )
            per_lang = num_samples // len(languages)
            count = 0
            for sample in ds:
                old = sample.get("old_contents", "") or ""
                new = sample.get("new_contents", "") or ""
                msg = sample.get("message", sample.get("subject", "")) or ""

                # Format as commit training text
                text = (
                    f"<|diff_start|>{old}<|old|>"
                    f"{new}<|new|>"
                    f"{msg}<|commit_msg|>"
                    f"<|endoftext|>"
                )
                texts.append(text)
                count += 1
                if count >= per_lang:
                    break
                if count % 10000 == 0:
                    log.info("  %s: %d/%d samples", lang, count, per_lang)
        except Exception as e:
            log.warning("Failed to load %s: %s", lang, e)

    log.info("Total texts collected: %d", len(texts))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-dir", default="tokenizer/neuron_sp_32k")
    args = parser.parse_args()

    # Step 1: Stream real data
    log.info("=== Step 1: Stream real CommitPack data ===")
    texts = stream_commitpack_texts(args.num_samples)
    if not texts:
        log.error("No data collected! Check network / HuggingFace access.")
        sys.exit(1)

    # Step 2: Learn BPE merges using our self-developed algorithm
    log.info("=== Step 2: Learn BPE merges (self-developed, pure Python) ===")
    from tokenizer.core.bpe_learn import learn_bpe_merges

    # 256 base bytes + special tokens + learned merges = vocab_size
    num_special = len(COMMIT_SPECIAL_TOKENS)
    num_base_bytes = 256
    num_merges = args.vocab_size - num_base_bytes - num_special - 1  # -1 for padding
    log.info(
        "Target: %d base bytes + %d merges + %d special + 1 pad = %d vocab",
        num_base_bytes, num_merges, num_special, args.vocab_size,
    )

    t0 = time.time()
    merges = learn_bpe_merges(texts, num_merges=num_merges)
    elapsed = time.time() - t0
    log.info("Learned %d merges in %.1f seconds", len(merges), elapsed)

    # Step 3: Build vocab
    log.info("=== Step 3: Build vocab ===")
    from tokenizer.core.vocab import Vocab, build_vocab
    vocab = build_vocab(merges, special_tokens=COMMIT_SPECIAL_TOKENS)
    log.info("Vocab size: %d", vocab.vocab_size)

    # Step 4: Save
    os.makedirs(args.output_dir, exist_ok=True)
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    vocab.save(vocab_path)
    log.info("Saved vocab to %s", vocab_path)

    # Save merges as text
    merges_path = os.path.join(args.output_dir, "merges.txt")
    with open(merges_path, "w") as f:
        f.write(f"# Neuron-SP BPE merges — {len(merges)} rules\n")
        f.write(f"# Trained on {len(texts)} real CommitPack samples\n")
        for a, b in merges:
            f.write(f"{a.hex()} {b.hex()}\n")
    log.info("Saved merges to %s", merges_path)

    # Step 5: Quick encode test
    log.info("=== Step 5: Encode test ===")
    from tokenizer.core.encoder import BPEEncoder
    encoder_vocab = {tok: vocab.token_to_id(tok) for tok in vocab._token_to_id}
    enc = BPEEncoder(encoder_vocab, merges)

    test_texts = [
        "def hello():\n    print('world')\n",
        "fix: resolve null pointer in parser",
        "<|diff_start|>old code<|old|>new code<|new|>fix bug<|commit_msg|><|endoftext|>",
    ]
    for t in test_texts:
        ids = enc.encode(t)
        log.info("  '%s' → %d tokens (%.2f tok/char)", t[:40], len(ids), len(ids)/len(t))

    log.info("=== Done! Tokenizer saved to %s ===", args.output_dir)


if __name__ == "__main__":
    main()
