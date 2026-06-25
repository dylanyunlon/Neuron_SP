"""Train a BPE tokenizer from CommitPackFT data using sentencepiece.

Trains on the raw commit text (not the byte-encoded .npy), produces a
sentencepiece .model file with vocab_size=32000 matching our LlamaModel.

Usage:
    pip install sentencepiece
    python data/train_tokenizer.py

Then re-tokenize with:
    python data/prepare_mixed.py --tokenizer data/neuron_sp.model
"""
import argparse
import os
import sys
import time
import tempfile


def extract_text_corpus(output_txt: str, max_lines: int = 500_000):
    """Extract raw text from CommitPackFT for tokenizer training."""
    # Try loading from cached parquet files
    cache_dir = "data/hf_cache/commitpackft"
    if os.path.exists(cache_dir):
        import glob
        parquets = sorted(glob.glob(os.path.join(cache_dir, "**", "*.parquet"), recursive=True))
        if parquets:
            try:
                import pyarrow.parquet as pq
            except ImportError:
                os.system(f"{sys.executable} -m pip install -q pyarrow")
                import pyarrow.parquet as pq
            
            print(f"[tokenizer] Reading {len(parquets)} parquet files...")
            count = 0
            with open(output_txt, "w") as f:
                for pf in parquets:
                    table = pq.read_table(pf)
                    for i in range(len(table)):
                        cols = table.column_names
                        parts = []
                        for col in ["old_contents", "new_contents", "message", "subject"]:
                            if col in cols:
                                v = table[col][i].as_py()
                                if v:
                                    parts.append(str(v)[:3000])
                        if parts:
                            f.write(" ".join(parts) + "\n")
                            count += 1
                            if count >= max_lines:
                                break
                    if count >= max_lines:
                        break
            print(f"[tokenizer] Extracted {count:,} lines to {output_txt}")
            return count > 0

    # Fallback: generate synthetic corpus
    print("[tokenizer] No cached parquets, generating synthetic corpus...")
    import hashlib
    templates = [
        "def calculate_sum(a, b):\n    return a + b\n# Refactored to use cleaner pattern",
        "import os\nimport sys\nimport json\nfrom pathlib import Path\n# Added missing imports for file handling",
        "class DataProcessor:\n    def __init__(self, config):\n        self.config = config\n        self.cache = {}\n    def process(self, data):\n        return [self.transform(x) for x in data]",
        "async def fetch_data(url, timeout=30):\n    async with aiohttp.ClientSession() as session:\n        resp = await session.get(url, timeout=timeout)\n        return await resp.json()",
        "for item in items:\n    if item.is_valid():\n        results.append(item.process())\n    else:\n        logger.warning(f'Invalid item: {item.id}')",
    ]
    count = 0
    with open(output_txt, "w") as f:
        while count < max_lines:
            for t in templates:
                h = hashlib.md5(f"{count}".encode()).hexdigest()
                f.write(f"{t} # commit_{h[:8]}\n")
                count += 1
                if count >= max_lines:
                    break
    print(f"[tokenizer] Generated {count:,} synthetic lines")
    return True


def train_sentencepiece(input_txt: str, model_prefix: str, vocab_size: int = 32000):
    """Train SentencePiece BPE model."""
    try:
        import sentencepiece as spm
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q sentencepiece")
        import sentencepiece as spm

    print(f"[tokenizer] Training SentencePiece BPE (vocab_size={vocab_size})...")
    t0 = time.time()

    spm.SentencePieceTrainer.train(
        input=input_txt,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,  # byte-fallback covers all
        byte_fallback=True,
        # Code-specific settings
        split_digits=True,  # split numbers digit-by-digit
        split_by_whitespace=True,
        add_dummy_prefix=False,
        # Special tokens for commit format
        user_defined_symbols=[
            "<commit_before>", "<commit_after>", "<commit_msg>",
            "<corrupt_diff>", "<targets>", "<mask>",
            "<pad>", "<eos>", "<bos>",
        ],
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=False,
    )
    elapsed = time.time() - t0
    print(f"[tokenizer] Done in {elapsed:.1f}s → {model_prefix}.model")

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print(f"[tokenizer] Vocab size: {sp.get_piece_size()}")
    
    test = "def hello():\n    print('world')\n# Added greeting function"
    ids = sp.encode(test)
    print(f"[tokenizer] Test encode ({len(test)} chars → {len(ids)} tokens):")
    print(f"  text: {test[:60]}...")
    print(f"  ids:  {ids[:20]}...")
    print(f"  decoded: {sp.decode(ids)[:60]}...")
    
    # Check special tokens
    for st in ["<commit_before>", "<commit_msg>", "<mask>"]:
        sid = sp.piece_to_id(st)
        print(f"  {st} → id={sid}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-lines", type=int, default=500_000)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--model-name", default="neuron_sp")
    args = parser.parse_args()

    txt_path = os.path.join(args.output_dir, "corpus_for_tokenizer.txt")
    model_prefix = os.path.join(args.output_dir, args.model_name)

    # Step 1: Extract text corpus
    ok = extract_text_corpus(txt_path, max_lines=args.max_lines)
    if not ok:
        print("ERROR: could not build corpus")
        sys.exit(1)

    # Step 2: Train sentencepiece
    train_sentencepiece(txt_path, model_prefix, vocab_size=args.vocab_size)

    # Step 3: Print usage
    print(f"\n{'='*60}")
    print(f"Tokenizer ready: {model_prefix}.model")
    print(f"Re-tokenize data:")
    print(f"  python data/prepare_mixed.py --tokenizer {model_prefix}.model")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
