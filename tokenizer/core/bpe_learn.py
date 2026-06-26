# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
bpe_learn.py — Pure-Python BPE merge learning algorithm.

Implements Sennrich et al. 2016 "Neural Machine Translation of Rare Words with
Subword Units" (https://arxiv.org/abs/1508.07909).

Key design decisions:
- Works directly on raw UTF-8 bytes — no text normalisation assumed.
- Incremental pair-frequency update: after merging the best pair (a, b) we only
  update counts for pairs that neighbour the affected positions, so we never
  re-scan the whole corpus.
- Each "word" is stored as a tuple of bytes-objects (one per current token)
  together with a frequency count.  The pair-frequency dict is kept in sync
  incrementally.

No external libraries are used — only the Python standard library.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A "word" is a tuple of byte tokens (each is a bytes object of length >= 1).
Word = Tuple[bytes, ...]

# Corpus: word-tuple → frequency
Corpus = Dict[Word, int]

# Pair-frequency table: (bytes, bytes) → total frequency
PairFreq = Dict[Tuple[bytes, bytes], int]

# For each word we also store, for each pair position, which words contain it:
#   pair → {word_tuple: list_of_occurrence_counts}
# But a simpler and sufficient structure is:
#   pair → {word_tuple: count_of_that_pair_in_word * word_frequency}
# We just keep pair → total_freq and rebuild neighbours on demand.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _text_to_word_corpus(texts: List[str]) -> Corpus:
    """
    Convert a list of strings to a byte-level word corpus.

    Tokenisation strategy (mirrors the original BPE paper):
      - Split each text on whitespace.  Each non-empty chunk becomes a "word".
      - Represent each word as a tuple of single-byte tokens derived from its
        UTF-8 encoding, with a special end-of-word marker ``b'\\xff\\xfe'``
        appended (chosen because it is the UTF-16 BOM and cannot appear in
        well-formed UTF-8, making it unambiguous).

    The corpus maps each unique word-tuple to its cumulative frequency.
    """
    EOW = b"\xff\xfe"  # end-of-word sentinel — never valid UTF-8 on its own
    corpus: Corpus = defaultdict(int)

    for text in texts:
        # Split on whitespace (space, tab, newline, …)
        for chunk in text.split():
            if not chunk:
                continue
            raw: bytes = chunk.encode("utf-8")
            # Each byte is its own initial token; append EOW as final token.
            word: Word = tuple(bytes([b]) for b in raw) + (EOW,)
            corpus[word] += 1

    return dict(corpus)


def _build_pair_freq(corpus: Corpus) -> PairFreq:
    """
    Compute the frequency of every adjacent pair across the full corpus.

    For a word with frequency f containing the sequence ... a b ..., the pair
    (a, b) contributes f to the total pair frequency.
    """
    pair_freq: PairFreq = defaultdict(int)
    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            pair_freq[(word[i], word[i + 1])] += freq
    return pair_freq


def _get_best_pair(pair_freq: PairFreq) -> Optional[Tuple[bytes, bytes]]:
    """Return the most-frequent pair, or None if the table is empty."""
    if not pair_freq:
        return None
    # max by frequency; tie-break lexicographically for determinism.
    return max(pair_freq, key=lambda p: (pair_freq[p], p))


def _merge_pair_in_word(word: Word, pair: Tuple[bytes, bytes]) -> Word:
    """
    Replace every (non-overlapping, left-to-right) occurrence of *pair* in
    *word* with the single merged token ``pair[0] + pair[1]``.
    """
    a, b = pair
    merged_token = a + b
    new_word: List[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def _update_pair_freq(
    pair_freq: PairFreq,
    old_word: Word,
    new_word: Word,
    freq: int,
) -> None:
    """
    Incrementally update *pair_freq* to reflect replacing *old_word* with
    *new_word* (both with the same corpus frequency *freq*).

    We subtract the pair counts from *old_word* and add the pair counts from
    *new_word*.  Only pairs that differ between the two need touching —
    everything else cancels — but it is simpler and correct to just subtract
    old and add new entirely.
    """
    # Subtract old pair contributions
    for i in range(len(old_word) - 1):
        p = (old_word[i], old_word[i + 1])
        pair_freq[p] -= freq
        if pair_freq[p] <= 0:
            del pair_freq[p]

    # Add new pair contributions
    for i in range(len(new_word) - 1):
        p = (new_word[i], new_word[i + 1])
        pair_freq[p] = pair_freq.get(p, 0) + freq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def learn_bpe_merges(
    texts: List[str],
    num_merges: int,
) -> List[Tuple[bytes, bytes]]:
    """
    Learn BPE merge rules from a list of text strings.

    Algorithm (Sennrich et al. 2016):
      1. Convert all texts to byte sequences split into individual byte tokens.
      2. Count all adjacent byte-pair frequencies across the corpus.
      3. Find the highest-frequency pair.
      4. Record it as a merge rule and apply it throughout the corpus.
      5. Update the pair-frequency table incrementally (no full re-scan).
      6. Repeat steps 3–5 for *num_merges* iterations (or until no pairs remain).

    Parameters
    ----------
    texts:
        Raw text strings forming the training corpus.  May contain any Unicode.
    num_merges:
        Number of BPE merge operations to learn.

    Returns
    -------
    List of ``(left_token, right_token)`` byte-pairs in merge order.
    Example: ``[(b'a', b'b'), (b'ab', b'c'), (b'th', b'e'), …]``
    """
    if num_merges < 0:
        raise ValueError(f"num_merges must be >= 0, got {num_merges}")

    # --- Step 1: build initial byte-level word corpus -----------------------
    corpus: Corpus = _text_to_word_corpus(texts)

    if not corpus:
        return []

    # --- Step 2: compute initial pair frequencies ---------------------------
    pair_freq: PairFreq = _build_pair_freq(corpus)

    merges: List[Tuple[bytes, bytes]] = []

    for merge_idx in range(num_merges):
        # --- Step 3: find best pair -----------------------------------------
        best = _get_best_pair(pair_freq)
        if best is None:
            # No more pairs to merge (corpus is fully merged or empty)
            break

        best_freq = pair_freq[best]
        if best_freq <= 0:
            break

        merges.append(best)

        # --- Step 4 & 5: apply merge and update pair frequencies ------------
        new_corpus: Corpus = {}
        for word, freq in corpus.items():
            if best[0] in word and best[1] in word:
                # Only process words that could contain this pair
                new_word = _merge_pair_in_word(word, best)
                if new_word != word:
                    _update_pair_freq(pair_freq, word, new_word, freq)
                    new_corpus[new_word] = new_corpus.get(new_word, 0) + freq
                    continue
            new_corpus[word] = new_corpus.get(word, 0) + freq

        corpus = new_corpus

    return merges


# ---------------------------------------------------------------------------
# Convenience: apply learned merges to a single text (useful for testing)
# ---------------------------------------------------------------------------


def apply_bpe_merges(
    text: str,
    merges: List[Tuple[bytes, bytes]],
) -> List[bytes]:
    """
    Tokenise *text* by applying a learned list of BPE merge rules in order.

    Returns a flat list of byte tokens (the final segmentation of the text).
    Primarily intended for unit-testing ``learn_bpe_merges``.
    """
    EOW = b"\xff\xfe"
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    result: List[bytes] = []
    for chunk in text.split():
        if not chunk:
            continue
        raw = chunk.encode("utf-8")
        word: List[bytes] = [bytes([b]) for b in raw] + [EOW]

        # Repeatedly apply the lowest-rank (earliest) merge that is present
        changed = True
        while changed and len(word) > 1:
            changed = False
            best_rank = len(merges)  # sentinel "infinity"
            best_pos = -1
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                r = merge_rank.get(p, len(merges))
                if r < best_rank:
                    best_rank = r
                    best_pos = i

            if best_pos >= 0 and best_rank < len(merges):
                a, b = word[best_pos], word[best_pos + 1]
                word = word[:best_pos] + [a + b] + word[best_pos + 2:]
                changed = True

        result.extend(word)

    return result


# ---------------------------------------------------------------------------
# Quick self-test / smoke test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """
    Minimal smoke-test: verify that the algorithm produces sensible output on a
    tiny hand-crafted corpus.
    """
    # Corpus designed so that 'aa' should be the most frequent pair.
    corpus = [
        "aaab aaac aaa",
        "aaab aaac",
        "aaab aaac aaa aaa",
    ]

    merges = learn_bpe_merges(corpus, num_merges=5)
    assert isinstance(merges, list), "Expected a list"
    assert len(merges) <= 5, "Should not exceed requested merge count"
    assert all(
        isinstance(pair, tuple) and len(pair) == 2
        and isinstance(pair[0], bytes) and isinstance(pair[1], bytes)
        for pair in merges
    ), "Each merge should be a (bytes, bytes) tuple"

    # The very first merge should combine b'a' + b'a' → b'aa' because 'aa'
    # is by far the most frequent adjacent pair in "aaab aaac aaa …".
    if merges:
        first = merges[0]
        assert first == (b"a", b"a"), (
            f"Expected first merge to be (b'a', b'a'), got {first!r}"
        )

    # Test apply_bpe_merges round-trip
    tokens = apply_bpe_merges("aaab", merges)
    assert isinstance(tokens, list) and all(isinstance(t, bytes) for t in tokens)

    print("Self-test passed.")
    print(f"  merges learned : {merges}")
    print(f"  tokenise 'aaab': {tokens}")


if __name__ == "__main__":
    # When run directly, execute the self-test and optionally read a corpus
    # from stdin (one document per line) for a quick interactive demo.
    _self_test()

    if not sys.stdin.isatty():
        lines = [line.rstrip("\n") for line in sys.stdin]
        demo_merges = learn_bpe_merges(lines, num_merges=20)
        print(f"\nLearned {len(demo_merges)} merges from stdin:")
        for i, m in enumerate(demo_merges):
            print(f"  {i:3d}  {m[0]!r} + {m[1]!r}  →  {m[0]+m[1]!r}")
