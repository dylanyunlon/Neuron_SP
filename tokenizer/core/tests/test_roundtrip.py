# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
tokenizer/core/tests/test_roundtrip.py — encode → decode lossless roundtrip tests.

For every test case:
    assert decode(encode(text)) == text

The encoder under test is BPEEncoder, exercised through a minimal but realistic
vocab (256 raw-byte tokens + the commit special tokens registered in vocab.py).
No external files or trained merges are required; the fixture builds the vocab
programmatically so the tests are fully self-contained and fast.

Test cases
----------
1. Pure ASCII           : "hello world"
2. Python source code   : "def f(x):\n    return x + 1\n"
3. Commit diff chunk    : "+    new_line\n-    old_line\n"
4. Special tokens       : "<|diff_start|>code<|old|>"
5. Unicode / CJK        : "变量 = 42"
6. Empty string         : ""
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Tuple

from tokenizer.core.encoder import BPEEncoder, Vocab, Merges
from tokenizer.core.vocab import _LEGACY_COMMIT_TOKENS


# ---------------------------------------------------------------------------
# Fixture: build a minimal encoder that covers every test case
# ---------------------------------------------------------------------------

def _build_minimal_encoder() -> BPEEncoder:
    """
    Construct a BPEEncoder with:

    * IDs 1-256  : the 256 raw single-byte tokens (byte 0x00 → id 1, …
                   byte 0xFF → id 256), matching the Neuron-SP vocab layout.
    * IDs 31744+ : the commit special tokens from ``_LEGACY_COMMIT_TOKENS``,
                   stored as their UTF-8 byte representations so that
                   encode() can match them after BPE tokenises the input.
    * No learned BPE merges — bytes pass through as-is, which is the
      correct baseline for a roundtrip test (any merge that can be applied
      can also be reversed through the inverse vocab lookup).

    The special tokens are inserted *before* the raw-byte fallback lookup
    so that the longest-match heuristic in the test helper finds them first.
    """
    vocab: Vocab = {}

    # Raw byte tokens: byte value b → id (b + 1) so that id 0 remains <pad>.
    for b in range(256):
        vocab[bytes([b])] = b + 1

    # Commit special tokens (ids 31744 …).
    special_start = 31744
    for offset, token_str in enumerate(_LEGACY_COMMIT_TOKENS):
        token_bytes = token_str.encode("utf-8")
        vocab[token_bytes] = special_start + offset

    merges: Merges = []  # no merges needed for roundtrip correctness
    return BPEEncoder(vocab=vocab, merges=merges)


# ---------------------------------------------------------------------------
# Helper: encode text with special-token pre-segmentation
# ---------------------------------------------------------------------------

def _encode(encoder: BPEEncoder, text: str) -> List[int]:
    """
    Encode *text*, handling commit special tokens as atomic units.

    BPEEncoder.encode() processes raw UTF-8 bytes and cannot natively
    recognise multi-byte special tokens like ``<|diff_start|>`` as a
    single vocabulary entry (they would be split byte-by-byte and each
    byte looked up individually).

    This helper:
    1. Splits *text* into segments: either a known special token string
       (looked up as a whole) or a run of ordinary text (passed to the
       standard encoder).
    2. Concatenates the resulting ID lists.

    This mirrors how production tokenisers handle special tokens and is
    the correct way to exercise the vocab entries at ids 31744+.
    """
    # Build sorted list of special token strings (longest first to avoid
    # greedy prefix ambiguity, e.g. "<|old|>" vs a hypothetical "<|o|>").
    special_tokens: List[str] = sorted(
        _LEGACY_COMMIT_TOKENS, key=len, reverse=True
    )

    # Build a fast bytes→id lookup for special tokens.
    special_vocab: Dict[bytes, int] = {
        tok.encode("utf-8"): encoder.vocab[tok.encode("utf-8")]
        for tok in special_tokens
        if tok.encode("utf-8") in encoder.vocab
    }

    ids: List[int] = []
    i = 0
    while i < len(text):
        matched = False
        for tok_str in special_tokens:
            if text.startswith(tok_str, i):
                tok_bytes = tok_str.encode("utf-8")
                if tok_bytes in special_vocab:
                    ids.append(special_vocab[tok_bytes])
                    i += len(tok_str)
                    matched = True
                    break
        if not matched:
            # Find the next special token boundary (or end of string).
            next_special = len(text)
            for tok_str in special_tokens:
                pos = text.find(tok_str, i)
                if pos != -1 and pos < next_special:
                    next_special = pos
            segment = text[i:next_special]
            ids.extend(encoder.encode(segment))
            i = next_special

    return ids


def _decode(encoder: BPEEncoder, ids: List[int]) -> str:
    """Thin wrapper around BPEEncoder.decode for symmetry with _encode."""
    return encoder.decode(ids)


# ---------------------------------------------------------------------------
# Shared encoder fixture (module-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def encoder() -> BPEEncoder:
    return _build_minimal_encoder()


# ---------------------------------------------------------------------------
# Parametrised roundtrip tests
# ---------------------------------------------------------------------------

TEST_CASES: List[Tuple[str, str]] = [
    ("pure_ascii",       "hello world"),
    ("python_code",      "def f(x):\n    return x + 1\n"),
    ("commit_diff",      "+    new_line\n-    old_line\n"),
    ("special_tokens",   "<|diff_start|>code<|old|>"),
    ("unicode_cjk",      "变量 = 42"),
    ("empty_string",     ""),
]


@pytest.mark.parametrize("label,text", TEST_CASES, ids=[t[0] for t in TEST_CASES])
def test_roundtrip(encoder: BPEEncoder, label: str, text: str) -> None:
    """encode → decode must reproduce the original text exactly."""
    ids = _encode(encoder, text)
    recovered = _decode(encoder, ids)
    assert recovered == text, (
        f"[{label}] Roundtrip failed.\n"
        f"  original : {text!r}\n"
        f"  ids      : {ids}\n"
        f"  recovered: {recovered!r}"
    )


# ---------------------------------------------------------------------------
# Additional invariant: encode("") must return []
# ---------------------------------------------------------------------------

def test_encode_empty_returns_empty_list(encoder: BPEEncoder) -> None:
    """Encoding the empty string must return an empty list (not raise)."""
    assert _encode(encoder, "") == []


# ---------------------------------------------------------------------------
# Additional invariant: decode([]) must return ""
# ---------------------------------------------------------------------------

def test_decode_empty_list_returns_empty_string(encoder: BPEEncoder) -> None:
    """Decoding an empty ID list must return the empty string."""
    assert _decode(encoder, []) == ""
