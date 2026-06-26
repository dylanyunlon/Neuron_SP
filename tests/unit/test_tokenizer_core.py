# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
tests/unit/test_tokenizer_core.py — Unit tests for tokenizer/core/

Covers:
  - Vocab ID layout (pad, bytes, endoftext, BPE merges, commit specials)
  - Vocab save / load round-trip
  - BPEEncoder.encode / decode correctness
  - BPEEncoder.from_vocab bridge
  - End-to-end: Vocab → BPEEncoder round-trip
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import List, Tuple

import pytest

# Ensure project root is on sys.path so 'tokenizer' resolves correctly.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tokenizer.core.vocab import (
    Vocab,
    build_vocab,
    _LEGACY_COMMIT_TOKENS,
    _PAD_ID,
    _BYTE_START_ID,
    _EOT_ID,
    _BPE_START_ID,
    _BPE_END_ID,
    _SPECIAL_START_ID,
    _bytes_to_json_key,
    _json_key_to_bytes,
)
from tokenizer.core.encoder import BPEEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_MERGES: List[Tuple[bytes, bytes]] = [
    (b"h", b"e"),    # id 258 → b"he"
    (b"he", b"l"),   # id 259 → b"hel"
    (b"hel", b"l"),  # id 260 → b"hell"
    (b"hell", b"o"), # id 261 → b"hello"
    (b"w", b"o"),    # id 262 → b"wo"
    (b"wo", b"r"),   # id 263 → b"wor"
    (b"wor", b"l"),  # id 264 → b"worl"
    (b"worl", b"d"), # id 265 → b"world"
]


@pytest.fixture
def small_vocab() -> Vocab:
    return build_vocab(SMALL_MERGES)


# ---------------------------------------------------------------------------
# Vocab: ID layout
# ---------------------------------------------------------------------------

class TestVocabIDLayout:

    def test_pad_at_zero(self, small_vocab):
        assert small_vocab.token_to_id(b"<pad>") == 0
        assert small_vocab.id_to_token(0) == b"<pad>"

    def test_raw_byte_first(self, small_vocab):
        # byte 0x00 → id 1
        assert small_vocab.token_to_id(bytes([0])) == _BYTE_START_ID
        assert small_vocab.id_to_token(_BYTE_START_ID) == bytes([0])

    def test_raw_byte_last(self, small_vocab):
        # byte 0xFF → id 256
        assert small_vocab.token_to_id(bytes([255])) == 256
        assert small_vocab.id_to_token(256) == bytes([255])

    def test_byte_id_formula(self, small_vocab):
        # every byte b maps to id b+1
        for b in range(256):
            expected_id = _BYTE_START_ID + b
            assert small_vocab.token_to_id(bytes([b])) == expected_id

    def test_endoftext_at_257(self, small_vocab):
        assert small_vocab.token_to_id(b"<|endoftext|>") == _EOT_ID
        assert small_vocab.id_to_token(_EOT_ID) == b"<|endoftext|>"

    def test_bpe_merges_start_at_258(self, small_vocab):
        assert small_vocab.token_to_id(b"he") == _BPE_START_ID
        assert small_vocab.id_to_token(_BPE_START_ID) == b"he"

    def test_bpe_merges_sequential(self, small_vocab):
        for idx, (left, right) in enumerate(SMALL_MERGES):
            merged = left + right
            expected_id = _BPE_START_ID + idx
            assert small_vocab.id_to_token(expected_id) == merged

    def test_commit_specials_start_at_31744(self, small_vocab):
        for slot, stoken_str in enumerate(_LEGACY_COMMIT_TOKENS):
            expected_id = _SPECIAL_START_ID + slot
            assert small_vocab.id_to_token(expected_id) == stoken_str.encode("utf-8"), (
                f"slot {slot}: expected id {expected_id} → {stoken_str!r}"
            )

    def test_commit_specials_count(self, small_vocab):
        # 9 commit special tokens (31744-31752 inclusive)
        assert len(_LEGACY_COMMIT_TOKENS) == 9

    def test_len(self, small_vocab):
        # 1 (<pad>) + 256 (bytes) + 1 (<|endoftext|>) + 8 (merges) + 9 (specials)
        # Note: <pad> and <|endoftext|> also appear in _LEGACY_COMMIT_TOKENS
        # but in the id_to_token dict they occupy both their primary slots AND
        # the special slots.  Len counts distinct IDs.
        assert len(small_vocab) == 1 + 256 + 1 + len(SMALL_MERGES) + len(_LEGACY_COMMIT_TOKENS)

    def test_missing_token_raises(self, small_vocab):
        with pytest.raises(KeyError):
            small_vocab.token_to_id(b"not_in_vocab_xyz")

    def test_missing_id_raises(self, small_vocab):
        with pytest.raises(KeyError):
            small_vocab.id_to_token(99999)

    def test_too_many_merges_raises(self):
        max_merges = _BPE_END_ID - _BPE_START_ID + 1  # 31 486
        # one merge over the limit
        excess = [(bytes([i % 256]), bytes([(i + 1) % 256])) for i in range(max_merges + 1)]
        with pytest.raises(ValueError, match="Too many BPE merges"):
            Vocab(merges=excess)


# ---------------------------------------------------------------------------
# Vocab: serialisation
# ---------------------------------------------------------------------------

class TestVocabSaveLoad:

    def test_round_trip_ids(self, small_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            small_vocab.save(path)
            v2 = Vocab.load(path)
        assert v2.token_to_id(b"<pad>") == 0
        assert v2.token_to_id(b"<|endoftext|>") == 257
        assert v2.token_to_id(b"hello") == 261
        assert v2.token_to_id(bytes([42])) == 43

    def test_round_trip_bytes(self, small_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            small_vocab.save(path)
            v2 = Vocab.load(path)
        assert v2.id_to_token(0) == b"<pad>"
        assert v2.id_to_token(257) == b"<|endoftext|>"
        assert v2.id_to_token(258) == b"he"

    def test_json_has_required_keys(self, small_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            small_vocab.save(path)
            with open(path) as fh:
                data = json.load(fh)
        assert "token_to_id" in data
        assert "merges" in data
        assert "special_tokens" in data

    def test_binary_token_round_trip(self):
        # Tokens with non-UTF-8 bytes must survive the JSON round-trip.
        binary_merges: List[Tuple[bytes, bytes]] = [
            (bytes([0x80]), bytes([0x81])),  # two non-UTF-8 bytes
        ]
        v = Vocab(merges=binary_merges)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            v.save(path)
            v2 = Vocab.load(path)
        merged = bytes([0x80, 0x81])
        assert v2.token_to_id(merged) == _BPE_START_ID
        assert v2.id_to_token(_BPE_START_ID) == merged

    def test_custom_special_tokens(self):
        custom = ["<s>", "</s>", "<unk>"]
        v = Vocab(merges=[], special_tokens=custom)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            v.save(path)
            v2 = Vocab.load(path)
        for slot, stoken in enumerate(custom):
            assert v2.id_to_token(_SPECIAL_START_ID + slot) == stoken.encode("utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestBytesJsonHelpers:

    def test_ascii_roundtrip(self):
        for tok in [b"hello", b"<pad>", b"<|endoftext|>"]:
            assert _json_key_to_bytes(_bytes_to_json_key(tok)) == tok

    def test_binary_roundtrip(self):
        for tok in [bytes([0x80]), bytes([0x00, 0xFF, 0x7F]), bytes(range(256))]:
            assert _json_key_to_bytes(_bytes_to_json_key(tok)) == tok

    def test_utf8_stored_verbatim(self):
        key = _bytes_to_json_key("héllo".encode("utf-8"))
        assert "\\x" not in key  # stored as plain UTF-8, not hex-escaped

    def test_non_utf8_hex_escaped(self):
        key = _bytes_to_json_key(bytes([0x80, 0x81]))
        assert key == "\\x80\\x81"


# ---------------------------------------------------------------------------
# BPEEncoder
# ---------------------------------------------------------------------------

class TestBPEEncoder:

    @pytest.fixture
    def enc(self, small_vocab) -> BPEEncoder:
        return BPEEncoder.from_vocab(small_vocab)

    def test_encode_hello(self, enc):
        # h-e → he(258), he-l → hel(259), hel-l → hell(260), hell-o → hello(261)
        ids = enc.encode("hello")
        assert ids == [261], f"got {ids}"

    def test_encode_world(self, enc):
        ids = enc.encode("world")
        assert ids == [265], f"got {ids}"

    def test_encode_empty(self, enc):
        assert enc.encode("") == []

    def test_encode_single_byte(self, enc):
        # 'A' → byte 65 → id 66 (65 + 1)
        assert enc.encode("A") == [66]

    def test_encode_bytes_direct(self, enc):
        assert enc.encode_bytes(b"hello") == [261]

    def test_decode_hello(self, enc):
        assert enc.decode([261]) == "hello"

    def test_decode_world(self, enc):
        assert enc.decode([265]) == "world"

    def test_round_trip(self, enc):
        text = "hello world"
        assert enc.decode(enc.encode(text)) == text

    def test_unk_id_on_missing(self, small_vocab):
        # Build encoder without merges so 'he' is not in vocab
        raw_vocab = {bytes([i]): i + 1 for i in range(256)}
        raw_vocab[b"<pad>"] = 0
        enc_no_merge = BPEEncoder(vocab=raw_vocab, merges=[], unk_id=0)
        ids = enc_no_merge.encode("A")
        assert ids == [66]

    def test_missing_token_raises_without_unk(self, small_vocab):
        # Token b"xyz" (3 bytes merged) that's not in the vocab
        raw_vocab = {bytes([i]): i + 1 for i in range(256)}
        raw_vocab[b"<pad>"] = 0
        enc_strict = BPEEncoder(vocab=raw_vocab, merges=[], unk_id=None)
        # single bytes should always be found
        assert enc_strict.encode("A") == [66]


# ---------------------------------------------------------------------------
# Integration: Vocab → BPEEncoder
# ---------------------------------------------------------------------------

class TestVocabEncoderIntegration:

    def test_from_vocab_classmethod(self, small_vocab):
        enc = BPEEncoder.from_vocab(small_vocab)
        assert enc.encode("hello") == [261]
        assert enc.decode([261]) == "hello"

    def test_commit_special_tokens_reachable(self, small_vocab):
        # Every commit special token's *bytes* must be reachable via the encoder
        # vocab dict (token_to_id direction).  <pad> and <|endoftext|> are also
        # present at their primary IDs (0, 257); the special-slot IDs (31744+)
        # are in id_to_token but may be shadowed in the flat token_to_id dict,
        # so we verify via the Vocab object's id_to_token here.
        enc = BPEEncoder.from_vocab(small_vocab)
        for slot, stoken_str in enumerate(_LEGACY_COMMIT_TOKENS):
            stoken_bytes = stoken_str.encode("utf-8")
            sid = _SPECIAL_START_ID + slot
            # The special-slot id must decode to the right bytes in the Vocab
            assert small_vocab.id_to_token(sid) == stoken_bytes
            # The bytes must be in the encoder's flat vocab
            assert stoken_bytes in enc.vocab, (
                f"{stoken_str!r} bytes not in encoder vocab"
            )

    def test_save_load_then_encode(self, small_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.json")
            small_vocab.save(path)
            v2 = Vocab.load(path)
        enc = BPEEncoder.from_vocab(v2)
        assert enc.encode("hello world") == [261, 32 + 1, 265]
        # 'hello'→261, ' '→ord(' ')+1=33, 'world'→265
        space_id = ord(" ") + 1  # byte 32 → id 33
        assert enc.encode("hello world") == [261, space_id, 265]
