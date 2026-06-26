# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
tokenizer/core/encoder.py — Pure-Python GPT-2-style BPE Encoder

Architecture
------------
BPEEncoder wraps a Vocab (bytes → int) and an ordered merge list to
implement byte-level BPE encoding identical to GPT-2 / tiktoken behaviour:

  text  ──► UTF-8 bytes  ──► List[bytes]  ──► BPE merges  ──► List[int]

Merge policy (GPT-2 greedy, NOT left-to-right):
  Each round, scan *all* adjacent pairs, find the one with the lowest
  merge rank (highest priority), merge every non-overlapping occurrence of
  that pair, then repeat until no pair has a known rank.

Types
-----
  Vocab  = Dict[bytes, int]          — byte-sequence → token ID
  Merges = List[Tuple[bytes, bytes]] — ordered BPE merge rules
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------
Vocab = Dict[bytes, int]
Merges = List[Tuple[bytes, bytes]]


# ---------------------------------------------------------------------------
# BPEEncoder
# ---------------------------------------------------------------------------

class BPEEncoder:
    """
    Byte-level BPE encoder.

    Parameters
    ----------
    vocab:
        Mapping from byte-sequences (tokens) to integer IDs.
        Must contain at least the 256 single-byte tokens.
    merges:
        Ordered list of BPE merge rules ``(left, right)``.
        Index 0 = highest priority merge.
    unk_id:
        ID returned when a token is not found in *vocab*.
        Defaults to ``None`` (raises KeyError on missing tokens).
    """

    def __init__(
        self,
        vocab: Vocab,
        merges: Merges,
        unk_id: Optional[int] = None,
    ) -> None:
        self.vocab: Vocab = vocab
        self.merges: Merges = merges
        self.unk_id: Optional[int] = unk_id

        # merge pair → rank (lower rank = applied first)
        self.merge_ranks: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_vocab(cls, neuron_vocab: "Any", unk_id: Optional[int] = None) -> "BPEEncoder":
        """Construct a :class:`BPEEncoder` from a :class:`vocab.Vocab` instance.

        This bridges the structured ``Vocab`` object (which owns the ID layout)
        and the flat ``Dict[bytes, int]`` that ``BPEEncoder.__init__`` expects,
        so callers do not have to manually unpack the vocab internals.

        Parameters
        ----------
        neuron_vocab:
            A ``tokenizer.core.vocab.Vocab`` instance.
        unk_id:
            Passed through to ``BPEEncoder.__init__``.
        """
        # neuron_vocab._token_to_id is the flat Dict[bytes, int] we need.
        flat_vocab: Vocab = dict(neuron_vocab._token_to_id)
        merges: Merges = list(neuron_vocab._merges)
        return cls(vocab=flat_vocab, merges=merges, unk_id=unk_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Encode *text* to a list of token IDs.

        Steps
        -----
        1. UTF-8 encode → one bytes-object per byte.
        2. Apply BPE merges (GPT-2 greedy strategy).
        3. Map each resulting token to its vocab ID.

        Parameters
        ----------
        text:
            Input string (arbitrary Unicode).

        Returns
        -------
        List[int]
            Token IDs.

        Raises
        ------
        KeyError
            If a token is absent from *vocab* and *unk_id* is ``None``.
        """
        if not text:
            return []

        # Step 1: UTF-8 bytes → list of single-byte tokens
        tokens: List[bytes] = [bytes([b]) for b in text.encode("utf-8")]

        # Step 2: greedy BPE merges
        tokens = self._apply_merges(tokens)

        # Step 3: vocab lookup
        ids: List[int] = []
        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            elif self.unk_id is not None:
                ids.append(self.unk_id)
            else:
                raise KeyError(
                    f"Token {tok!r} not found in vocab. "
                    "Pass unk_id= to suppress this error."
                )
        return ids

    def encode_bytes(self, data: bytes) -> List[int]:
        """
        Encode raw *data* bytes directly (skips UTF-8 step).

        Useful for binary payloads or pre-encoded data.
        """
        if not data:
            return []
        tokens = [bytes([b]) for b in data]
        tokens = self._apply_merges(tokens)
        ids: List[int] = []
        for tok in tokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            elif self.unk_id is not None:
                ids.append(self.unk_id)
            else:
                raise KeyError(f"Token {tok!r} not found in vocab.")
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to a string.

        Builds the reverse vocab on first call (cached).

        Parameters
        ----------
        ids:
            List of token IDs as returned by :meth:`encode`.

        Returns
        -------
        str
            Decoded string.  Undecodable byte sequences are replaced with
            the Unicode replacement character (``errors='replace'``).
        """
        if not hasattr(self, "_id_to_token"):
            self._id_to_token: Dict[int, bytes] = {
                v: k for k, v in self.vocab.items()
            }
        raw = b"".join(self._id_to_token.get(i, b"") for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Core BPE merge logic
    # ------------------------------------------------------------------

    def _apply_merges(self, tokens: List[bytes]) -> List[bytes]:
        """
        Apply BPE merges to *tokens* using the GPT-2 greedy strategy.

        Algorithm
        ---------
        Repeat until no adjacent pair has a known merge rank:

        1. Scan all adjacent pairs ``(tokens[i], tokens[i+1])``.
        2. Among those present in ``merge_ranks``, pick the one with the
           **lowest rank** (= highest priority, = earliest in the merge
           list).
        3. Merge **every non-overlapping occurrence** of that pair from
           left to right, producing ``left + right`` as a single token.
        4. Repeat from step 1 with the updated token list.

        This is O(n * M) in the worst case (n = token length, M = number
        of applicable merges), but identical in output to the reference
        GPT-2 tokenizer.

        Parameters
        ----------
        tokens:
            Initial list of single-byte tokens (each a ``bytes`` of
            length 1) or partially merged tokens.

        Returns
        -------
        List[bytes]
            Fully merged token list.
        """
        if len(tokens) < 2:
            return tokens

        while True:
            # --- find best pair this round ---
            best_pair: Optional[Tuple[bytes, bytes]] = None
            best_rank: int = sys.maxsize

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                # No more applicable merges.
                break

            # --- merge all non-overlapping occurrences of best_pair ---
            tokens = self._merge_pair(tokens, best_pair)

        return tokens

    @staticmethod
    def _merge_pair(
        tokens: List[bytes],
        pair: Tuple[bytes, bytes],
    ) -> List[bytes]:
        """
        Merge every non-overlapping occurrence of *pair* in *tokens*
        from left to right.

        Parameters
        ----------
        tokens:
            Current token list.
        pair:
            ``(left, right)`` to merge into ``left + right``.

        Returns
        -------
        List[bytes]
            Token list with all occurrences of *pair* merged.
        """
        left, right = pair
        merged = left + right
        result: List[bytes] = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == left and tokens[i + 1] == right:
                result.append(merged)
                i += 2  # skip both members of the pair
            else:
                result.append(tokens[i])
                i += 1
        return result


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_encoder_from_json(tokenizer_json_path: str) -> BPEEncoder:
    """
    Build a :class:`BPEEncoder` from a HuggingFace ``tokenizer.json`` file
    (as produced by ``tokenizers``).

    Parameters
    ----------
    tokenizer_json_path:
        Path to ``tokenizer.json``.

    Returns
    -------
    BPEEncoder
    """
    import json

    with open(tokenizer_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    model = data.get("model", {})

    # ----- vocab -----
    raw_vocab: Dict[str, int] = model.get("vocab", {})
    # HF byte-level BPE stores token strings; we need to convert to bytes.
    # The HF ByteLevel pre-tokenizer uses a unicode <-> byte mapping.
    byte_decoder = _build_byte_decoder()
    vocab: Vocab = {}
    for token_str, token_id in raw_vocab.items():
        token_bytes = bytes([byte_decoder[c] for c in token_str])
        vocab[token_bytes] = token_id

    # ----- merges -----
    raw_merges: List[str] = model.get("merges", [])
    merges: Merges = []
    for merge_str in raw_merges:
        parts = merge_str.split(" ", 1)
        if len(parts) == 2:
            left = bytes([byte_decoder[c] for c in parts[0]])
            right = bytes([byte_decoder[c] for c in parts[1]])
            merges.append((left, right))

    return BPEEncoder(vocab=vocab, merges=merges)


def _build_byte_decoder() -> Dict[str, int]:
    """
    Build the GPT-2 unicode→byte decoder table.

    GPT-2's ByteLevel representation maps each byte 0–255 to a printable
    unicode character so that the vocabulary can be stored as plain text.
    """
    # Printable ASCII ranges that map to themselves
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # cs[i] is the unicode code-point for byte bs[i]
    return {chr(c): b for b, c in zip(bs, cs)}


# ---------------------------------------------------------------------------
# Minimal self-test (python -m tokenizer.core.encoder)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Smoke-test BPEEncoder with a hand-crafted mini vocabulary."""
    # Build a tiny vocab: 256 raw bytes + 2 merge tokens
    vocab: Vocab = {}
    for i in range(256):
        vocab[bytes([i])] = i

    # Teach the encoder: 'h'+'e' → 'he' (id=256), 'he'+'l' → 'hel' (id=257)
    vocab[b"he"] = 256
    vocab[b"hel"] = 257

    merges: Merges = [
        (b"h", b"e"),   # rank 0 — highest priority
        (b"he", b"l"),  # rank 1
    ]

    enc = BPEEncoder(vocab=vocab, merges=merges)

    # Encode "hello"
    ids = enc.encode("hello")
    # Expected: 'hel'(257)  'l'(108)  'o'(111)
    assert ids == [257, 108, 111], f"encode failed: {ids}"

    # Round-trip
    decoded = enc.decode(ids)
    assert decoded == "hello", f"decode failed: {decoded!r}"

    # Empty string
    assert enc.encode("") == []

    # Single byte
    assert enc.encode("A") == [65]

    print("BPEEncoder self-test PASSED ✓")
    print(f"  'hello' → ids {ids} → {decoded!r}")


if __name__ == "__main__":
    _self_test()
