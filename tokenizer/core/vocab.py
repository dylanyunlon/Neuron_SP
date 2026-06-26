# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
tokenizer/core/vocab.py — BPE vocab table with bidirectional token↔id lookup.

ID layout (32 000 total slots, inclusive):
┌───────────────────────────────────────────────────────────────────┐
│  id     0          : <pad>                                        │
│  ids    1 –   256  : raw bytes 0x00 – 0xFF  (byte_0 … byte_255)  │
│  id   257          : <|endoftext|>  (also kept in specials below) │
│  ids  258 – 31743  : learned BPE merge tokens                    │
│  ids 31744 – 31752 : commit special tokens (from _LEGACY_…)      │
└───────────────────────────────────────────────────────────────────┘

The 9 commit special tokens (ids 31744-31752) mirror
``pipeline.unified_tokenizer._LEGACY_COMMIT_TOKENS`` plus ``<pad>``
(which is already at id 0 but included for completeness in the special
token registry).

Serialisation: JSON with two top-level keys:
  "token_to_id" : { "<token-str-or-hex-escaped-bytes>" : int, … }
  "merges"      : [ ["part_a", "part_b"], … ]   (in merge order)

Byte tokens are stored as their raw hex representation ``\\xNN`` when
they are not valid UTF-8 to ensure lossless round-trip through JSON.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Commit special tokens — mirrors pipeline/unified_tokenizer._LEGACY_COMMIT_TOKENS
# ---------------------------------------------------------------------------
_LEGACY_COMMIT_TOKENS: List[str] = [
    "<|diff_start|>",
    "<|diff_end|>",
    "<|old|>",
    "<|new|>",
    "<|commit_msg|>",
    "<|file_path|>",
    "<|lang|>",
    "<|endoftext|>",
    "<pad>",
]

# IDs for the fixed prefix of the vocab layout
_PAD_ID: int = 0
_BYTE_START_ID: int = 1         # byte 0x00 → id 1, byte 0xFF → id 256
_EOT_ID: int = 257              # <|endoftext|>
_BPE_START_ID: int = 258        # first learned BPE merge
_BPE_END_ID: int = 31743        # last  learned BPE merge (inclusive)
_SPECIAL_START_ID: int = 31744  # first commit special token


# ---------------------------------------------------------------------------
# Helpers for encoding / decoding token bytes ↔ JSON-safe strings
# ---------------------------------------------------------------------------

def _bytes_to_json_key(token: bytes) -> str:
    """Encode *token* bytes to a JSON-safe string.

    Printable ASCII and valid UTF-8 strings are stored verbatim.
    Everything else is hex-escaped as ``\\xNN`` sequences so that the
    round-trip through JSON is lossless.
    """
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{b:02x}" for b in token)


def _json_key_to_bytes(key: str) -> bytes:
    """Inverse of :func:`_bytes_to_json_key`."""
    if "\\x" not in key:
        return key.encode("utf-8")
    # Parse hex-escape sequences
    result = bytearray()
    i = 0
    while i < len(key):
        if key[i : i + 2] == "\\x" and i + 4 <= len(key):
            result.append(int(key[i + 2 : i + 4], 16))
            i += 4
        else:
            result.extend(key[i].encode("utf-8"))
            i += 1
    return bytes(result)


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

class Vocab:
    """BPE vocabulary with bidirectional token↔id lookup.

    ID layout
    ---------
    ``id 0``
        ``<pad>`` — padding sentinel.
    ``ids 1-256``
        Raw byte tokens: byte ``b`` maps to id ``b + 1``.
    ``id 257``
        ``<|endoftext|>`` — document boundary.
    ``ids 258-31743``
        Learned BPE merge tokens, ordered by merge priority (earlier
        merges get lower IDs).
    ``ids 31744-31752``
        Commit special tokens from
        ``pipeline.unified_tokenizer._LEGACY_COMMIT_TOKENS``
        (9 tokens total, including ``<pad>`` and ``<|endoftext|>``
        which are also referenced at their canonical IDs 0 and 257).

    Parameters
    ----------
    merges:
        Ordered list of BPE merge pairs ``(left_bytes, right_bytes)``.
        The first merge pair gets ID 258, the second 259, etc.
    special_tokens:
        List of special-token *strings* (typically
        ``_LEGACY_COMMIT_TOKENS``).  Overrides default if supplied.
    """

    def __init__(
        self,
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        if special_tokens is None:
            special_tokens = _LEGACY_COMMIT_TOKENS

        self._merges: List[Tuple[bytes, bytes]] = list(merges)
        self._special_token_strings: List[str] = list(special_tokens)

        # Forward map:  token bytes → int id
        self._token_to_id: Dict[bytes, int] = {}
        # Reverse map:  int id → token bytes
        self._id_to_token: Dict[int, bytes] = {}

        self._build_tables()

    # ------------------------------------------------------------------
    # Internal table construction
    # ------------------------------------------------------------------

    def _build_tables(self) -> None:
        """Populate ``_token_to_id`` and ``_id_to_token`` from scratch."""
        t2i = self._token_to_id
        i2t = self._id_to_token

        t2i.clear()
        i2t.clear()

        # id 0 → <pad>
        pad_bytes = b"<pad>"
        t2i[pad_bytes] = _PAD_ID
        i2t[_PAD_ID] = pad_bytes

        # ids 1-256 → raw bytes
        for byte_val in range(256):
            token = bytes([byte_val])
            token_id = _BYTE_START_ID + byte_val   # 1 … 256
            t2i[token] = token_id
            i2t[token_id] = token

        # id 257 → <|endoftext|>
        eot_bytes = b"<|endoftext|>"
        t2i[eot_bytes] = _EOT_ID
        i2t[_EOT_ID] = eot_bytes

        # ids 258-31743 → learned BPE merge tokens
        max_merges = _BPE_END_ID - _BPE_START_ID + 1  # 31 486
        if len(self._merges) > max_merges:
            raise ValueError(
                f"Too many BPE merges: got {len(self._merges)}, "
                f"max allowed is {max_merges} (ids {_BPE_START_ID}-{_BPE_END_ID})."
            )
        for idx, (left, right) in enumerate(self._merges):
            merged: bytes = left + right
            token_id = _BPE_START_ID + idx
            # Byte tokens can be overridden by BPE merges in higher slots
            if merged not in t2i:
                t2i[merged] = token_id
            i2t[token_id] = merged

        # ids 31744+ → commit special tokens
        for slot, stoken_str in enumerate(self._special_token_strings):
            token_id = _SPECIAL_START_ID + slot
            stoken_bytes = stoken_str.encode("utf-8")
            # Allow the special slot to shadow any earlier mapping for the
            # same byte sequence (the canonical IDs 0 and 257 remain
            # primary for <pad> and <|endoftext|>; the special-slot entries
            # serve as named aliases at the high end of the range).
            t2i[stoken_bytes] = t2i.get(stoken_bytes, token_id)
            i2t[token_id] = stoken_bytes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def token_to_id(self, token: bytes) -> int:
        """Return the vocabulary ID for *token*.

        Raises
        ------
        KeyError
            If *token* is not in the vocabulary.
        """
        try:
            return self._token_to_id[token]
        except KeyError:
            raise KeyError(f"Token not in vocabulary: {token!r}") from None

    def id_to_token(self, id: int) -> bytes:  # noqa: A002
        """Return the token bytes for vocabulary *id*.

        Raises
        ------
        KeyError
            If *id* is outside the known vocabulary range.
        """
        try:
            return self._id_to_token[id]
        except KeyError:
            raise KeyError(f"ID not in vocabulary: {id}") from None

    def __len__(self) -> int:
        """Number of *distinct* IDs registered (may differ from max_id+1)."""
        return len(self._id_to_token)

    @property
    def vocab_size(self) -> int:
        """Alias for ``len(self)``."""
        return len(self)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the vocabulary to *path* as a JSON file.

        The JSON contains two top-level keys:

        ``token_to_id``
            Mapping from a JSON-safe string representation of the token
            bytes to its integer id.  Byte tokens that are not valid
            UTF-8 are hex-escaped (``\\xNN``).
        ``merges``
            Ordered list of ``[left_hex, right_hex]`` pairs where each
            element is the hex encoding of the byte sequence.  Storing
            as hex keeps the JSON unambiguous regardless of encoding.
        ``special_tokens``
            The list of commit special-token strings.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        serial_t2i: Dict[str, int] = {
            _bytes_to_json_key(tok): tid
            for tok, tid in self._token_to_id.items()
        }
        serial_merges: List[List[str]] = [
            [left.hex(), right.hex()]
            for left, right in self._merges
        ]

        payload = {
            "token_to_id": serial_t2i,
            "merges": serial_merges,
            "special_tokens": self._special_token_strings,
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Reconstruct a :class:`Vocab` from a JSON file written by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the JSON file produced by :meth:`save`.

        Returns
        -------
        Vocab
            Fully initialised vocabulary object.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        merges: List[Tuple[bytes, bytes]] = [
            (bytes.fromhex(left_hex), bytes.fromhex(right_hex))
            for left_hex, right_hex in payload["merges"]
        ]
        special_tokens: List[str] = payload.get(
            "special_tokens", _LEGACY_COMMIT_TOKENS
        )
        return cls(merges=merges, special_tokens=special_tokens)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_vocab(
    merges: List[Tuple[bytes, bytes]],
    special_tokens: Optional[List[str]] = None,
) -> Vocab:
    """Construct a :class:`Vocab` with the standard Neuron-SP ID layout.

    Parameters
    ----------
    merges:
        Ordered BPE merge pairs, exactly as produced by the BPE trainer.
    special_tokens:
        Override the default ``_LEGACY_COMMIT_TOKENS`` list.

    Returns
    -------
    Vocab
    """
    return Vocab(merges=merges, special_tokens=special_tokens)
