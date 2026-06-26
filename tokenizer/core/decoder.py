# SPDX-License-Identifier: Apache-2.0
# Neuron-SP Team
"""
tokenizer/core/decoder.py — Pure-Python BPE decoder.

Converts a sequence of vocabulary IDs back to a Unicode string by
looking up each ID in the :class:`~tokenizer.core.vocab.Vocab` table,
concatenating the raw byte payloads, and decoding the result as UTF-8.

Special IDs that are silently skipped
--------------------------------------
``id 0``   ``<pad>``          — padding sentinel, carries no content.
``id 257`` ``<|endoftext|>``  — document-boundary marker, carries no content.
``ids 31744-31752``           — commit special tokens (diff markers, etc.);
                                these delimit structure but are not part of
                                the decoded surface string.

All other IDs are resolved through :meth:`~tokenizer.core.vocab.Vocab.id_to_token`
and their byte payloads are appended in order.  Unknown IDs (not present in
the vocabulary) are silently skipped rather than raising an exception, so
that a partially-corrupt token stream still yields a best-effort decode.
"""

from __future__ import annotations

from typing import Iterable, List

from tokenizer.core.vocab import (
    Vocab,
    _EOT_ID,
    _PAD_ID,
    _SPECIAL_START_ID,
)

# IDs that carry structural / padding semantics and must be excluded from
# the decoded surface text.
_SKIP_IDS: frozenset[int] = frozenset(
    [_PAD_ID, _EOT_ID] + list(range(_SPECIAL_START_ID, _SPECIAL_START_ID + 9))
)


class BPEDecoder:
    """Decode a sequence of BPE token IDs to a Unicode string.

    Parameters
    ----------
    vocab:
        A fully initialised :class:`~tokenizer.core.vocab.Vocab` instance
        whose :meth:`~tokenizer.core.vocab.Vocab.id_to_token` method will
        be used for ID→bytes resolution.

    Examples
    --------
    >>> from tokenizer.core.vocab import Vocab
    >>> vocab = Vocab(merges=[])
    >>> decoder = BPEDecoder(vocab)
    >>> decoder.decode([72, 101, 108, 108, 111])  # H e l l o (byte IDs = val+1)
    'Hello'
    """

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, ids: Iterable[int]) -> str:
        """Convert a list of vocabulary IDs to a decoded string.

        Parameters
        ----------
        ids:
            Any iterable of integer token IDs (e.g. the output of
            :class:`~tokenizer.core.encoder.BPEEncoder`).

        Returns
        -------
        str
            The decoded Unicode string.  Byte sequences that are not valid
            UTF-8 are replaced with the U+FFFD replacement character
            (``errors="replace"`` semantics).

        Notes
        -----
        * ``id 0`` (``<pad>``) is always skipped.
        * ``id 257`` (``<|endoftext|>``) is always skipped.
        * ``ids 31744-31752`` (commit special tokens) are always skipped.
        * IDs not present in the vocabulary are silently skipped.
        """
        raw: bytes = b""
        for id in ids:
            # Skip padding, end-of-text, and all commit special tokens.
            if id in _SKIP_IDS:
                continue
            # Resolve id → bytes; silently ignore unknown IDs.
            try:
                token: bytes = self.vocab.id_to_token(id)
            except KeyError:
                continue
            raw += token
        return raw.decode("utf-8", errors="replace")

    def decode_batch(self, batch: Iterable[List[int]]) -> List[str]:
        """Decode a batch of ID sequences.

        Parameters
        ----------
        batch:
            An iterable of ID lists (e.g. a padded 2-D array sliced along
            the batch dimension).

        Returns
        -------
        List[str]
            One decoded string per input sequence, in the same order.
        """
        return [self.decode(ids) for ids in batch]
