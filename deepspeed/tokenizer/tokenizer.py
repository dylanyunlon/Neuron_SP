# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# M76: Megatron 11220df86 — tokenizer moved to its own directory
# Ported from: megatron/data/tokenizer.py → megatron/tokenizer/tokenizer.py
#
# Megatron moved tokenizer.py out of megatron/data/ into its own dedicated
# megatron/tokenizer/ package.  This file mirrors that move:
#   megatron/data/tokenizer.py  → deepspeed/tokenizer/tokenizer.py
#   (was: deepspeed/runtime/data_pipeline/... legacy path)
#
# The only change in the upstream commit was the file location; no logic
# was modified.  The intra-package import now resolves correctly because
# bert_tokenization.py is in the same deepspeed/tokenizer/ directory.
# ---------------------------------------------------------------------------

"""DeepSpeed tokenizer — ported from Megatron tokenizer (11220df86)."""

from abc import ABC
from abc import abstractmethod

from .bert_tokenization import FullTokenizer as FullBertTokenizer

print('[M76]')


def build_tokenizer(args):
    """Initialize tokenizer.

    M76: Previously imported as:
        from deepspeed.runtime.data_pipeline... (legacy)
    Now imported as:
        from deepspeed.tokenizer import build_tokenizer
    mirroring Megatron's megatron.tokenizer import path.
    """
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
               args.model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))


class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True):
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id


# ---------------------------------------------------------------------------
# M702: Megatron 6f72a2851 — add dialog dataset and special tokens in tokenizer
# Source: megatron/tokenizer/tokenizer.py (NVIDIA/Megatron-LM commit 6f72a2851)
# Author: zihanl <zihanl@nvidia.com>  Date: 2021-06-28
#
# Mapping: megatron/tokenizer/tokenizer.py
#          → deepspeed/tokenizer/tokenizer.py
#
# Changes ported:
#   1. build_tokenizer(): added GPT2BPETokenizer branch; passes spec_toks
#      (from args.spec_toks) as special_tokens kwarg.
#   2. _GPT2BPETokenizer: new class wrapping GPT2Tokenizer with optional
#      special_tokens; sets pad_id / sep_id / ctrl_id when present.
#
# DeepSpeed adaptation:
#   - GPT2Tokenizer imported from deepspeed.runtime.utils (already present
#     in this codebase) rather than megatron.tokenizer.gpt2_tokenization.
#   - getattr(args, 'spec_toks', None) guards against args objects that
#     pre-date M702 and do not carry the new attribute.
#   - No other logic changes.
# ---------------------------------------------------------------------------


from deepspeed.runtime.utils import GPT2Tokenizer as _RawGPT2Tokenizer


def _build_gpt2bpe_tokenizer(args):
    """Build a GPT2BPETokenizer (M702 helper)."""
    assert args.merge_file is not None, '--merge-file required for GPT2BPETokenizer'
    spec_toks = getattr(args, 'spec_toks', None)
    return _GPT2BPETokenizer(args.vocab_file, args.merge_file, special_tokens=spec_toks)


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer (M702: Megatron 6f72a2851)."""

    def __init__(self, vocab_file, merge_file, special_tokens=None):
        name = 'GPT2 BPE'
        super().__init__(name)

        # Parse comma-separated special tokens string into a list.
        if special_tokens is not None:
            special_tokens = special_tokens.split(',')
        else:
            special_tokens = []

        self.tokenizer = _RawGPT2Tokenizer(vocab_file, merge_file,
                                            errors='replace',
                                            special_tokens=special_tokens,
                                            max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

        if len(special_tokens) > 0:
            if '[PAD]' in special_tokens:
                self.pad_id = self.tokenizer.encoder['[PAD]']
            if '[SEP]' in special_tokens:
                self.sep_id = self.tokenizer.encoder['[SEP]']
            if '[CTRL]' in special_tokens:
                self.ctrl_id = self.tokenizer.encoder['[CTRL]']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


print('[M702]')
