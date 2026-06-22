# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
unified_tokenizer.py — 三阶段共用 tokenizer

选用 bigcode/starcoderbase 的 BPE tokenizer (vocab=49152):
  - 代码优化 (Python/JS/Java/Go/Rust/C++ 等)
  - 包含 <fim_prefix> <fim_middle> <fim_suffix> 等 fill-in-middle 特殊 token
  - 兼容 Megatron 的 tokenizer 接口

新增 commit 专用特殊 token:
  <|diff_start|>  <|diff_end|>  <|old|>  <|new|>
  <|commit_msg|>  <|file_path|>  <|lang|>
"""

import os
from typing import Optional, List

TOKENIZER_NAME = "bigcode/starcoderbase"

COMMIT_SPECIAL_TOKENS = [
    "<|diff_start|>", "<|diff_end|>",
    "<|old|>", "<|new|>",
    "<|commit_msg|>", "<|file_path|>", "<|lang|>",
    "<|endoftext|>",
]


def get_tokenizer(
    name_or_path: str = TOKENIZER_NAME,
    cache_dir: Optional[str] = None,
    add_commit_tokens: bool = True,
):
    """Load tokenizer, add commit special tokens, return with metadata."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if add_commit_tokens:
        existing = set(tok.get_vocab().keys())
        new_tokens = [t for t in COMMIT_SPECIAL_TOKENS if t not in existing]
        if new_tokens:
            tok.add_special_tokens({"additional_special_tokens": new_tokens})
            print(f"[tokenizer] added {len(new_tokens)} commit tokens, "
                  f"vocab: {len(tok)} tokens")

    # Megatron 兼容: 设置 eod
    if not hasattr(tok, "eod"):
        tok.eod = tok.eos_token_id

    return tok


class MegatronTokenizerWrapper:
    """适配 Megatron 的 tokenizer 接口.

    Megatron 期望 tokenizer 有:
      .vocab_size, .eod, .tokenize(text) -> list[int], .detokenize(ids) -> str
    """

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.eod = hf_tokenizer.eos_token_id
        self.vocab_size = len(hf_tokenizer)

        # commit token ids 缓存
        self.diff_start_id = hf_tokenizer.convert_tokens_to_ids("<|diff_start|>")
        self.diff_end_id = hf_tokenizer.convert_tokens_to_ids("<|diff_end|>")
        self.old_id = hf_tokenizer.convert_tokens_to_ids("<|old|>")
        self.new_id = hf_tokenizer.convert_tokens_to_ids("<|new|>")
        self.msg_id = hf_tokenizer.convert_tokens_to_ids("<|commit_msg|>")

    @property
    def tokenizer(self):
        return self._tok

    def tokenize(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def encode_commit(self, old_code: str, new_code: str, message: str,
                      file_path: str = "", lang: str = "") -> List[int]:
        """将一条 commit 编码为 token 序列.

        格式: <|diff_start|> <|file_path|> path <|lang|> lang
              <|old|> old_code <|new|> new_code
              <|commit_msg|> message <|diff_end|>
        """
        parts = [self.diff_start_id]
        if file_path:
            fp_id = self._tok.convert_tokens_to_ids("<|file_path|>")
            parts += [fp_id] + self.tokenize(file_path)
        if lang:
            lang_id = self._tok.convert_tokens_to_ids("<|lang|>")
            parts += [lang_id] + self.tokenize(lang)
        parts += [self.old_id] + self.tokenize(old_code)
        parts += [self.new_id] + self.tokenize(new_code)
        parts += [self.msg_id] + self.tokenize(message)
        parts += [self.diff_end_id]
        return parts


def build_megatron_tokenizer(
    name_or_path: str = TOKENIZER_NAME,
    cache_dir: Optional[str] = None,
) -> MegatronTokenizerWrapper:
    """一步到位: load HF tokenizer → wrap for Megatron."""
    hf_tok = get_tokenizer(name_or_path, cache_dir, add_commit_tokens=True)
    wrapper = MegatronTokenizerWrapper(hf_tok)
    print(f"[tokenizer] {name_or_path} loaded, vocab={wrapper.vocab_size}, "
          f"eod={wrapper.eod}, diff_start={wrapper.diff_start_id}")
    return wrapper


if __name__ == "__main__":
    tok = build_megatron_tokenizer()
    # 快速验证
    ids = tok.encode_commit(
        old_code="def foo():\n    pass",
        new_code="def foo():\n    return 42",
        message="implement foo",
        file_path="src/main.py",
        lang="python",
    )
    print(f"commit tokens: {len(ids)}")
    print(f"decoded: {tok.detokenize(ids)[:200]}")
