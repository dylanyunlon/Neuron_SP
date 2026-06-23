# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
unified_tokenizer.py — 三阶段共用 tokenizer

选用 bigcode/starcoderbase 的 BPE tokenizer (vocab=49152):
  - 代码优化 (Python/JS/Java/Go/Rust/C++ 等)
  - 包含 <fim_prefix> <fim_middle> <fim_suffix> 等 fill-in-middle 特殊 token
  - 兼容 Megatron 的 tokenizer 接口

新增 commit 专用特殊 token (来自 commit_tokenizer.COMMIT_SPECIAL_TOKENS):
  <COMMIT> <MSG> <FILE> <ADD> <DEL> <CTX> 等结构化 diff token
  <|diff_start|>  <|diff_end|>  <|old|>  <|new|>
  <|commit_msg|>  <|file_path|>  <|lang|>
"""

import os
import sys
from typing import Optional, List

# ---------------------------------------------------------------------------
# Import commit_tokenizer helpers.
# commit_tokenizer lives in Megatron-LM/megatron/training/tokenizer/ which
# may not be on sys.path when this module is imported directly, so we add
# that directory dynamically.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEGATRON_TOK_PATH = os.path.join(
    _HERE, "..", "Megatron-LM", "megatron", "training", "tokenizer"
)
if os.path.isdir(_MEGATRON_TOK_PATH) and _MEGATRON_TOK_PATH not in sys.path:
    sys.path.insert(0, _MEGATRON_TOK_PATH)

try:
    from commit_tokenizer import (  # type: ignore[import]
        CommitTokenizer,
        validate_commit_tokens,
        COMMIT_SPECIAL_TOKENS as _COMMIT_SPECIAL_TOKENS_DICT,
    )
    _COMMIT_TOKENIZER_AVAILABLE = True
except ImportError:
    _COMMIT_TOKENIZER_AVAILABLE = False
    _COMMIT_SPECIAL_TOKENS_DICT = {}
    CommitTokenizer = None  # type: ignore[assignment,misc]
    validate_commit_tokens = None  # type: ignore[assignment]

TOKENIZER_NAME = "bigcode/starcoderbase"

# Legacy diff-style tokens kept for backward-compat with existing checkpoints
_LEGACY_COMMIT_TOKENS = [
    "<|diff_start|>", "<|diff_end|>",
    "<|old|>", "<|new|>",
    "<|commit_msg|>", "<|file_path|>", "<|lang|>",
    "<|endoftext|>",
]

# Full set: legacy tokens + structural commit tokens from commit_tokenizer
COMMIT_SPECIAL_TOKENS: List[str] = _LEGACY_COMMIT_TOKENS + [
    t for t in list(_COMMIT_SPECIAL_TOKENS_DICT.keys())
    if t not in _LEGACY_COMMIT_TOKENS
]


def get_tokenizer(
    name_or_path: str = TOKENIZER_NAME,
    cache_dir: Optional[str] = None,
    add_commit_tokens: bool = True,
):
    """Load tokenizer, add commit special tokens, return with metadata.

    When ``add_commit_tokens=True`` this function registers *all* tokens
    from both the legacy diff-token set and ``commit_tokenizer``'s full
    ``COMMIT_SPECIAL_TOKENS`` dictionary so that the resulting tokenizer is
    immediately usable for commit-aware pretraining.
    """
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(
            name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except (OSError, Exception) as e:
        # Fallback to GPT-2 tokenizer if StarCoder is gated/unavailable
        fallback = "gpt2"
        print(f"[tokenizer] {name_or_path} unavailable ({type(e).__name__}), "
              f"falling back to {fallback}")
        tok = AutoTokenizer.from_pretrained(fallback, cache_dir=cache_dir)
        tok.pad_token = tok.eos_token

    if add_commit_tokens:
        existing = set(tok.get_vocab().keys())
        new_tokens = [t for t in COMMIT_SPECIAL_TOKENS if t not in existing]
        if new_tokens:
            tok.add_special_tokens({"additional_special_tokens": new_tokens})
            print(f"[tokenizer] added {len(new_tokens)} commit tokens, "
                  f"vocab: {len(tok)} tokens")
        else:
            print(f"[tokenizer] all commit tokens already present in vocab "
                  f"({len(tok)} tokens)")

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
        # Alias so eval code can use the explicit name
        self.commit_msg_id = self.msg_id

    def commit_token_ids(self):
        """Return a dict of the three primary commit boundary token IDs
        used by the eval pipeline to segment generated sequences."""
        return {
            "commit_msg": self.commit_msg_id,
            "diff_start": self.diff_start_id,
            "diff_end": self.diff_end_id,
        }

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
    run_validation: bool = True,
) -> MegatronTokenizerWrapper:
    """一步到位: load HF tokenizer → register commit tokens → wrap for Megatron.

    Steps
    -----
    1. Load the HF base tokenizer (StarCoder BPE or GPT-2 fallback).
    2. Add *all* COMMIT_SPECIAL_TOKENS (legacy + structural) via
       ``get_tokenizer``.
    3. Use ``CommitTokenizer.add_special_tokens()`` to sync the
       CommitTokenizer ID map with the HF vocab so that the two token
       systems stay consistent.
    4. Optionally run ``validate_commit_tokens()`` to assert the six core
       commit tokens (<COMMIT>, <MSG>, <FILE>, <ADD>, <DEL>, <CTX>) are
       correctly registered.
    5. Wrap in ``MegatronTokenizerWrapper`` and return.

    Args:
        name_or_path: HuggingFace model name or local path.
        cache_dir: Optional cache directory for HF downloads.
        run_validation: If True, run validate_commit_tokens() after
            registration and raise RuntimeError on failure.

    Returns:
        A fully initialised ``MegatronTokenizerWrapper`` ready for
        Megatron pretraining.
    """
    # Step 1 & 2: load base tokenizer and register all commit tokens
    hf_tok = get_tokenizer(name_or_path, cache_dir, add_commit_tokens=True)

    # Step 3: sync CommitTokenizer ID map with the HF vocab
    if _COMMIT_TOKENIZER_AVAILABLE and CommitTokenizer is not None:
        _commit_tok = CommitTokenizer()
        _commit_tok.add_special_tokens(hf_tok)
    else:
        print("[tokenizer] WARNING: commit_tokenizer not available, "
              "skipping CommitTokenizer.add_special_tokens() sync")

    # Step 4: validate core commit tokens
    if run_validation:
        if _COMMIT_TOKENIZER_AVAILABLE and validate_commit_tokens is not None:
            ok = validate_commit_tokens(hf_tok)
            if not ok:
                raise RuntimeError(
                    "[build_megatron_tokenizer] Core commit token validation "
                    "FAILED — check that commit_tokenizer is importable and "
                    "all six tokens (<COMMIT>, <MSG>, <FILE>, <ADD>, <DEL>, "
                    "<CTX>) are in the HF tokenizer vocab."
                )
        else:
            print("[tokenizer] WARNING: validate_commit_tokens not available, "
                  "skipping validation")

    # Step 5: wrap for Megatron
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
