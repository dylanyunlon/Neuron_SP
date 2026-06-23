# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
tools/generate_sample.py — Commit-completion inference from a trained checkpoint.

Given a commit message and optional file context, the model generates the
corresponding diff in structured commit format:

    <COMMIT>
    <MSG> fix: resolve off-by-one in attention mask </MSG>
    <FILE path="megatron/core/transformer/attention.py"> <lang:python>
    <HUNK @@ -100,5 +100,7 @@>
    <CTX>    def forward(self, x):</CTX>
    <DEL>        mask = torch.ones(x.size(0))</DEL>
    <ADD>        mask = torch.ones(x.size(1))</ADD>
    </HUNK>
    </FILE>
    </COMMIT>

Checkpoint formats supported
-----------------------------
- DesLocEngine format: ``{"model_state": {...}, "global_step": N, ...}``
- Plain state-dict: ``{param_name: tensor, ...}``

Tokenizer
---------
The script wraps whichever HuggingFace-compatible tokenizer the user
provides (``--tokenizer-name``, default ``gpt2``) and registers all
commit special tokens defined in
``Megatron-LM/megatron/training/tokenizer/commit_tokenizer.py``.

Sampling modes
--------------
- **greedy**   (``--strategy greedy``): argmax at each step — deterministic,
  highest-probability token always chosen.
- **nucleus**  (``--strategy nucleus``): top-p sampling — draw from the
  smallest prefix of the vocabulary whose cumulative probability ≥ top-p.
  Temperature scales logits before softmax; top-k hard-caps candidates first
  when both are set.

Usage
-----
    # greedy decoding
    python tools/generate_sample.py \\
        --checkpoint checkpoints/step_010000.pt \\
        --commit-message "fix: handle None return in backward pass" \\
        --file-context "def backward(ctx, grad): ..." \\
        --strategy greedy

    # nucleus sampling
    python tools/generate_sample.py \\
        --checkpoint checkpoints/step_010000.pt \\
        --commit-message "refactor: split large function into helpers" \\
        --strategy nucleus \\
        --temperature 0.8 \\
        --top-p 0.95 \\
        --max-new-tokens 512

    # auto-discover latest checkpoint in a directory
    python tools/generate_sample.py \\
        --checkpoint checkpoints/ \\
        --commit-message "perf: vectorise inner loop" \\
        --strategy nucleus \\
        --num-samples 3
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_sample")


# ---------------------------------------------------------------------------
# Inline model definition (matches run_pretrain.py LlamaModel exactly so
# that checkpoint state-dicts load without key remapping)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class _SwiGLU(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        ffn = int(hidden * 8 / 3)
        ffn = (ffn + 63) // 64 * 64
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.up   = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn,    hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class _TransformerBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(hidden)
        self.attn  = _CausalSelfAttention(hidden, n_heads)
        self.norm2 = _RMSNorm(hidden)
        self.mlp   = _SwiGLU(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LlamaModel(nn.Module):
    """
    Llama-style causal LM identical to the one in run_pretrain.py.

    Weight-ties embedding ↔ lm_head (standard LLaMA practice).
    Architecture defaults match the 7B preset in configs/7b_commitpack.yaml:
      hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32064.
    """

    def __init__(
        self,
        vocab_size: int = 32064,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embedding     = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        self.layers = nn.ModuleList(
            [_TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.norm    = _RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tie

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.embedding(input_ids) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Model size presets (mirrors convert_to_hf.py _MODEL_SIZE_PRESETS)
# ---------------------------------------------------------------------------

_MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "70m": dict(hidden_size=512,  num_layers=8,  num_heads=8),
    "1b":  dict(hidden_size=2048, num_layers=16, num_heads=16),
    "7b":  dict(hidden_size=4096, num_layers=32, num_heads=32),
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(directory: Path) -> Path:
    """Return the ``step_*.pt`` file with the highest step number."""
    pattern = re.compile(r"step_(\d+)\.pt$")
    candidates: List[Tuple[int, Path]] = []
    for p in directory.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        logger.error("No step_*.pt files found in %s", directory)
        sys.exit(1)
    candidates.sort(key=lambda x: x[0])
    chosen = candidates[-1][1]
    logger.info("Auto-selected checkpoint: %s", chosen)
    return chosen


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a Neuron_SP checkpoint.

    Handles two formats:
    - DesLocEngine: ``{"model_state": {...}, "global_step": N, ...}``
    - Plain state-dict: ``{param_name: tensor, ...}``

    Returns ``(state_dict, meta)`` where *meta* carries ``global_step``,
    ``tokens_seen``, and an optional embedded ``config`` dict.
    """
    logger.info("Loading checkpoint from %s …", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # DesLocEngine format
        state_dict = ckpt["model_state"]
        meta: Dict[str, Any] = {
            "global_step": ckpt.get("global_step", 0),
            "tokens_seen": ckpt.get("tokens_seen", 0),
            "config":      ckpt.get("config", None),
        }
        logger.info(
            "DesLocEngine checkpoint — step=%d, tokens_seen=%.2fM",
            meta["global_step"],
            meta["tokens_seen"] / 1e6,
        )
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        meta = {"global_step": 0, "tokens_seen": 0, "config": None}
        logger.info("Plain state-dict checkpoint (%d keys)", len(state_dict))
    else:
        logger.error(
            "Unrecognised checkpoint format: expected dict, got %s", type(ckpt)
        )
        sys.exit(1)

    return state_dict, meta


def _infer_model_config(
    state_dict: Dict[str, torch.Tensor],
    override: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Infer ``vocab_size``, ``hidden_size``, ``num_layers``, ``num_heads`` from
    the weight shapes in *state_dict*.  *override* keys take precedence.
    """
    cfg: Dict[str, int] = {}

    if "embedding.weight" in state_dict:
        v, h = state_dict["embedding.weight"].shape
        cfg["vocab_size"]  = v
        cfg["hidden_size"] = h

    layer_indices: set = set()
    for k in state_dict:
        m = re.match(r"layers\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    if layer_indices:
        cfg["num_layers"] = max(layer_indices) + 1

    if "pos_embedding.weight" in state_dict:
        cfg["seq_len"] = state_dict["pos_embedding.weight"].shape[0]

    if override:
        cfg.update(override)

    # Defaults (7B) for anything still missing
    defaults = dict(
        vocab_size=32064, hidden_size=4096, num_layers=32, num_heads=32, seq_len=2048
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)

    return cfg


def build_model(
    state_dict: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    model_size: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
) -> LlamaModel:
    """
    Instantiate *LlamaModel*, load *state_dict*, move to *device* and cast to
    *dtype*.  Architecture is inferred from weight shapes; an explicit
    ``--model-size`` preset overrides individual dims.
    """
    # Priority: embedded config > size preset > shape inference
    override: Dict[str, int] = {}
    if meta.get("config"):
        embedded = meta["config"]
        for k in ("vocab_size", "hidden_size", "num_layers", "num_heads", "seq_len"):
            if k in embedded:
                override[k] = int(embedded[k])

    if model_size:
        if model_size not in _MODEL_SIZE_PRESETS:
            logger.error("Unknown model-size %r; choices: %s", model_size, list(_MODEL_SIZE_PRESETS))
            sys.exit(1)
        override.update(_MODEL_SIZE_PRESETS[model_size])

    cfg = _infer_model_config(state_dict, override=override)
    logger.info("Model config: %s", cfg)

    model = LlamaModel(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        seq_len=cfg["seq_len"],
    )
    logger.info("Parameters: %.2fM", model.num_parameters / 1e6)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s …", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:5])

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _build_tokenizer(tokenizer_name: str):
    """
    Load a HuggingFace tokenizer and register all commit special tokens.

    Requires ``transformers``; imports are deferred so the script can still
    parse arguments and print usage without the library installed.
    """
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        logger.error(
            "transformers is not installed.  Install with:  pip install transformers"
        )
        sys.exit(1)

    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register commit special tokens (import from repo path)
    _register_commit_tokens(tokenizer)

    return tokenizer


def _register_commit_tokens(tokenizer) -> None:
    """
    Try to import COMMIT_SPECIAL_TOKENS from the repo and add them to the
    tokenizer.  Gracefully skips when the path is not available.
    """
    repo_root = Path(__file__).resolve().parents[1]
    commit_tok_path = (
        repo_root / "Megatron-LM" / "megatron" / "training" / "tokenizer" / "commit_tokenizer.py"
    )
    if not commit_tok_path.exists():
        logger.warning(
            "commit_tokenizer.py not found at %s; commit special tokens will not be registered.",
            commit_tok_path,
        )
        return

    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("commit_tokenizer", commit_tok_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    commit_tokens = list(mod.COMMIT_SPECIAL_TOKENS.keys())
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in commit_tokens if t not in existing]
    if new_tokens:
        added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info("Registered %d commit special tokens (total vocab: %d)", added, len(tokenizer))
    else:
        logger.info("All %d commit special tokens already in vocab.", len(commit_tokens))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(
    commit_message: str,
    file_context: Optional[str] = None,
) -> str:
    """
    Construct the model input for commit-completion inference.

    The prompt follows the training format from commit_tokenizer.py:
    it opens the <COMMIT> structure and supplies <MSG> (and optionally a
    <REF> file context block) so the model generates the <FILE>…</FILE>
    diff region.

    Args:
        commit_message: The commit subject / body the model should expand.
        file_context: Optional surrounding code (function body, class, etc.)
            that helps the model pick the right change location.

    Returns:
        A prompt string ready to be tokenized and fed to the model.
    """
    parts = ["<COMMIT>"]
    parts.append(f"<MSG> {commit_message.strip()} </MSG>")
    if file_context and file_context.strip():
        # Use <REF> to supply caller-provided context (Seed-Coder style)
        parts.append(f"<REF>\n{file_context.strip()}\n</REF>")
    # Leave the sequence open — the model continues from here
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def _top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Zero-out logits outside the top-k candidates and/or the nucleus (top-p).

    Args:
        logits: Shape ``(vocab_size,)`` — single-step logits.
        top_k:  Keep only the top-k highest-probability tokens (0 = disabled).
        top_p:  Keep the smallest set of tokens whose cumulative probability
                is at least *top_p* (1.0 = disabled, i.e. full vocabulary).

    Returns:
        Filtered logits tensor (same shape; excluded positions set to -inf).
    """
    if top_k > 0:
        # Zero out everything below the k-th largest value
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens once cumulative prob exceeds top_p; shift by 1 to keep the token
        # that pushes the cumsum *over* the threshold (inclusive nucleus)
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_indices, sorted_logits)

    return logits


# ---------------------------------------------------------------------------
# Greedy decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: LlamaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens by greedily picking ``argmax`` at each step.

    Args:
        model:          The loaded LlamaModel (eval mode, on device).
        input_ids:      Shape ``(1, prompt_len)`` — prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate beyond prompt.
        eos_token_id:   Stop early when this token is produced.

    Returns:
        Tensor of shape ``(1, prompt_len + generated_len)`` — full sequence
        including the original prompt tokens.
    """
    seq = input_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate context to model's maximum sequence length
        ctx = seq[:, -model.seq_len:]
        logits = model(ctx)                     # (1, T, vocab)
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # (1, 1)
        seq = torch.cat([seq, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return seq


# ---------------------------------------------------------------------------
# Nucleus (top-p) sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def nucleus_sample(
    model: LlamaModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens via nucleus (top-p) sampling with optional temperature.

    Procedure at each step:
    1. Forward pass → logits for the last position.
    2. Divide by *temperature* (lower = sharper distribution).
    3. Apply top-k hard cap (if ``top_k > 0``).
    4. Apply top-p nucleus filter.
    5. Multinomial sample from the resulting distribution.

    Args:
        model:          The loaded LlamaModel (eval mode, on device).
        input_ids:      Shape ``(1, prompt_len)`` — prompt token IDs.
        max_new_tokens: Maximum number of tokens to generate beyond prompt.
        temperature:    Softmax temperature (>1 = more random, <1 = sharper).
        top_p:          Nucleus probability threshold (0 < top_p ≤ 1.0).
        top_k:          Hard cap on candidate tokens before nucleus filter
                        (0 = disabled).
        eos_token_id:   Stop early when this token is produced.

    Returns:
        Tensor of shape ``(1, prompt_len + generated_len)`` — full sequence
        including the original prompt tokens.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1]; got {top_p}")

    seq = input_ids.clone()

    for _ in range(max_new_tokens):
        ctx = seq[:, -model.seq_len:]
        logits = model(ctx)                          # (1, T, vocab)
        next_logits = logits[0, -1, :].float()       # (vocab,)

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Top-k + top-p filtering
        next_logits = _top_k_top_p_filter(next_logits, top_k=top_k, top_p=top_p)

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1, 1)
        seq = torch.cat([seq, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return seq


# ---------------------------------------------------------------------------
# High-level generate wrapper
# ---------------------------------------------------------------------------

def generate(
    model: LlamaModel,
    tokenizer,
    prompt: str,
    strategy: str = "nucleus",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    device: Optional[torch.device] = None,
) -> str:
    """
    Tokenize *prompt*, run the selected decoding strategy, and decode output.

    Args:
        model:          LlamaModel in eval mode.
        tokenizer:      HuggingFace-compatible tokenizer.
        prompt:         Prompt string (commit header, possibly with <REF>).
        strategy:       ``"greedy"`` or ``"nucleus"``.
        max_new_tokens: Maximum tokens to generate beyond the prompt.
        temperature:    Nucleus sampling temperature (ignored for greedy).
        top_p:          Nucleus probability threshold (ignored for greedy).
        top_k:          Hard-cap on candidates before nucleus filter.
        device:         Target device; defaults to model's first parameter device.

    Returns:
        Decoded text containing only the *newly generated* tokens (prompt
        excluded), stripped of leading/trailing whitespace.
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    eos_id = tokenizer.eos_token_id

    if strategy == "greedy":
        out_ids = greedy_decode(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
        )
    elif strategy == "nucleus":
        out_ids = nucleus_sample(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_id,
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; choose 'greedy' or 'nucleus'")

    # Decode only the generated portion (not the prompt)
    generated_ids = out_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return generated_text.strip()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a step_*.pt checkpoint file, or a directory from which the "
             "latest step_*.pt is auto-selected.",
    )
    parser.add_argument(
        "--model-size",
        choices=list(_MODEL_SIZE_PRESETS.keys()),
        default=None,
        help="Model size preset (overrides shape inference).",
    )

    # Input
    parser.add_argument(
        "--commit-message",
        required=True,
        help="Commit message / subject line to complete (the model generates the diff).",
    )
    parser.add_argument(
        "--file-context",
        default=None,
        help="Optional code context injected as a <REF> block (function body, class, etc.).",
    )

    # Sampling
    parser.add_argument(
        "--strategy",
        choices=["greedy", "nucleus"],
        default="nucleus",
        help="Decoding strategy.  'greedy' = argmax; 'nucleus' = top-p sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for nucleus sampling (default: 1.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        dest="top_p",
        help="Nucleus (top-p) probability threshold (default: 0.95).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        dest="top_k",
        help="Hard-cap on top-k candidates before nucleus filter (0 = disabled).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        dest="max_new_tokens",
        help="Maximum number of tokens to generate beyond the prompt (default: 256).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent samples to generate (default: 1; greedy always 1).",
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="HuggingFace tokenizer name or local path (default: gpt2).",
    )

    # Hardware
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string, e.g. 'cuda', 'cuda:1', 'cpu' (auto-detected by default).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model weight dtype (default: float32).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Device / dtype
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Resolve checkpoint path
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = _find_latest_checkpoint(ckpt_path)

    # Load checkpoint and build model
    state_dict, meta = load_checkpoint(ckpt_path)
    model = build_model(state_dict, meta, args.model_size, device, dtype)

    # Tokenizer
    tokenizer = _build_tokenizer(args.tokenizer_name)

    # Resize token embeddings when extra commit tokens were added
    vocab_size_tok = len(tokenizer)
    vocab_size_model = model.embedding.weight.shape[0]
    if vocab_size_tok > vocab_size_model:
        logger.warning(
            "Tokenizer vocab (%d) > model vocab (%d); "
            "the additional commit special tokens will map to out-of-range IDs.  "
            "Resize embeddings or use a checkpoint trained with the full vocab.",
            vocab_size_tok,
            vocab_size_model,
        )

    # Build prompt
    prompt = build_prompt(args.commit_message, file_context=args.file_context)
    logger.info("Prompt (%d chars):\n%s", len(prompt), prompt)

    # Determine how many samples to run
    num_samples = 1 if args.strategy == "greedy" else max(1, args.num_samples)
    if args.strategy == "greedy" and args.num_samples > 1:
        logger.info("Greedy decoding is deterministic; ignoring --num-samples > 1.")

    # Generate
    print("\n" + "=" * 72)
    print(f"Checkpoint : {ckpt_path}  (step {meta['global_step']})")
    print(f"Strategy   : {args.strategy}")
    if args.strategy == "nucleus":
        print(f"Temperature: {args.temperature}  top_p={args.top_p}  top_k={args.top_k}")
    print(f"Max tokens : {args.max_new_tokens}")
    print("=" * 72 + "\n")

    for i in range(num_samples):
        t0 = time.perf_counter()
        generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            strategy=args.strategy,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.perf_counter() - t0

        # Count generated tokens for throughput
        gen_ids = tokenizer.encode(generated, add_special_tokens=False)
        tok_per_sec = len(gen_ids) / elapsed if elapsed > 0 else float("inf")

        if num_samples > 1:
            print(f"── Sample {i + 1}/{num_samples} ── ({len(gen_ids)} tokens, {elapsed:.2f}s, {tok_per_sec:.1f} tok/s)")
        else:
            print(f"── Generated ({len(gen_ids)} tokens, {elapsed:.2f}s, {tok_per_sec:.1f} tok/s)")

        print(generated)
        print()


if __name__ == "__main__":
    main()
