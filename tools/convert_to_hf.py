#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Neuron_SP Team
"""
tools/convert_to_hf.py — Convert Neuron_SP checkpoints to HuggingFace format.

Reads checkpoints saved by DesLocEngine or the standalone training loop
(``checkpoints/step_*.pt``) and writes a HuggingFace ``LlamaForCausalLM``
directory (``config.json`` + ``model.safetensors``).

Naming-mapping handled:
  RMSNorm  : layers.N.norm1 / norm2 / norm  →  input_layernorm / post_attention_layernorm / model.norm
  Attention: layers.N.attn.qkv (fused)      →  q_proj / k_proj / v_proj (split along dim 0)
             layers.N.attn.proj             →  o_proj
  SwiGLU   : layers.N.mlp.gate / up / down  →  gate_proj / up_proj / down_proj
  Embedding: embedding                       →  model.embed_tokens
  LM-head  : lm_head                        →  lm_head (weight-tied, de-duplicated)
  Positional: pos_embedding is a learned absolute-position table; it has no
              direct HF-Llama equivalent (HF Llama uses RoPE).  The tensor is
              stored in the output directory as ``pos_embedding.safetensors``
              so it is not silently discarded.

Usage
-----
  python tools/convert_to_hf.py \\
      --checkpoint-path checkpoints/step_0001000.pt \\
      --output-dir      hf_model/

  # Auto-discover the latest checkpoint:
  python tools/convert_to_hf.py \\
      --checkpoint-path checkpoints/ \\
      --output-dir      hf_model/

  # Override model size for checkpoints that lack embedded config:
  python tools/convert_to_hf.py \\
      --checkpoint-path checkpoints/step_0001000.pt \\
      --output-dir      hf_model/ \\
      --hidden-size 4096 --num-layers 32 --num-heads 32 --vocab-size 32000

Output
------
  hf_model/
    config.json           # HuggingFace LlamaConfig
    model.safetensors     # All mapped weights
    pos_embedding.safetensors  # Absolute pos-embed (kept for reproducibility)
    generation_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_to_hf")

# ---------------------------------------------------------------------------
# Dependency checks (safetensors is the only non-stdlib dep beyond torch)
# ---------------------------------------------------------------------------

def _require_safetensors() -> Any:
    try:
        from safetensors.torch import save_file  # type: ignore[import]
        return save_file
    except ImportError:
        logger.error(
            "safetensors is not installed.  Install with:  pip install safetensors"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(directory: Path) -> Path:
    """Return the step_*.pt file with the highest step number in *directory*."""
    pattern = re.compile(r"step_(\d+)\.pt$")
    candidates = []
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


def load_checkpoint(checkpoint_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a Neuron_SP checkpoint and return ``(model_state_dict, meta)``.

    The checkpoint may be:
      * A dict with keys ``model_state``, ``global_step``, ``config``, …
        (format written by :class:`DesLocEngine`)
      * A plain ``state_dict`` mapping param-name → tensor
        (format written by ``pipeline/train_three_stage.py``)
    """
    logger.info("Loading checkpoint from %s …", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # DesLocEngine format
        state_dict = ckpt["model_state"]
        meta: Dict[str, Any] = {
            "global_step":  ckpt.get("global_step", 0),
            "tokens_seen":  ckpt.get("tokens_seen", 0),
            "config":       ckpt.get("config", None),
        }
        logger.info(
            "DesLocEngine checkpoint — step=%d, tokens_seen=%.2fM",
            meta["global_step"],
            meta["tokens_seen"] / 1e6,
        )
    elif isinstance(ckpt, dict):
        # Assume it is already a plain state dict (pipeline or standalone save)
        state_dict = ckpt
        meta = {"global_step": 0, "tokens_seen": 0, "config": None}
        logger.info("Plain state-dict checkpoint (%d keys)", len(state_dict))
    else:
        logger.error(
            "Unrecognised checkpoint format: expected dict, got %s", type(ckpt)
        )
        sys.exit(1)

    return state_dict, meta


# ---------------------------------------------------------------------------
# Shape inference from state dict
# ---------------------------------------------------------------------------

_MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "70m": dict(hidden_size=512,  num_layers=8,  num_heads=8,  intermediate_size=1344),
    "1b":  dict(hidden_size=2048, num_layers=16, num_heads=16, intermediate_size=5504),
    "7b":  dict(hidden_size=4096, num_layers=32, num_heads=32, intermediate_size=11008),
}


def _infer_model_config(
    state_dict: Dict[str, torch.Tensor],
    override: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Infer model hyperparameters from the weight shapes in *state_dict*.

    ``override`` keys take precedence over inferred values.
    """
    cfg: Dict[str, int] = {}

    # vocab_size, hidden_size from embedding
    if "embedding.weight" in state_dict:
        v, h = state_dict["embedding.weight"].shape
        cfg["vocab_size"]  = v
        cfg["hidden_size"] = h

    # num_layers — count how many layer blocks exist
    layer_indices = set()
    for k in state_dict:
        m = re.match(r"layers\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    if layer_indices:
        cfg["num_layers"] = max(layer_indices) + 1

    # num_heads, head_dim from qkv shape
    for k in state_dict:
        if re.match(r"layers\.\d+\.attn\.qkv\.weight", k):
            # shape: (3 * hidden, hidden)
            qkv_out, hidden = state_dict[k].shape
            h = cfg.get("hidden_size", hidden)
            # qkv_out == 3 * hidden, but hidden might equal h
            # num_heads cannot be read directly; try common factors
            # We store and derive it from head_dim assumption below
            cfg["_qkv_out"] = qkv_out
            break

    # intermediate_size from MLP gate projection
    for k in state_dict:
        if re.match(r"layers\.\d+\.mlp\.gate\.weight", k):
            cfg["intermediate_size"] = state_dict[k].shape[0]
            break

    if override:
        cfg.update({k: v for k, v in override.items() if v is not None})

    # Derive num_heads if not explicitly provided
    if "num_heads" not in cfg and "hidden_size" in cfg and "_qkv_out" in cfg:
        hidden = cfg["hidden_size"]
        # For standard LLaMA: head_dim == 128 (7B), 128 (1B), 64 (70M)
        for head_dim in (128, 64, 32):
            if hidden % head_dim == 0:
                cfg["num_heads"] = hidden // head_dim
                break
    cfg.pop("_qkv_out", None)

    # Validate required keys
    required = ("vocab_size", "hidden_size", "num_layers", "num_heads", "intermediate_size")
    missing = [r for r in required if r not in cfg]
    if missing:
        logger.error(
            "Cannot infer model config, missing: %s.  "
            "Pass --hidden-size / --num-layers / --num-heads / --vocab-size "
            "/ --intermediate-size explicitly.",
            missing,
        )
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# Weight remapping
# ---------------------------------------------------------------------------

def remap_state_dict(
    src: Dict[str, torch.Tensor],
    num_layers: int,
    num_heads: int,
    hidden_size: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Map a Neuron_SP ``LlamaModel`` state dict to the HuggingFace
    ``LlamaForCausalLM`` naming convention.

    Returns
    -------
    hf_dict : dict
        Remapped weights ready for safetensors serialisation.
    pos_embed : Tensor or None
        The learned absolute position embedding (no HF Llama equivalent);
        returned separately so the caller can persist it.
    """
    hf: Dict[str, torch.Tensor] = {}
    pos_embed: Optional[torch.Tensor] = None
    head_dim = hidden_size // num_heads

    for src_key, tensor in src.items():
        # ------------------------------------------------------------------ #
        # Embedding
        # ------------------------------------------------------------------ #
        if src_key == "embedding.weight":
            hf["model.embed_tokens.weight"] = tensor
            # lm_head shares this weight; add a copy under lm_head namespace
            # (de-duplicated: safetensors allows sharing if the same data)
            hf["lm_head.weight"] = tensor
            logger.debug("  %s  →  model.embed_tokens.weight  +  lm_head.weight", src_key)
            continue

        if src_key == "pos_embedding.weight":
            # No RoPE equivalent; stash for separate file
            pos_embed = tensor
            logger.debug("  %s  →  (stored in pos_embedding.safetensors)", src_key)
            continue

        # ------------------------------------------------------------------ #
        # Final layer norm
        # ------------------------------------------------------------------ #
        if src_key == "norm.weight":
            hf["model.norm.weight"] = tensor
            logger.debug("  %s  →  model.norm.weight", src_key)
            continue

        # ------------------------------------------------------------------ #
        # lm_head (weight-tied duplicate — skip; already added above)
        # ------------------------------------------------------------------ #
        if src_key == "lm_head.weight":
            # Already handled via embedding.weight.  If the checkpoint *does*
            # contain a de-tied lm_head (e.g. after fine-tuning), use it.
            if "lm_head.weight" not in hf:
                hf["lm_head.weight"] = tensor
            logger.debug("  %s  →  lm_head.weight (de-tied or redundant)", src_key)
            continue

        # ------------------------------------------------------------------ #
        # Per-layer weights
        # ------------------------------------------------------------------ #
        m = re.match(r"layers\.(\d+)\.(.*)", src_key)
        if not m:
            logger.warning("Unmapped key (skipped): %s", src_key)
            continue

        layer_idx = int(m.group(1))
        suffix = m.group(2)

        # RMSNorm: pre-attention
        if suffix == "norm1.weight":
            hf_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            hf[hf_key] = tensor
            logger.debug("  %s  →  %s", src_key, hf_key)
            continue

        # RMSNorm: post-attention
        if suffix == "norm2.weight":
            hf_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            hf[hf_key] = tensor
            logger.debug("  %s  →  %s", src_key, hf_key)
            continue

        # Attention — fused QKV → split
        if suffix == "attn.qkv.weight":
            # Shape: (3 * hidden_size, hidden_size)
            # Split equally into Q, K, V along dim 0.
            q, k, v = tensor.chunk(3, dim=0)
            hf[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q
            hf[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k
            hf[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
            logger.debug(
                "  %s  →  q_proj / k_proj / v_proj  (each %s)",
                src_key, tuple(q.shape),
            )
            continue

        # Attention bias (unlikely but handle gracefully)
        if suffix == "attn.qkv.bias":
            q_b, k_b, v_b = tensor.chunk(3, dim=0)
            hf[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = q_b
            hf[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = k_b
            hf[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = v_b
            continue

        # Output projection
        if suffix == "attn.proj.weight":
            hf_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            hf[hf_key] = tensor
            logger.debug("  %s  →  %s", src_key, hf_key)
            continue

        if suffix == "attn.proj.bias":
            hf[f"model.layers.{layer_idx}.self_attn.o_proj.bias"] = tensor
            continue

        # SwiGLU MLP: gate projection
        if suffix == "mlp.gate.weight":
            hf_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            hf[hf_key] = tensor
            logger.debug("  %s  →  %s", src_key, hf_key)
            continue

        # SwiGLU MLP: up projection
        if suffix == "mlp.up.weight":
            hf_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            hf[hf_key] = tensor
            logger.debug("  %s  →  %s", src_key, hf_key)
            continue

        # SwiGLU MLP: down projection
        if suffix == "mlp.down.weight":
            hf_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            hf[hf_key] = tensor
            logger.debug("  %s  →  %s", src_key, hf_key)
            continue

        # MLP biases (uncommon)
        if suffix in ("mlp.gate.bias", "mlp.up.bias", "mlp.down.bias"):
            proj_map = {"mlp.gate.bias": "gate_proj", "mlp.up.bias": "up_proj", "mlp.down.bias": "down_proj"}
            hf[f"model.layers.{layer_idx}.mlp.{proj_map[suffix]}.bias"] = tensor
            continue

        logger.warning("Unmapped layer key (skipped): %s", src_key)

    return hf, pos_embed


# ---------------------------------------------------------------------------
# HuggingFace config generation
# ---------------------------------------------------------------------------

def build_hf_config(model_cfg: Dict[str, int], global_step: int) -> Dict[str, Any]:
    """Produce a ``config.json`` compatible with ``LlamaForCausalLM``."""
    hidden   = model_cfg["hidden_size"]
    n_heads  = model_cfg["num_heads"]
    head_dim = hidden // n_heads

    # RoPE theta — HF Llama default; matches most LLaMA2/3 deployments
    rope_theta = 10000.0

    return {
        "_name_or_path": "neuron_sp_converted",
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "initializer_range": 0.02,
        "intermediate_size": model_cfg["intermediate_size"],
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": n_heads,
        "num_hidden_layers": model_cfg["num_layers"],
        "num_key_value_heads": n_heads,   # MHA (not GQA/MQA)
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": rope_theta,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
        "use_cache": True,
        "vocab_size": model_cfg["vocab_size"],
        # Metadata
        "_neuron_sp_global_step": global_step,
    }


def build_generation_config(vocab_size: int) -> Dict[str, Any]:
    return {
        "_from_model_config": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "transformers_version": "4.40.0",
    }


# ---------------------------------------------------------------------------
# Main conversion routine
# ---------------------------------------------------------------------------

def convert(
    checkpoint_path: Path,
    output_dir: Path,
    override_cfg: Optional[Dict[str, Optional[int]]] = None,
    verbose: bool = False,
) -> None:
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    save_file = _require_safetensors()

    # Resolve checkpoint path (may be a directory → pick latest)
    if checkpoint_path.is_dir():
        checkpoint_path = _find_latest_checkpoint(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    # Load
    state_dict, meta = load_checkpoint(checkpoint_path)
    global_step = meta["global_step"]

    # Infer/override model config
    model_cfg = _infer_model_config(state_dict, override=override_cfg)
    logger.info(
        "Model config — hidden=%d  layers=%d  heads=%d  vocab=%d  intermediate=%d",
        model_cfg["hidden_size"],
        model_cfg["num_layers"],
        model_cfg["num_heads"],
        model_cfg["vocab_size"],
        model_cfg["intermediate_size"],
    )

    # Remap weights
    logger.info("Remapping %d weight tensors …", len(state_dict))
    hf_dict, pos_embed = remap_state_dict(
        state_dict,
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        hidden_size=model_cfg["hidden_size"],
    )
    logger.info("Mapped to %d HF tensors.", len(hf_dict))

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights as safetensors
    model_safetensors_path = output_dir / "model.safetensors"
    logger.info("Saving model.safetensors → %s", model_safetensors_path)
    # safetensors requires contiguous tensors
    hf_dict_contig = {k: v.contiguous() for k, v in hf_dict.items()}
    save_file(hf_dict_contig, str(model_safetensors_path))

    # Save pos_embedding separately (no HF equivalent)
    if pos_embed is not None:
        pos_path = output_dir / "pos_embedding.safetensors"
        logger.info(
            "Saving pos_embedding.safetensors → %s  (shape=%s, no RoPE equivalent)",
            pos_path, tuple(pos_embed.shape),
        )
        save_file({"pos_embedding.weight": pos_embed.contiguous()}, str(pos_path))

    # Write config.json
    hf_config = build_hf_config(model_cfg, global_step=global_step)
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    logger.info("config.json written → %s", config_path)

    # Write generation_config.json
    gen_config_path = output_dir / "generation_config.json"
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(build_generation_config(model_cfg["vocab_size"]), f, indent=2)
    logger.info("generation_config.json written → %s", gen_config_path)

    # Summary
    total_params = sum(t.numel() for k, t in hf_dict.items() if k != "lm_head.weight")
    logger.info(
        "Conversion complete.  ~%.2fB parameters  →  %s",
        total_params / 1e9,
        output_dir,
    )
    logger.info(
        "Load with:  AutoModelForCausalLM.from_pretrained('%s', trust_remote_code=False)",
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a Neuron_SP step_*.pt checkpoint to HuggingFace "
            "LlamaForCausalLM safetensors format."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-path",
        required=True,
        metavar="PATH",
        help=(
            "Path to a step_*.pt file OR a directory; if a directory, the "
            "checkpoint with the highest step number is used automatically."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Destination directory for the HuggingFace model files.",
    )
    # Optional overrides
    p.add_argument("--hidden-size",       type=int, default=None, metavar="N",
                   help="Override hidden size (inferred from checkpoint if omitted).")
    p.add_argument("--num-layers",        type=int, default=None, metavar="N",
                   help="Override number of transformer layers.")
    p.add_argument("--num-heads",         type=int, default=None, metavar="N",
                   help="Override number of attention heads.")
    p.add_argument("--vocab-size",        type=int, default=None, metavar="N",
                   help="Override vocabulary size.")
    p.add_argument("--intermediate-size", type=int, default=None, metavar="N",
                   help="Override MLP intermediate (SwiGLU hidden) size.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    override_cfg: Dict[str, Optional[int]] = {
        "hidden_size":       args.hidden_size,
        "num_layers":        args.num_layers,
        "num_heads":         args.num_heads,
        "vocab_size":        args.vocab_size,
        "intermediate_size": args.intermediate_size,
    }
    # Strip None values; _infer_model_config handles missing keys
    override_cfg = {k: v for k, v in override_cfg.items() if v is not None}

    convert(
        checkpoint_path=Path(args.checkpoint_path),
        output_dir=Path(args.output_dir),
        override_cfg=override_cfg or None,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
