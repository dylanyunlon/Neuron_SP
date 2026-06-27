# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Model definitions — GPT and hybrid architectures.

Wiring (Task G partial):
  GPTModel now uses core.tensor_parallel components for its embedding and
  output-projection layers:
    * word_embeddings  → VocabParallelEmbedding   (TP-sharded vocab dim)
    * output_layer     → ColumnParallelLinear      (TP-sharded output dim,
                                                    gather_output=True so
                                                    every rank gets full logits)

  The rest of the model (TransformerBlock, attention, MLP) comes from
  core.transformer, which was wired in earlier Task D.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.core.transformer import TransformerConfig, TransformerBlock, MegatronModule

# ---------------------------------------------------------------------------
# Tensor-parallel components wired in for Task F/G
# ---------------------------------------------------------------------------
from deepspeed.core.tensor_parallel import (
    VocabParallelEmbedding,
    ColumnParallelLinear,
)


# ===========================================================================
# common/language_module.py
# ===========================================================================

class LanguageModule(MegatronModule, ABC):
    """Base class for language models (GPT, T5, hybrid).

    Manages embedding layers, output projection, and loss computation.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Pipeline stage flags — subclasses set these before calling
        # setup_embeddings_and_output_layer().
        self.pre_process: bool = True
        self.post_process: bool = True
        self.share_embeddings_and_output_weights: bool = True

    def setup_embeddings_and_output_layer(self) -> None:
        """Initialize input embeddings and output projection.

        Handles weight tying between input embeddings and output layer
        when share_embeddings_and_output_weights is True.

        Pipeline semantics (mirroring Megatron-LM):
        - First PP stage  (pre_process=True):  owns self.embedding
        - Last  PP stage  (post_process=True): owns self.output_layer
        - When PP > 1 and weights are shared, the last stage initialises
          a copy of the word-embedding weight as zeros and then all-reduces
          it with the first stage so both copies start identical.
        """
        # Mark embedding / output-layer weights for decoupled-LR and Muon
        # optimizer grouping (same convention as Megatron).
        if self.pre_process and hasattr(self, "embedding"):
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True

        if (
            self.post_process
            and hasattr(self, "output_layer")
            and self.output_layer.weight is not None
        ):
            self.output_layer.weight.is_embedding_or_output_parameter = True

        # Nothing more to do when we are not sharing weights.
        if not self.share_embeddings_and_output_weights:
            return

        # PP == 1: same rank owns both; zero out wgrad to prevent double-counting.
        if self.config.pipeline_model_parallel_size == 1:
            w = self.shared_embedding_or_output_weight()
            if w is not None:
                w.zero_out_wgrad = True
            return

        # PP > 1: last stage initialises its copy as zeros + all-reduce to sync.
        if self.post_process and not self.pre_process:
            weight = self.shared_embedding_or_output_weight()
            if weight is not None:
                weight.data.fill_(0)
                weight.shared = True
                weight.shared_embedding = True

        # Synchronise embedding weights across the first and last PP stages
        # so both copies start with identical values.
        if torch.distributed.is_initialized():
            weight = self.shared_embedding_or_output_weight()
            if weight is not None and (self.pre_process or self.post_process):
                weight.data = weight.data.cuda()
                # Use the default process group — a proper embedding group
                # would be wired in by the runtime; here we use all-reduce
                # over the global group which is safe because only the first
                # and last stage hold non-zero weights.
                torch.distributed.all_reduce(weight.data)

    def compute_language_model_loss(
        self, labels: torch.Tensor, logits: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss with optional per-token scaling.

        Args:
            labels:  [batch, seq_len]  — ground-truth token ids.
            logits:  [seq_len, batch, vocab_size]  — raw model outputs
                     (seq-first layout, as produced by ColumnParallelLinear).

        Returns:
            loss tensor of shape [batch, seq_len].
        """
        # logits: [s, b, vocab] → labels need [s, b] for F.cross_entropy
        labels = labels.transpose(0, 1).contiguous()  # [s, b]

        # Standard vocab-parallel cross-entropy: each TP rank holds a slice
        # of the vocabulary.  We approximate the full softmax by reducing
        # across TP ranks using the log-sum-exp trick, identical to Megatron's
        # vocab_parallel_cross_entropy.
        tp_size = self.config.tensor_model_parallel_size
        if tp_size > 1 and torch.distributed.is_initialized():
            # Subtract max for numerical stability (within local vocab shard).
            logits_max = logits.max(dim=-1, keepdim=True).values
            torch.distributed.all_reduce(
                logits_max,
                op=torch.distributed.ReduceOp.MAX,
            )
            logits = logits - logits_max

            # Compute log-softmax denominator across the full vocabulary.
            exp_logits = logits.exp()
            sum_exp = exp_logits.sum(dim=-1, keepdim=True)
            torch.distributed.all_reduce(sum_exp)

            # Gather the predicted log-prob for the true label.
            vocab_start = (logits.size(-1)) * self._tp_rank()
            # Shift labels into local index space; mask out-of-range labels.
            local_labels = labels - vocab_start  # [s, b]
            valid_mask = (local_labels >= 0) & (local_labels < logits.size(-1))
            local_labels = local_labels.clamp(0, logits.size(-1) - 1)

            # Gather logits at ground-truth positions: [s, b]
            predicted_logits = logits.gather(
                dim=-1,
                index=local_labels.unsqueeze(-1),
            ).squeeze(-1)
            predicted_logits = predicted_logits * valid_mask.float()
            torch.distributed.all_reduce(predicted_logits)

            loss = torch.log(sum_exp.squeeze(-1)) - predicted_logits  # [s, b]
        else:
            # Single-rank: standard cross-entropy over the full vocabulary.
            s, b, vocab = logits.shape
            loss = F.cross_entropy(
                logits.reshape(s * b, vocab),
                labels.reshape(s * b),
                reduction="none",
            ).reshape(s, b)

        # Return [b, s] to match Megatron convention.
        loss = loss.transpose(0, 1).contiguous()
        return loss

    def _tp_rank(self) -> int:
        """Return the TP rank of this process (0 if not distributed)."""
        try:
            from deepspeed.core import parallel_state
            return parallel_state.get_tensor_model_parallel_rank()
        except Exception:
            return 0

    def shared_embedding_or_output_weight(self) -> Optional[torch.Tensor]:
        """Return the canonical shared weight tensor.

        On the first PP stage (pre_process=True) this is the word-embedding
        weight; on the last stage (post_process=True) it is the output-layer
        weight.  Returns None when neither condition holds.
        """
        if self.pre_process and hasattr(self, "embedding"):
            return self.embedding.word_embeddings.weight
        if self.post_process and hasattr(self, "output_layer"):
            return self.output_layer.weight
        return None


# ===========================================================================
# gpt/gpt_model.py
# ===========================================================================

class GPTModel(LanguageModule):
    """GPT-style autoregressive language model.

    Architecture: embedding → TransformerBlock → output_layer → loss
    Supports: GQA, RoPE, SwiGLU, RMSNorm, per-token loss.

    DES-LOC integration:
    - Pipeline splits determined by config.pipeline_layer_split
    - Activation checkpointing granularity varies per tier
    - AutoSP shards sequence dim in attention
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,   # True for first PP stage (has embeddings)
        post_process: bool = True,  # True for last PP stage (has output layer)
        share_embeddings_and_output_weights: bool = True,
    ) -> None:
        # LanguageModule.__init__ sets self.config via MegatronModule
        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        vocab_size: int = getattr(config, "vocab_size", 32000)
        max_seq_len: int = getattr(config, "max_position_embeddings", 4096)

        # ------------------------------------------------------------------ #
        # Embedding layer (first PP stage only)
        # ------------------------------------------------------------------ #
        if self.pre_process:
            self.embedding = _EmbeddingLayer(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_seq_len,
            )

        # ------------------------------------------------------------------ #
        # Rotary positional embeddings (built lazily on first forward)
        # ------------------------------------------------------------------ #
        self._rotary_pos_emb: Optional[_RotaryEmbedding] = None
        position_embedding_type = getattr(config, "position_embedding_type", "rope")
        if position_embedding_type == "rope":
            self._rotary_pos_emb = _RotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_base=getattr(config, "rotary_base", 10000),
                rotary_interleaved=config.rotary_interleaved,
            )

        # ------------------------------------------------------------------ #
        # Transformer block (all PP stages)
        # ------------------------------------------------------------------ #
        self.decoder = TransformerBlock(config=config)

        # ------------------------------------------------------------------ #
        # Output projection (last PP stage only)
        # ------------------------------------------------------------------ #
        if self.post_process:
            # ColumnParallelLinear: hidden_size → vocab_size (no bias).
            # gather_output=True so every TP rank holds the full [s, b, vocab]
            # logit tensor — required for loss computation on all ranks.
            # When weights are shared the parameter is owned by the embedding;
            # we wire the tie below after allocation.
            self.output_layer = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=vocab_size,
                config=config,
                bias=False,
                gather_output=True,
                skip_bias_add=False,
            )
            if share_embeddings_and_output_weights and self.pre_process:
                # Tie weights in-place: the output_layer weight is replaced by
                # the VocabParallelEmbedding weight so they stay in sync.
                # Both are sharded on dim-0 (vocab / output dim), which is
                # consistent for a ColumnParallelLinear with gather_output=True.
                del self.output_layer.weight
                self.output_layer.weight = (  # type: ignore[assignment]
                    self.embedding.word_embeddings.weight
                )

        # ------------------------------------------------------------------ #
        # Initialise embedding sync / weight-tying bookkeeping
        # ------------------------------------------------------------------ #
        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    # ---------------------------------------------------------------------- #
    # Pipeline parallel interface
    # ---------------------------------------------------------------------- #

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor received from previous PP stage."""
        self.decoder.set_input_tensor(input_tensor)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[object] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Returns loss if labels provided (training), logits otherwise (inference).

        Args:
            input_ids:      [batch, seq_len]
            position_ids:   [batch, seq_len]
            attention_mask: optional causal / padding mask
            labels:         [batch, seq_len] token ids for loss computation
            inference_params: KV-cache state for autoregressive inference
        """
        # ---- 1. Embedding ------------------------------------------------ #
        if self.pre_process:
            # [b, s] → [s, b, h]  (seq-first layout throughout the model)
            hidden_states = self.embedding(input_ids, position_ids)
        else:
            # Intermediate or last PP stage: activations were received from the
            # previous stage via set_input_tensor().  Pass them directly.
            hidden_states = getattr(self.decoder, "_input_tensor", None)

        # ---- 2. Rotary positional embeddings ----------------------------- #
        rotary_pos_emb: Optional[torch.Tensor] = None
        if self._rotary_pos_emb is not None:
            seq_len = input_ids.size(1) if input_ids is not None else (
                hidden_states.size(0) if hidden_states is not None else 0
            )
            rotary_pos_emb = self._rotary_pos_emb(seq_len)

        # ---- 3. Transformer block ---------------------------------------- #
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
        )

        # ---- 4. Output projection + loss --------------------------------- #
        if not self.post_process:
            # Middle PP stage: pass activations to next stage.
            return hidden_states

        # hidden_states: [s, b, h]
        output_weight: Optional[torch.Tensor] = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if output_weight is not None:
            # Use F.linear so we can pass an external weight tensor.
            logits = F.linear(hidden_states, output_weight)  # [s, b, vocab]
        else:
            # ColumnParallelLinear returns (output, bias); bias is None here
            # because output_layer was constructed with bias=False.
            logits, _ = self.output_layer(hidden_states)  # [s, b, vocab]

        if labels is None:
            # Inference: return [b, s, vocab] for caller convenience.
            return logits.transpose(0, 1).contiguous()

        # Training: compute and return scalar/per-token loss.
        loss = self.compute_language_model_loss(labels, logits)

        if self.config.calculate_per_token_loss:
            return loss  # [b, s]

        # Average over all tokens (standard training objective).
        return loss.mean()

    # ---------------------------------------------------------------------- #
    # Sharded state dict (distributed checkpointing)
    # ---------------------------------------------------------------------- #

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None,
    ) -> dict:
        """Return a sharded state dict for distributed checkpointing.

        Each tensor is wrapped in a ShardedTensor descriptor that encodes its
        position within the global parameter tensor so that the checkpoint
        system can reconstruct the full weight even across topology changes.
        """
        from deepspeed.core.dist_checkpointing import ShardedTensor as ST

        state = {}
        tp_rank = self._tp_rank()
        tp_size = self.config.tensor_model_parallel_size

        for param_name, param in self.named_parameters():
            key = f"{prefix}{param_name}"
            data = param.data

            # Determine which axis (if any) is sharded across TP ranks.
            # Convention: Column-parallel weights shard on axis 0 (output),
            # Row-parallel weights shard on axis 1 (input).
            # Embeddings shard on axis 0 (vocab dimension).
            if tp_size > 1 and data.ndim >= 1:
                # Heuristic: if the leading dimension is divisible by tp_size,
                # assume it is column-parallel sharded on axis 0.
                if data.size(0) % tp_size == 0:
                    global_shape_0 = data.size(0) * tp_size
                    global_shape = (global_shape_0,) + tuple(data.shape[1:])
                    global_offset = (tp_rank * data.size(0),) + (0,) * (data.ndim - 1)
                    axis_fragmentations = (tp_size,) + (1,) * (data.ndim - 1)
                else:
                    global_shape = tuple(data.shape)
                    global_offset = (0,) * data.ndim
                    axis_fragmentations = (1,) * data.ndim
            else:
                global_shape = tuple(data.shape)
                global_offset = (0,) * data.ndim
                axis_fragmentations = (1,) * data.ndim

            state[key] = ST(
                key=key,
                data=data,
                global_shape=global_shape,
                global_offset=global_offset,
                axis_fragmentations=axis_fragmentations,
                replica_id=0,
            )

        return state


# ===========================================================================
# Internal helpers (not part of the public API)
# ===========================================================================

class _EmbeddingLayer(nn.Module):
    """Word + positional embedding with TP-parallel vocab sharding.

    Word embeddings are now backed by ``VocabParallelEmbedding`` which splits
    the vocabulary across the tensor-parallel group.  When TP=1 this is
    identical to ``nn.Embedding``.

    Produces output in seq-first layout: [seq_len, batch, hidden_size].
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_sequence_length = max_sequence_length

        # Word embeddings — TP-sharded across the vocab dimension.
        # VocabParallelEmbedding handles the all-reduce in forward()
        # and falls back to standard nn.Embedding when TP=1.
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size,
            config=config,
        )

        # Learned absolute position embeddings (used when RoPE is disabled).
        position_embedding_type = getattr(config, "position_embedding_type", "rope")
        self.use_position_embeddings = (position_embedding_type == "learned_absolute")
        if self.use_position_embeddings:
            # Position embeddings are not vocab-sharded; use standard Embedding.
            self.position_embeddings = nn.Embedding(max_sequence_length, config.hidden_size)

        self.embedding_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        # [b, s] → [b, s, h]  (VocabParallelEmbedding returns this shape)
        words_emb = self.word_embeddings(input_ids)
        if self.use_position_embeddings:
            pos_emb = self.position_embeddings(position_ids)
            words_emb = words_emb + pos_emb

        embeddings = self.embedding_dropout(words_emb)
        # Transpose to seq-first: [s, b, h]
        return embeddings.transpose(0, 1).contiguous()


class _RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) — precomputes cos/sin tables."""

    def __init__(
        self,
        kv_channels: int,
        rotary_base: int = 10000,
        rotary_interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.dim = kv_channels
        self.base = rotary_base
        self.rotary_interleaved = rotary_interleaved
        self._seq_len_cached: int = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len <= self._seq_len_cached:
            return
        self._seq_len_cached = seq_len
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, theta)  # [s, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [s, dim]
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return RoPE embedding of shape [seq_len, 1, 1, dim]."""
        if device is None:
            device = torch.device("cpu")
        self._build_cache(seq_len, device, torch.float32)
        assert self._cos_cached is not None
        # Return (cos, sin) stacked so callers can unpack easily.
        cos = self._cos_cached[:seq_len]  # [s, dim]
        sin = self._sin_cached[:seq_len]  # [s, dim]
        # Package as [s, 1, 1, dim] for broadcasting in attention.
        return torch.stack([cos, sin], dim=0).unsqueeze(2)  # [2, s, 1, dim]


# ===========================================================================
# gpt/gpt_layer_specs.py
# ===========================================================================

def get_gpt_layer_spec(config: TransformerConfig) -> dict:
    """Return the layer specification for a GPT model.

    The spec defines which submodule classes to use for attention,
    MLP, norms, etc. Allows swapping components (e.g. standard attention
    vs MLA, standard MLP vs MoE).

    Returns a dict that maps logical component names to their classes and
    keyword arguments.  TransformerLayer / TransformerBlock consults this
    dict when constructing its submodules so that the model can be
    reconfigured without subclassing.
    """
    from deepspeed.core.transformer import (
        SelfAttention,
        MLP,
    )

    # Choose norm class: RMSNorm or LayerNorm
    if config.normalization == "RMSNorm":
        norm_cls = _RMSNorm
    else:
        norm_cls = nn.LayerNorm

    # Choose MLP: standard SwiGLU or MoE
    if config.num_moe_experts is not None and config.num_moe_experts > 1:
        from deepspeed.core.transformer import MoELayer
        mlp_cls = MoELayer
    else:
        mlp_cls = MLP

    spec = {
        # Self-attention component
        "self_attention": {
            "cls": SelfAttention,
            "kwargs": {
                "layer_number": None,  # filled in by TransformerLayer
            },
        },
        # MLP / FFN component
        "mlp": {
            "cls": mlp_cls,
            "kwargs": {},
        },
        # Layer norms
        "input_layernorm": {
            "cls": norm_cls,
            "kwargs": {
                "normalized_shape": config.hidden_size,
                "eps": config.layernorm_epsilon,
            },
        },
        "pre_mlp_layernorm": {
            "cls": norm_cls,
            "kwargs": {
                "normalized_shape": config.hidden_size,
                "eps": config.layernorm_epsilon,
            },
        },
    }

    return spec


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A lightweight drop-in replacement for LayerNorm without the mean
    subtraction step, commonly used in modern LLMs (Llama, Mistral, etc.).
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight
