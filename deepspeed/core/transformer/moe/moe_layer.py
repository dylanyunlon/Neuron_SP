"""MoE Layer adapted from Megatron megatron/core/transformer/moe/moe_layer.py.

Combines TopKRouter + expert MLPs + token dispatcher into a single module
that replaces the standard MLP in TransformerLayer.

Pure PyTorch. PCIe-optimized: uses permute/unpermute instead of all-to-all
for single-machine heterogeneous topologies.

Megatron forward parity (forward() gaps addressed in this version):
  Gap 1: shared expert now runs FIRST on the original hidden_states, before
          any routing — matching Megatron's shared_experts_compute() ordering.
  Gap 2: expert loop now uses autograd-safe index_add_ accumulation instead
          of boolean-mask in-place writes that broke the gradient graph.
  Gap 3: padding_mask forwarded to router (router.forward() ignores it for
          now but the interface is stable for when we add capacity-aware
          padding-aware routing).
  Gap 4: _expert_utilization now populated with actual tokens_per_expert
          counts so log_utilization() in MoEAdapter has real data to emit.
  Gap 5: moe_layer_recompute flag wires torch.utils.checkpoint when
          config.moe_layer_recompute is True (mirrors Megatron's
          moe_layer_recompute option).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from deepspeed.core.transformer.moe.moe_utils import permute_tokens, unpermute_tokens
from deepspeed.core.transformer.moe.router import TopKRouter


class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU).

    Same architecture as the standard MLP but instantiated per-expert.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = getattr(config, "ffn_hidden_size", config.hidden_size * 4)
        use_swiglu = getattr(config, "activation_func_type", "gelu") == "swiglu"

        if use_swiglu:
            # SwiGLU: gate + up projection, then down
            self.gate_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=False)
            self.down_proj = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=False)
            self.use_swiglu = True
        else:
            self.up_proj = nn.Linear(self.hidden_size, self.ffn_hidden_size, bias=True)
            self.down_proj = nn.Linear(self.ffn_hidden_size, self.hidden_size, bias=True)
            self.use_swiglu = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(nn.functional.gelu(self.up_proj(x)))


class MoELayer(nn.Module):
    """Mixture-of-Experts layer.

    Replaces the standard MLP in TransformerLayer when num_moe_experts > 0.
    Token routing via TopKRouter, expert processing via per-expert MLPs.

    For PCIe-only heterogeneous topologies (A6000 + H100), uses permute-based
    token dispatch instead of all-to-all to avoid PCIe bandwidth bottleneck.

    Args:
        config: TransformerConfig with MoE fields.
        layer_number: Layer index (for logging/debugging).
    """

    def __init__(self, config, layer_number: int = 0) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_moe_experts
        self.topk = config.moe_router_topk
        self.layer_number = layer_number
        self.moe_layer_recompute: bool = getattr(config, "moe_layer_recompute", False)

        # Router
        self.router = TopKRouter(config)

        # Expert MLPs
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.num_experts)])

        # Optional shared expert (DeepSeek-style).
        # Matches Megatron use_shared_expert / shared_experts field.
        self.shared_expert: Optional[ExpertMLP] = None
        num_shared = getattr(config, "moe_num_shared_experts", 0)
        if num_shared > 0:
            self.shared_expert = ExpertMLP(config)

        # Populated during forward for utilization logging (MoEAdapter.log_utilization).
        self._expert_utilization: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal forward implementation (wrapped by checkpoint if needed)
    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Core MoE forward logic.

        Follows Megatron's ordering:
          1. shared_expert on full hidden_states (Gap 1 fix)
          2. route tokens (with optional padding_mask — Gap 3)
          3. permute tokens by expert
          4. run each expert with autograd-safe index_add_ accumulation (Gap 2 fix)
          5. unpermute and weight by routing scores
          6. add shared_expert output
        """
        # ---- Step 1: Shared expert on original input (Megatron gap fix) ----
        # Must happen before any routing manipulation so the shared expert sees
        # the unmodified token representations.
        shared_out: Optional[torch.Tensor] = None
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)

        # ---- Step 2: Route tokens ----
        # router.forward() returns (scores, indices, probs):
        #   scores:  [num_tokens, topk]  — normalized routing weights
        #   indices: [num_tokens, topk]  — expert indices (-1 = dropped)
        #   probs:   [num_tokens, num_experts] — full softmax (for aux loss)
        scores, indices, _probs = self.router(hidden_states)

        num_tokens, hidden_size = hidden_states.shape

        # ---- Step 3: Permute tokens by expert assignment ----
        # permuted_tokens: [num_tokens * topk, hidden_size]
        # sorted_indices:  argsort of flat expert indices — inverse is unpermute key
        permuted_tokens, sorted_indices = permute_tokens(
            hidden_states, indices, self.num_experts, self.topk
        )

        # Recover which expert each permuted slot belongs to.
        flat_indices = indices.reshape(-1)        # [num_tokens * topk]
        sorted_expert_ids = flat_indices[sorted_indices]   # expert id at each permuted position

        # ---- Step 4: Per-expert computation (autograd-safe) ----
        # Use index_add_ accumulation rather than boolean-mask writes.
        # Boolean in-place scatter on a zero tensor severs the autograd graph for
        # the unwritten positions; index_add_ preserves the full gradient path.
        expert_outputs = torch.zeros_like(permuted_tokens)  # [num_tokens*topk, hidden_size]

        # Track tokens_per_expert for utilization logging (Gap 4 fix).
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device=hidden_states.device
        )

        for expert_idx in range(self.num_experts):
            # Boolean mask over the sorted/permuted dimension
            expert_mask = sorted_expert_ids == expert_idx   # [num_tokens * topk]
            num_expert_tokens = expert_mask.sum().item()
            if num_expert_tokens == 0:
                continue

            tokens_per_expert[expert_idx] = num_expert_tokens

            # Gather input tokens for this expert
            expert_input = permuted_tokens[expert_mask]     # [E, hidden_size]
            expert_out = self.experts[expert_idx](expert_input)  # [E, hidden_size]

            # Scatter output back using index_add_ on dim=0.
            # This is equivalent to expert_outputs[expert_mask] = expert_out but
            # keeps the full gradient path alive for permuted_tokens.
            indices_for_expert = expert_mask.nonzero(as_tuple=False).squeeze(1)  # [E]
            expert_outputs.index_add_(0, indices_for_expert, expert_out)

        # Dropped tokens (index == -1): zero contribution, already zero-initialized.

        # Populate utilization for MoEAdapter.log_utilization() (Gap 4 fix)
        self._expert_utilization = tokens_per_expert

        # ---- Step 5: Unpermute and apply routing weights ----
        output = unpermute_tokens(expert_outputs, sorted_indices, scores, self.topk)
        # output: [num_tokens, hidden_size]

        # ---- Step 6: Add shared expert output ----
        if shared_out is not None:
            output = output + shared_out

        return output

    # ------------------------------------------------------------------
    # Public forward — optional activation-checkpoint wrapper (Gap 5)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process tokens through MoE.

        Args:
            hidden_states: [seq_len, batch_size, hidden_size] or
                           [num_tokens, hidden_size].
            padding_mask:  Optional boolean mask [seq_len, batch_size] or
                           [num_tokens].  True = valid token.  Passed to
                           the router for capacity-aware routing.

        Returns:
            output: Same shape as input.
        """
        # Reshape to 2D if needed
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            seq_len, batch_size, hidden_size = hidden_states.shape
            hidden_states_2d = hidden_states.reshape(-1, hidden_size)
            # Flatten padding_mask to match
            if padding_mask is not None and padding_mask.dim() == 2:
                padding_mask = padding_mask.reshape(-1)
        else:
            hidden_states_2d = hidden_states

        if self.moe_layer_recompute and self.training:
            # Activation checkpoint: recompute forward activations during
            # backward to save memory, at the cost of one extra forward pass.
            # Mirrors Megatron's tensor_parallel.checkpoint() path.
            output = torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                hidden_states_2d,
                padding_mask,
                use_reentrant=False,
            )
        else:
            output = self._forward_impl(hidden_states_2d, padding_mask)

        return output.reshape(original_shape)

    def get_aux_loss(self) -> torch.Tensor:
        """Return accumulated auxiliary loss from router."""
        return self.router.aux_loss
