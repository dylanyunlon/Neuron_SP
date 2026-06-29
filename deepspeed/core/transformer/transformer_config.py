# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""TransformerConfig — ported from Megatron-LM with DES-LOC per-layer tier assignment.

Evolution summary (80 Megatron commits → this file)
----------------------------------------------------
Key milestones ported from Megatron-LM evolution:

* M2274–M2312  : softmax_type, QK-logit clipping, sink attention, window_attn_skip_freq,
                  quick-geglu activation, isolation RNG for sampling.
* M2343–M2387  : SWA + full attention mixing, expert-MLP bias, max_position_embeddings
                  deprecation, avoid RoPE split-concat.
* M2407–M2459  : unified CUDA-graph scope API (cuda_graph_impl / cuda_graph_modules /
                  inference_cuda_graph_scope).
* M2636–M2718  : TE custom quantization recipe, inference linear layer, HybridEP flex
                  dispatcher, NVFP4 MOE, shared-expert overlap for FlexDispatcher.
* M2799–M2879  : per-module TE quant config, QKV sub-sharding, latent MoEs, QK-logit
                  clipping, batch-invariance, NVLS fused RS+residual+norm+AG kernel.
* M2906–M3068  : CUDA-graph scope refactor, MTP support in hybrid models, misc inference
                  cleanup, router replay, various CUDA-graph improvements.
* M3096–M3253  : CUDA-graph improvements (retake), TransformerConfig generate_arguments,
                  first/last PP-stage layer count checking, ModuleSpec→Protocols for MLP.
* M3371–M3539  : fp32 residuals fix, fVPP, μP (Maximal Update Parameterization), MoE
                  inference-optimized layers, removed deprecated GroupedMLP, fused
                  dLN+add backward, MLA DOWN-proj GEMM fusion, torch grouped-gemm BF16/MXFP8.
* M3573–M3712  : A2A overlap assertion fix, Forced load imbalance, misc MXFP8 inference,
                  CPU-offloading + full-iteration CUDA-graph, dense/MoE upcycling validation.
* M3713–M3868  : Fix TransformerConfig validation for mixed dense/MoE upcycling, new A2A
                  high-priority stream, CUDA-graph API decompose, per-layer TE quant.
* M3884–M4147  : vLLM grouped-gemm backend, chunked MLP, allgather dispatcher for inference,
                  HybridEP permute fusion, GEMM+SwiGLU fused MLP, NVFP4 fixes, TEFusedDenseMLP,
                  paged stashing, fine-grained offload throttle, mxfp8 LM-head output projection,
                  GDN selective-recompute for norm_out, mtp_detach_heads.

DES-LOC extension (Neuron_SP-specific)
---------------------------------------
Two complementary extensions live in this file:

1. **Per-layer GPU-tier assignment** (desloc_h100_layers / desloc_a6000_layers)
   — Zero-based global layer indices pinned to H100 or A6000 GPUs.
   — Populated automatically by ``_resolve_desloc_tiers()`` from the chosen
     ``desloc_tier_strategy`` ("front_heavy" | "back_heavy" | "interleave" | "manual").

2. **Parameter-tier tagging** (desloc_tier_enabled + keyword lists)
   — When ``desloc_tier_enabled=True`` the model constructor annotates each
     named parameter with a ``desloc_tier`` attribute ('x' | 'u' | 'v') so the
     DES-LOC tiered all-reduce scheduler can bucket communications by tier and
     apply the (Kx, Ku, Kv) decomposed sync periods from ``DesLocConfig``.

Usage example::

    cfg = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        desloc_tier_strategy="front_heavy",
        desloc_h100_layer_fraction=0.5,   # put first 50% on H100
        desloc_tier_enabled=True,
    )
    # TransformerConfig.__post_init__ calls _resolve_desloc_tiers()
    h100 = cfg.desloc_h100_layers   # [0..15]
    a6000 = cfg.desloc_a6000_layers # [16..31]
"""

from __future__ import annotations

import math
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from deepspeed.core.model_parallel_config import ModelParallelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight shims for Megatron utilities not present in deepspeed.core
# ---------------------------------------------------------------------------

def _init_method_normal(std: float) -> Callable:
    """Return a weight-init callable: normal(0, std)."""
    import torch.nn.init as init
    def _fn(tensor: torch.Tensor) -> None:
        init.normal_(tensor, mean=0.0, std=std)
    return _fn


def _scaled_init_method_normal(std: float, num_layers: int, multiplier: float = 2.0) -> Callable:
    """Return output-layer init: normal(0, std / sqrt(multiplier * num_layers))."""
    import torch.nn.init as init
    scaled_std = std / math.sqrt(multiplier * max(num_layers, 1))
    def _fn(tensor: torch.Tensor) -> None:
        init.normal_(tensor, mean=0.0, std=scaled_std)
    return _fn


def _mup_scaled_init_method_normal(
    std: float,
    num_layers: int,
    width_mult: float,
    multiplier: float = 2.0,
) -> Callable:
    """MuP output-layer init with both depth- and width-scaling."""
    import torch.nn.init as init
    scaled_std = std / math.sqrt(max(width_mult, 1e-8)) / math.sqrt(multiplier * max(num_layers, 1))
    def _fn(tensor: torch.Tensor) -> None:
        init.normal_(tensor, mean=0.0, std=scaled_std)
    return _fn


@dataclass
class TransformerConfig(ModelParallelConfig):
    """Full transformer configuration extending ModelParallelConfig.

    Closely mirrors Megatron-LM's TransformerConfig (2 700+ lines, 80 commits)
    with the following DeepSpeed / DES-LOC additions:

    * ``desloc_h100_layers`` / ``desloc_a6000_layers`` — explicit per-layer
      tier assignment lists (zero-based global indices).
    * ``desloc_tier_strategy`` — automatic assignment strategy.
    * ``desloc_h100_layer_fraction`` — fraction of layers placed on H100 when
      using "front_heavy" or "back_heavy" strategies.
    * ``desloc_tier_enabled`` + keyword lists — parameter-tier tagging for the
      decomposed all-reduce scheduler (DESLOCAdamW / engine.py).
    * ``pipeline_layer_split`` — heterogeneous PP split for TransformerBlock.
    """

    # ------------------------------------------------------------------
    # Core model architecture
    # ------------------------------------------------------------------

    num_layers: int = 0
    """Number of transformer layers in a transformer block."""

    hidden_size: int = 0
    """Transformer hidden size."""

    num_attention_heads: int = 0
    """Number of transformer attention heads."""

    num_query_groups: Optional[int] = None
    """Number of query groups for group query attention (GQA/MQA).
    If None, falls back to num_attention_heads (standard MHA)."""

    ffn_hidden_size: Optional[int] = None
    """Feed-Forward Network hidden size. Defaults to 4*hidden_size for
    standard transformers."""

    kv_channels: Optional[int] = None
    """Projection weights dimension in multi-head attention.
    Defaults to hidden_size // num_attention_heads."""

    # ------------------------------------------------------------------
    # Multi-Token Prediction (MTP)
    # ------------------------------------------------------------------

    mtp_num_layers: Optional[int] = None
    """Number of Multi-Token Prediction (MTP) layers.
    MTP extends the prediction scope to multiple future tokens at each position."""

    mtp_loss_scaling_factor: Optional[float] = 0.1
    """Weighting factor of MTP loss.  Average of MTP losses across all depths
    multiplied by this scaling factor gives the overall MTP auxiliary loss."""

    mtp_use_repeated_layer: bool = False
    """Use a single MTP layer repeatedly instead of multiple separate layers."""

    mtp_detach_heads: bool = False
    """If True, detach MTP head inputs from the main model graph.
    This prevents MTP loss gradients from flowing back to the main model,
    only training the MTP heads themselves.

    When True, MTP model parameters should be tagged with
    ``param.grad_norm_group = 'mtp'`` so that the optimizer clips their
    gradients independently from the main model norm.  See
    ``MTP_GRAD_NORM_GROUP`` in ``deepspeed.core.optimizer``.
    # From Megatron M4171: Clip mtp grads separately when mtp_detach_heads=True.
    """

    mtp_hybrid_override_pattern: Optional[str] = None
    """DEPRECATED: Use unified hybrid_layer_pattern instead.
    Legacy argument for loading old checkpoints. Force a specific hybrid layer
    pattern for MTP layers."""

    # ------------------------------------------------------------------
    # Pipeline parallel layout
    # ------------------------------------------------------------------

    num_layers_in_first_pipeline_stage: Optional[int] = None
    """Number of transformer layers on first pipeline stage.
    None implies equal layer division across PP ranks."""

    num_layers_in_last_pipeline_stage: Optional[int] = None
    """Number of transformer layers on last pipeline stage.
    None implies equal layer division across PP ranks."""

    account_for_embedding_in_pipeline_split: bool = False
    """If set, the embedding layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    account_for_loss_in_pipeline_split: bool = False
    """If set, the loss layer will be treated as a standard transformer
    layer in the context of partition and placement for pipeline parallelism."""

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    softmax_scale: Optional[float] = None
    """Softmax scale for attention. If None, defaults to 1/sqrt(kv_channels)."""

    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = 'vanilla'
    """Softmax variant.
    'off-by-one': from https://www.evanmiller.org/attention-is-off-by-one.html.
    'learnable': learnable offset."""

    add_qkv_bias: bool = False
    """Add a bias term only for QKV projections."""

    attention_output_gate: bool = False
    """Whether to apply output gate to the attention layers."""

    qk_layernorm: bool = False
    """Whether to apply normalization to the query and key embeddings."""

    qk_l2_norm: bool = False
    """Whether to apply Llama-4-style QK L2 norm."""

    qk_clip: bool = False
    """Whether to clip the query and key weights (Muon MLA training)."""

    qk_clip_alpha: float = 0.5
    """Balancing alpha for qk-clip: Q = Q * (eta ** alpha)."""

    qk_clip_threshold: float = 100.0
    """Threshold for qk-clip: eta = min(threshold / max_attention_logits, 1.0)."""

    log_max_attention_logit: bool = False
    """Whether to log the max attention logit across the whole model."""

    no_rope_freq: Optional[Union[int, List[int]]] = None
    """Controls which layers perform RoPE.  Integer N: skip every (N-1) layers.
    List: custom pattern, 1=skip, 0=apply."""

    mrope_section: Optional[List[int]] = None
    """Multimodal rope section for temporal, height, and width in rope calculation."""

    fused_single_qkv_rope: bool = False
    """If set, avoid splitting QKV before ROPE forward and avoid concatenating ROPE dgrads."""

    flash_decode: bool = False
    """Use the optimised flash decoding kernel during inference."""

    # ------------------------------------------------------------------
    # Sliding-window / sink attention
    # ------------------------------------------------------------------

    window_size: Optional[Tuple[int, int]] = None
    """Sliding window attention size; -1 = infinite window.  None = full attention."""

    window_attn_skip_freq: Optional[Union[int, List[int]]] = None
    """Frequency of full attention layers among sliding-window attention layers.
    Integer N → (N-1):1 ratio.  List → custom pattern where 1 = SWA layer."""

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    hidden_dropout: float = 0.0
    """Dropout probability for transformer hidden state."""

    attention_dropout: float = 0.0
    """Post-attention dropout probability."""

    layernorm_epsilon: float = 1e-5
    """Epsilon value for any LayerNorm / RMSNorm operations."""

    layernorm_zero_centered_gamma: bool = False
    """If True, the LayerNorm gamma is centered around 0 (improves numerical stability)."""

    # ------------------------------------------------------------------
    # Linear layers
    # ------------------------------------------------------------------

    add_bias_linear: bool = False
    """Include a bias term in all linear layers."""

    # ------------------------------------------------------------------
    # Activation / gating
    # ------------------------------------------------------------------

    gated_linear_unit: bool = False
    """Use a gated linear unit (SwiGLU / GeGLU) for the first MLP linear."""

    activation_func: Callable = F.gelu
    """Activation function for the MLP non-linearity."""

    activation_func_fp8_input_store: bool = False
    """Store the input of MLP activation function in FP8 for backprop to save memory."""

    glu_linear_offset: float = 0.0
    """Offset term in the GLU activation function: activation_func(x[0]) * (x[1] + offset).
    Only used when gated_linear_unit is True."""

    activation_func_clamp_value: Optional[float] = None
    """Clamp the output of linear_fc1 in the activation function.
    Only used when activation_func is quick_gelu."""

    use_te_activation_func: bool = False
    """Whether to use FFN activation functions implemented by TransformerEngine."""

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    normalization: Literal["LayerNorm", "RMSNorm"] = "RMSNorm"
    """Which norm to use, valid options are LayerNorm and RMSNorm."""

    apply_residual_connection_post_layernorm: bool = False
    """If True, applies residual connection after the layer norm (post-norm).
    Default is pre-norm (residual before norm)."""

    # ------------------------------------------------------------------
    # Positional embeddings
    # ------------------------------------------------------------------

    rotary_interleaved: bool = False
    """True: rotate pairs of even/odd dims (RoFormer style).
    False: rotate first-half/second-half (LLaMA style)."""

    apply_rope_fusion: bool = False
    """If True, use fused RoPE kernel."""

    # ------------------------------------------------------------------
    # Linear attention / gated delta net
    # ------------------------------------------------------------------

    experimental_attention_variant: Optional[Literal['gated_delta_net', 'dsa']] = None
    """Type of attention variant to use."""

    linear_attention_freq: Optional[Union[int, List[int]]] = None
    """Frequency between LA layers and SDPA layers."""

    linear_conv_kernel_dim: Optional[int] = 4
    """Conv kernel dimension for the gated delta net."""

    linear_key_head_dim: Optional[int] = 128
    """Query and key head dimension for the gated delta net."""

    linear_value_head_dim: Optional[int] = 128
    """Value and gate head dimension for the gated delta net."""

    linear_num_key_heads: Optional[int] = 16
    """Number of query and key heads for the gated delta net."""

    linear_num_value_heads: Optional[int] = 32
    """Number of value and gate heads for the gated delta net."""

    # DSA-specific
    dsa_indexer_n_heads: Optional[int] = None
    dsa_indexer_head_dim: Optional[int] = None
    dsa_indexer_topk: Optional[int] = None
    dsa_indexer_loss_coeff: Optional[float] = None
    dsa_indexer_use_sparse_loss: bool = False

    # ------------------------------------------------------------------
    # Mixture-of-Experts
    # ------------------------------------------------------------------

    num_moe_experts: Optional[int] = None
    """Number of MoE experts. When set, MLP is replaced by MoELayer."""

    moe_layer_freq: Union[int, List[int]] = 1
    """Frequency between MoE layers and Dense layers."""

    moe_ffn_hidden_size: Optional[int] = None
    """MoE Feed-Forward Network hidden size. Defaults to ffn_hidden_size."""

    moe_shared_expert_intermediate_size: Optional[int] = None
    """Shared expert total FFN hidden size.  None = no shared expert.
    It should be equal to 'num_shared_experts * ffn_size_of_each_shared_expert' if
    there are multiple shared experts.
    By default, the shared experts execute before the router. However, when
    moe_shared_expert_overlap or overlap_moe_expert_parallel_comm is set,
    the shared experts execute after the router, before the routed experts.
    This makes the gradients from the router and the shared experts added in
    different orders to the hidden_states, causing minor numerical differences
    in the hidden_states gradient.
    # From Megatron M2444: document shared-expert-before-router default execution order."""

    moe_shared_expert_gate: bool = False
    """Enable gate for shared expert. Only effective when
    moe_shared_expert_intermediate_size is set."""

    moe_shared_expert_overlap: bool = False
    """Enable overlapping between shared expert computations and dispatcher communications.
    Without this, the shared experts execute before the router.
    Only effective when moe_shared_expert_intermediate_size is set.
    # From Megatron M2444: shared experts run before router by default; overlap changes order."""

    moe_router_load_balancing_type: Union[str, List[str]] = "aux_loss"
    """Load balancing strategy for the router (aux_loss / seq_aux_loss /
    global_aux_loss / sinkhorn / none)."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_enable_routing_replay: bool = False
    """If True, enable the routing replay feature for MoE layers."""

    moe_router_topk_limited_devices: Optional[int] = None
    """DEPRECATED: replaced by moe_router_num_groups and moe_router_group_topk."""

    moe_router_padding_for_quantization: Optional[bool] = False
    """Pad routing_map so each expert receives a multiple of 16/32 tokens
    (needed for FP8/FP4 quantized precision)."""

    moe_router_padding_for_fp8: Optional[bool] = False
    """[Compat alias] Enables moe_router_padding_for_quantization."""

    moe_router_num_groups: Optional[int] = None
    """Number of groups to divide experts into for group-limited routing."""

    moe_router_group_topk: Optional[int] = None
    """Number of selected groups for group-limited routing."""

    moe_router_pre_softmax: bool = False
    """Enable pre-softmax(pre-sigmoid) routing for MoE."""

    moe_router_topk_scaling_factor: Optional[float] = None
    """Scaling factor for routing score in top-k selection (pre_softmax only)."""

    moe_router_score_function: Literal['softmax', 'sigmoid', 'sqrtsoftplus'] = "softmax"
    """Score function for MoE routing."""

    moe_router_dtype: Optional[Literal['fp32', 'fp64']] = None
    """Data type for routing and expert output weighted averaging."""

    moe_router_enable_expert_bias: bool = False
    """TopK routing with dynamic per-expert bias (aux-loss-free load balancing)."""

    moe_router_bias_update_rate: float = 1e-3
    """Expert bias update rate.  Default matches DeepSeekV3."""

    moe_router_force_load_balancing: bool = False
    """[Experimental] Force load balancing with random logits for benchmark."""

    moe_router_force_biased: Optional[float] = None
    """[Experimental] Apply random expert bias in normal distribution for benchmark."""

    moe_router_fusion: bool = False
    """Enable fusion for MoE TopK routing and aux-loss computation (TE >= 2.7.0)."""

    moe_grouped_gemm: bool = False
    """Use grouped GEMM for multiple local experts in a single kernel launch."""

    moe_single_grouped_weight: bool = False
    """Store expert weights as a single grouped parameter via TE GroupedTensor."""

    moe_single_grouped_bias: bool = False
    """Store expert biases as a single grouped parameter via TE GroupedTensor."""

    use_grouped_gemm_for_dense_mlp: bool = False
    """Use GroupedLinear(num_groups=1) for dense MLP to trigger TEFusedDenseMLP fusion."""

    moe_aux_loss_coeff: Union[float, List[float]] = 0.0
    """Scaling coefficient for the aux loss."""

    moe_z_loss_coeff: Optional[float] = None
    """Scaling coefficient for the z-loss."""

    moe_input_jitter_eps: Optional[float] = None
    """Add noise to the input tensor by applying jitter with a specified epsilon.
    # From Megatron M2378: router jitter Uniform distribution bounds must be created
    # with dtype=input.dtype to avoid float32/bf16 mismatch when input is bf16.
    # Correct: torch.distributions.uniform.Uniform(
    #     torch.tensor(1.0 - eps, dtype=input.dtype, device=input.device),
    #     torch.tensor(1.0 + eps, dtype=input.dtype, device=input.device),
    # ) — both bounds must match input dtype, not default float32."""

    moe_token_dropping: bool = False
    """Selectively drop and pad tokens for each expert (unsupported, must be False)."""

    moe_token_dispatcher_type: Literal['allgather', 'alltoall', 'flex'] = "allgather"
    """Type of token dispatcher to use."""

    moe_enable_deepep: bool = False
    """[Experimental] Enable DeepEP for efficient token dispatching."""

    moe_flex_dispatcher_backend: Literal['deepep', 'hybridep'] = "deepep"
    """[Experimental] Backend to use for flex token dispatcher."""

    moe_permute_fusion_into_hybridep: bool = False
    """Fuse token rearrangement ops during token dispatching for HybridEP."""

    moe_per_layer_logging: bool = False
    """Enable per-layer logging for MoE (aux loss and z loss)."""

    moe_expert_capacity_factor: Optional[float] = None
    """Capacity factor for each expert; None = no token drop."""

    moe_pad_expert_input_to_capacity: bool = False
    """Pad input for each expert to expert capacity length."""

    moe_pad_experts_for_cuda_graph_inference: bool = False
    """Switch router to dropping+padding during decode so no D2H sync is needed."""

    moe_token_drop_policy: Literal['probs', 'position'] = "probs"
    """Policy to drop tokens: 'probs' or 'position'."""

    moe_layer_recompute: bool = False
    """DEPRECATED. Memory optimisation: checkpoint moe_layer to save activation memory."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_apply_probs_on_input: bool = False
    """Apply probs on input of experts instead of applying after activation and GLU."""

    moe_latent_size: Optional[int] = None
    """Latent projection dimension for MoE. None = no latent projections."""

    moe_deepep_num_sms: int = 20
    """Number of SMs to use for DeepEP."""

    moe_hybridep_num_sms: Optional[int] = None
    """Number of SMs to use for HybridEP. None uses the DeepEP default."""

    moe_hybridep_num_blocks_permute: Optional[int] = None
    """Number of CUDA thread blocks for the permute part in HybridEP."""

    moe_hybridep_num_blocks_unpermute: Optional[int] = None
    """Number of CUDA thread blocks for the unpermute part in HybridEP."""

    moe_hybridep_num_sms_preprocessing: int = 108
    """Number of SMs for HybridEP preprocessing (metadata scan kernel)."""

    moe_mlp_glu_interleave_size: Optional[int] = None
    """Block-interleaved GLU activations in the MoE grouped MLP layer."""

    moe_expert_rank_capacity_factor: Optional[float] = None
    """Capacity factor for each expert rank; None = no token drop."""

    # Insight I7: per-GPU align_size (Megatron M3707)
    # M3707 revealed that a global align_size=16 caused unnecessary padding on
    # non-quantized paths and broke non-FP8 dispatch on A6000 (no TE quantization).
    # In our heterogeneous topology (A6000 / H100 / Blackwell) each GPU type has
    # different memory-alignment requirements:
    #   A6000  → 0   (non-quantized; padding is wasteful and triggers TE compat issues)
    #   H100   → 16  (FP8 quantized dispatch; TE fused_permute_and_pad_with_probs)
    #   Blackwell → 16 (FP4 quantized dispatch)
    # Using a per-GPU-type dict instead of a global constant prevents the mismatch
    # where one GPU type's alignment requirement is silently imposed on the others,
    # which in DES-LOC produces divergent tensor strides across tiers.
    moe_align_size: Dict[str, int] = field(default_factory=lambda: {
        "A6000": 0, "H100": 16, "Blackwell": 16
    })
    """Per-GPU-type token alignment size for MoE dispatch/permute.
    Zero means no padding (non-quantized path); 16 activates TE fused
    permute+pad kernels required for FP8/FP4 quantized dispatch.
    Keys: 'A6000', 'H100', 'Blackwell'."""

    # Paged stash
    moe_paged_stash: bool = False
    """If True, enable paged stash for all routed-expert activations needed for backward."""

    moe_paged_stash_page_size: int = 64
    """Number of tokens per page for paged stash memory management."""

    moe_paged_stash_buffer_size_factor_cuda: float = 1.10
    """Scale factor for paged stash CUDA buffer allocation."""

    moe_paged_stash_buffer_size_factor_cpu: float = 0.0
    """Scale factor for paged stash host buffer. 0 disables host buffer."""

    # ------------------------------------------------------------------
    # Multi-Latent Attention (MLA / DeepSeek style)
    # ------------------------------------------------------------------

    multi_latent_attention: bool = False
    """Whether to use Multi-Latent Attention."""

    # ------------------------------------------------------------------
    # Per-token loss
    # ------------------------------------------------------------------

    calculate_per_token_loss: bool = False
    """Whether cross-entropy loss is calculated over non-padded tokens only."""

    # ------------------------------------------------------------------
    # Precision / mixed-precision
    # ------------------------------------------------------------------

    fp32_residual_connection: bool = False
    """If True, move residual connections to fp32."""

    attention_softmax_in_fp32: bool = True
    """If True, run attention masking and softmax in fp32."""

    apply_query_key_layer_scaling: bool = False
    """If True, scale Q * K^T by 1 / layer-number (improves fp16 numeric stability).
    Also sets attention_softmax_in_fp32 to True."""

    disable_bf16_reduced_precision_matmul: bool = False
    """If True, prevent matmul from using reduced precision accumulation in BF16."""

    # FP8
    fp8: Optional[Literal['e4m3', 'hybrid']] = None
    """Enable FP8 precision through Transformer Engine."""

    fp8_recipe: Optional[Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise', 'custom']] = "delayed"
    """FP8 scaling recipe."""

    fp8_param: bool = False
    """Keep parameters in FP8 precision to save memory (requires fp8 mode)."""

    fp8_quantizer_factory: Optional[str] = None
    """Python import path to callable quantizer factory (required when fp8_recipe='custom')."""

    fp8_margin: int = 0
    """Margin for the scaling factor computation."""

    fp8_interval: int = 1
    """DEPRECATED from TE v1.8.0. Controls how often the scaling factor is recomputed."""

    fp8_amax_history_len: int = 1
    """Length of the amax history window used for scaling factor computation."""

    fp8_amax_compute_algo: Literal['most_recent', 'max'] = "most_recent"
    """Algorithm used for choosing the amax value for the scaling factor computation."""

    fp8_wgrad: bool = True
    """When False, override FP8 config and do wgrad in higher precision."""

    fp8_output_proj: bool = False
    """If True, run the LM-head output projection with TE ColumnParallelLinear under MXFP8."""

    fp8_dot_product_attention: bool = False
    """When True, use the FP8 implementation of Dot Product Attention."""

    fp8_multi_head_attention: bool = False
    """When True, use the FP8 implementation of Multi Head Attention."""

    tp_only_amax_red: bool = False
    """When True, reduce FP8 AMAX only in the TP or TP-CP domain."""

    first_last_layers_bf16: bool = False
    """If True, retain first and last N TransformerBlocks in BF16 (not FP8)."""

    num_layers_at_start_in_bf16: int = 1
    """Layers at model start to keep in BF16 when first_last_layers_bf16 is True."""

    num_layers_at_end_in_bf16: int = 1
    """Layers at model end to keep in BF16 when first_last_layers_bf16 is True."""

    use_kitchen: bool = False
    """Use the kitchen extension for transformer quantization."""

    use_kitchen_attention: bool = False
    """Use the kitchen extension for attention (instead of TE's attention)."""

    kitchen_attention_backend: Literal["sdpa", "fa"] = "sdpa"
    """Kitchen attention backend when use_kitchen_attention=True."""

    # FP4
    fp4: Optional[Literal['e2m1']] = None
    """Enable FP4 precision through Transformer Engine (Blackwell+ only)."""

    fp4_recipe: Optional[Literal['nvfp4', 'custom']] = "nvfp4"
    """FP4 scaling recipe."""

    fp4_param: bool = False
    """Keep parameters in FP4 precision to save memory (requires fp4 mode)."""

    fp4_quantizer_factory: Optional[str] = None
    """Python import path to callable quantizer factory (required when fp4_recipe='custom')."""

    # Per-module quantization
    quant_recipe: Optional[object] = None
    """Configuration of per-module quantization settings to be applied to the model."""

    # ------------------------------------------------------------------
    # Fusion kernels
    # ------------------------------------------------------------------

    bias_activation_fusion: bool = False
    """If True, fuses bias addition and the activation function when possible."""

    masked_softmax_fusion: bool = False
    """If True, uses softmax fusion."""

    persist_layer_norm: bool = False
    """If True, uses the persistent fused layer norm kernel."""

    memory_efficient_layer_norm: bool = False
    """If True, uses Apex memory-efficient fused LayerNorm kernel."""

    bias_dropout_fusion: bool = False
    """If True, uses bias dropout fusion."""

    fused_residual_rmsnorm: bool = False
    """If True, fuses residual connection and RMSNorm backward pass when TE is used."""

    use_transformer_engine_op_fuser: bool = False
    """If True, submodules may use Transformer Engine's operation fuser API."""

    use_fused_weighted_squared_relu: bool = False
    """If True, uses fused weighted squared relu kernel when using MoE."""

    # ------------------------------------------------------------------
    # Context Parallelism
    # ------------------------------------------------------------------

    cp_comm_type: Optional[Union[str, List[str]]] = None
    """Inter-GPU communication type for context parallelism."""

    high_priority_a2a_comm_stream: bool = False
    """If True, A2A communication stream is created with CUDA high priority."""

    overlap_moe_expert_parallel_comm: bool = False
    """Enable A2A overlap combine backprop with wgrad GEMM for EP."""

    delay_wgrad_compute: bool = False
    """Delay wgrad computation (requires overlap_moe_expert_parallel_comm)."""

    overlap_dispatch_backward_with_experts_wgrad: bool = False
    """Overlap dispatch backward with experts wgrad (mutually exclusive with delay_wgrad_compute)."""

    ep_overlap_early_attn_memory_release: bool = False
    """Release attention memory early when EP overlap is enabled."""

    # ------------------------------------------------------------------
    # CUDA Graphs
    # ------------------------------------------------------------------

    enable_cuda_graph: bool = False
    """DEPRECATED: replaced by cuda_graph_impl."""

    external_cuda_graph: bool = False
    """DEPRECATED: replaced by cuda_graph_impl."""

    cuda_graph_impl: Literal['none', 'local', 'transformer_engine', 'full_iteration'] = "none"
    """CUDA graph capture implementation."""

    cuda_graph_modules: Union[str, List[str]] = "full"
    """Training capture coverage within per-layer CUDA graphs."""

    cuda_graph_scope: Optional[Union[str, List[str]]] = None
    """DEPRECATED: renamed to cuda_graph_modules."""

    cuda_graph_use_single_mempool: bool = True
    """For local full_iteration graphs, share memory pool between training and optimizer."""

    cuda_graph_retain_backward_graph: bool = False
    """Retain grad for cudagraph backward passes."""

    cuda_graph_warmup_steps: int = 3
    """Number of warmup steps for CUDA graphs."""

    inference_cuda_graph_scope: Optional[str] = None
    """Controls CUDA graph scope during inference (none / layer / block)."""

    # ------------------------------------------------------------------
    # Activation recomputation
    # ------------------------------------------------------------------

    recompute_granularity: Optional[Literal["full", "selective"]] = None
    """Activation recomputation granularity. None = no recompute."""

    recompute_method: Optional[Literal["uniform", "block"]] = None
    """Which layers to recompute. Only relevant when recompute_granularity != selective."""

    recompute_num_layers: Optional[int] = None
    """Number of transformer layers per recompute unit."""

    recompute_modules: Optional[List[str]] = None
    """Submodules to recompute when recompute_granularity='selective'.
    Choices: core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts, gdn_norm_out."""

    distribute_saved_activations: Optional[bool] = False
    """If True, distribute recomputed activations across the model parallel group."""

    # ------------------------------------------------------------------
    # Fine-grained activation offloading
    # ------------------------------------------------------------------

    fine_grained_activation_offloading: bool = False
    """If True, offload input of specified modules to CPU at module level."""

    offload_modules: Optional[List[str]] = field(default_factory=list)
    """Submodules to offload: attn_norm, qkv_linear, core_attn, attn_proj,
    mlp_norm, expert_fc1, moe_act."""

    min_offloaded_tensor_size: int = 1024 * 1024
    """Minimum size of tensor to be offloaded."""

    fine_grained_offloading_max_inflight_offloads: Optional[int] = None
    """Max number of inflight offloads not yet joined on the main stream. None = no joins."""

    # ------------------------------------------------------------------
    # CPU offloading (layer-level)
    # ------------------------------------------------------------------

    cpu_offloading: bool = False
    """If True, offload activation memory to CPU."""

    cpu_offloading_num_layers: int = 0
    """Number of Transformer layers to offload to CPU."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    init_method_std: float = 0.02
    """Standard deviation for the default normal initialisation method."""

    init_method: Optional[Callable] = None
    """Weight initialisation callable. Defaults to normal(init_method_std)."""

    output_layer_init_method: Optional[Callable] = None
    """Initialisation for output layers (attention + MLP output projections)."""

    embedding_init_method: Optional[Callable] = None
    """Method to initialise weights of the embedding layer."""

    embedding_init_method_std: Optional[float] = None
    """Std for embedding layer. If None, uses init_method_std."""

    init_model_with_meta_device: bool = False
    """If True, initialises the model with the meta device (for very large models)."""

    # ------------------------------------------------------------------
    # MuP (Maximal Update Parameterization)  — M3402
    # ------------------------------------------------------------------

    use_mup: bool = False
    """Enable Maximal Update Parameterization (MuP) for hyperparameter transfer."""

    mup_width_mult: float = 1.0
    """Width multiplier for MuP scaling, computed as hidden_size / mup_base_hidden_size."""

    mup_base_hidden_size: Optional[int] = None
    """Base hidden size for MuP width scaling."""

    mup_embedding_mult: float = 1.0
    """Multiplier for embedding layer output. Applied after the embedding lookup."""

    mup_output_mult: float = 1.0
    """Multiplier for output logits before softmax."""

    mup_base_head_dim: Optional[float] = None
    """Base head dimension for MuP attention scaling."""

    mup_attn_scale_power: float = 1.0
    """Power for attention scaling: softmax_scale = 1 / (kv_channels ** mup_attn_scale_power)."""

    # ------------------------------------------------------------------
    # Inference optimisation
    # ------------------------------------------------------------------

    transformer_impl: Literal['local', 'transformer_engine', 'inference_optimized'] = "local"
    """Transformer implementation to use."""

    use_inference_optimized_layers: bool = False
    """If True, use inference-optimised transformer layers during inference."""

    inference_rng_tracker: bool = False
    """Whether to instantiate a separate RNG tracker for inference."""

    inference_sampling_seed: int = 42
    """Random seed for sampling during inference."""

    inference_fuse_tp_communication: bool = False
    """If True, uses a fused RS-residual-norm-AG kernel during inference."""

    inference_disable_triton_nvls_kernels: bool = False
    """If True, disables Triton NVLS kernels during inference."""

    inference_grouped_gemm_backend: Literal['flashinfer', 'torch', 'vllm'] = "vllm"
    """Backend for grouped GEMM operations during inference."""

    inference_moe_disable_fused_quant_kernels: bool = False
    """When True, disable fused kernels combining permute/activation with MXFP8 quant."""

    inference_moe_token_dispatcher_type: Literal['nccl', 'nvls'] = 'nvls'
    """Token dispatcher for MoE expert parallelism during inference."""

    # Chunked MLP (M3894)
    mlp_chunks_for_prefill: int = 1
    """Number of chunks along sequence dimension for MLP computation during prefill."""

    mlp_chunks_for_training: int = 1
    """Number of chunks along sequence dimension for MLP computation during training."""

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    test_mode: bool = False
    """Whether to run real-time tests."""

    clone_scatter_output_in_embedding: bool = True
    """When True, clone the output of scatter in embedding layer to aid GC."""

    disable_parameter_transpose_cache: bool = False
    """When True, parameter transposes are not cached for subsequent iterations."""

    config_logger_dir: str = ""
    """When non-empty, dump entry-point configs to config_logger_dir."""

    batch_invariant_mode: bool = False
    """If True, use batch-invariant kernels for deterministic forward execution."""

    use_te_rng_tracker: bool = False
    """Whether to use the TE or MCore version of the RNG tracker."""

    symmetric_ar_type: Optional[Literal['two_shot', 'one_shot', 'multimem_all_reduce']] = None
    """Type of symmetric all reduce to use. None = disabled."""

    nccl_all_reduce_for_prefill: bool = False
    """If True, use NCCL all-reduce kernels when symmetric all-reduce is enabled."""

    is_hybrid_model: bool = False
    """Indicates whether this is a hybrid model."""

    # Mamba
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_groups: int = 8
    mamba_num_heads: Optional[int] = None
    use_mamba_mem_eff_path: bool = True

    heterogeneous_block_specs: bool = False
    """Whether to use heterogeneous block specs (nemotron-nas architecture)."""

    hetereogenous_dist_checkpoint: bool = False
    """Whether to use heterogenous layers in distributed checkpoint."""

    expert_tensor_parallel_size: int = 1
    """Tensor parallelism size for MoE expert layers."""

    variable_seq_lengths: bool = False
    """Support variable sequence lengths across micro-batches."""

    pipeline_dtype: Optional[torch.dtype] = None
    """Data type for pipeline parallel communication."""

    fp16: bool = False
    """Use FP16 precision."""

    bf16: bool = False
    """Use BF16 precision."""

    # ------------------------------------------------------------------
    # PP layer split (backwards compat with existing TransformerBlock)
    # ------------------------------------------------------------------

    pipeline_layer_split: Optional[List[int]] = None
    """Explicit heterogeneous PP split: list of per-stage layer counts.
    len must equal pipeline_model_parallel_size and sum must equal num_layers."""

    mtp_standalone: bool = False
    """Set by __post_init__ when MTP layers occupy a standalone pipeline stage."""

    # ------------------------------------------------------------------
    # DES-LOC: per-layer GPU tier assignment
    # ------------------------------------------------------------------

    # Insight I8: default flight_recorder for PCIe (Megatron M3499)
    # PCIe clusters (our 2×A6000 + 1×H100 NVL + 2×Blackwell topology) are
    # significantly more prone to NCCL collective hangs than NVLink clusters:
    # higher latency, noisier fabric, and asymmetric bandwidth across tiers all
    # increase the probability of a timeout.  Megatron M3499 added
    # flight_recorder_dump_on_timeout as an opt-in; we default it ON here because
    # silent hangs are much harder to debug than verbose dumps.
    # Buffer size 65536 > Megatron default 36864 to retain enough trace history
    # over the slower PCIe all-reduce steps that precede a timeout event.
    flight_recorder_dump_on_timeout: bool = True
    """Dump NCCL flight-recorder traces on collective timeout (PCIe topology default).
    Set False on NVLink clusters where hangs are rare and dump overhead matters."""

    flight_recorder_trace_buffer_size: int = 65536
    """NCCL flight-recorder ring-buffer capacity (entries). Default 65536 exceeds
    Megatron's 36864 to capture sufficient history across slow PCIe all-reduces."""

    desloc_h100_layers: Optional[List[int]] = field(default=None, repr=False)
    """Zero-based global layer indices assigned to H100 GPUs.
    Populated automatically by _resolve_desloc_tiers() unless set manually."""

    desloc_a6000_layers: Optional[List[int]] = field(default=None, repr=False)
    """Zero-based global layer indices assigned to A6000 GPUs.
    Populated automatically by _resolve_desloc_tiers() unless set manually."""

    desloc_tier_strategy: Literal[
        "front_heavy", "back_heavy", "interleave", "manual"
    ] = "front_heavy"
    """Automatic tier-assignment strategy.

    front_heavy : First fraction of layers → H100, rest → A6000.
    back_heavy  : Last fraction of layers → H100, rest → A6000.
    interleave  : Even-indexed layers → H100, odd-indexed → A6000.
    manual      : No automatic assignment (caller fills the lists).
    """

    desloc_h100_layer_fraction: float = 0.5
    """Fraction [0, 1] of total layers placed on H100 GPUs when using
    front_heavy or back_heavy strategies. Ignored for interleave/manual."""

    desloc_tier_map: Optional[Dict[int, str]] = field(default=None, repr=False)
    """Read-only dict built by _resolve_desloc_tiers(): layer_idx → 'h100' | 'a6000'.
    Useful for quick per-layer look-ups at run time."""

    # ------------------------------------------------------------------
    # DES-LOC: parameter-tier tagging for decomposed all-reduce
    # ------------------------------------------------------------------

    desloc_tier_enabled: bool = False
    """If True, annotate each named parameter with a ``desloc_tier`` attribute during
    model construction so the DES-LOC tiered all-reduce scheduler can bucket comms by tier.

    Three tiers are defined:
      - 'x': weight/embedding/norm (synced every Kx steps)
      - 'u': attention-related (q/k/v projections, synced every Ku steps)
      - 'v': MLP/FFN weights (fc1/fc2/gate, synced every Kv steps)

    See engine.py::_desloc_tiered_ar and DESLOCAdamW.sync_if_needed.
    """

    desloc_tier_x_keywords: Optional[List[str]] = None
    """Name substrings that map a parameter to tier 'x' (params/norms/embeddings).
    Defaults to ['norm', 'embed', 'wpe', 'wte', 'ln_', 'position'] when
    desloc_tier_enabled is True.  Earlier entries take priority."""

    desloc_tier_u_keywords: Optional[List[str]] = None
    """Name substrings that map a parameter to tier 'u' (attention weights).
    Defaults to ['attn', 'attention', 'query', 'key', 'value', 'qkv',
    'linear_q', 'linear_k', 'linear_v', 'linear_proj', 'out_proj'] when
    desloc_tier_enabled is True."""

    desloc_tier_v_keywords: Optional[List[str]] = None
    """Name substrings that map a parameter to tier 'v' (MLP/FFN weights).
    Defaults to ['mlp', 'ffn', 'fc', 'dense', 'expert', 'router',
    'linear_fc1', 'linear_fc2', 'gate'] when desloc_tier_enabled is True."""

    desloc_default_tier: str = 'x'
    """Fallback tier assigned when no keyword matches. One of 'x', 'u', 'v'."""

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate and derive fields; mirrors Megatron TransformerConfig.__post_init__."""
        # Call parent __post_init__ only if it exists.
        parent_post = getattr(super(), '__post_init__', None)
        if callable(parent_post):
            parent_post()

        # --- fp16/bf16 mutual exclusion ------------------------------------
        if getattr(self, 'fp16', False) and getattr(self, 'bf16', False):
            raise ValueError(
                f"Only one of fp16: {self.fp16} and bf16: {self.bf16} should be True."
            )

        # --- fp32 residual → pipeline_dtype --------------------------------
        if self.fp32_residual_connection and self.pipeline_dtype is not None:
            if self.pipeline_dtype != torch.float:
                logger.warning(
                    "fp32_residual_connection is enabled, overriding pipeline_dtype "
                    "from %s to torch.float to match the residual stream dtype.",
                    self.pipeline_dtype,
                )
            self.pipeline_dtype = torch.float

        # --- Derived size defaults -----------------------------------------
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        if self.kv_channels is None and self.num_attention_heads > 0:
            self.kv_channels = self.hidden_size // self.num_attention_heads

        if self.num_query_groups is None:
            self.num_query_groups = self.num_attention_heads

        # --- Attention heads validation ------------------------------------
        if (
            self.num_attention_heads > 0
            and self.tensor_model_parallel_size > 0
            and self.num_attention_heads % self.tensor_model_parallel_size != 0
        ):
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"tensor_model_parallel_size ({self.tensor_model_parallel_size})."
            )

        # --- apply_query_key_layer_scaling ---------------------------------
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        # --- gated delta net checks ----------------------------------------
        if self.experimental_attention_variant == "gated_delta_net":
            assert self.linear_attention_freq is not None, (
                "linear_attention_freq must be set for gated_delta_net."
            )
            assert self.linear_conv_kernel_dim is not None
            assert self.linear_key_head_dim is not None
            assert self.linear_value_head_dim is not None
            assert self.linear_num_key_heads is not None
            assert self.linear_num_value_heads is not None
            assert self.linear_num_value_heads % self.linear_num_key_heads == 0, (
                f"linear_num_value_heads ({self.linear_num_value_heads}) must be a multiple "
                f"of linear_num_key_heads ({self.linear_num_key_heads})."
            )

        # --- MoE field defaults & validations ------------------------------
        if self.num_moe_experts is not None and self.num_moe_experts <= 0:
            raise ValueError("num_moe_experts must be non-negative.")

        if self.num_moe_experts is not None and self.moe_ffn_hidden_size is None:
            self.moe_ffn_hidden_size = self.ffn_hidden_size
            warnings.warn("moe_ffn_hidden_size is not set, using ffn_hidden_size instead.")

        if self.num_moe_experts is None and self.moe_ffn_hidden_size is not None:
            is_mixed = (isinstance(self.moe_layer_freq, list) and 0 in self.moe_layer_freq) or (
                isinstance(self.moe_layer_freq, int) and self.moe_layer_freq > 1
            )
            if is_mixed:
                warnings.warn(
                    "moe_ffn_hidden_size is set but num_moe_experts is None. "
                    "This is expected for dense layers in a mixed dense/MoE model. "
                    "moe_ffn_hidden_size will be ignored for dense layers."
                )
                self.moe_ffn_hidden_size = None
            else:
                raise ValueError(
                    "moe_ffn_hidden_size is set but num_moe_experts is None. "
                    "Please set num_moe_experts or remove moe_ffn_hidden_size."
                )

        # Expert bias only supports sigmoid / sqrtsoftplus score functions
        if self.moe_router_enable_expert_bias and self.moe_router_score_function not in (
            "sigmoid", "sqrtsoftplus"
        ):
            raise ValueError(
                "Expert bias routing only supports 'sigmoid' and 'sqrtsoftplus' "
                "score functions."
            )

        # Moe aux_loss coeff list must match load_balancing_type list
        if isinstance(self.moe_router_load_balancing_type, list):
            assert isinstance(self.moe_aux_loss_coeff, list) and len(
                self.moe_aux_loss_coeff
            ) == len(self.moe_router_load_balancing_type), (
                "moe_aux_loss_coeff must be a list of the same length as "
                "moe_router_load_balancing_type"
            )

        # Clamp negative capacity factor
        if self.moe_expert_capacity_factor is not None and self.moe_expert_capacity_factor < 0:
            self.moe_expert_capacity_factor = None

        if self.moe_pad_expert_input_to_capacity:
            if self.moe_expert_capacity_factor is None:
                raise ValueError(
                    "moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity"
                )

        # Group-limited routing validation
        if self.moe_router_group_topk:
            if self.moe_router_topk_limited_devices:
                raise ValueError(
                    "moe_router_topk_limited_devices is deprecated and replaced by "
                    "moe_router_group_topk and moe_router_num_groups."
                )
            if not self.moe_router_num_groups:
                raise ValueError(
                    "When using group limited routing, moe_router_num_groups must be specified."
                )
            else:
                assert self.num_moe_experts % self.moe_router_num_groups == 0, (
                    f"num_moe_experts ({self.num_moe_experts}) should be divisible by "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
                assert self.moe_router_group_topk <= self.moe_router_num_groups, (
                    f"moe_router_group_topk ({self.moe_router_group_topk}) should be ≤ "
                    f"moe_router_num_groups ({self.moe_router_num_groups})."
                )
        elif self.moe_router_topk_limited_devices:
            warnings.warn(
                "moe_router_topk_limited_devices is deprecated. Use moe_router_group_topk "
                "and moe_router_num_groups instead."
            )
            self.moe_router_group_topk = self.moe_router_topk_limited_devices
            self.moe_router_num_groups = self.expert_model_parallel_size

        # topk=1 with softmax needs pre_softmax
        if (
            self.moe_router_topk == 1
            and self.moe_router_score_function == "softmax"
            and not self.moe_router_pre_softmax
            and self.moe_router_load_balancing_type != "sinkhorn"
        ):
            raise ValueError("Please use moe_router_pre_softmax when topk is 1.")

        # moe_router_padding_for_fp8 compat alias
        if self.moe_router_padding_for_fp8:
            warnings.warn(
                "--moe-router-padding-for-fp8 is going to be deprecated. "
                "Use --moe-router-padding-for-quantization instead."
            )
            self.moe_router_padding_for_quantization = True

        # DeepEP deprecation migration
        if self.moe_enable_deepep:
            if self.moe_token_dispatcher_type != "flex":
                raise ValueError("DeepEP backend is only supported with flex token dispatcher.")
            if self.moe_flex_dispatcher_backend == "hybridep":
                raise ValueError("Only one backend is supported for flex token dispatcher.")
            self.moe_flex_dispatcher_backend = "deepep"
            warnings.warn(
                "moe_enable_deepep is deprecated. "
                "Please use --moe-flex-dispatcher-backend=deepep instead."
            )

        # Shared-expert overlap needs alltoall/flex dispatcher
        if self.moe_shared_expert_intermediate_size is not None:
            if self.moe_shared_expert_intermediate_size <= 0:
                raise ValueError(
                    "moe_shared_expert_intermediate_size must be "
                    "num_shared_experts * ffn_size_of_each_shared_expert."
                )
            if self.moe_shared_expert_overlap and self.moe_token_dispatcher_type not in [
                "alltoall", "flex"
            ]:
                raise ValueError(
                    "moe_shared_expert_overlap only works with alltoall or flex token dispatcher."
                )

        # moe_layer_recompute deprecation
        if self.moe_layer_recompute:
            warnings.warn(
                "--moe-layer-recompute is deprecated. "
                "Use --recompute-granularity selective --recompute-modules moe_layer instead."
            )
            if self.recompute_granularity == "full":
                raise ValueError(
                    "Do not set --moe-layer-recompute with full recompute granularity."
                )
            self.recompute_granularity = "selective"
            if self.recompute_modules is None:
                self.recompute_modules = []
            if "moe" not in self.recompute_modules:
                self.recompute_modules.append("moe")

        # --- Recompute defaults & validations ------------------------------
        if self.recompute_modules is None:
            self.recompute_modules = ["core_attn"]

        if self.recompute_granularity is not None:
            if self.recompute_granularity not in ["full", "selective"]:
                raise ValueError(
                    f'recompute_granularity must be "full" or "selective", '
                    f'got {self.recompute_granularity!r}.'
                )
            if self.recompute_method is not None and self.recompute_method not in [
                "block", "uniform"
            ]:
                raise ValueError(
                    f'recompute_method must be "block" or "uniform", got {self.recompute_method!r}.'
                )
            elif self.recompute_granularity != "selective" and self.recompute_method is None:
                raise ValueError(
                    f"Using recompute_granularity '{self.recompute_granularity}' so "
                    'recompute_method must be "block" or "uniform".'
                )
            if self.recompute_granularity != "selective" and self.recompute_num_layers is None:
                raise ValueError(
                    f"When using recompute_granularity '{self.recompute_granularity}', "
                    "recompute_num_layers must be set."
                )
            elif (
                self.recompute_granularity == "selective"
                and self.recompute_num_layers is not None
            ):
                raise ValueError(
                    f"When using recompute_granularity 'selective', "
                    "recompute_num_layers must be None."
                )
            if self.recompute_granularity == "selective" and self.recompute_modules:
                allowed_modules = {
                    "core_attn", "moe_act", "layernorm", "mla_up_proj",
                    "mlp", "moe", "shared_experts", "gdn_norm_out",
                }
                invalid = set(self.recompute_modules) - allowed_modules
                assert not invalid, (
                    f"Invalid recompute_modules: {invalid}. Allowed: {allowed_modules}"
                )
                if "moe_act" in self.recompute_modules and not self.moe_grouped_gemm:
                    raise ValueError(
                        "moe_act in recompute_modules is only supported with moe_grouped_gemm."
                    )
                if "mla_up_proj" in self.recompute_modules and not self.multi_latent_attention:
                    raise ValueError(
                        "mla_up_proj in recompute_modules is only supported with "
                        "multi_latent_attention."
                    )
                if (
                    "gdn_norm_out" in self.recompute_modules
                    and self.experimental_attention_variant != "gated_delta_net"
                ):
                    raise ValueError(
                        "gdn_norm_out in recompute_modules is only supported with "
                        "experimental_attention_variant='gated_delta_net'."
                    )
                if "shared_experts" in self.recompute_modules:
                    if (
                        self.moe_shared_expert_intermediate_size is not None
                        and self.moe_shared_expert_overlap
                    ):
                        raise ValueError(
                            "shared_experts recompute cannot work with --moe-shared-expert-overlap."
                        )

        # --- Fine-grained activation offloading validation -----------------
        if self.fine_grained_activation_offloading:
            assert not self.cpu_offloading, (
                "fine_grained_activation_offloading cannot be enabled with cpu_offloading."
            )
            assert self.offload_modules is not None and len(self.offload_modules) > 0
            allowed = {
                "core_attn", "attn_proj", "expert_fc1", "moe_act",
                "attn_norm", "mlp_norm", "qkv_linear",
            }
            invalid = set(self.offload_modules) - allowed
            assert not invalid, (
                f"Invalid offload_modules: {invalid}. Allowed: {allowed}"
            )
            if "attn_proj" in self.offload_modules and "core_attn" not in self.offload_modules:
                raise ValueError(
                    "attn_proj cannot be set to offload_modules alone without core_attn."
                )

        # --- Paged stash validation ----------------------------------------
        if self.moe_paged_stash:
            if self.cpu_offloading:
                raise ValueError("moe_paged_stash cannot be enabled with cpu_offloading.")
            if self.moe_expert_rank_capacity_factor is None:
                raise ValueError(
                    "moe_paged_stash requires moe_expert_rank_capacity_factor to be set."
                )

        # --- CPU offloading validation -------------------------------------
        if self.cpu_offloading:
            if self.pipeline_model_parallel_size > 1:
                raise ValueError(
                    "Currently there is no support for Pipeline parallelism with CPU offloading."
                )
            if self.recompute_granularity is not None:
                raise ValueError(
                    "CPU offloading does not work when activation recomputation is enabled."
                )

        # --- FP8 validations -----------------------------------------------
        if self.fp8_param and not self.fp8:
            raise ValueError("fp8_param must be used together with fp8 mode.")

        if self.fp8_output_proj:
            if not self.fp8:
                raise ValueError("fp8_output_proj must be used together with fp8 mode.")

        # --- FP4 validations -----------------------------------------------
        if self.fp4_param and not self.fp4:
            raise ValueError("fp4_param must be used together with fp4 mode.")

        if self.fp4 and self.fp8:
            raise ValueError("fp4 and fp8 cannot be used simultaneously.")

        # --- bias_activation_fusion checks ---------------------------------
        if self.bias_activation_fusion:
            if self.use_te_activation_func:
                raise ValueError(
                    "bias_activation_fusion and use_te_activation_func cannot both be True."
                )

        if self.fused_residual_rmsnorm:
            if self.normalization != "RMSNorm":
                raise ValueError(
                    "fused_residual_rmsnorm is only supported when normalization is RMSNorm."
                )

        # --- RoPE fusion checks -------------------------------------------
        if self.apply_rope_fusion:
            if self.multi_latent_attention:
                warnings.warn(
                    "apply_rope_fusion for multi-latent attention only supports training. "
                    "It is experimental."
                )

        if self.multi_latent_attention and self.rotary_interleaved:
            raise ValueError("rotary_interleaved does not work with multi_latent_attention.")

        # --- MuP configuration  (M3402) ------------------------------------
        if self.use_mup:
            if self.mup_base_hidden_size is None:
                self.mup_base_hidden_size = self.hidden_size
            assert self.mup_base_hidden_size > 0, "--mup-base-hidden-size must be positive."
            self.mup_width_mult = self.hidden_size / self.mup_base_hidden_size

            # MuP attention scaling: 1/d_head instead of 1/sqrt(d_head)
            if self.softmax_scale is None and self.kv_channels:
                base_head_scale = (
                    1.0 if self.mup_base_head_dim is None else self.mup_base_head_dim ** 0.5
                )
                self.softmax_scale = base_head_scale / (self.kv_channels ** self.mup_attn_scale_power)

            # MuP output scaling
            if self.mup_output_mult == 1.0 and self.mup_width_mult != 1.0:
                self.mup_output_mult = 1.0 / self.mup_width_mult

        # --- Embedding init method -----------------------------------------
        if self.embedding_init_method_std is None:
            self.embedding_init_method_std = self.init_method_std

        if self.embedding_init_method is None:
            if self.init_method is None or self.embedding_init_method_std != self.init_method_std:
                self.embedding_init_method = _init_method_normal(self.embedding_init_method_std)
            else:
                self.embedding_init_method = self.init_method

        # --- Weight initialisation methods ---------------------------------
        if self.init_method is None:
            if self.use_mup:
                self.init_method = _init_method_normal(
                    self.init_method_std / math.sqrt(max(self.mup_width_mult, 1e-8))
                )
            else:
                self.init_method = _init_method_normal(self.init_method_std)

        if self.output_layer_init_method is None:
            if self.use_mup:
                self.output_layer_init_method = _mup_scaled_init_method_normal(
                    self.init_method_std,
                    self.num_layers,
                    self.mup_width_mult,
                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
                )
            else:
                self.output_layer_init_method = _scaled_init_method_normal(
                    self.init_method_std,
                    self.num_layers,
                    multiplier=2.0 if not self.is_hybrid_model else 1.0,
                )

        # --- no_rope_freq normalisation ------------------------------------
        if self.no_rope_freq:
            if isinstance(self.no_rope_freq, int) and self.num_layers > 0:
                assert self.num_layers % self.no_rope_freq == 0, (
                    f"no_rope_freq={self.no_rope_freq} should be divisible by "
                    f"num_layers={self.num_layers}."
                )
                pattern = [0] * (self.no_rope_freq - 1) + [1]
                self.no_rope_freq = pattern * (self.num_layers // self.no_rope_freq)
            elif isinstance(self.no_rope_freq, list) and self.num_layers > 0:
                assert len(self.no_rope_freq) == self.num_layers, (
                    f"Length of no_rope list ({len(self.no_rope_freq)}) must match "
                    f"the number of layers ({self.num_layers})"
                )

        # --- Pipeline layout checks ----------------------------------------
        if (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ) and (
            self.account_for_embedding_in_pipeline_split
            or self.account_for_loss_in_pipeline_split
        ):
            raise ValueError(
                "num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage "
                "cannot be set at the same time with account_for_embedding_in_pipeline_split "
                "and account_for_loss_in_pipeline_split."
            )

        # Uneven PP layer validation
        if (
            self.num_layers_in_first_pipeline_stage is not None
            or self.num_layers_in_last_pipeline_stage is not None
        ):
            pp_size = self.pipeline_model_parallel_size
            remaining = self.num_layers
            if self.num_layers_in_first_pipeline_stage is not None:
                if self.num_layers_in_first_pipeline_stage <= 0:
                    raise ValueError("num_layers_in_first_pipeline_stage must be > 0.")
                remaining -= self.num_layers_in_first_pipeline_stage
                pp_size -= 1
            if self.num_layers_in_last_pipeline_stage is not None:
                if self.num_layers_in_last_pipeline_stage <= 0:
                    raise ValueError("num_layers_in_last_pipeline_stage must be > 0.")
                remaining -= self.num_layers_in_last_pipeline_stage
                pp_size -= 1
            if bool(remaining) != bool(pp_size):
                raise ValueError(
                    f"Mismatch: {remaining} middle layers remaining but {pp_size} "
                    "middle PP stages available."
                )
            if pp_size and remaining % pp_size != 0:
                raise ValueError(
                    f"Middle layers ({remaining}) must be divisible by "
                    f"middle PP stage count ({pp_size})."
                )

        # --- Expert parallelism checks -------------------------------------
        if self.expert_model_parallel_size > 1 and self.num_moe_experts is None:
            raise ValueError("num_moe_experts must be non-None to use expert-parallel.")

        # --- A2A overlap checks -------------------------------------------
        if self.overlap_moe_expert_parallel_comm:
            assert self.expert_model_parallel_size > 1, (
                "overlap_moe_expert_parallel_comm is only supported with expert model parallelism."
            )
            assert self.moe_token_dispatcher_type in ["alltoall", "flex"], (
                "overlap_moe_expert_parallel_comm is supported with alltoall/flex token dispatcher."
            )
            assert self.recompute_granularity != "full", (
                "Disable full recomputation when enabling overlap_moe_expert_parallel_comm."
            )
            assert not self.moe_shared_expert_overlap, (
                "Disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm."
            )
            # M3573 (45b8eac95): recompute_modules must not include "moe" when A2A overlap is on,
            # because the MoE recompute path re-runs dispatch which is incompatible with overlap.
            assert (
                self.recompute_modules is None or "moe" not in self.recompute_modules
            ), "disable moe in recompute_modules when enabling overlap_moe_expert_parallel_comm"
            # From Megatron M4044 (supersedes M2924): EP A2A overlap with MTP is
            # supported even with PP=1. Only mtp_num_layers <= 1 is required.
            assert (
                self.mtp_num_layers is None or self.mtp_num_layers == 1
            ), "MTP layernum only supports 1 when enabling overlap_moe_expert_parallel_comm."

        if self.delay_wgrad_compute:
            assert self.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled when enabling delay_wgrad_compute."
            )

        if self.overlap_dispatch_backward_with_experts_wgrad:
            assert not self.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be disabled when enabling "
                "overlap_dispatch_backward_with_experts_wgrad."
            )
            assert not self.delay_wgrad_compute, (
                "delay_wgrad_compute and overlap_dispatch_backward_with_experts_wgrad "
                "are mutually exclusive."
            )

        if self.ep_overlap_early_attn_memory_release:
            assert self.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled when enabling "
                "ep_overlap_early_attn_memory_release."
            )

        # --- Context parallelism checks ------------------------------------
        if self.context_parallel_size > 1 and self.cp_comm_type is not None:
            if isinstance(self.cp_comm_type, list):
                assert len(self.cp_comm_type) == self.num_layers, (
                    f"Length of cp_comm_type ({len(self.cp_comm_type)}) should equal "
                    f"the total number of transformer layers ({self.num_layers})!"
                )

        # --- inference_optimized transformer checks ------------------------
        if self.transformer_impl == "inference_optimized":
            assert self.normalization == "RMSNorm"
            assert not self.layernorm_zero_centered_gamma
            assert not self.add_bias_linear
            assert not self.add_qkv_bias
            assert not self.use_kitchen
            # M3617 (09cce75b1): MXFP8 with inference_optimized requires fp8_param (param AG).
            if self.fp8 == "mxfp8":
                if not self.fp8_param:
                    raise ValueError(
                        "fp8_param must be enabled when using "
                        "--transformer-impl='inference_optimized' with --fp8-recipe='mxfp8'. "
                        "Please set --fp8-param-gather."
                    )

        if self.inference_fuse_tp_communication:
            assert self.transformer_impl == "inference_optimized", (
                "inference_fuse_tp_communication is only supported for "
                "inference_optimized transformer implementation."
            )
            assert self.num_moe_experts is None, (
                "--inference-fuse-tp-communication is not supported for MoE models."
            )

        if self.inference_disable_triton_nvls_kernels:
            assert self.transformer_impl == "inference_optimized", (
                "inference_disable_triton_nvls_kernels is only supported for "
                "inference_optimized transformer implementation."
            )

        # --- MuP bias-in-MoE check -----------------------------------------
        if self.num_moe_experts is not None and self.add_bias_linear:
            assert self.expert_tensor_parallel_size == 1, (
                "Bias in MoE is only supported when ETP==1."
            )

        # --- CUDA graph deprecated field migration -------------------------
        if self.enable_cuda_graph or self.external_cuda_graph:
            assert self.cuda_graph_impl == "none", (
                "Do not use enable_cuda_graph or external_cuda_graph with cuda_graph_impl."
            )
            assert not (self.enable_cuda_graph and self.external_cuda_graph), (
                "enable_cuda_graph and external_cuda_graph cannot both be enabled."
            )
            if self.enable_cuda_graph:
                warnings.warn("enable_cuda_graph is deprecated, use cuda_graph_impl='local'.")
                self.cuda_graph_impl = "local"
            if self.external_cuda_graph:
                warnings.warn(
                    "external_cuda_graph is deprecated, "
                    "use cuda_graph_impl='transformer_engine'."
                )
                self.cuda_graph_impl = "transformer_engine"

        if self.cuda_graph_scope is not None:
            warnings.warn(
                "cuda_graph_scope is deprecated, use cuda_graph_modules instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(self.cuda_graph_scope, list):
                self.cuda_graph_modules = list(self.cuda_graph_scope)
            else:
                self.cuda_graph_modules = self.cuda_graph_scope
            self.cuda_graph_scope = None

        # Normalise "full" shorthand for cuda_graph_modules
        if self.cuda_graph_modules == "full":
            self.cuda_graph_modules = []

        assert self.cuda_graph_impl in [
            "none", "transformer_engine", "local", "full_iteration",
        ], f"Invalid cuda_graph_impl: {self.cuda_graph_impl!r}"

        if self.cuda_graph_impl == "full_iteration":
            assert not self.cuda_graph_modules, (
                'cuda_graph_modules must be empty when cuda_graph_impl="full_iteration".'
            )

        if self.cuda_graph_impl != "none":
            if self.cpu_offloading and self.cuda_graph_impl != "full_iteration":
                raise ValueError("CUDA graphs not supported with CPU offloading.")

            if self.fine_grained_activation_offloading:
                assert self.cuda_graph_impl in ("transformer_engine", "full_iteration"), (
                    "fine-grained activation offloading is only supported with "
                    "transformer_engine or full_iteration CUDA graph implementation."
                )
                assert self.cuda_graph_warmup_steps > 0, (
                    "cuda_graph_warmup_steps must be > 0 when enabling "
                    "fine-grained activation offloading."
                )
                if self.cuda_graph_impl == "full_iteration":
                    assert (
                        self.fine_grained_offloading_max_inflight_offloads is not None
                        and self.fine_grained_offloading_max_inflight_offloads >= 0
                    ), (
                        "fine_grained_offloading_max_inflight_offloads must be a non-negative "
                        "integer when using fine-grained activation offloading with "
                        "full-iteration CUDA graphs."
                    )

            if self.recompute_granularity:
                if self.recompute_granularity != "selective":
                    assert self.cuda_graph_impl == "full_iteration", (
                        "Full recompute is only supported with full iteration CUDA graph."
                    )

        # --- DSA variant checks -------------------------------------------
        if self.experimental_attention_variant == "dsa":
            assert self.context_parallel_size == 1, (
                "Currently context parallelism is not supported by DSAttention!"
            )
            assert not self.apply_rope_fusion, "RoPE fusion is not supported for DSAttention."

        # --- batch_invariant_mode -----------------------------------------
        if self.batch_invariant_mode:
            # Warn rather than hard-fail since AttnBackend enum is Megatron-specific
            logger.info(
                "batch_invariant_mode=True: ensure attention_backend is set to flash."
            )

        # --- MoE allgather dispatcher + variable seq lengths check --------
        if self.moe_token_dispatcher_type == "allgather" and self.variable_seq_lengths:
            raise ValueError(
                f"Token dispatcher type 'allgather' does not support variable sequence length."
            )

        # --- DES-LOC parameter tier defaults --------------------------------
        if self.desloc_tier_enabled:
            if self.desloc_tier_x_keywords is None:
                self.desloc_tier_x_keywords = [
                    'norm', 'embed', 'wpe', 'wte', 'ln_', 'position',
                ]
            if self.desloc_tier_u_keywords is None:
                self.desloc_tier_u_keywords = [
                    'attn', 'attention', 'query', 'key', 'value', 'qkv',
                    'linear_q', 'linear_k', 'linear_v', 'linear_proj', 'out_proj',
                ]
            if self.desloc_tier_v_keywords is None:
                self.desloc_tier_v_keywords = [
                    'mlp', 'ffn', 'fc', 'dense', 'expert', 'router',
                    'linear_fc1', 'linear_fc2', 'gate',
                ]
            assert self.desloc_default_tier in ('x', 'u', 'v'), (
                f"desloc_default_tier must be one of 'x', 'u', 'v', "
                f"got {self.desloc_default_tier!r}."
            )

        # --- Feature-combination validation (M2309/M2354/M2368/M2444) ----
        self._validate_overlap_combinations()

        # --- DES-LOC tier resolution (per-layer GPU assignment) -----------
        self._resolve_desloc_tiers()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_overlap_combinations(self) -> None:
        """Validate that overlap feature combinations are supported.
        From Megatron M2309/M2354/M2368/M2444: feature interaction bugs are
        the #1 correctness hazard in distributed training config.
        # From Megatron M2309/M2354/M2368/M2444: feature combination guard.
        """
        import warnings

        # M2309: overlap_grad_reduce + delay_wgrad_compute requires TE >= 2.8
        if (getattr(self, 'overlap_grad_reduce', False)
                and getattr(self, 'delay_wgrad_compute', False)):
            warnings.warn(
                "overlap_grad_reduce=True with delay_wgrad_compute=True requires "
                "TransformerEngine >= 2.8. Below that version this combination "
                "silently produces wrong gradients. From Megatron M2309.",
                UserWarning, stacklevel=4,
            )

        # M2444: moe_shared_expert_overlap requires shared experts to be configured
        if (getattr(self, 'moe_shared_expert_overlap', False)
                and getattr(self, 'moe_shared_expert_intermediate_size', None) is None):
            raise ValueError(
                "moe_shared_expert_overlap=True requires moe_shared_expert_intermediate_size "
                "to be set. From Megatron M2444: overlap only applies when shared experts exist."
            )

        # M2444: these two overlap modes conflict
        if (getattr(self, 'moe_shared_expert_overlap', False)
                and getattr(self, 'overlap_moe_expert_parallel_comm', False)):
            raise ValueError(
                "moe_shared_expert_overlap and overlap_moe_expert_parallel_comm cannot "
                "both be True. From Megatron M2444: conflicting execution order assumptions."
            )

    def validate_for_desloc_topology(self) -> None:
        """Validate config for DES-LOC heterogeneous PCIe topology.
        Call this after constructing TransformerConfig in your training script.
        # From Megatron batch_aa analysis: PCIe topology invalidates NVLink assumptions.
        """
        import warnings
        tp = getattr(self, 'tensor_model_parallel_size', 1)
        ep = getattr(self, 'expert_model_parallel_size', 1)
        dispatcher = getattr(self, 'moe_token_dispatcher_type', 'allgather')

        if getattr(self, 'sequence_parallel', False) and tp > 1:
            warnings.warn(
                f"sequence_parallel=True with tensor_model_parallel_size={tp}: "
                "requires high TP bandwidth. On PCIe topology (no NVLink) this "
                "may bottleneck. Consider reducing TP degree or disabling SP.",
                UserWarning, stacklevel=2,
            )

        if dispatcher == 'allgather' and ep > 1:
            warnings.warn(
                "moe_token_dispatcher_type='allgather' with expert_model_parallel_size>1 "
                "broadcasts ALL tokens across EP ranks. On PCIe topology use 'alltoall' "
                "dispatcher to reduce bandwidth. From Megatron batch_aa MoE analysis.",
                UserWarning, stacklevel=2,
            )

        pp = getattr(self, 'pipeline_model_parallel_size', 1)
        vpp = getattr(self, 'virtual_pipeline_model_parallel_size', None)
        num_layers = getattr(self, 'num_layers', None)
        if pp > 1 and vpp is not None and num_layers is not None:
            assert num_layers % (pp * vpp) == 0, (
                f"num_layers={num_layers} must be divisible by "
                f"pipeline_model_parallel_size={pp} * virtual_pipeline_model_parallel_size={vpp}={pp*vpp}. "
                "From Megatron M2350: VPP layer divisibility requirement."
            )

    def _resolve_desloc_tiers(self) -> None:
        """Populate desloc_h100_layers / desloc_a6000_layers / desloc_tier_map.

        Skipped when num_layers == 0 (config not yet fully specified) or
        when strategy is "manual" and both lists are already provided.
        """
        if self.num_layers <= 0:
            return  # config not ready yet

        # If both lists already set (manual or pre-filled), just build the map
        if self.desloc_h100_layers is not None and self.desloc_a6000_layers is not None:
            self._build_tier_map()
            return

        if self.desloc_tier_strategy == "manual":
            # Caller promised to fill in manually; nothing to do here
            return

        n = self.num_layers
        strategy = self.desloc_tier_strategy
        frac = max(0.0, min(1.0, self.desloc_h100_layer_fraction))
        n_h100 = int(round(n * frac))

        if strategy == "front_heavy":
            h100 = list(range(n_h100))
            a6000 = list(range(n_h100, n))

        elif strategy == "back_heavy":
            n_a6000 = n - n_h100
            a6000 = list(range(n_a6000))
            h100 = list(range(n_a6000, n))

        elif strategy == "interleave":
            h100 = list(range(0, n, 2))    # even layers
            a6000 = list(range(1, n, 2))   # odd layers

        else:
            raise ValueError(
                f"Unknown desloc_tier_strategy: {self.desloc_tier_strategy!r}. "
                "Valid values: 'front_heavy', 'back_heavy', 'interleave', 'manual'."
            )

        self.desloc_h100_layers = h100
        self.desloc_a6000_layers = a6000
        self._build_tier_map()

        logger.debug(
            "DES-LOC tier assignment (%s): %d layers → H100, %d layers → A6000",
            strategy,
            len(h100),
            len(a6000),
        )

    def _build_tier_map(self) -> None:
        """Build the layer_idx → tier string look-up dict."""
        tier_map: Dict[int, str] = {}
        if self.desloc_h100_layers:
            for idx in self.desloc_h100_layers:
                tier_map[idx] = "h100"
        if self.desloc_a6000_layers:
            for idx in self.desloc_a6000_layers:
                tier_map[idx] = "a6000"
        self.desloc_tier_map = tier_map

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_layer_tier(self, layer_idx: int) -> Optional[str]:
        """Return the tier string ('h100' | 'a6000') for a global layer index.

        Args:
            layer_idx: Zero-based global layer index.

        Returns:
            'h100', 'a6000', or None if the layer has no tier assignment.
        """
        if self.desloc_tier_map is None:
            return None
        return self.desloc_tier_map.get(layer_idx)

    def is_h100_layer(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* is assigned to an H100 GPU."""
        return self.get_layer_tier(layer_idx) == "h100"

    def is_a6000_layer(self, layer_idx: int) -> bool:
        """Return True if *layer_idx* is assigned to an A6000 GPU."""
        return self.get_layer_tier(layer_idx) == "a6000"

    def get_parameter_tier(self, param_name: str) -> str:
        """Return the DES-LOC tier ('x' | 'u' | 'v') for a parameter by name.

        This is a convenience helper for the model constructor to call when
        ``desloc_tier_enabled`` is True.

        Args:
            param_name: Fully qualified parameter name (e.g. from named_parameters()).

        Returns:
            'x', 'u', or 'v' tier string.
        """
        if not self.desloc_tier_enabled:
            return self.desloc_default_tier

        name_lower = param_name.lower()

        for kw in (self.desloc_tier_x_keywords or []):
            if kw in name_lower:
                return 'x'
        for kw in (self.desloc_tier_u_keywords or []):
            if kw in name_lower:
                return 'u'
        for kw in (self.desloc_tier_v_keywords or []):
            if kw in name_lower:
                return 'v'

        return self.desloc_default_tier
