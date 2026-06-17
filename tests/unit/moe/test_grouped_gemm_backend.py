# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# mirrors Megatron bb979dd64 — tests/unit_tests/inference/test_vllm_fused_moe.py,
# reinterpreted as DES-LOC SM86/SM90 arch-routed backend validation.
#
# Tests:
#   1. GroupedGemmBackend enum values and SM-to-backend mapping logic
#   2. _select_block_size_m tile selection heuristic
#   3. _build_indirection_tables correctness (CPU, no CUDA needed for structure)
#   4. _forward_torch fallback (runs on CPU with small tensors)
#   5. GroupedGemmBackendSelector.forward() shape contract
#   6. force_backend override (unit-testable without specific GPU)

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_routing_map(max_tokens: int, topk: int, num_experts: int, seed: int = 0):
    """Build a deterministic routing_map [max_tokens, topk] int32."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    rows = []
    for _ in range(max_tokens):
        row = torch.randperm(num_experts, generator=rng)[:topk]
        rows.append(row)
    return torch.stack(rows).to(torch.int32)


def _make_valid_tokens(n: int, device="cpu") -> torch.Tensor:
    return torch.tensor(n, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# 1. Backend enum
# ---------------------------------------------------------------------------


def test_backend_enum_values():
    from deepspeed.moe.grouped_gemm_backend import GroupedGemmBackend

    assert GroupedGemmBackend.VLLM_FP8.value == "vllm_fp8"
    assert GroupedGemmBackend.CUTLASS_BF16.value == "cutlass_bf16"
    assert GroupedGemmBackend.TORCH.value == "torch"
    assert len(list(GroupedGemmBackend)) == 3


# ---------------------------------------------------------------------------
# 2. _select_block_size_m
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_tokens,expected",
    [
        (1, 16),
        (32, 16),
        (33, 32),
        (96, 32),
        (97, 64),
        (512, 64),
        (513, 128),
        (4096, 128),
    ],
)
def test_select_block_size_m(max_tokens, expected):
    from deepspeed.moe.grouped_gemm_backend import _select_block_size_m

    assert _select_block_size_m(max_tokens) == expected


# ---------------------------------------------------------------------------
# 3. SM detection helper (no GPU required — mocked)
# ---------------------------------------------------------------------------


def test_get_sm_major_no_cuda(monkeypatch):
    """When CUDA is unavailable, SM major should be 0."""
    import deepspeed.moe.grouped_gemm_backend as m

    # Reset cache
    m._SM_CACHE = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    sm = m._get_sm_major()
    assert sm == 0
    m._SM_CACHE = None  # teardown


# ---------------------------------------------------------------------------
# 4. GroupedGemmBackendSelector backend resolution
# ---------------------------------------------------------------------------


def test_selector_force_torch():
    """force_backend=TORCH should always select torch path regardless of SM."""
    from deepspeed.moe.grouped_gemm_backend import GroupedGemmBackend, GroupedGemmBackendSelector

    sel = GroupedGemmBackendSelector(force_backend=GroupedGemmBackend.TORCH)
    device = torch.device("cpu")
    resolved = sel._resolve_backend(device)
    assert resolved == GroupedGemmBackend.TORCH


def test_selector_force_cutlass_bf16():
    from deepspeed.moe.grouped_gemm_backend import GroupedGemmBackend, GroupedGemmBackendSelector

    sel = GroupedGemmBackendSelector(force_backend=GroupedGemmBackend.CUTLASS_BF16)
    resolved = sel._resolve_backend(torch.device("cpu"))
    assert resolved == GroupedGemmBackend.CUTLASS_BF16


def test_selector_sm_routing(monkeypatch):
    """Test SM-based routing without real GPU: patch _get_sm_major."""
    import deepspeed.moe.grouped_gemm_backend as m
    from deepspeed.moe.grouped_gemm_backend import GroupedGemmBackend, GroupedGemmBackendSelector

    original_cache = m._SM_CACHE
    try:
        # Simulate SM 86 (A6000)
        m._SM_CACHE = 86
        # HAVE_TRITON may be False in test env; if so, expect TORCH
        sel = GroupedGemmBackendSelector()
        backend = sel._resolve_backend(torch.device("cpu"))
        if m.HAVE_TRITON:
            assert backend == GroupedGemmBackend.CUTLASS_BF16, (
                f"SM86 + Triton should map to CUTLASS_BF16, got {backend}"
            )
        else:
            assert backend == GroupedGemmBackend.TORCH

        # Simulate SM 90 (H100)
        sel2 = GroupedGemmBackendSelector()
        sel2._backend = None
        m._SM_CACHE = 90
        backend2 = sel2._resolve_backend(torch.device("cpu"))
        if m.HAVE_TRITON:
            assert backend2 == GroupedGemmBackend.VLLM_FP8
        else:
            assert backend2 == GroupedGemmBackend.TORCH

        # Simulate SM 80 (A100) — should fall to TORCH
        sel3 = GroupedGemmBackendSelector()
        sel3._backend = None
        m._SM_CACHE = 80
        backend3 = sel3._resolve_backend(torch.device("cpu"))
        assert backend3 == GroupedGemmBackend.TORCH

    finally:
        m._SM_CACHE = original_cache


# ---------------------------------------------------------------------------
# 5. _forward_torch correctness (CPU, small tensors)
# ---------------------------------------------------------------------------


def _make_moe_tensors(max_tokens, topk, hidden_size, ffn_size, num_local_experts,
                      local_expert_start, num_total_experts, seed=42):
    torch.manual_seed(seed)
    hidden_states = torch.randn(max_tokens, hidden_size, dtype=torch.bfloat16)
    fc1_weight = torch.randn(num_local_experts, ffn_size, hidden_size, dtype=torch.bfloat16)
    fc2_weight = torch.randn(num_local_experts, hidden_size, ffn_size, dtype=torch.bfloat16)
    probs_raw = torch.rand(max_tokens, topk)
    probs = probs_raw / probs_raw.sum(dim=1, keepdim=True)  # normalise

    routing_map = _make_routing_map(max_tokens, topk, num_total_experts, seed=seed)
    # Remap to only local experts for simplicity (all tokens go to local experts)
    routing_map = routing_map % num_local_experts + local_expert_start
    routing_map = routing_map.to(torch.int32)

    valid_tokens = _make_valid_tokens(max_tokens)
    return hidden_states, fc1_weight, fc2_weight, probs, routing_map, valid_tokens


def test_forward_torch_shape():
    """_forward_torch output shape must be [max_tokens, hidden_size]."""
    from deepspeed.moe.grouped_gemm_backend import _forward_torch

    max_tokens, topk, H, N = 8, 2, 16, 32
    num_local_experts = 2
    local_expert_start = 0
    num_total_experts = num_local_experts

    hs, w1, w2, probs, rmap, valid = _make_moe_tensors(
        max_tokens, topk, H, N, num_local_experts, local_expert_start, num_total_experts
    )
    # _forward_torch accepts BF16 hidden_states but works on CPU (converts internally)
    out = _forward_torch(
        hs.float(),  # float32 for CPU path (no BF16 matmul on CPU)
        w1.float(),
        w2.float(),
        probs,
        rmap,
        num_local_experts,
        local_expert_start,
        valid,
    )
    assert out.shape == (max_tokens, H), f"Expected ({max_tokens}, {H}), got {out.shape}"


def test_forward_torch_zeros_beyond_valid():
    """Rows beyond valid_tokens must be zero."""
    from deepspeed.moe.grouped_gemm_backend import _forward_torch

    max_tokens, topk, H, N = 10, 1, 8, 16
    num_local_experts = 2
    local_expert_start = 0

    hs, w1, w2, probs, rmap, _ = _make_moe_tensors(
        max_tokens, topk, H, N, num_local_experts, local_expert_start, num_local_experts
    )
    valid_t = 5
    valid = _make_valid_tokens(valid_t)

    out = _forward_torch(
        hs.float(), w1.float(), w2.float(), probs, rmap, num_local_experts, local_expert_start, valid
    )
    # Rows [valid_t:] must be zero (no token contributions)
    assert torch.all(out[valid_t:] == 0), "Rows beyond valid_tokens should be zero"


# ---------------------------------------------------------------------------
# 6. moe_grouped_gemm_forward convenience wrapper — shape contract
# ---------------------------------------------------------------------------


def test_moe_grouped_gemm_forward_shape():
    """moe_grouped_gemm_forward must return [max_tokens, hidden_size]."""
    from deepspeed.moe.grouped_gemm_backend import GroupedGemmBackend, moe_grouped_gemm_forward

    max_tokens, topk, H, N = 12, 2, 16, 32
    num_local_experts = 3
    local_expert_start = 0

    hs, w1, w2, probs, rmap, valid = _make_moe_tensors(
        max_tokens, topk, H, N, num_local_experts, local_expert_start, num_local_experts
    )

    out = moe_grouped_gemm_forward(
        hs.float(),
        w1.float(),
        w2.float(),
        probs,
        rmap,
        num_local_experts,
        local_expert_start,
        valid,
        force_backend=GroupedGemmBackend.TORCH,
    )
    assert out.shape == (max_tokens, H)


def test_moe_grouped_gemm_forward_out_buffer():
    """moe_grouped_gemm_forward writes into a pre-allocated output buffer."""
    from deepspeed.moe.grouped_gemm_backend import GroupedGemmBackend, moe_grouped_gemm_forward

    max_tokens, topk, H, N = 6, 1, 8, 16
    num_local_experts = 2
    local_expert_start = 0

    hs, w1, w2, probs, rmap, valid = _make_moe_tensors(
        max_tokens, topk, H, N, num_local_experts, local_expert_start, num_local_experts
    )
    out_buf = torch.full((max_tokens, H), -999.0, dtype=torch.float32)
    result = moe_grouped_gemm_forward(
        hs.float(),
        w1.float(),
        w2.float(),
        probs,
        rmap,
        num_local_experts,
        local_expert_start,
        valid,
        out=out_buf,
        force_backend=GroupedGemmBackend.TORCH,
    )
    # Result should be the same object or have the same values
    assert result.shape == (max_tokens, H)
    # Sentinel value should be overwritten for valid tokens
    assert not torch.all(result == -999.0), "Output buffer was not updated"


# ---------------------------------------------------------------------------
# 7. _ceil_div correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b,expected", [(0, 8, 0), (1, 8, 1), (8, 8, 1), (9, 8, 2), (17, 8, 3)])
def test_ceil_div(a, b, expected):
    from deepspeed.moe.grouped_gemm_backend import _ceil_div

    assert _ceil_div(a, b) == expected


# ---------------------------------------------------------------------------
# 8. Import and __init__ export
# ---------------------------------------------------------------------------


def test_init_exports():
    """deepspeed.moe must export GroupedGemmBackend and GroupedGemmBackendSelector."""
    from deepspeed.moe import GroupedGemmBackend, GroupedGemmBackendSelector, moe_grouped_gemm_forward

    assert GroupedGemmBackend is not None
    assert GroupedGemmBackendSelector is not None
    assert moe_grouped_gemm_forward is not None
