# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Tests for issue #121: save/load with ShardedTensor.

Runs without CUDA / distributed: uses a single-rank stub so the full
save → load round-trip can be validated in CI without GPUs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Add repo root to sys.path so we can import deepspeed directly
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Stub torch.distributed for single-process tests
# ---------------------------------------------------------------------------

def _patch_distributed(monkeypatch):
    """Make torch.distributed appear initialised with world_size=1, rank=0."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None, raising=False)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, obj: [out.__setitem__(i, obj) for i in range(len(out))],
        raising=False,
    )


# ---------------------------------------------------------------------------
# Tiny model
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=True)
        self.fc2 = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShardedStateDict:
    """Unit tests for sharded_state_dict()."""

    def test_returns_sharded_tensor_for_all_params(self, monkeypatch):
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import (
            sharded_state_dict,
            ShardedTensor,
        )
        model = TinyModel()
        ssd = sharded_state_dict(model)

        param_names = {n for n, _ in model.named_parameters()}
        assert set(ssd.keys()) == param_names, (
            f"sharded_state_dict keys {set(ssd.keys())} != param names {param_names}"
        )
        for k, v in ssd.items():
            assert isinstance(v, ShardedTensor), (
                f"Expected ShardedTensor for key '{k}', got {type(v)}"
            )

    def test_global_shape_matches_param_shape_no_tp(self, monkeypatch):
        """With TP=1 (default) global_shape == local param shape."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import sharded_state_dict
        model = TinyModel()
        ssd = sharded_state_dict(model)
        for name, param in model.named_parameters():
            st = ssd[name]
            assert tuple(st.global_shape) == tuple(param.shape), (
                f"Key '{name}': global_shape {st.global_shape} != param shape {tuple(param.shape)}"
            )

    def test_prefix_prepended(self, monkeypatch):
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import sharded_state_dict
        model = TinyModel()
        ssd = sharded_state_dict(model, prefix="model.")
        for k in ssd:
            assert k.startswith("model."), f"Key '{k}' missing prefix 'model.'"

    def test_tp_param_sharded_along_partition_dim(self, monkeypatch):
        """TP-marked parameter should report tp_size as axis_fragmentation."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import (
            sharded_state_dict,
            ShardedTensor,
        )
        model = TinyModel()
        # Mark fc1.weight as TP-sharded along axis 0 with tp_size=2.
        param = model.fc1.weight
        param.tensor_model_parallel = True
        param.partition_dim = 0
        param.partition_stride = 1

        # Simulate tp_rank=0, tp_size=2 by patching parallel_state accessors.
        import deepspeed.core.distributed.sharded_checkpoint as sc
        monkeypatch.setattr(sc, "_tp_rank", lambda: 0)
        monkeypatch.setattr(sc, "_tp_size", lambda: 2)
        monkeypatch.setattr(sc, "_dp_rank", lambda: 0)
        monkeypatch.setattr(sc, "_dp_size", lambda: 1)

        ssd = sharded_state_dict(model)
        st = ssd["fc1.weight"]
        # axis_fragmentations[0] should be tp_size=2 for the TP axis.
        assert st.axis_fragmentations[0] == 2, (
            f"Expected axis_fragmentations[0]=2, got {st.axis_fragmentations}"
        )
        # global_shape along TP axis should be 2× the local shape.
        assert st.global_shape[0] == param.shape[0] * 2, (
            f"Expected global_shape[0]={param.shape[0]*2}, got {st.global_shape[0]}"
        )


class TestSaveLoadRoundtrip:
    """Integration: save_checkpoint → load_checkpoint round-trip."""

    def test_basic_roundtrip(self, monkeypatch, tmp_path):
        """Model params survive save → load unchanged (single rank, no TP)."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import (
            save_checkpoint,
            load_checkpoint,
        )

        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir)

        model_save = TinyModel()
        # Record original values.
        orig_params = {n: p.detach().clone() for n, p in model_save.named_parameters()}

        save_checkpoint(model_save, ckpt_dir)

        # Create a fresh model and load into it.
        model_load = TinyModel()
        # Scramble weights to confirm load actually writes.
        with torch.no_grad():
            for p in model_load.parameters():
                p.fill_(99.0)

        load_checkpoint(model_load, ckpt_dir)

        for name, orig in orig_params.items():
            loaded = dict(model_load.named_parameters())[name]
            assert torch.allclose(orig, loaded), (
                f"Param '{name}' mismatch after round-trip. "
                f"max_diff={( orig - loaded).abs().max():.6f}"
            )

    def test_extra_state_saved_and_loaded(self, monkeypatch, tmp_path):
        """extra_state dict survives save → load."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import (
            save_checkpoint,
            load_checkpoint,
        )
        ckpt_dir = str(tmp_path / "ckpt2")
        os.makedirs(ckpt_dir)

        model = TinyModel()
        extra = {"iteration": 42, "rng_state": "abc123"}
        save_checkpoint(model, ckpt_dir, extra_state=extra)

        model2 = TinyModel()
        recovered = load_checkpoint(model2, ckpt_dir)
        assert recovered.get("iteration") == 42
        assert recovered.get("rng_state") == "abc123"

    def test_checkpoint_directory_layout(self, monkeypatch, tmp_path):
        """Verify expected files are created in checkpoint directory."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import save_checkpoint
        ckpt_dir = str(tmp_path / "ckpt3")
        os.makedirs(ckpt_dir)

        model = TinyModel()
        save_checkpoint(model, ckpt_dir)

        files = {f.name for f in Path(ckpt_dir).iterdir()}
        # rank 0 should produce shard_00000.pt, common.pt, metadata.json
        assert "shard_00000.pt" in files, f"Missing shard file; got: {files}"
        assert "common.pt" in files, f"Missing common.pt; got: {files}"
        assert "metadata.json" in files, f"Missing metadata.json; got: {files}"

    def test_metadata_json_format(self, monkeypatch, tmp_path):
        """metadata.json must be valid JSON with expected keys."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import save_checkpoint
        ckpt_dir = str(tmp_path / "ckpt4")
        os.makedirs(ckpt_dir)

        save_checkpoint(TinyModel(), ckpt_dir)

        with open(Path(ckpt_dir) / "metadata.json") as f:
            meta = json.load(f)

        assert "world_size" in meta, f"metadata.json missing 'world_size': {meta}"
        assert meta["world_size"] == 1  # single-rank test

    def test_prefix_roundtrip(self, monkeypatch, tmp_path):
        """Keys saved with prefix are loaded back correctly."""
        _patch_distributed(monkeypatch)
        from deepspeed.core.distributed.sharded_checkpoint import (
            save_checkpoint,
            load_checkpoint,
        )
        ckpt_dir = str(tmp_path / "ckpt5")
        os.makedirs(ckpt_dir)

        model_save = TinyModel()
        orig = {n: p.detach().clone() for n, p in model_save.named_parameters()}
        save_checkpoint(model_save, ckpt_dir, prefix="model.")

        model_load = TinyModel()
        with torch.no_grad():
            for p in model_load.parameters():
                p.fill_(-1.0)

        load_checkpoint(model_load, ckpt_dir, prefix="model.")

        for name, orig_val in orig.items():
            loaded = dict(model_load.named_parameters())[name]
            assert torch.allclose(orig_val, loaded), (
                f"Prefix round-trip: param '{name}' mismatch"
            )


class TestDDPMethods:
    """Verify DistributedDataParallel exposes sharded_state_dict/save/load."""

    def test_ddp_has_sharded_checkpoint_methods(self):
        from deepspeed.core.distributed.distributed_data_parallel import (
            DistributedDataParallel,
        )
        assert hasattr(DistributedDataParallel, "sharded_state_dict")
        assert hasattr(DistributedDataParallel, "save_checkpoint")
        assert hasattr(DistributedDataParallel, "load_checkpoint")

    def test_module_level_exports(self):
        """All three symbols must be importable directly from the package."""
        from deepspeed.core.distributed import (
            sharded_state_dict,
            save_checkpoint,
            load_checkpoint,
        )
        assert callable(sharded_state_dict)
        assert callable(save_checkpoint)
        assert callable(load_checkpoint)
