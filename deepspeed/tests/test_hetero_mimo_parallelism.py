# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""Unit tests for deepspeed/pipe/hetero_mimo_parallelism.py.

Mirrors Megatron upstream test files:
  tests/unit_tests/models/test_mimo_role.py
  tests/unit_tests/models/test_mimo_submodules.py

All tests are pure-Python, no GPU / dist.init required, using mock ranks
passed directly to RankRole.from_grid_map and HeteroParallelismConfig.build_role.
"""

import warnings
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from deepspeed.pipe.hetero_mimo_parallelism import (
    LANGUAGE_MODULE_KEY,
    DeviceGrid,
    HeteroMIMOPipelineStageBuilder,
    HeteroParallelismConfig,
    MIMORankAssigner,
    ModalitySubmoduleBase,
    ModuleLayout,
    ModuleStageInfo,
    RankRole,
)


# ---------------------------------------------------------------------------
# DeviceGrid tests
# ---------------------------------------------------------------------------

class TestDeviceGrid:
    def test_size_single_dim(self):
        grid = DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4])
        assert grid.size == 4

    def test_size_multi_dim(self):
        grid = DeviceGrid(rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2])
        assert grid.size == 4

    def test_contains_rank(self):
        grid = DeviceGrid(rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2])
        assert not grid.contains_rank(3)
        assert grid.contains_rank(4)
        assert grid.contains_rank(7)
        assert not grid.contains_rank(8)

    def test_local_rank(self):
        grid = DeviceGrid(rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2])
        assert grid.local_rank(4) == 0
        assert grid.local_rank(7) == 3

    def test_local_rank_out_of_grid_raises(self):
        grid = DeviceGrid(rank_offset=4, dim_names=["tp"], dim_sizes=[4])
        with pytest.raises(ValueError):
            grid.local_rank(3)

    def test_pp_rank_and_size_no_pp(self):
        grid = DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4])
        pp_rank, pp_size = grid.pp_rank_and_size(2)
        assert pp_rank == 0
        assert pp_size == 1

    def test_pp_rank_and_size_with_pp(self):
        # Grid layout: tp=2, pp=2; ranks 4,5,6,7
        # local 0 → tp=0, pp=0 (first stage)
        # local 1 → tp=1, pp=0 (first stage)
        # local 2 → tp=0, pp=1 (last stage)
        # local 3 → tp=1, pp=1 (last stage)
        grid = DeviceGrid(rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2])
        pp_rank_4, pp_size = grid.pp_rank_and_size(4)  # local=0, stride(pp)=1, pp_rank=(0//1)%2=0
        assert pp_size == 2
        pp_rank_6, _ = grid.pp_rank_and_size(6)  # local=2, stride(pp)=1, pp_rank=(2//1)%2=0
        pp_rank_7, _ = grid.pp_rank_and_size(7)  # local=3, stride(pp)=1, pp_rank=(3//1)%2=1
        assert pp_rank_6 == 0
        assert pp_rank_7 == 1

    def test_mismatched_dim_lengths_raises(self):
        with pytest.raises(ValueError):
            DeviceGrid(rank_offset=0, dim_names=["tp", "pp"], dim_sizes=[2])


# ---------------------------------------------------------------------------
# ModuleStageInfo tests
# ---------------------------------------------------------------------------

class TestModuleStageInfo:
    def test_fields(self):
        info = ModuleStageInfo(is_first_stage=True, is_last_stage=False)
        assert info.is_first_stage is True
        assert info.is_last_stage is False


# ---------------------------------------------------------------------------
# RankRole tests
# ---------------------------------------------------------------------------

class TestRankRole:
    def _make_simple_grid_map(self):
        """Ranks 0-3: vision encoder (TP=4).  Ranks 4-7: language (TP=2, PP=2)."""
        return {
            "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
            LANGUAGE_MODULE_KEY: DeviceGrid(
                rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2]
            ),
        }

    def test_unified_has_all_modules(self):
        role = RankRole.unified(["vision", "audio", LANGUAGE_MODULE_KEY])
        assert role.mode == ModuleLayout.UNIFIED
        assert role.has_modality_modules
        assert role.has_language_module
        for name in ["vision", "audio", LANGUAGE_MODULE_KEY]:
            assert name in role.modules
            assert role.is_first_stage(name)
            assert role.is_last_stage(name)

    def test_from_grid_map_encoder_rank(self):
        grid_map = self._make_simple_grid_map()
        role = RankRole.from_grid_map(grid_map, ["vision"], global_rank=2)
        assert role.mode == ModuleLayout.NON_COLOCATED
        assert role.has_modality_modules
        assert not role.has_language_module
        assert role.is_first_stage("vision")
        assert role.is_last_stage("vision")

    def test_from_grid_map_lm_first_stage(self):
        grid_map = self._make_simple_grid_map()
        # Rank 4: local=0, pp_rank=0 → first stage
        role = RankRole.from_grid_map(grid_map, ["vision"], global_rank=4)
        assert role.mode == ModuleLayout.NON_COLOCATED
        assert not role.has_modality_modules
        assert role.has_language_module
        assert role.is_first_stage(LANGUAGE_MODULE_KEY)
        assert not role.is_last_stage(LANGUAGE_MODULE_KEY)

    def test_from_grid_map_lm_last_stage(self):
        grid_map = self._make_simple_grid_map()
        # Rank 7: local=3; pp_rank = (3 // 1) % 2 = 1 → last stage
        role = RankRole.from_grid_map(grid_map, ["vision"], global_rank=7)
        assert role.mode == ModuleLayout.NON_COLOCATED
        assert role.has_language_module
        assert not role.is_first_stage(LANGUAGE_MODULE_KEY)
        assert role.is_last_stage(LANGUAGE_MODULE_KEY)

    def test_from_grid_map_rank_not_in_any_grid_raises(self):
        grid_map = self._make_simple_grid_map()
        with pytest.raises(RuntimeError, match="not present in any module grid"):
            RankRole.from_grid_map(grid_map, ["vision"], global_rank=99)

    def test_from_grid_map_wrong_keys_raises(self):
        grid_map = {
            "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
            # missing LANGUAGE_MODULE_KEY
        }
        with pytest.raises(ValueError, match="Missing"):
            RankRole.from_grid_map(grid_map, ["vision"], global_rank=0)

    def test_is_first_stage_unknown_module(self):
        role = RankRole.unified(["vision"])
        assert role.is_first_stage("nonexistent") is False

    def test_modality_module_names(self):
        role = RankRole.unified(["vision", "audio", LANGUAGE_MODULE_KEY])
        names = role.modality_module_names
        assert LANGUAGE_MODULE_KEY not in names
        assert set(names) == {"vision", "audio"}

    def test_colocated_mode_when_rank_in_both_grids(self):
        # Overlapping grids: rank 2 is in both
        grid_map = {
            "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
            LANGUAGE_MODULE_KEY: DeviceGrid(
                rank_offset=0, dim_names=["tp"], dim_sizes=[4]
            ),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            role = RankRole.from_grid_map(grid_map, ["vision"], global_rank=2)
        assert role.mode == ModuleLayout.COLOCATED


# ---------------------------------------------------------------------------
# HeteroParallelismConfig tests
# ---------------------------------------------------------------------------

class TestHeteroParallelismConfig:
    def _make_config(self):
        return HeteroParallelismConfig(
            modality_module_names=["vision", "audio"],
            module_to_grid_map={
                "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                "audio": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                LANGUAGE_MODULE_KEY: DeviceGrid(
                    rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2]
                ),
            },
            module_tp_degrees={"vision": 1, "audio": 1, LANGUAGE_MODULE_KEY: 2},
        )

    def test_build_role_unified_when_no_grid_map(self):
        config = HeteroParallelismConfig(
            modality_module_names=["vision"],
            module_to_grid_map=None,
        )
        role = config.build_role(global_rank=0)
        assert role.mode == ModuleLayout.UNIFIED

    def test_build_role_encoder_rank(self):
        config = self._make_config()
        role = config.build_role(global_rank=1)
        assert role.has_modality_modules
        assert not role.has_language_module

    def test_build_role_lm_rank(self):
        config = self._make_config()
        role = config.build_role(global_rank=5)
        assert role.has_language_module
        assert not role.has_modality_modules

    def test_tensor_parallel_degree_explicit_override(self):
        config = self._make_config()
        assert config.tensor_parallel_degree("vision") == 1
        assert config.tensor_parallel_degree(LANGUAGE_MODULE_KEY) == 2

    def test_tensor_parallel_degree_from_grid(self):
        config = HeteroParallelismConfig(
            modality_module_names=["vision"],
            module_to_grid_map={
                "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[3]),
                LANGUAGE_MODULE_KEY: DeviceGrid(
                    rank_offset=3, dim_names=["tp"], dim_sizes=[4]
                ),
            },
            module_tp_degrees=None,  # no override → infer from grid
        )
        assert config.tensor_parallel_degree("vision") == 3
        assert config.tensor_parallel_degree(LANGUAGE_MODULE_KEY) == 4

    def test_tensor_parallel_degree_defaults_to_1(self):
        config = HeteroParallelismConfig(
            modality_module_names=[],
            module_to_grid_map=None,
            module_tp_degrees=None,
        )
        assert config.tensor_parallel_degree("unknown") == 1


# ---------------------------------------------------------------------------
# MIMORankAssigner tests
# ---------------------------------------------------------------------------

class TestMIMORankAssigner:
    def test_assign_encoder_rank(self):
        config = HeteroParallelismConfig(
            modality_module_names=["vision"],
            module_to_grid_map={
                "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                LANGUAGE_MODULE_KEY: DeviceGrid(
                    rank_offset=4, dim_names=["tp"], dim_sizes=[4]
                ),
            },
        )
        assigner = MIMORankAssigner(config)
        # Patch build_all_groups so no dist needed
        with patch.object(config, "build_all_groups"):
            role = assigner.assign(global_rank=2)
        assert role.has_modality_modules

    def test_assign_colocated_raises_not_implemented(self):
        config = HeteroParallelismConfig(
            modality_module_names=["vision"],
            module_to_grid_map={
                "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                LANGUAGE_MODULE_KEY: DeviceGrid(
                    rank_offset=0, dim_names=["tp"], dim_sizes=[4]
                ),
            },
        )
        assigner = MIMORankAssigner(config)
        with patch.object(config, "build_all_groups"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(NotImplementedError, match="COLOCATED"):
                    assigner.assign(global_rank=0)


# ---------------------------------------------------------------------------
# HeteroMIMOPipelineStageBuilder tests
# ---------------------------------------------------------------------------

class TestHeteroMIMOPipelineStageBuilder:
    def _dummy_enc_factory(self, mod_name, is_first, is_last, tp):
        return [nn.Linear(4, 4)]

    def _dummy_lm_factory(self, is_first, is_last, tp):
        return [nn.Linear(4, 4), nn.Linear(4, 4)]

    def test_build_unified(self):
        role = RankRole.unified(["vision", LANGUAGE_MODULE_KEY])
        config = HeteroParallelismConfig(modality_module_names=["vision"])
        builder = HeteroMIMOPipelineStageBuilder(role, config)
        layers = builder.build(self._dummy_enc_factory, self._dummy_lm_factory)
        # 1 encoder layer + 2 LM layers
        assert len(layers) == 3

    def test_build_non_colocated_encoder_only(self):
        role = RankRole(
            modules={"vision": ModuleStageInfo(True, True)},
            mode=ModuleLayout.NON_COLOCATED,
        )
        config = HeteroParallelismConfig(modality_module_names=["vision"])
        builder = HeteroMIMOPipelineStageBuilder(role, config)
        layers = builder.build(self._dummy_enc_factory, self._dummy_lm_factory)
        assert len(layers) == 1  # only encoder

    def test_build_non_colocated_lm_only(self):
        role = RankRole(
            modules={LANGUAGE_MODULE_KEY: ModuleStageInfo(True, True)},
            mode=ModuleLayout.NON_COLOCATED,
        )
        config = HeteroParallelismConfig(modality_module_names=["vision"])
        builder = HeteroMIMOPipelineStageBuilder(role, config)
        layers = builder.build(self._dummy_enc_factory, self._dummy_lm_factory)
        assert len(layers) == 2  # only LM

    def test_build_colocated_raises(self):
        role = RankRole(
            modules={
                "vision": ModuleStageInfo(True, True),
                LANGUAGE_MODULE_KEY: ModuleStageInfo(True, True),
            },
            mode=ModuleLayout.COLOCATED,
        )
        config = HeteroParallelismConfig(modality_module_names=["vision"])
        builder = HeteroMIMOPipelineStageBuilder(role, config)
        with pytest.raises(NotImplementedError):
            builder.build(self._dummy_enc_factory, self._dummy_lm_factory)


# ---------------------------------------------------------------------------
# ModalitySubmoduleBase tests
# ---------------------------------------------------------------------------

class ConcreteSubmodule(ModalitySubmoduleBase):
    """Minimal concrete submodule for testing."""

    def __init__(self, is_first_stage=True, is_last_stage=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(is_first_stage=is_first_stage, is_last_stage=is_last_stage)
        self.encoder = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 16)

    def encode(self, inputs):
        x = inputs.get("x")
        return self.encoder(x)

    def project(self, embeddings):
        return self.proj(embeddings)


class TestModalitySubmoduleBase:
    def test_stage_flags(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConcreteSubmodule(is_first_stage=True, is_last_stage=False)
        assert m.is_first_stage is True
        assert m.is_last_stage is False

    def test_forward_first_and_last_stage(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConcreteSubmodule(is_first_stage=True, is_last_stage=True)
        x = torch.randn(3, 8)
        out = m.forward(encoder_inputs={"x": x})
        assert out.shape == (3, 16)

    def test_forward_first_stage_only(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConcreteSubmodule(is_first_stage=True, is_last_stage=False)
        x = torch.randn(3, 8)
        out = m.forward(encoder_inputs={"x": x})
        # Not last stage: no projection, returns raw encoder output
        assert out.shape == (3, 8)

    def test_forward_last_stage_only(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConcreteSubmodule(is_first_stage=False, is_last_stage=True)
        hidden = torch.randn(3, 8)
        out = m.forward(hidden_states=hidden)
        assert out.shape == (3, 16)

    def test_forward_first_stage_no_inputs_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ConcreteSubmodule(is_first_stage=True, is_last_stage=True)
        with pytest.raises(ValueError, match="encoder_inputs"):
            m.forward(encoder_inputs=None)

    def test_forward_returns_none_when_encode_returns_none(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            class NullEncoder(ModalitySubmoduleBase):
                def encode(self, inputs):
                    return None
                def project(self, embeddings):
                    return embeddings

            m = NullEncoder(is_first_stage=True, is_last_stage=True)
        out = m.forward(encoder_inputs={"x": torch.randn(2, 4)})
        assert out is None

    def test_emits_experimental_warning(self):
        with pytest.warns(UserWarning, match="experimental"):
            ConcreteSubmodule()


# ---------------------------------------------------------------------------
# Integration: end-to-end role + builder without dist
# ---------------------------------------------------------------------------

class TestIntegrationNoDist:
    """Simulate a 2-node cluster (8 ranks) without dist.init_process_group."""

    def _make_config(self):
        return HeteroParallelismConfig(
            modality_module_names=["vision", "audio"],
            module_to_grid_map={
                "vision": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                "audio": DeviceGrid(rank_offset=0, dim_names=["tp"], dim_sizes=[4]),
                LANGUAGE_MODULE_KEY: DeviceGrid(
                    rank_offset=4, dim_names=["tp", "pp"], dim_sizes=[2, 2]
                ),
            },
            module_tp_degrees={"vision": 1, "audio": 1, LANGUAGE_MODULE_KEY: 2},
        )

    def test_all_eight_ranks_get_valid_roles(self):
        config = self._make_config()
        for rank in range(8):
            role = config.build_role(global_rank=rank)
            assert role.mode != ModuleLayout.COLOCATED
            if rank < 4:
                assert role.has_modality_modules
                assert not role.has_language_module
            else:
                assert role.has_language_module
                assert not role.has_modality_modules

    def test_tp_degrees_are_correct_per_module(self):
        config = self._make_config()
        assert config.tensor_parallel_degree("vision") == 1
        assert config.tensor_parallel_degree("audio") == 1
        assert config.tensor_parallel_degree(LANGUAGE_MODULE_KEY) == 2

    def test_pp_stage_positions_for_lm_ranks(self):
        config = self._make_config()
        # Grid: dim_names=["tp","pp"], dim_sizes=[2,2], rank_offset=4.
        # pp has stride=1 (varies fastest), so:
        #   local 0 (rank 4): pp_rank=0 → first stage
        #   local 1 (rank 5): pp_rank=1 → last stage
        #   local 2 (rank 6): pp_rank=0 → first stage
        #   local 3 (rank 7): pp_rank=1 → last stage
        for rank in [4, 6]:
            role = config.build_role(global_rank=rank)
            assert role.is_first_stage(LANGUAGE_MODULE_KEY), f"rank {rank} should be first stage"
            assert not role.is_last_stage(LANGUAGE_MODULE_KEY), f"rank {rank} should not be last"

        for rank in [5, 7]:
            role = config.build_role(global_rank=rank)
            assert not role.is_first_stage(LANGUAGE_MODULE_KEY), f"rank {rank} should not be first"
            assert role.is_last_stage(LANGUAGE_MODULE_KEY), f"rank {rank} should be last stage"
