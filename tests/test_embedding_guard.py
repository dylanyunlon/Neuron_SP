"""
Unit tests for deepspeed.core.distributed.embedding_guard.

Tests the embedding grad-sync guard (fix #591 Blocker 3) without
requiring a real distributed environment.

Test matrix:
  - EmbeddingGradSyncConfig defaults
  - validate_embedding_sync_flags non-distributed passthrough
  - safe_model_parallel_config default safety flags
  - safe_model_parallel_config with overrides
  - safe_model_parallel_config share_embeddings explicitly False
"""

import pytest

from deepspeed.core.distributed.embedding_guard import (
    EmbeddingGradSyncConfig,
    safe_model_parallel_config,
    validate_embedding_sync_flags,
)


class TestEmbeddingGradSyncConfig:
    """Tests for EmbeddingGradSyncConfig dataclass."""

    def test_defaults(self):
        cfg = EmbeddingGradSyncConfig()
        assert cfg.share_embeddings_and_output_weights is False
        assert cfg.has_position_embeddings is False
        assert cfg.has_cond_embedder is False
        assert cfg.validated is False

    def test_explicit_true(self):
        cfg = EmbeddingGradSyncConfig(
            share_embeddings_and_output_weights=True,
            has_position_embeddings=True,
            validated=True,
        )
        assert cfg.share_embeddings_and_output_weights is True
        assert cfg.validated is True


class TestValidateEmbeddingSyncFlags:
    """Tests for validate_embedding_sync_flags (non-distributed)."""

    def test_non_distributed_passthrough(self):
        """Without torch.distributed, should return input flags unchanged."""
        result = validate_embedding_sync_flags(
            share_embeddings=True,
            has_position_embeddings=False,
            has_cond_embedder=True,
        )
        assert result.share_embeddings_and_output_weights is True
        assert result.has_position_embeddings is False
        assert result.has_cond_embedder is True
        assert result.validated is True

    def test_all_false(self):
        result = validate_embedding_sync_flags(
            share_embeddings=False,
            has_position_embeddings=False,
            has_cond_embedder=False,
        )
        assert result.share_embeddings_and_output_weights is False
        assert result.validated is True


class TestSafeModelParallelConfig:
    """Tests for safe_model_parallel_config."""

    def test_default_safety_flags(self):
        """Default config should have all safety flags set correctly."""
        cfg = safe_model_parallel_config()
        # Sequence parallel disabled (prevents conditional TP allreduce)
        assert cfg.sequence_parallel is False
        # Pipeline parallel size 1 (no PP stages -> no embedding cross-PP allreduce)
        assert cfg.pipeline_model_parallel_size == 1
        # TP size 1 (no tensor parallelism)
        assert cfg.tensor_model_parallel_size == 1
        # share_embeddings explicitly False
        assert cfg.share_embeddings_and_output_weights is False

    def test_override_hidden_size(self):
        """Should accept overrides for non-safety fields."""
        cfg = safe_model_parallel_config(hidden_size=2048)
        assert cfg.hidden_size == 2048
        # Safety flags still intact
        assert cfg.sequence_parallel is False

    def test_override_safety_flag(self):
        """Overrides should work even for safety flags (user knows best)."""
        cfg = safe_model_parallel_config(sequence_parallel=True)
        assert cfg.sequence_parallel is True

    def test_share_embeddings_field_exists(self):
        """The config must have share_embeddings_and_output_weights."""
        cfg = safe_model_parallel_config()
        assert hasattr(cfg, "share_embeddings_and_output_weights")
        assert cfg.share_embeddings_and_output_weights is False

    def test_returns_model_parallel_config(self):
        """Return type should be ModelParallelConfig."""
        from deepspeed.core.model_parallel_config import ModelParallelConfig
        cfg = safe_model_parallel_config()
        assert isinstance(cfg, ModelParallelConfig)


class TestFinalizeModelGradsEmbeddingGuard:
    """Test that the embedding guard prevents conditional allreduce asymmetry.

    These tests verify the guard logic WITHOUT calling NCCL, by checking
    that the config flags finalize_model_grads reads are set to values
    that either fire on ALL ranks or fire on NONE.
    """

    def test_gate_config_skips_word_embedding_allreduce(self):
        """With share_embeddings=False and PP=1, word embedding allreduce should be skipped."""
        cfg = safe_model_parallel_config()
        # The guard condition in finalize_model_grads.py:
        #   _skip_embedding_allreduce = (
        #       config.share_embeddings_and_output_weights is False
        #       and PP <= 1
        #       and not has_cond_embedder
        #       and mtp_num_layers in (None, 0)
        #   )
        skip = (
            getattr(cfg, 'share_embeddings_and_output_weights', None) is False
            and cfg.pipeline_model_parallel_size <= 1
            and not getattr(cfg, 'has_cond_embedder', False)
            and getattr(cfg, 'mtp_num_layers', None) in (None, 0)
        )
        assert skip is True, (
            "Gate config should cause finalize_model_grads to skip "
            "embedding allreduce (prevents asymmetric NCCL)"
        )

    def test_pp_gt1_does_not_skip(self):
        """With PP>1, embedding allreduce should NOT be skipped."""
        cfg = safe_model_parallel_config(pipeline_model_parallel_size=2)
        skip = (
            getattr(cfg, 'share_embeddings_and_output_weights', None) is False
            and cfg.pipeline_model_parallel_size <= 1
        )
        assert skip is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
