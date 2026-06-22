# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
pipeline/ — DES-LOC 三阶段预训练管线

Stage 1: Base code pretraining  (Stack v2 完整代码文件)
Stage 2: Commit continued-pretrain (CommitPack 4TB diff 序列)
Stage 3: Instruction tuning     (CommitPackFT 高质量 commit)

Usage:
    python -m pipeline.train_three_stage --config pipeline/configs/7b.yaml
    python -m pipeline.smoke_test       # 70M 模型端到端验证
"""

from .unified_tokenizer import get_tokenizer, TOKENIZER_NAME

try:
    from .engine_bridge import DESLOCEngine
except ImportError:
    DESLOCEngine = None  # deepspeed not installed, bridge unavailable

__all__ = ["get_tokenizer", "TOKENIZER_NAME", "DESLOCEngine"]
