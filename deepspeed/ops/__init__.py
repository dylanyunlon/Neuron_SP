# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from . import adam
from . import adagrad
from . import lamb
from . import lion
from . import sparse_attention
from . import transformer
from . import fp_quantizer
from .transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig

# SM-dispatched FlashAttention-2 kernels for heterogeneous GPU clusters
# (SM 8.6 A6000 / SM 9.0 H100 / SM 12.0 Blackwell) — addresses #145
from .fused_attention import FlashAttentionOp

from ..git_version_info import compatible_ops as __compatible_ops__
