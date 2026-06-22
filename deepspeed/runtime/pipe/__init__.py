# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .module import PipelineModule, LayerSpec, TiedLayerSpec
from .topology import ProcessTopology
# DES-LOC M716: BLOOM ALiBi position encoding
from .alibi import ALiBiEmbedding, build_alibi_bias, get_alibi_slopes
