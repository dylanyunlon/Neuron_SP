# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''Copyright The Microsoft DeepSpeed Team'''

# M460: re-export C++-accelerated helpers so callers can use them without
# importing the internal module directly.
from .indexed_dataset import (  # noqa: F401
    build_sample_idx,
    build_blending_indices,
    build_mapping,
)
