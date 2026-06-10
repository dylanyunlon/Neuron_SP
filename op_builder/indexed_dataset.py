# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
OpBuilder for indexed_dataset_helpers – C++ accelerated index construction.
Megatron f51ceb7c9 adaptation for DeepSpeed data_pipeline.
"""

from .builder import TorchCPUOpBuilder


class IndexedDatasetBuilder(TorchCPUOpBuilder):
    BUILD_VAR = "DS_BUILD_INDEXED_DATASET"
    NAME = "indexed_dataset_helpers"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.data_pipeline.{self.NAME}'

    def sources(self):
        return ['csrc/data_pipeline/helpers.cpp']

    def include_paths(self):
        return ['csrc/includes']

    def cxx_args(self):
        args = super().cxx_args() if hasattr(super(), 'cxx_args') else []
        # pybind11 needs at least C++14; keep consistent with rest of codebase
        args += ['-O3', '-std=c++14']
        return args

    def extra_ldflags(self):
        # CPU-only extension, no CUDA linkage needed
        return ['-fopenmp'] if not self.is_rocm_pytorch() else []
