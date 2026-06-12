# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ---------------------------------------------------------------------------
# M610: Megatron 0d5188c15 — refactored the fused kernels build
# Source: megatron/fused_kernels/__init__.py (NVIDIA/Megatron-LM 0d5188c15)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2021-03-17
#
# Mapping: megatron/fused_kernels/__init__.py → deepspeed/fused_kernels/__init__.py
#
# Changes ported:
#   1. Reorder imports: os before pathlib, blank line before cpp_extension.
#   2. Rewrite comment block for TORCH_CUDA_ARCH_LIST as single paragraph.
#   3. Replace three separate load_*() functions with a single load(args) that:
#        a. Checks CUDA 11 / cc 8.0 flag once.
#        b. Builds srcpath / buildpath once.
#        c. Defines inner _cpp_extention_load_helper(name, sources, extra_cuda_flags).
#        d. Conditionally builds fused softmax (args.masked_softmax_fusion) and
#           fused layer norm (args.fp32_residual_connection) kernels.
#   4. Rename get_cuda_bare_metal_version → _get_cuda_bare_metal_version (private).
#   5. Rename create_build_dir → _create_build_dir (private).
#
# DeepSpeed adaptation:
#   - _cpp_extention_load_helper passes verbose=(args.rank == 0) as upstream.
#   - Extra cuda flags for softmax include -U__CUDA_NO_HALF_OPERATORS__ etc.
#   - Layer norm extra flags: ['-maxrregcount=50'] (no half-op flags needed).
#   - Adds print('[M610]') marker.
# ---------------------------------------------------------------------------

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(args):

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(
        cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_70,code=sm_70',
                               '--use_fast_math'] + extra_cuda_flags + cc_flag,
            verbose=(args.rank == 0)
        )

    # ==============
    # Fused softmax.
    # ==============

    if args.masked_softmax_fusion:
        extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '--expt-relaxed-constexpr',
                            '--expt-extended-lambda']

        # Upper triangular softmax.
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']
        scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_upper_triang_masked_softmax_cuda",
            sources, extra_cuda_flags)

        # Masked softmax.
        sources=[srcpath / 'scaled_masked_softmax.cpp',
                 srcpath / 'scaled_masked_softmax_cuda.cu']
        scaled_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_masked_softmax_cuda", sources, extra_cuda_flags)

    # =================================
    # Mixed precision fused layer norm.
    # =================================

    extra_cuda_flags = ['-maxrregcount=50']
    sources = [srcpath / 'layer_norm_cuda.cpp',
               srcpath / 'layer_norm_cuda_kernel.cu']
    fused_mix_prec_layer_norm_cuda = _cpp_extention_load_helper(
        "fused_mix_prec_layer_norm_cuda", sources, extra_cuda_flags)
    print('[M612] load: fused_mix_prec_layer_norm_cuda loaded unconditionally (bfloat support)')


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


print('[M610]')
