# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team
#
# CMakeLists.txt — csrc/hetero_reduce  (addresses #39)
#
# Builds a static library  hetero_reduce_kernels  containing all five CUDA
# translation units.  This is the canonical build target for the CUDA kernels;
# the Python extension (op_builder/hetero_reduce.py) uses setup.py / JIT
# compilation instead, so this file is for:
#   - standalone C++ integration tests
#   - the benchmarks/ CMake project (which add_subdirectory() this file)
#   - CI compilation checks without a PyTorch install
#
# Usage:
#   # From the repo root:
#   mkdir -p build/hetero_reduce && cd build/hetero_reduce
#   cmake ../../csrc/hetero_reduce        \
#         -DCMAKE_CUDA_ARCHITECTURES="86;90;120"
#   make -j$(nproc)
#
#   # Or add to a parent CMakeLists:
#   add_subdirectory(csrc/hetero_reduce)
#   target_link_libraries(my_target PRIVATE hetero_reduce_kernels)

cmake_minimum_required(VERSION 3.18)

# Allow being included as a sub-project.
if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
    project(HeteroReduceKernels CUDA CXX)
endif()

# ─── Language standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# ─── CUDA toolkit ────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ─── Default CUDA architectures ──────────────────────────────────────────────
# Caller can override: cmake … -DCMAKE_CUDA_ARCHITECTURES="86;90;120"
# SM12.0 native SASS requires CUDA ≥ 12.8; the PTX fallback (compute_90)
# handles Blackwell when built with older toolchains.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # Detect whether CUDA ≥ 12.8 is present for SM12.0 native SASS.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "86;90;120" CACHE STRING
            "CUDA architectures (SM8.6 A6000, SM9.0 H100, SM12.0 Blackwell)")
    else()
        # PTX forward-compatibility covers SM12.0 via compute_90 PTX.
        set(CMAKE_CUDA_ARCHITECTURES "86;90" CACHE STRING
            "CUDA architectures (SM8.6 A6000, SM9.0 H100; SM12.0 via PTX)")
    endif()
endif()

message(STATUS "[hetero_reduce] CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "[hetero_reduce] CUDA compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")

# ─── Include directories ─────────────────────────────────────────────────────
# csrc/includes contains ds_kernel_utils.h (DS_D_INLINE, hw_warp_size, …).
# Resolve the path relative to this CMakeLists.txt regardless of where cmake
# is invoked from.
get_filename_component(HETERO_REDUCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(CSRC_ROOT         "${HETERO_REDUCE_DIR}/.."     ABSOLUTE)

set(HETERO_INCLUDE_DIRS
    "${HETERO_REDUCE_DIR}"          # hetero_reduce.h
    "${CSRC_ROOT}/includes"         # ds_kernel_utils.h
)

# ─── Compile flags ───────────────────────────────────────────────────────────
set(HETERO_CUDA_FLAGS
    -O3
    --use_fast_math
    --expt-relaxed-constexpr        # needed for constexpr device lambdas
    -lineinfo                        # keeps line numbers for Nsight / cuda-gdb
    -Xptxas=-v                       # print register / smem usage per kernel
    -Xcompiler=-Wall
)

# ─── Source files ────────────────────────────────────────────────────────────
set(HETERO_REDUCE_SOURCES
    "${HETERO_REDUCE_DIR}/hetero_reduce.cu"
    "${HETERO_REDUCE_DIR}/fused_swiglu_ln.cu"
    "${HETERO_REDUCE_DIR}/fused_rope_hetero.cu"
    "${HETERO_REDUCE_DIR}/pcie_adaptive_allreduce.cu"
    "${HETERO_REDUCE_DIR}/tier_activation_offload.cu"
)

# ─── Static library target ───────────────────────────────────────────────────
add_library(hetero_reduce_kernels STATIC ${HETERO_REDUCE_SOURCES})

target_include_directories(hetero_reduce_kernels
    PUBLIC  ${HETERO_INCLUDE_DIRS}
)

target_compile_options(hetero_reduce_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${HETERO_CUDA_FLAGS}>
)

set_target_properties(hetero_reduce_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
)

target_link_libraries(hetero_reduce_kernels
    PUBLIC CUDA::cudart
)

# ─── Optional: install rules (skip when used as sub-project) ─────────────────
if(CMAKE_PROJECT_NAME STREQUAL "HeteroReduceKernels")
    install(TARGETS hetero_reduce_kernels
            ARCHIVE DESTINATION lib)
    install(FILES hetero_reduce.h
            DESTINATION include/hetero_reduce)
endif()
