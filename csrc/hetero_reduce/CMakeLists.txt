# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSpeed Team
#
# CMakeLists.txt — csrc/hetero_reduce  (NeurIPS 2026 DES-LOC + AutoSP)
#
# Builds a static library  hetero_reduce_kernels  containing all CUDA
# translation units for heterogeneous GPU training:
#   hetero_reduce.cu              — warp-shuffle + CAS cross-warp reduction
#   pcie_adaptive_allreduce.cu    — SM12.0 async-bulk-copy ring allreduce
#   cross_entropy_tp.cu           — online Milakov-Gimelshein TP softmax
#   fused_swiglu_ln.cu            — float4-vectorised SwiGLU + RMSNorm
#   fused_layernorm_residual.cu   — fused pre-LN residual add
#   fused_gradient_allreduce.cu   — INT8 compressed gradient allreduce
#   fused_rope_hetero.cu          — heterogeneous-head RoPE
#   tier_activation_offload.cu    — tier-aware activation checkpoint offload
#   fused_adam_heterogeneous.cu   — per-tier LR-scaled Adam optimizer
#
# SM targets:
#   sm_86  — NVIDIA A6000 (Ampere)
#   sm_90  — NVIDIA H100  (Hopper)
#   sm_120 — NVIDIA GB200 (Blackwell)  [requires CUDA >= 12.8]
#
# Usage:
#   mkdir -p build/hetero_reduce && cd build/hetero_reduce
#   cmake ../../csrc/hetero_reduce \
#         -DCMAKE_CUDA_ARCHITECTURES="86;90;120"   # override if needed
#   make -j$(nproc)

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED PROJECT_NAME OR PROJECT_NAME STREQUAL "")
    project(HeteroReduceKernels CUDA CXX)
endif()

# ─── Language standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
# Enable constexpr device lambdas (needed for if constexpr in kernels)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# ─── CUDA toolkit ────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

message(STATUS "[hetero_reduce] CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "[hetero_reduce] CUDA compiler: ${CMAKE_CUDA_COMPILER_VERSION}")

# ─── SM architecture selection ───────────────────────────────────────────────
# • sm_86  (Ampere A6000/A100-PCIe): always included
# • sm_90  (Hopper H100):            always included
# • sm_120 (Blackwell GB200/GB100):  requires CUDA >= 12.8
#   If CUDA < 12.8, we still generate PTX for compute_90 which JIT-compiles
#   on SM12.0 via Blackwell's forward-compatibility mode.
#
# Caller can override via:
#   cmake ... -DCMAKE_CUDA_ARCHITECTURES="86;90;120"

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "86;90;120" CACHE STRING
            "CUDA SM targets: sm_86 (A6000), sm_90 (H100), sm_120 (Blackwell)")
        message(STATUS "[hetero_reduce] CUDA >= 12.8 detected: enabling sm_120 native SASS")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "86;90" CACHE STRING
            "CUDA SM targets: sm_86 (A6000), sm_90 (H100) [sm_120 via PTX fallback]")
        message(STATUS "[hetero_reduce] CUDA < 12.8: sm_120 Blackwell via PTX forward-compat")
    endif()
endif()

message(STATUS "[hetero_reduce] CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ─── Validate SM120 requirement ──────────────────────────────────────────────
# If the user explicitly requests sm_120 but has CUDA < 12.8, error out with
# a clear message instead of a cryptic nvcc error.
if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
        message(FATAL_ERROR
            "[hetero_reduce] SM120 (Blackwell) native SASS requires CUDA >= 12.8.\n"
            "  Current CUDA: ${CMAKE_CUDA_COMPILER_VERSION}\n"
            "  To use PTX forward-compatibility, remove 120 from CMAKE_CUDA_ARCHITECTURES:\n"
            "    cmake ... -DCMAKE_CUDA_ARCHITECTURES=\"86;90\"")
    endif()
endif()

# ─── Include directories ─────────────────────────────────────────────────────
get_filename_component(HETERO_REDUCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(CSRC_ROOT         "${HETERO_REDUCE_DIR}/.."     ABSOLUTE)

set(HETERO_INCLUDE_DIRS
    "${HETERO_REDUCE_DIR}"          # hetero_reduce.h, fused_adam_heterogeneous.cu
    "${CSRC_ROOT}/includes"         # ds_kernel_utils.h (DS_D_INLINE, hw_warp_size, …)
)

# ─── Per-file compile options ─────────────────────────────────────────────────
# Base flags: applied to all CUDA translation units.
set(HETERO_CUDA_BASE_FLAGS
    -O3                              # full optimisation
    --use_fast_math                  # map sin/cos/exp to hardware instructions
    --expt-relaxed-constexpr         # constexpr device lambdas + if constexpr
    -lineinfo                        # PTX line numbers for Nsight Compute / nvprof
    -Xptxas=-v                       # print register / smem / spill stats per kernel
    -Xcompiler=-Wall                 # host compiler warnings
    -Xcompiler=-Wno-unknown-pragmas  # suppress "#pragma unroll" warnings on host
)

# SM120-specific flags: enable TMA / cp.async.bulk syntax.
# Only added for the pcie_adaptive_allreduce.cu which uses cp.async pipeline.
# Other TUs don't need it and it increases compile time.
set(HETERO_SM120_EXTRA_FLAGS
    -DCUDA_SM120_ASYNC_AVAILABLE=1
)

# ─── Source files ────────────────────────────────────────────────────────────
set(HETERO_REDUCE_SOURCES
    # Core reduce kernels
    "${HETERO_REDUCE_DIR}/hetero_reduce.cu"              # warp-coop + CAS reduction
    "${HETERO_REDUCE_DIR}/pcie_adaptive_allreduce.cu"    # ring allreduce + SM12 async
    "${HETERO_REDUCE_DIR}/cross_entropy_tp.cu"           # online log-softmax TP loss
    "${HETERO_REDUCE_DIR}/fused_swiglu_ln.cu"            # SwiGLU + RMSNorm
    "${HETERO_REDUCE_DIR}/fused_layernorm_residual.cu"   # pre-LN residual add
    "${HETERO_REDUCE_DIR}/fused_gradient_allreduce.cu"   # INT8 compressed allreduce
    "${HETERO_REDUCE_DIR}/fused_rope_hetero.cu"          # heterogeneous RoPE
    "${HETERO_REDUCE_DIR}/tier_activation_offload.cu"    # activation checkpoint offload
    "${HETERO_REDUCE_DIR}/fused_adam_heterogeneous.cu"   # per-tier Adam optimizer
)

# ─── Static library target ───────────────────────────────────────────────────
add_library(hetero_reduce_kernels STATIC ${HETERO_REDUCE_SOURCES})

target_include_directories(hetero_reduce_kernels
    PUBLIC  ${HETERO_INCLUDE_DIRS}
)

target_compile_options(hetero_reduce_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${HETERO_CUDA_BASE_FLAGS}>
)

set_target_properties(hetero_reduce_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE  ON
    CUDA_SEPARABLE_COMPILATION OFF    # off = faster link, no device-side cross-TU calls
)

target_link_libraries(hetero_reduce_kernels
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cuda_driver       # for cudaMemcpyPeerAsync, cudaEventRecord
)

# ─── CUB include path (for fused_gradient_allreduce.cu's cub::BlockReduce) ───
# CUDA >= 12.0 ships CUB as part of the toolkit. For older versions we
# rely on the system CUB. CUDAToolkit_INCLUDE_DIRS covers both.
target_include_directories(hetero_reduce_kernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# ─── Optional: enable SM120 async flags for pcie_adaptive_allreduce.cu ───────
if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES)
    set_source_files_properties(
        "${HETERO_REDUCE_DIR}/pcie_adaptive_allreduce.cu"
        PROPERTIES COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:CUDA>:${HETERO_SM120_EXTRA_FLAGS}>"
    )
endif()

# ─── Install rules ───────────────────────────────────────────────────────────
if(CMAKE_PROJECT_NAME STREQUAL "HeteroReduceKernels")
    install(TARGETS hetero_reduce_kernels
            ARCHIVE DESTINATION lib)
    install(FILES
                hetero_reduce.h
            DESTINATION include/hetero_reduce)
endif()

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "[hetero_reduce] Configured ${CMAKE_PROJECT_NAME}")
message(STATUS "[hetero_reduce]   Sources: ${HETERO_REDUCE_SOURCES}")
message(STATUS "[hetero_reduce]   SM targets: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "[hetero_reduce]   CUB: ${CUDAToolkit_INCLUDE_DIRS}")
