# CMakeLists.txt for hetero_reduce benchmarks
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="86;90;120"
#   make -j4

cmake_minimum_required(VERSION 3.18)
project(HeteroReduceBenchmarks CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Common include dirs
set(HETERO_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/hetero_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/includes
)

# Common compile flags
set(CUDA_FLAGS
    -O3
    -lineinfo
    --use_fast_math
    -Xptxas=-v        # Show register/smem usage
    -Xcompiler=-Wall
)

# ─── Kernel source library (via csrc/hetero_reduce/CMakeLists.txt) ──
# All five kernel TUs + correct __launch_bounds__ for SM8.6/9.0/12.0.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../csrc/hetero_reduce
                 ${CMAKE_CURRENT_BINARY_DIR}/hetero_reduce_lib)
# Alias so benchmark targets can use the same name as before.
add_library(hetero_kernels ALIAS hetero_reduce_kernels)

# ─── Benchmarks ──────────────────────────────────────────────────────
foreach(bench hetero_reduce fused_swiglu_ln pcie_allreduce
              fused_layernorm_residual cross_entropy_tp)
    add_executable(bench_${bench} bench_${bench}.cu)
    target_link_libraries(bench_${bench} PRIVATE hetero_kernels CUDA::cudart)
    target_include_directories(bench_${bench} PRIVATE ${HETERO_INCLUDE_DIRS})
    target_compile_options(bench_${bench} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
    set_property(TARGET bench_${bench} PROPERTY
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
endforeach()

# ─── reduce-scatter vs naive baseline (issue #21) ────────────────────
# Compares hetero_reduce_scatter_kernel against two naive single-GPU
# baselines that model what non-fused NCCL reduce-scatter does on device.
# Produces throughput (GB/s) and speedup numbers for each SM dispatch path.
add_executable(bench_reduce_scatter_vs_naive
    bench_reduce_scatter_vs_naive.cu)
target_link_libraries(bench_reduce_scatter_vs_naive
    PRIVATE hetero_kernels CUDA::cudart)
target_include_directories(bench_reduce_scatter_vs_naive
    PRIVATE ${HETERO_INCLUDE_DIRS})
target_compile_options(bench_reduce_scatter_vs_naive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
set_property(TARGET bench_reduce_scatter_vs_naive PROPERTY
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# ─── NVTX profiling support (optional) ───────────────────────────────
find_library(NVTX_LIB nvToolsExt PATHS ${CUDAToolkit_ROOT}/lib64)
if(NVTX_LIB)
    message(STATUS "NVTX found: ${NVTX_LIB}")
    foreach(bench hetero_reduce fused_swiglu_ln pcie_allreduce
              fused_layernorm_residual cross_entropy_tp)
        target_compile_definitions(bench_${bench} PRIVATE NVTX_ENABLED=1)
        target_link_libraries(bench_${bench} PRIVATE ${NVTX_LIB})
    endforeach()
endif()

# Print CUDA arch summary
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
