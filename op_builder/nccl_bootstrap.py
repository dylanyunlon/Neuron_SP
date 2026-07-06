# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""OpBuilder for the csrc/parallel_state/nccl_bootstrap C++ extension.

Addresses issue #139: C++ NCCL group bootstrap.

Bulk NCCL communicator creation via ncclGroupStart/End, replacing sequential
ncclCommInitRank calls to cut init latency on pure-PCIe heterogeneous clusters
(A6000 / H100 / Blackwell).

Exposed Python API (after NcclBootstrapBuilder().load())
---------------------------------------------------------
op.get_unique_id() -> bytes
    Generate a fresh ncclUniqueId.  Broadcast result from rank 0.

op.bulk_create_comms(rank_lists, this_rank, nccl_id_data, device_idx=-1)
    -> List[int]
    Create len(rank_lists) NCCL communicators inside a single
    ncclGroupStart/End block.  Returns opaque int64 handles; 0 = not a member.

op.destroy_comms(handles) -> None
    Destroy all non-zero handles.

op.comm_count(handle) -> int
    Number of ranks in the communicator.

op.comm_rank(handle) -> int
    This process's local rank within the communicator.

op.abort_comm(handle) -> None
    Non-collective abort (error recovery path).

op.unique_id_size() -> int
    sizeof(ncclUniqueId) in bytes (128 for NCCL ≥ 2.x).
"""

import sys
from .builder import CUDAOpBuilder


class NcclBootstrapBuilder(CUDAOpBuilder):
    """Builds the nccl_bootstrap C++ extension (issue #139).

    Compiled by the host CXX compiler only — no CUDA device code.
    Links against NCCL (via ``-lnccl``) and CUDA runtime already present in
    every GPU-capable PyTorch installation.
    """

    BUILD_VAR = "DS_BUILD_NCCL_BOOTSTRAP"
    NAME = "nccl_bootstrap"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"deepspeed.ops.parallel_state.{self.NAME}_op"

    def sources(self):
        return [
            "csrc/parallel_state/nccl_bootstrap.cpp",
        ]

    def include_paths(self):
        return [
            # DeepSpeed shared headers (DS_D_INLINE, hw_warp_size, …)
            "csrc/includes",
        ]

    def cxx_args(self):
        args = super().cxx_args()
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        # No CUDA device code in this extension.
        return []

    def extra_ldflags(self):
        # Link against NCCL shared library bundled with PyTorch CUDA wheels.
        # On systems where NCCL is installed separately the linker will still
        # find it via LD_LIBRARY_PATH / rpath set by the PyTorch wheel.
        if self.is_rocm_pytorch():
            # ROCm uses rccl which exposes an identical ncclGroupStart API.
            return ["-lrccl"]
        return ["-lnccl"]

    def is_compatible(self, verbose=True):
        """Require CUDA ≥ 11.0 (ncclGroupStart stable, BF16 available)."""
        try:
            cuda_major, cuda_minor = self.installed_cuda_version()
            if cuda_major < 11:
                if verbose:
                    self.warning(
                        f"{self.NAME}: CUDA {cuda_major}.{cuda_minor} < 11.0; "
                        "ncclGroupStart requires CUDA ≥ 11.0 / NCCL ≥ 2.1."
                    )
                return False
        except Exception:
            pass  # Let base class handle missing nvcc.
        return super().is_compatible(verbose)
