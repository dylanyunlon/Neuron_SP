# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""NcclBootstrapOp — lazy-loading wrapper for the nccl_bootstrap C++ extension.

Addresses issue #139: C++ NCCL group bootstrap.

The extension is JIT-compiled on first use when not pre-built during
``pip install deepspeed``.

Bulk communicator creation example::

    import torch.distributed as dist
    from deepspeed.ops.parallel_state import NcclBootstrapOp

    dist.init_process_group("nccl")
    rank = dist.get_rank()

    op = NcclBootstrapOp()

    # Rank 0 generates the shared ncclUniqueId; broadcast to all ranks.
    import torch
    uid_tensor = torch.zeros(op.unique_id_size(), dtype=torch.uint8, device="cuda")
    if rank == 0:
        uid_bytes = op.get_unique_id()            # 128 bytes
        uid_tensor[:] = torch.frombuffer(uid_bytes, dtype=torch.uint8)
    dist.broadcast(uid_tensor, src=0)
    uid_bytes = bytes(uid_tensor.cpu().numpy().tobytes())

    # Create TP (ranks 0–3) and DP (ranks 0,4 / 1,5 / 2,6 / 3,7) groups.
    handles = op.bulk_create_comms(
        rank_lists=[[0, 1, 2, 3], [0, 4], [1, 5], [2, 6], [3, 7]],
        this_rank=rank,
        nccl_id_data=uid_bytes,
    )
    tp_handle = handles[0]   # 0 if rank not in [0,1,2,3]
    dp_handle = next(h for h in handles[1:] if h != 0)  # this rank's DP comm
"""

# Compiled C++ module, loaded on first NcclBootstrapOp instantiation.
_nccl_bootstrap_module = None


def _load_module():
    global _nccl_bootstrap_module
    if _nccl_bootstrap_module is None:
        try:
            from deepspeed.ops.op_builder import NcclBootstrapBuilder
        except ImportError:
            import sys
            import os
            _repo_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                )
            )
            if _repo_root not in sys.path:
                sys.path.insert(0, _repo_root)
            from op_builder.nccl_bootstrap import NcclBootstrapBuilder  # noqa: F401

        _nccl_bootstrap_module = NcclBootstrapBuilder().load()
    return _nccl_bootstrap_module


class NcclBootstrapOp:
    """Python wrapper for C++ bulk NCCL communicator bootstrap (issue #139).

    All methods delegate to the JIT-compiled ``nccl_bootstrap`` extension.
    The module is loaded (and compiled if necessary) on the first method call.

    See ``csrc/parallel_state/nccl_bootstrap.cpp`` for full documentation.
    """

    # ------------------------------------------------------------------ #
    # Unique-id helpers                                                    #
    # ------------------------------------------------------------------ #

    def get_unique_id(self) -> bytes:
        """Generate a fresh ncclUniqueId and return it as raw bytes.

        Call on rank 0 only, then broadcast the result before calling
        :meth:`bulk_create_comms`.
        """
        return _load_module().get_unique_id()

    def unique_id_size(self) -> int:
        """Return sizeof(ncclUniqueId) in bytes (128 for NCCL ≥ 2.x)."""
        return _load_module().unique_id_size()

    # ------------------------------------------------------------------ #
    # Bulk communicator creation                                           #
    # ------------------------------------------------------------------ #

    def bulk_create_comms(
        self,
        rank_lists,
        this_rank: int,
        nccl_id_data: bytes,
        device_idx: int = -1,
    ):
        """Create K NCCL communicators inside a single ncclGroupStart/End block.

        All K communicators are bootstrapped in parallel, cutting init time
        from O(K × latency) to O(latency) on pure-PCIe clusters.

        Parameters
        ----------
        rank_lists : List[List[int]]
            One list of global ranks per desired communicator.
        this_rank : int
            The calling process's global rank.
        nccl_id_data : bytes
            Serialised ncclUniqueId obtained from :meth:`get_unique_id` on
            rank 0 and broadcast to all ranks.
        device_idx : int, optional
            CUDA device index to activate before init (default -1 = current).

        Returns
        -------
        List[int]
            Opaque int64 handles, one per entry in *rank_lists*.
            A handle of 0 means this rank is not a member of that communicator.
        """
        return _load_module().bulk_create_comms(
            rank_lists, this_rank, nccl_id_data, device_idx
        )

    # ------------------------------------------------------------------ #
    # Communicator lifecycle                                               #
    # ------------------------------------------------------------------ #

    def destroy_comms(self, handles) -> None:
        """Destroy all non-zero communicator handles.

        Parameters
        ----------
        handles : List[int]
            Handles returned by :meth:`bulk_create_comms`.
        """
        _load_module().destroy_comms(handles)

    def abort_comm(self, handle: int) -> None:
        """Non-collectively abort a communicator (error-recovery path).

        Parameters
        ----------
        handle : int
            A single non-zero handle returned by :meth:`bulk_create_comms`.
        """
        _load_module().abort_comm(handle)

    # ------------------------------------------------------------------ #
    # Communicator introspection                                           #
    # ------------------------------------------------------------------ #

    def comm_count(self, handle: int) -> int:
        """Return the number of ranks in the communicator.

        Parameters
        ----------
        handle : int
            Non-zero handle returned by :meth:`bulk_create_comms`.
        """
        return _load_module().comm_count(handle)

    def comm_rank(self, handle: int) -> int:
        """Return this process's local rank within the communicator.

        Parameters
        ----------
        handle : int
            Non-zero handle returned by :meth:`bulk_create_comms`.
        """
        return _load_module().comm_rank(handle)
