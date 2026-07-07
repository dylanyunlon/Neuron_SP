# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""CPU-only unit tests for parallel_state tier-aware group management.

Tests all three tier-group creation paths using the gloo backend (no GPU
required), verifying the complete API surface for issue #34.

Topology under test — 5-rank flat TP=1 PP=1 DP=5 setup
============================================================
Our ags1 cluster has 5 GPUs across two NUMA nodes:

    rank  GPU          SM     VRAM   NUMA  tier
    ----  -----------  -----  -----  ----  ---------
    0     A6000        8.6    48 GB  0     a6000
    1     A6000        8.6    48 GB  0     a6000
    2     H100         9.0    94 GB  1     h100
    3     Blackwell    12.0   96 GB  1     blackwell
    4     Blackwell    12.0   96 GB  1     blackwell

TP=1, PP=1, DP=5 means all ranks sit in a single DP group.
Tier sub-groups overlay on top:

    DP group       : [0, 1, 2, 3, 4]   (one NCCL communicator, all ranks)
    tier 'a6000'   : [0, 1]             (intra-tier for async gradient sync)
    tier 'h100'    : [2]                (single-rank group, but still created)
    tier 'blackwell': [3, 4]            (intra-tier for fast collective)

DES-LOC gradient sync uses tier sub-groups for intra-tier all-reduce (fast,
same VRAM class) then cross-tier reduce-scatter at a lower frequency (Kx/Ku/Kv
schedule).  The tier groups are independent NCCL communicators that can run
concurrently with the main DP communicator.

Running these tests
-------------------
    pytest tests/unit/model_parallelism/test_tier_groups_cpu.py -v

No GPU required.  The gloo backend with a temporary file store is used for
intra-process distributed communication.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
from datetime import timedelta
from typing import List

import pytest
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Helpers: spin up a tiny N-rank gloo world in subprocesses
# ---------------------------------------------------------------------------

def _worker(
    rank: int,
    world_size: int,
    store_path: str,
    fn,
    result_queue,
) -> None:
    """Subprocess worker: init gloo dist, run *fn(rank, world_size)*, put result."""
    try:
        store = dist.FileStore(store_path, world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=rank,
            world_size=world_size,
        )
        result = fn(rank, world_size)
        result_queue.put((rank, "ok", result))
    except Exception as exc:  # pragma: no cover
        result_queue.put((rank, "error", str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_in_world(world_size: int, fn) -> List:
    """Run *fn(rank, world_size)* across *world_size* gloo processes.

    Returns a list of ``(rank, status, result)`` tuples sorted by rank.
    Raises ``AssertionError`` if any worker returns ``'error'``.
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()

    with tempfile.NamedTemporaryFile(delete=False) as f:
        store_path = f.name

    procs = []
    for r in range(world_size):
        p = ctx.Process(
            target=_worker,
            args=(r, world_size, store_path, fn, q),
            daemon=True,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=60)

    results = []
    while not q.empty():
        results.append(q.get_nowait())

    results.sort(key=lambda x: x[0])

    errors = [(r, msg) for r, status, msg in results if status == "error"]
    assert not errors, f"Worker errors: {errors}"
    return results


# ---------------------------------------------------------------------------
# TierMap factory: simulate the 5-GPU ags1 topology without real GPUs
# ---------------------------------------------------------------------------

def _make_tier_map_5gpu():
    """Build a TierMap matching the ags1 5-GPU topology.

    Imported lazily so the test file can be collected even if hetero_bridge
    is not installed.
    """
    from deepspeed.core.hetero_bridge.tier_map import GPUTier, TierInfo, TierMap, _BYTES_PER_GB

    infos = [
        TierInfo(rank=0, tier=GPUTier.A6000,    total_vram_bytes=48 * _BYTES_PER_GB, numa_node=0, peak_bf16_tflops=309.7),
        TierInfo(rank=1, tier=GPUTier.A6000,    total_vram_bytes=48 * _BYTES_PER_GB, numa_node=0, peak_bf16_tflops=309.7),
        TierInfo(rank=2, tier=GPUTier.H100,     total_vram_bytes=94 * _BYTES_PER_GB, numa_node=1, peak_bf16_tflops=989.0),
        TierInfo(rank=3, tier=GPUTier.BLACKWELL, total_vram_bytes=96 * _BYTES_PER_GB, numa_node=1, peak_bf16_tflops=2250.0),
        TierInfo(rank=4, tier=GPUTier.BLACKWELL, total_vram_bytes=96 * _BYTES_PER_GB, numa_node=1, peak_bf16_tflops=2250.0),
    ]
    return TierMap.from_infos(infos)


# ---------------------------------------------------------------------------
# Test 1: desloc_config path — static tier assignment
# ---------------------------------------------------------------------------

def _fn_desloc_config_path(rank, world_size):
    """Path 3: DesLocConfig.tiers — static tier assignment."""
    from deepspeed.core.desloc_config import DesLocConfig, TierSpec, TierType
    import deepspeed.core.parallel_state as ps

    desloc_cfg = DesLocConfig(
        tiers=[
            TierSpec(
                tier_type=TierType.PROFESSIONAL,
                gpu_indices=[0, 1],
                sm_capability=(8, 6),
                vram_gb=48.0,
                bf16_tflops=309.7,
                pcie_gen=4,
                pcie_width=16,
                numa_node=0,
            ),
            TierSpec(
                tier_type=TierType.DATACENTER,
                gpu_indices=[2],
                sm_capability=(9, 0),
                vram_gb=94.0,
                bf16_tflops=989.0,
                pcie_gen=5,
                pcie_width=16,
                numa_node=1,
            ),
            TierSpec(
                tier_type=TierType.BLACKWELL,
                gpu_indices=[3, 4],
                sm_capability=(12, 0),
                vram_gb=96.0,
                bf16_tflops=2250.0,
                pcie_gen=5,
                pcie_width=16,
                numa_node=1,
            ),
        ]
    )

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        desloc_config=desloc_cfg,
    )

    assert ps.is_tier_aware(), "tier-aware should be True after desloc_config init"
    keys = ps.get_tier_group_keys()
    assert len(keys) > 0, f"expected tier groups, got {keys}"

    # Every rank should have a local_tier
    local = ps.get_local_tier()
    assert local is not None, f"rank {rank} has no local_tier"

    # Check DP world size (all 5 ranks in one DP group)
    assert ps.get_data_parallel_world_size() == world_size

    result = {
        "rank": rank,
        "tier_keys": keys,
        "local_tier_type": local.tier_type.name,
        "dp_world_size": ps.get_data_parallel_world_size(),
    }
    ps.destroy_model_parallel()
    return result


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_desloc_config_path():
    """desloc_config.tiers path creates the correct tier sub-groups."""
    results = run_in_world(5, _fn_desloc_config_path)
    rank_results = {r: data for r, _status, data in results}

    # Ranks 0, 1 → PROFESSIONAL (a6000)
    assert rank_results[0]["local_tier_type"] == "PROFESSIONAL"
    assert rank_results[1]["local_tier_type"] == "PROFESSIONAL"
    # Rank 2 → DATACENTER (h100)
    assert rank_results[2]["local_tier_type"] == "DATACENTER"
    # Ranks 3, 4 → BLACKWELL
    assert rank_results[3]["local_tier_type"] == "BLACKWELL"
    assert rank_results[4]["local_tier_type"] == "BLACKWELL"

    # DP world size = 5 for all ranks
    for r, data in rank_results.items():
        assert data["dp_world_size"] == 5, f"rank {r}: wrong DP world size"


# ---------------------------------------------------------------------------
# Test 2: TierMap path — pre-built TierMap passed to initialize_model_parallel
# ---------------------------------------------------------------------------

def _fn_tier_map_path(rank, world_size):
    """Path 1: pre-built TierMap passed as tier_map=..."""
    import deepspeed.core.parallel_state as ps

    try:
        tm = _make_tier_map_5gpu()
    except ImportError:
        return {"skipped": True}

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        tier_map=tm,
    )

    assert ps.is_tier_aware()
    keys = ps.get_tier_group_keys()
    # Expect 3 tiers: a6000, h100, blackwell
    assert set(keys) == {"a6000", "h100", "blackwell"}, f"unexpected tier keys: {keys}"

    # get_tier_group() must return a ProcessGroup for active tiers
    a6000_pg = ps.get_tier_group("a6000")
    assert a6000_pg is not None

    h100_pg = ps.get_tier_group("h100")
    assert h100_pg is not None

    # get_tier_group('datacenter') must return None (wrong key — old docstring style)
    assert ps.get_tier_group("datacenter") is None, \
        "'datacenter' is not a valid GPUTier.value key; should return None"

    # Verify per-tier ranks
    a6000_ranks = ps.get_tier_ranks("a6000")
    h100_ranks = ps.get_tier_ranks("h100")
    bw_ranks = ps.get_tier_ranks("blackwell")

    result = {
        "rank": rank,
        "tier_keys": keys,
        "a6000_ranks": a6000_ranks,
        "h100_ranks": h100_ranks,
        "blackwell_ranks": bw_ranks,
        "mem_budget_mb": tm.mem_budget(rank) // (1024 ** 2),
    }
    ps.destroy_model_parallel()
    return result


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_tier_map_path():
    """Pre-built TierMap path creates exactly the right per-tier groups."""
    results = run_in_world(5, _fn_tier_map_path)
    rank_results = {r: data for r, _status, data in results}

    if rank_results[0].get("skipped"):
        pytest.skip("hetero_bridge not available")

    for r, data in rank_results.items():
        assert data["a6000_ranks"] == [0, 1], f"rank {r}: a6000 ranks wrong"
        assert data["h100_ranks"] == [2],     f"rank {r}: h100 ranks wrong"
        assert data["blackwell_ranks"] == [3, 4], f"rank {r}: blackwell ranks wrong"

    # Memory budget: A6000 < H100 < Blackwell
    assert rank_results[0]["mem_budget_mb"] < rank_results[2]["mem_budget_mb"]
    assert rank_results[2]["mem_budget_mb"] < rank_results[3]["mem_budget_mb"]


# ---------------------------------------------------------------------------
# Test 3: destroy → re-init stress test (Gap 2 regression)
# ---------------------------------------------------------------------------

def _fn_destroy_reinit_cycle(rank, world_size):
    """Verify initialize → destroy → initialize does not crash."""
    import deepspeed.core.parallel_state as ps

    try:
        tm = _make_tier_map_5gpu()
    except ImportError:
        return {"skipped": True}

    errors = []
    for cycle in range(3):
        try:
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                tier_map=tm,
            )
            assert ps.is_tier_aware(), f"cycle {cycle}: not tier-aware after init"
            assert ps.is_initialized(), f"cycle {cycle}: not initialized"
            ps.destroy_model_parallel()
            assert not ps.is_initialized(), f"cycle {cycle}: still initialized after destroy"
            assert not ps.is_tier_aware(), f"cycle {cycle}: tier groups still present after destroy"
        except Exception as exc:
            errors.append(f"cycle {cycle}: {exc}")

    return {"rank": rank, "errors": errors, "cycles_completed": 3}


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_destroy_reinit_cycle():
    """Three initialize→destroy→initialize cycles must not crash.

    Regression for Gap 2: _set_global_memory_buffer() had an assert that
    fired on the second initialize_model_parallel() call.
    """
    results = run_in_world(5, _fn_destroy_reinit_cycle)
    for rank, status, data in results:
        if data.get("skipped"):
            pytest.skip("hetero_bridge not available")
        assert data["errors"] == [], \
            f"rank {rank} errors: {data['errors']}"
        assert data["cycles_completed"] == 3


# ---------------------------------------------------------------------------
# Test 4: is_tier_aware / get_tier_group_keys without tier config
# ---------------------------------------------------------------------------

def _fn_no_tier_config(rank, world_size):
    """Without tier config, is_tier_aware() must return False."""
    import deepspeed.core.parallel_state as ps

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        # no tier_map, no desloc_config, no discover_tier_map
    )

    result = {
        "rank": rank,
        "is_tier_aware": ps.is_tier_aware(),
        "tier_keys": ps.get_tier_group_keys(),
        "local_tier": ps.get_local_tier(),
        "tier_map": ps.get_tier_map(),
    }
    ps.destroy_model_parallel()
    return result


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_no_tier_config():
    """Without tier configuration, tier API returns safe no-op values."""
    results = run_in_world(3, _fn_no_tier_config)
    for rank, status, data in results:
        assert data["is_tier_aware"] is False
        assert data["tier_keys"] == []
        assert data["local_tier"] is None
        assert data["tier_map"] is None


# ---------------------------------------------------------------------------
# Test 5: update_pg_timeout_per_tier
# ---------------------------------------------------------------------------

def _fn_timeout_per_tier(rank, world_size):
    """update_pg_timeout_per_tier should succeed after tier init."""
    import deepspeed.core.parallel_state as ps

    try:
        tm = _make_tier_map_5gpu()
    except ImportError:
        return {"skipped": True}

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        tier_map=tm,
    )

    # Should not raise
    ps.update_pg_timeout_per_tier(
        intra_tier_timeout=timedelta(minutes=5),
        inter_tier_timeout=timedelta(minutes=20),
    )

    ps.destroy_model_parallel()
    return {"rank": rank, "ok": True}


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_update_pg_timeout_per_tier():
    """update_pg_timeout_per_tier must not raise after tier group init."""
    results = run_in_world(5, _fn_timeout_per_tier)
    for rank, status, data in results:
        if data.get("skipped"):
            pytest.skip("hetero_bridge not available")
        assert data["ok"] is True


# ---------------------------------------------------------------------------
# Test 6: topology documentation — 5-GPU TP=1 PP=1 DP=5 group structure
# ---------------------------------------------------------------------------

def _fn_topology_5gpu(rank, world_size):
    """Document and verify the expected group structure for TP=1 PP=1 DP=5."""
    import deepspeed.core.parallel_state as ps

    try:
        tm = _make_tier_map_5gpu()
    except ImportError:
        return {"skipped": True}

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        tier_map=tm,
    )

    #  Expected topology for TP=1, PP=1, DP=5:
    #
    #  Group type        Ranks               Purpose
    #  ----------------  ------------------- ------------------------------------------
    #  TP                [rank]              Tensor parallel (size=1 → trivial)
    #  PP                [rank]              Pipeline stage (size=1 → trivial)
    #  DP                [0,1,2,3,4]         Weight-replica all-reduce (all 5 GPUs)
    #  tier 'a6000'      [0,1]               DES-LOC intra-tier fast all-reduce (A6000s)
    #  tier 'h100'       [2]                 DES-LOC intra-tier (trivial single-rank)
    #  tier 'blackwell'  [3,4]               DES-LOC intra-tier fast all-reduce (BWs)
    #
    #  DES-LOC gradient sync schedule (Kx/Ku/Kv):
    #    - Every step:  intra-tier all-reduce within each tier's pg
    #    - Every Kx=8:  cross-tier reduce-scatter via DP pg
    #    - Every Ku=32: cross-tier first moment sync
    #    - Every Kv=64: cross-tier second moment sync

    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_pipeline_model_parallel_world_size() == 1
    assert ps.get_data_parallel_world_size() == 5

    tier_keys = ps.get_tier_group_keys()
    assert "a6000" in tier_keys
    assert "h100" in tier_keys
    assert "blackwell" in tier_keys

    result = {
        "rank": rank,
        "tp_size": ps.get_tensor_model_parallel_world_size(),
        "pp_size": ps.get_pipeline_model_parallel_world_size(),
        "dp_size": ps.get_data_parallel_world_size(),
        "tier_keys": tier_keys,
        "a6000_ranks": ps.get_tier_ranks("a6000"),
        "h100_ranks": ps.get_tier_ranks("h100"),
        "blackwell_ranks": ps.get_tier_ranks("blackwell"),
    }
    ps.destroy_model_parallel()
    return result


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_topology_5gpu_tp1_pp1_dp5():
    """Verify the canonical 5-GPU TP=1 PP=1 DP=5 topology group structure.

    This test doubles as documentation: it asserts the exact rank membership
    of every group created for our ags1 cluster configuration.
    """
    results = run_in_world(5, _fn_topology_5gpu)
    for rank, status, data in results:
        if data.get("skipped"):
            pytest.skip("hetero_bridge not available")

        assert data["tp_size"] == 1, f"rank {rank}: TP size"
        assert data["pp_size"] == 1, f"rank {rank}: PP size"
        assert data["dp_size"] == 5, f"rank {rank}: DP size"
        assert data["a6000_ranks"] == [0, 1]
        assert data["h100_ranks"] == [2]
        assert data["blackwell_ranks"] == [3, 4]


# ---------------------------------------------------------------------------
# Test 7: get_tier_group_keys() — the new helper added in this PR
# ---------------------------------------------------------------------------

def _fn_get_tier_group_keys(rank, world_size):
    """get_tier_group_keys() returns sorted list of active tier names."""
    import deepspeed.core.parallel_state as ps

    try:
        tm = _make_tier_map_5gpu()
    except ImportError:
        return {"skipped": True}

    # Before init: should return empty
    pre_init_keys = ps.get_tier_group_keys()

    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        tier_map=tm,
    )

    post_init_keys = ps.get_tier_group_keys()

    ps.destroy_model_parallel()
    post_destroy_keys = ps.get_tier_group_keys()

    return {
        "rank": rank,
        "pre_init": pre_init_keys,
        "post_init": post_init_keys,
        "post_destroy": post_destroy_keys,
    }


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_get_tier_group_keys_lifecycle():
    """get_tier_group_keys() reflects the init/destroy lifecycle."""
    results = run_in_world(5, _fn_get_tier_group_keys)
    for rank, status, data in results:
        if data.get("skipped"):
            pytest.skip("hetero_bridge not available")

        assert data["pre_init"] == [], \
            f"rank {rank}: expected [] before init, got {data['pre_init']}"

        assert data["post_init"] == ["a6000", "blackwell", "h100"], \
            f"rank {rank}: wrong keys after init: {data['post_init']}"

        assert data["post_destroy"] == [], \
            f"rank {rank}: expected [] after destroy, got {data['post_destroy']}"
