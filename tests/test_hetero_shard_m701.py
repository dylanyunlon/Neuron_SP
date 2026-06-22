"""
DES-LOC M701 — hetero_shard_ratio unit test
Simulates: 2x A6000 (49GB) + 1x H100 (96GB)
Ratio: [1.0, 1.0, 2.0]  →  A6000s each ~25%, H100 ~50%

Run: python tests/test_hetero_shard_m701.py
"""

import torch
import math


# ── Minimal stub of the partitioning logic ───────────────────────────────────

class FakeDistributed:
    def __init__(self, world_size, rank):
        self._ws = world_size
        self._rank = rank
    def get_world_size(self, group=None):
        return self._ws
    def get_rank(self, group=None):
        return self._rank


class HeteroShardMixin:
    """
    Extracted hetero partition logic from DeepSpeedZeroOptimizer_Stage3.
    Used here without the full DeepSpeed stack.
    """
    def _init_hetero(self, hetero_shard_ratio, dp_world_size):
        self.partition_count = dp_world_size
        self._hetero_shard_ratio = hetero_shard_ratio
        if hetero_shard_ratio is not None:
            assert len(hetero_shard_ratio) == dp_world_size
            ratio_sum = sum(hetero_shard_ratio)
            self._hetero_shard_fracs = [r / ratio_sum for r in hetero_shard_ratio]
        else:
            self._hetero_shard_fracs = None
        self._hetero_shard_diag_printed = False

    def get_data_parallel_partitions(self, tensor, fake_dist, verbose=True):
        partitions = []
        dp = fake_dist.get_world_size()
        total_num_elements = tensor.numel()

        if self._hetero_shard_fracs is not None:
            raw_sizes = [int(total_num_elements * frac) for frac in self._hetero_shard_fracs]
            allocated = sum(raw_sizes)
            remainder = total_num_elements - allocated
            order = sorted(range(dp), key=lambda i: self._hetero_shard_fracs[i], reverse=True)
            for i in range(remainder):
                raw_sizes[order[i]] += 1
            start = 0
            for id in range(dp):
                partitions.append(tensor.narrow(0, start, raw_sizes[id]))
                start += raw_sizes[id]
            if verbose and not self._hetero_shard_diag_printed:
                self._hetero_shard_diag_printed = True
                for r in range(dp):
                    print(f"[DES-LOC M701] rank={r} hetero_shard_params={raw_sizes[r]} "
                          f"frac={self._hetero_shard_fracs[r]:.4f} total={total_num_elements}")
        else:
            base_size = total_num_elements // dp
            remaining = total_num_elements % dp
            start = 0
            for id in range(dp):
                partition_size = base_size + (1 if id < remaining else 0)
                partitions.append(tensor.narrow(0, start, partition_size))
                start += partition_size
        return partitions


class MockStage3(HeteroShardMixin):
    pass


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_hetero_partitioning():
    print("\n=== TEST 1: Hetero shard ratio [1, 1, 2] (A6000, A6000, H100) ===")
    obj = MockStage3()
    # ratio: A6000=49GB ≈ 1 unit, H100=96GB ≈ 2 units
    obj._init_hetero([1.0, 1.0, 2.0], dp_world_size=3)
    fake_dist = FakeDistributed(world_size=3, rank=0)

    total_params = 10000
    tensor = torch.arange(total_params, dtype=torch.float32)
    partitions = obj.get_data_parallel_partitions(tensor, fake_dist)

    sizes = [p.numel() for p in partitions]
    print(f"  Partition sizes: {sizes}")
    assert sum(sizes) == total_params, "Partition sizes must sum to total"
    # A6000 ranks should each get ~25%, H100 should get ~50%
    assert sizes[0] == sizes[1], "Both A6000 ranks should be equal"
    assert sizes[2] == sizes[0] * 2, "H100 rank should get 2x A6000 rank"
    print("  PASS: sizes correct")

    # Check contiguity: each partition is a slice of the original tensor
    start = 0
    for i, p in enumerate(partitions):
        expected = tensor[start:start + sizes[i]]
        assert torch.equal(p, expected), f"Partition {i} content mismatch"
        start += sizes[i]
    print("  PASS: partition contents match original tensor")


def test_uniform_fallback():
    print("\n=== TEST 2: Uniform partitioning (hetero_shard_ratio=None) ===")
    obj = MockStage3()
    obj._init_hetero(None, dp_world_size=3)
    fake_dist = FakeDistributed(world_size=3, rank=0)

    total_params = 10001  # not evenly divisible
    tensor = torch.arange(total_params, dtype=torch.float32)
    partitions = obj.get_data_parallel_partitions(tensor, fake_dist, verbose=False)

    sizes = [p.numel() for p in partitions]
    print(f"  Partition sizes: {sizes}")
    assert sum(sizes) == total_params
    # Sizes should differ by at most 1
    assert max(sizes) - min(sizes) <= 1
    print("  PASS: uniform partition with remainder distributed correctly")


def test_gradient_consistency():
    """
    Gradient consistency mock: ensure each rank owns a disjoint, contiguous slice
    and that reassembling gives back the full gradient.
    """
    print("\n=== TEST 3: Gradient consistency (hetero shard reassembly) ===")
    obj = MockStage3()
    obj._init_hetero([1.0, 1.0, 2.0], dp_world_size=3)
    fake_dist = FakeDistributed(world_size=3, rank=0)

    total_params = 1024
    grad_flat = torch.randn(total_params)
    partitions = obj.get_data_parallel_partitions(grad_flat, fake_dist, verbose=False)

    # Reassemble
    reassembled = torch.cat(partitions)
    assert torch.allclose(reassembled, grad_flat), "Reassembled gradient != original"
    print("  PASS: cat(partitions) == original gradient tensor")


def test_remainder_distribution():
    """
    When total_params % ratio_sum != 0, remainder elements must be allocated.
    """
    print("\n=== TEST 4: Remainder allocation with non-divisible total ===")
    obj = MockStage3()
    obj._init_hetero([1.0, 1.0, 2.0], dp_world_size=3)
    fake_dist = FakeDistributed(world_size=3, rank=0)

    for total in [7, 11, 100, 999, 10007]:
        tensor = torch.zeros(total)
        partitions = obj.get_data_parallel_partitions(tensor, fake_dist, verbose=False)
        assert sum(p.numel() for p in partitions) == total, f"total={total} failed"
    print("  PASS: all totals accounted for")


if __name__ == "__main__":
    test_hetero_partitioning()
    test_uniform_fallback()
    test_gradient_consistency()
    test_remainder_distribution()
    print("\n✅ All M701 hetero_shard_ratio tests passed.")
