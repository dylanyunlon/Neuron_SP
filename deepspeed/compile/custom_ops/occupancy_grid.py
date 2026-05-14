import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch


@dataclass
class KernelOccupancy:
    blocks_per_sm: int = 0
    max_occupancy: int = 0
    grid_size: int = 0
    block_threads: int = 256
    num_sms: int = 0
    shared_mem_per_block: int = 0
    register_pressure: int = 0


@dataclass
class GPUCapabilitySnapshot:
    num_sms: int = 0
    max_threads_per_sm: int = 0
    max_blocks_per_sm: int = 0
    max_shared_memory_per_sm: int = 0
    warp_size: int = 32
    compute_capability: Tuple[int, int] = (0, 0)


def probe_gpu_capability(device_id: int = -1) -> GPUCapabilitySnapshot:
    if not torch.cuda.is_available():
        return GPUCapabilitySnapshot()

    if device_id < 0:
        device_id = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device_id)

    _MAX_BLOCKS_TABLE = {
        (7, 0): 32, (7, 5): 16, (8, 0): 32, (8, 6): 16,
        (8, 9): 16, (9, 0): 32, (10, 0): 32,
    }

    cc = (props.major, props.minor)
    max_blocks = _MAX_BLOCKS_TABLE.get(cc, 16)

    return GPUCapabilitySnapshot(
        num_sms=props.multi_processor_count,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        max_blocks_per_sm=max_blocks,
        max_shared_memory_per_sm=props.max_shared_memory_per_block * 2,
        warp_size=props.warp_size if hasattr(props, 'warp_size') else 32,
        compute_capability=cc,
    )


def compute_occupancy(num_elements: int,
                      block_threads: int = 256,
                      shared_mem_per_block: int = 0,
                      capability: Optional[GPUCapabilitySnapshot] = None) -> KernelOccupancy:
    if capability is None:
        capability = probe_gpu_capability()

    if capability.num_sms == 0:
        return KernelOccupancy(grid_size=1, block_threads=block_threads)

    warps_per_block = math.ceil(block_threads / capability.warp_size)
    max_warps_per_sm = capability.max_threads_per_sm // capability.warp_size

    blocks_by_warps = max_warps_per_sm // warps_per_block if warps_per_block > 0 else 0
    blocks_by_limit = capability.max_blocks_per_sm

    blocks_by_smem = capability.max_blocks_per_sm
    if shared_mem_per_block > 0 and capability.max_shared_memory_per_sm > 0:
        blocks_by_smem = capability.max_shared_memory_per_sm // shared_mem_per_block

    blocks_per_sm = min(blocks_by_warps, blocks_by_limit, blocks_by_smem)
    blocks_per_sm = max(blocks_per_sm, 1)

    max_occupancy = blocks_per_sm * capability.num_sms
    num_tiles = math.ceil(num_elements / block_threads)
    grid_size = min(max_occupancy, num_tiles)

    return KernelOccupancy(
        blocks_per_sm=blocks_per_sm,
        max_occupancy=max_occupancy,
        grid_size=grid_size,
        block_threads=block_threads,
        num_sms=capability.num_sms,
        shared_mem_per_block=shared_mem_per_block,
    )


def compute_histogram_grid(num_elements: int,
                           capability: Optional[GPUCapabilitySnapshot] = None) -> int:
    occ = compute_occupancy(
        num_elements=num_elements,
        block_threads=256,
        shared_mem_per_block=256 * 8,
        capability=capability,
    )
    return occ.grid_size


def compute_filter_grid(num_elements: int,
                        capability: Optional[GPUCapabilitySnapshot] = None) -> int:
    occ = compute_occupancy(
        num_elements=num_elements,
        block_threads=256,
        shared_mem_per_block=256 * 4,
        capability=capability,
    )
    return occ.grid_size


def compute_a2a_grid_for_tier(num_elements: int, tier: int,
                              capability: Optional[GPUCapabilitySnapshot] = None) -> int:
    _TIER_BLOCK_THREADS = {1: 128, 2: 256, 3: 512}
    block_threads = _TIER_BLOCK_THREADS.get(tier, 256)

    occ = compute_occupancy(
        num_elements=num_elements,
        block_threads=block_threads,
        capability=capability,
    )
    return occ.grid_size


_CAPABILITY_CACHE: Dict[int, GPUCapabilitySnapshot] = {}


def get_cached_capability(device_id: int = -1) -> GPUCapabilitySnapshot:
    if device_id < 0 and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
    if device_id not in _CAPABILITY_CACHE:
        _CAPABILITY_CACHE[device_id] = probe_gpu_capability(device_id)
    return _CAPABILITY_CACHE[device_id]
