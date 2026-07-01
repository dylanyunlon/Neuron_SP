"""
TierDiscovery: detects available GPUs and builds a ranked list of TierSpec objects.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List, Tuple

import torch

from deepspeed.runtime.desloc_types import TierClass, TierSpec

logger = logging.getLogger(__name__)


class TierDiscovery:
    """
    Detects available GPUs using torch.cuda and cross-references with nvidia-smi
    to build a ranked list of TierSpec objects.

    Discovery logic:
    - SM 9.x + >= 80GB → H100-class
    - SM 8.6 + >= 40GB → A6000-class
    - Everything else → UNKNOWN (still usable, degraded performance)
    """

    # Known BF16 TFLOPs and PCIe BW by (sm_major, sm_minor, approx_mem_gb)
    _PERF_TABLE: Dict[Tuple[int, int], Tuple[float, float]] = {
        (9, 0): (835.0, 50.0),   # H100 NVL  PCIe5
        (8, 6): (38.7, 25.0),    # A6000     PCIe4
        (8, 0): (312.0, 40.0),   # A100      PCIe4
        (7, 0): (14.1, 16.0),    # V100
    }

    def discover(self) -> List[TierSpec]:
        """
        Run full GPU discovery and return sorted list (highest-tier first).

        Returns:
            List of TierSpec, sorted by bf16_tflops descending.

        Raises:
            RuntimeError: If no CUDA-capable GPUs are found.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA-capable GPUs detected. Cannot run DES-LOC engine.")

        n_gpus = torch.cuda.device_count()
        logger.info("TierDiscovery: found %d CUDA device(s).", n_gpus)

        numa_map = self._query_numa_map()
        specs: List[TierSpec] = []

        for idx in range(n_gpus):
            try:
                spec = self._inspect_device(idx, numa_map)
                specs.append(spec)
                logger.info("  %s", spec)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to inspect GPU %d: %s", idx, exc)

        if not specs:
            raise RuntimeError("TierDiscovery found zero usable GPUs.")

        specs.sort(key=lambda s: s.bf16_tflops, reverse=True)
        return specs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _inspect_device(self, idx: int, numa_map: Dict[int, int]) -> TierSpec:
        """Build a TierSpec for a single CUDA device index."""
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = props.total_memory / (1 << 30)
        sm_major = props.major
        sm_minor = props.minor
        name = props.name

        torch.cuda.synchronize(idx)
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        free_mem_gb = free_bytes / (1 << 30)

        bf16_tflops, pcie_bw = self._PERF_TABLE.get(
            (sm_major, sm_minor),
            (self._estimate_tflops(props), 16.0),
        )

        tier = self._classify(sm_major, sm_minor, total_mem_gb)
        numa_node = numa_map.get(idx, -1)

        return TierSpec(
            device_index=idx,
            tier=tier,
            total_mem_gb=total_mem_gb,
            free_mem_gb=free_mem_gb,
            sm_major=sm_major,
            sm_minor=sm_minor,
            bf16_tflops=bf16_tflops,
            pcie_bw_gbs=pcie_bw,
            numa_node=numa_node,
            name=name,
        )

    @staticmethod
    def _classify(sm_major: int, sm_minor: int, mem_gb: float) -> TierClass:
        """Classify a GPU into a TierClass based on SM version and memory."""
        if sm_major >= 12 and mem_gb >= 90:
            return TierClass.RTX_PRO_6000_BW
        if sm_major == 9 and mem_gb >= 80:
            return TierClass.H100
        if sm_major == 8 and sm_minor == 6 and mem_gb >= 40:
            return TierClass.A6000
        return TierClass.UNKNOWN

    @staticmethod
    def _estimate_tflops(props: Any) -> float:
        """Rough BF16 TFLOPs estimate when not in the perf table."""
        return 2 * props.multi_processor_count * 128 * 1.5 / 1e3

    @staticmethod
    def _query_numa_map() -> Dict[int, int]:
        """
        Query NUMA affinity for each GPU via nvidia-smi.

        Returns:
            Dict mapping device_index -> numa_node.
        """
        numa: Dict[int, int] = {}
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,numa_affinity",
                 "--format=csv,noheader,nounits"],
                timeout=10,
                stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    try:
                        numa[int(parts[0])] = int(parts[1])
                    except ValueError:
                        pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("nvidia-smi NUMA query failed (non-fatal): %s", exc)
        return numa
