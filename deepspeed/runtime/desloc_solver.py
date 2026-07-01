"""
PartitionSolver: evaluates ZeRO-3 and Pipeline 1F1B strategies and selects
the one with higher estimated training throughput.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from deepspeed.runtime.desloc_config import (
    PartitionPlan,
    PartitionStrategy,
    TierClass,
    TierSpec,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


class PartitionSolver:
    """
    Evaluates two partition strategies for the discovered GPU tiers and
    selects the one with higher estimated training throughput.

    Strategy A — ZeRO-3 + Heterogeneous Gradient Accumulation:
        H100 processes large micro-batches (22 per accumulation step);
        each A6000 processes 1 micro-batch. AllReduce synchronizes gradients.

    Strategy B — Pipeline 1F1B:
        H100 hosts 30 transformer layers (largest share due to BF16 headroom);
        each A6000 hosts 1 layer. 1F1B schedule used to overlap forward/backward.

    Throughput estimation is analytical (not profiled), based on:
        tokens/s ≈ Σ_device(micro_bs * seq_len * grad_accum / step_time_device)
    where step_time is estimated from BF16 TFLOPs and model FLOPs per token.
    """

    def __init__(self, tiers: List[TierSpec], config: TrainingConfig) -> None:
        self.tiers = tiers
        self.config = config

    def solve(self) -> PartitionPlan:
        """
        Compare both strategies and return the better PartitionPlan.

        If config.strategy_override is set, that strategy is used directly.

        Returns:
            PartitionPlan with strategy, layer assignments, and grad accum steps.
        """
        if self.config.strategy_override is not None:
            logger.info("Strategy override: %s", self.config.strategy_override)
            if self.config.strategy_override == PartitionStrategy.ZERO3_HETERO:
                return self._plan_zero3()
            return self._plan_pipeline()

        plan_a = self._plan_zero3()
        plan_b = self._plan_pipeline()

        logger.info(
            "PartitionSolver — ZeRO-3 est. %.1f tok/s  |  Pipeline est. %.1f tok/s",
            plan_a.estimated_throughput,
            plan_b.estimated_throughput,
        )

        chosen = plan_a if plan_a.estimated_throughput >= plan_b.estimated_throughput else plan_b
        logger.info("Selected strategy: %s  (%s)", chosen.strategy, chosen.notes)
        return chosen

    # ------------------------------------------------------------------
    # Strategy A: ZeRO-3 + Heterogeneous Gradient Accumulation
    # ------------------------------------------------------------------
    def _plan_zero3(self) -> PartitionPlan:
        """ZeRO-3 plan: uniform grad_accum, per-tier micro_batch_size."""
        _TIER_MULTIPLIER: Dict[TierClass, int] = {
            TierClass.H100:            8,
            TierClass.RTX_PRO_6000_BW: 4,
            TierClass.A6000:           1,
            TierClass.UNKNOWN:         1,
        }

        cfg = self.config
        tier_layer_map: Dict[int, List[int]] = {}
        grad_accum: Dict[int, int] = {}
        micro_bs: Dict[int, int] = {}

        all_layers = list(range(cfg.num_layers))
        for spec in self.tiers:
            tier_layer_map[spec.device_index] = all_layers[:]
            grad_accum[spec.device_index] = cfg.grad_accum_steps

            if (cfg.micro_batch_size_per_gpu is not None
                    and spec.device_index < len(cfg.micro_batch_size_per_gpu)):
                micro_bs[spec.device_index] = cfg.micro_batch_size_per_gpu[spec.device_index]
            else:
                multiplier = _TIER_MULTIPLIER.get(spec.tier, 1)
                micro_bs[spec.device_index] = cfg.micro_batch_size * multiplier

        throughput = self._estimate_zero3_throughput(micro_bs, grad_accum)

        return PartitionPlan(
            strategy=PartitionStrategy.ZERO3_HETERO,
            tier_layer_map=tier_layer_map,
            grad_accum_steps=grad_accum,
            micro_batch_sizes=micro_bs,
            estimated_throughput=throughput,
            notes=(
                f"ZeRO-3 hetero: grad_accum={cfg.grad_accum_steps}, "
                f"per-device micro_bs={dict(sorted(micro_bs.items()))} "
                f"(source={'yaml' if cfg.micro_batch_size_per_gpu else 'tflops_multiplier'})"
            ),
        )

    # ------------------------------------------------------------------
    # Strategy B: Pipeline 1F1B
    # ------------------------------------------------------------------
    def _plan_pipeline(self) -> PartitionPlan:
        """Build the Pipeline 1F1B partition plan."""
        cfg = self.config
        tier_layer_map: Dict[int, List[int]] = {}
        grad_accum: Dict[int, int] = {}
        micro_bs: Dict[int, int] = {}

        layers = list(range(cfg.num_layers))
        h100_specs = [s for s in self.tiers if s.tier == TierClass.H100]
        a6000_specs = [s for s in self.tiers if s.tier == TierClass.A6000]
        other_specs = [s for s in self.tiers
                       if s.tier not in (TierClass.H100, TierClass.A6000)]

        n_h100 = len(h100_specs)
        n_a6000 = len(a6000_specs)
        n_other = len(other_specs)
        total_devices = n_h100 + n_a6000 + n_other

        if total_devices == 0:
            total_devices = 1

        h100_share = min(30, cfg.num_layers - n_a6000 - n_other)
        a6000_share = 1 if n_a6000 > 0 else 0
        leftover = cfg.num_layers - h100_share * n_h100 - a6000_share * n_a6000

        cursor = 0
        for spec in h100_specs:
            n = h100_share
            tier_layer_map[spec.device_index] = layers[cursor: cursor + n]
            cursor += n
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            micro_bs[spec.device_index] = cfg.micro_batch_size

        for spec in a6000_specs:
            n = a6000_share
            tier_layer_map[spec.device_index] = layers[cursor: cursor + n]
            cursor += n
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            micro_bs[spec.device_index] = cfg.micro_batch_size

        per_other = (leftover // n_other) if n_other else 0
        for spec in other_specs:
            n = per_other
            tier_layer_map[spec.device_index] = layers[cursor: cursor + n]
            cursor += n
            grad_accum[spec.device_index] = cfg.grad_accum_steps
            micro_bs[spec.device_index] = cfg.micro_batch_size

        throughput = self._estimate_pipeline_throughput(micro_bs, grad_accum)

        return PartitionPlan(
            strategy=PartitionStrategy.PIPELINE_1F1B,
            tier_layer_map=tier_layer_map,
            grad_accum_steps=grad_accum,
            micro_batch_sizes=micro_bs,
            estimated_throughput=throughput,
            notes=f"Pipeline 1F1B: H100 {h100_share} layers, A6000 {a6000_share} each",
        )

    # ------------------------------------------------------------------
    # Throughput estimators (analytical)
    # ------------------------------------------------------------------
    def _flops_per_token(self) -> float:
        """
        Approximate FLOPs per token for a transformer model.

        Using the standard 6*N approximation (N = parameter count).
        N ≈ 12 * num_layers * hidden_size^2 for a dense transformer.
        """
        n_params = 12 * self.config.num_layers * self.config.hidden_size ** 2
        return 6 * n_params

    def _estimate_zero3_throughput(
        self,
        micro_bs: Dict[int, int],
        grad_accum: Dict[int, int],
    ) -> float:
        """Estimate tokens/s for ZeRO-3 strategy (bottlenecked by slowest device)."""
        fpt = self._flops_per_token()
        step_times = []
        for spec in self.tiers:
            idx = spec.device_index
            tokens_per_step = (
                micro_bs.get(idx, 1)
                * self.config.seq_len
                * grad_accum.get(idx, 1)
            )
            flops_per_step = fpt * tokens_per_step
            tflops = spec.bf16_tflops * 1e12
            step_time = flops_per_step / (tflops * 0.35)
            step_times.append((tokens_per_step, step_time))

        if not step_times:
            return 0.0

        bottleneck_time = max(t for _, t in step_times)
        total_tokens = sum(tok for tok, _ in step_times)
        return total_tokens / bottleneck_time if bottleneck_time > 0 else 0.0

    def _estimate_pipeline_throughput(
        self,
        micro_bs: Dict[int, int],
        grad_accum: Dict[int, int],
    ) -> float:
        """Estimate tokens/s for Pipeline strategy with bubble overhead."""
        fpt = self._flops_per_token()
        n_stages = len(self.tiers)
        step_times = []
        for spec in self.tiers:
            idx = spec.device_index
            n_layers = len(self.config.num_layers > 0 and [] or [])  # placeholder
            tokens_per_micro = micro_bs.get(idx, 1) * self.config.seq_len
            flops_per_micro = fpt * tokens_per_micro / max(n_stages, 1)
            tflops = spec.bf16_tflops * 1e12
            micro_time = flops_per_micro / (tflops * 0.35)
            step_times.append(micro_time)

        if not step_times:
            return 0.0

        t_stage = max(step_times)
        n_micro = grad_accum.get(list(grad_accum.keys())[0], 1) if grad_accum else 1
        total_step_time = t_stage * (n_micro + n_stages - 1)
        total_tokens = sum(
            micro_bs.get(s.device_index, 1) * self.config.seq_len * n_micro
            for s in self.tiers
        )
        return total_tokens / total_step_time if total_step_time > 0 else 0.0
