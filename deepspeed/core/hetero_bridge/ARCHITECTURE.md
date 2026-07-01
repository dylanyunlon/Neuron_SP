# hetero_bridge — Phase 1 Architecture & API Contract

**Status:** SKELETON. Interfaces are frozen. Sub-Claudes fill implementations only.
**Author:** Manager Claude (Phase 1). **Do NOT change signatures below without manager sign-off.**

## Why this subsystem exists

`deepspeed/core/optimizer/` (Megatron `DistributedOptimizer`, `param_layout`, `clip_grads`)
and `deepspeed/core/pipeline_parallel/schedules.py` were migrated but are **not driving
training**. `run_pretrain.py` still uses plain `torch.optim.AdamW` + FSDP. `desloc_engine.py`
*references* `core.optimizer.DistributedOptimizer` and `core.pipeline_parallel.schedules`
(lines ~1742–2179) but the connective tissue is incomplete.

`hetero_bridge` is that connective tissue: a real subsystem (not one wired file) that lets
the Megatron distributed optimizer + pipeline schedules run correctly on the heterogeneous
PCIe cluster (2×A6000 + H100 + 2×Blackwell, no NVLink, P2P disabled), integrated with BOTH
DES-LOC (per-parameter sync periods) and AutoSP (Ulysses sequence parallel).

## Module layout (files to create under deepspeed/core/hetero_bridge/)

```
hetero_bridge/
├── __init__.py                 # public exports only
├── tier_map.py                 # TierMap: which rank is which GPU tier + mem budget
├── shard_planner.py            # HeteroShardPlanner: param-shard assignment across tiers
├── dist_opt_adapter.py         # DistOptAdapter: wraps core.optimizer.DistributedOptimizer
├── desloc_sync_policy.py       # DesLocSyncPolicy: per-param Kx/Ku/Kv sync schedule
├── pp_schedule_adapter.py      # PPScheduleAdapter: hetero 1F1B layer-split bridge
├── autosp_hook.py              # AutoSPHook: connect SP group to optimizer/grad flow
└── engine_integration.py       # install(engine): single entrypoint, wires all of the above
```

Each `.py` is a real module with classes, docstrings, and `raise NotImplementedError("<what>")`
in method bodies for Phase 2. NO business logic in Phase 1.

## Frozen public API (the contract)

### tier_map.py
```python
class GPUTier(enum.Enum):
    A6000 = "a6000"        # SM8.6, 48GB
    H100  = "h100"         # SM9.0, 96GB
    BLACKWELL = "blackwell"# SM12.0, 96GB

@dataclass
class TierInfo:
    rank: int
    tier: GPUTier
    total_vram_bytes: int
    numa_node: int
    peak_bf16_tflops: float

class TierMap:
    def __init__(self, world_size: int) -> None: ...
    @classmethod
    def discover(cls) -> "TierMap":
        """Enumerate local+distributed ranks → TierInfo via torch.cuda + NVML."""
        raise NotImplementedError
    def tier_of(self, rank: int) -> GPUTier: raise NotImplementedError
    def mem_budget(self, rank: int) -> int:
        """Usable bytes for optimizer state after model+activations. Tier-specific."""
        raise NotImplementedError
    def ranks_of_tier(self, tier: GPUTier) -> list[int]: raise NotImplementedError
```

### shard_planner.py
```python
@dataclass
class ShardPlan:
    rank_to_param_ids: dict[int, list[int]]   # which params' fp32 shard each rank owns
    rank_to_bytes: dict[int, int]
    rationale: str

class HeteroShardPlanner:
    def __init__(self, tier_map: "TierMap") -> None: ...
    def plan(self, named_params: list[tuple[str, "torch.Tensor"]]) -> "ShardPlan":
        """Assign fp32 optimizer shards so bigger-VRAM tiers hold more.
        Must respect TierMap.mem_budget. Deterministic across ranks (same input→same plan)."""
        raise NotImplementedError
```

### dist_opt_adapter.py
```python
class DistOptAdapter:
    """Adapts core.optimizer.DistributedOptimizer to the hetero ShardPlan.
    On A6000 ranks: optimizer state resident on CPU (DeepSpeedCPUAdam path).
    On H100/Blackwell ranks: fused AdamW on GPU. (Matches HEAD commit a52efee1.)"""
    def __init__(self, model: "nn.Module", shard_plan: "ShardPlan",
                 tier_map: "TierMap", lr: float, betas: tuple[float,float],
                 weight_decay: float) -> None: ...
    def build(self) -> "torch.optim.Optimizer":
        """Construct the underlying DistributedOptimizer with per-rank optimizer type."""
        raise NotImplementedError
    def step(self) -> None: raise NotImplementedError
    def zero_grad(self, set_to_none: bool = True) -> None: raise NotImplementedError
    def reduce_scatter_grads(self) -> None:
        """Hetero reduce-scatter: grads → owning rank's fp32 shard, PCIe-aware."""
        raise NotImplementedError
    def all_gather_params(self) -> None:
        """Gather updated bf16 params back to all ranks after step."""
        raise NotImplementedError
```

### desloc_sync_policy.py
```python
@dataclass
class SyncPeriods:
    kx: int   # fast params (attention) sync period
    ku: int   # slow params
    kv: int   # very-slow params

class DesLocSyncPolicy:
    """Decides which params sync this step, per DES-LOC decomposed local SGD."""
    def __init__(self, periods: "SyncPeriods") -> None: ...
    def classify(self, named_params: list[tuple[str, "torch.Tensor"]]) -> dict[int, str]:
        """param_id → {'x','u','v'} class by gradient-variance heuristic."""
        raise NotImplementedError
    def should_sync(self, param_id: int, step: int) -> bool: raise NotImplementedError
```

### pp_schedule_adapter.py
```python
class PPScheduleAdapter:
    """Bridge to core.pipeline_parallel.schedules for hetero layer split.
    Blackwell/H100 stages get more layers than A6000 stages (bubble optimization)."""
    def __init__(self, tier_map: "TierMap", num_layers: int) -> None: ...
    def layer_split(self) -> list[int]:
        """Layers-per-stage, VRAM-weighted. e.g. [6,6,20] for 2×A6000+H100."""
        raise NotImplementedError
    def forward_backward(self, *, data_iterator, model, num_microbatches: int,
                         seq_length: int, micro_batch_size: int) -> dict:
        """Delegate to schedules.forward_backward_pipelining_* with the hetero split."""
        raise NotImplementedError
```

### autosp_hook.py
```python
class AutoSPHook:
    """Connects the AutoSP/Ulysses SP group to the optimizer + grad reduction so
    SP-sharded params reduce correctly and DES-LOC sync respects the SP group."""
    def __init__(self, sp_group, tier_map: "TierMap") -> None: ...
    def wrap_grad_reduction(self, adapter: "DistOptAdapter") -> None:
        raise NotImplementedError
    def sp_aware_sync(self, policy: "DesLocSyncPolicy") -> None:
        raise NotImplementedError
```

### engine_integration.py
```python
def install(engine, *, lr: float, betas=(0.9,0.95), weight_decay=0.1,
            sync_periods: "SyncPeriods" | None = None) -> None:
    """SINGLE entrypoint. desloc_engine calls this once.
    Builds TierMap.discover() → HeteroShardPlanner.plan() → DistOptAdapter.build()
    → DesLocSyncPolicy → PPScheduleAdapter → AutoSPHook, and attaches them to `engine`.
    After install(), engine.optimizer is the hetero DistributedOptimizer and the
    training loop uses adapter.reduce_scatter_grads / step / all_gather_params."""
    raise NotImplementedError
```

## Integration point (already half-present)

`desloc_engine.py` will call `hetero_bridge.engine_integration.install(self, lr=...)` in
its optimizer-setup path, replacing the ad-hoc `AdamW`/FSDP setup. The references at
lines 1742–2179 (DistributedOptimizer owns FP32 shards) become real once `install` runs.

## Success criteria (Phase 2 done means ALL of these)

1. `python -c "from deepspeed.core.hetero_bridge.engine_integration import install"` works.
2. `desloc_engine` imports and calls `install()`; NO `torch.optim.AdamW` fallback remains
   in the DES-LOC path.
3. A 70M smoke test (`--model-size 70m --steps 5`) runs to completion using the bridge.
4. `grep -r "NotImplementedError" hetero_bridge/` returns nothing in method bodies.
5. Optimizer state is CPU-resident on A6000 ranks, GPU-fused on H100/Blackwell ranks.
6. No new files with v2/v3/vN suffixes. No new branches. Everything on `main`.

## Task split for Phase 2 (one sub-Claude per group, own conversation)

- **Sub-Claude A:** `tier_map.py` + `shard_planner.py` (discovery + shard math).
  Read: `deepspeed/core/optimizer/param_layout.py`, `desloc_discovery.py`, `desloc_solver.py`.
- **Sub-Claude B:** `dist_opt_adapter.py` (the core — wrap DistributedOptimizer, per-tier
  optimizer). Read: `deepspeed/core/optimizer/distrib_optimizer.py` (full history), HEAD
  commit `a52efee1` (VRAM-adaptive optimizer), `zero3_hetero_shard.py`.
- **Sub-Claude C:** `desloc_sync_policy.py` + `autosp_hook.py`. Read: `desloc_engine.py`
  finalize_model_grads path, `deepspeed/sequence/auto_sp.py`.
- **Sub-Claude D:** `pp_schedule_adapter.py` + `engine_integration.py`. Read:
  `deepspeed/core/pipeline_parallel/schedules.py`, `references/.../combined_1f1b.py`.

Because every signature above is frozen, the four outputs compose without interface drift.
