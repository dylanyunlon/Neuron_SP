# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
engine_bridge.py — Megatron ↔ DeepSpeed 异构引擎桥接

解决的问题:
  - pretrain_commit.py 用 Megatron 框架 (model_provider, forward_step)
  - hetero_shard_ratio / HeteroMemoryManager 在 DeepSpeed runtime 里
  - 需要一个桥把两边接起来

工作方式:
  1. 用 Megatron 构建模型 (GPTModel)
  2. 用 DeepSpeed 初始化引擎 (带 hetero 配置)
  3. 训练时用 DeepSpeed 的 engine.step() 替代 Megatron 的 optimizer.step()

硬件目标:
  GPU0: RTX A6000 49GB  (NUMA0)  — shard_ratio=1.0
  GPU1: RTX PRO 6000 Blackwell 98GB (NUMA0) — shard_ratio=2.0
  GPU2: H100 NVL 96GB   (NUMA0)  — shard_ratio=2.0
  GPU3: RTX A6000 49GB  (NUMA1)  — shard_ratio=1.0
  GPU4: RTX PRO 6000 Blackwell 98GB (NUMA1) — shard_ratio=2.0
"""

import os
import json
import torch
from typing import Optional, Dict, Any, List

try:
    import deepspeed
    import deepspeed.comm as dist
    _HAS_DEEPSPEED = True
except ImportError:
    _HAS_DEEPSPEED = False


# ── GPU tier 自动检测 ──

def detect_gpu_tiers() -> List[Dict[str, Any]]:
    """自动检测每张 GPU 的型号/显存/compute capability."""
    tiers = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / (1024 ** 3)
        tiers.append({
            "index": i,
            "name": props.name,
            "mem_gb": round(mem_gb, 1),
            "compute": f"{props.major}.{props.minor}",
            "is_ampere": props.major == 8,
            "is_hopper": props.major == 9,
            "is_blackwell": props.major >= 10,
        })
        print(f"  [GPU{i}] {props.name}  {mem_gb:.0f}GB  SM{props.major}.{props.minor}")
    return tiers


def compute_shard_ratios(tiers: List[Dict]) -> List[float]:
    """按显存比例计算 hetero_shard_ratio.

    49GB GPU → 1.0, 96GB GPU → 2.0, 98GB → 2.0
    """
    min_mem = min(t["mem_gb"] for t in tiers)
    ratios = [round(t["mem_gb"] / min_mem, 1) for t in tiers]
    print(f"  [bridge] shard_ratios: {ratios} (min={min_mem:.0f}GB)")
    return ratios


def compute_dtype_per_gpu(tiers: List[Dict]) -> List[torch.dtype]:
    """每张 GPU 的最优精度.

    Ampere (A6000): FP16 (TF32 for matmul)
    Hopper (H100): BF16
    Blackwell: BF16
    """
    dtypes = []
    for t in tiers:
        if t["is_hopper"] or t["is_blackwell"]:
            dtypes.append(torch.bfloat16)
        else:
            dtypes.append(torch.float16)
    names = [str(d).split(".")[-1] for d in dtypes]
    print(f"  [bridge] dtype_map: {names}")
    return dtypes


# ── DeepSpeed 配置生成 ──

def build_ds_config(
    train_batch_size: int = 8,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    hetero_shard_ratio: Optional[List[float]] = None,
    offload_optimizer: bool = True,
    offload_param: bool = False,
    fp16: bool = False,
    bf16: bool = True,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 2000,
    total_steps: int = 100000,
) -> Dict[str, Any]:
    """生成 DeepSpeed JSON 配置 (ZeRO-3 + 异构)."""
    config = {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 100,
        "wall_clock_breakdown": True,

        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 50_000_000,
            "stage3_prefetch_bucket_size": 50_000_000,
            "stage3_param_persistence_threshold": 100_000,
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": weight_decay,
            },
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": warmup_steps,
                "total_num_steps": total_steps,
            },
        },

        "gradient_clipping": 1.0,
    }

    if bf16:
        config["bf16"] = {"enabled": True}
    elif fp16:
        config["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}

    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if offload_param:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    # DES-LOC 异构扩展 (非标准字段, 由修改过的 stage3.py 读取)
    if hetero_shard_ratio is not None:
        config["zero_optimization"]["hetero_shard_ratio"] = hetero_shard_ratio

    return config


# ── 引擎桥接 ──

class DESLOCEngine:
    """将 Megatron 模型包装为 DeepSpeed 引擎, 带异构支持.

    Usage:
        model = megatron_model_provider()
        engine = DESLOCEngine(model)
        engine.init()

        for batch in dataloader:
            loss = engine.train_step(batch)
            if engine.step_count % save_interval == 0:
                engine.save_checkpoint(path)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ds_config: Optional[Dict] = None,
        auto_detect_hetero: bool = True,
    ):
        self.model = model
        self.ds_config = ds_config
        self.auto_detect_hetero = auto_detect_hetero
        self.engine = None
        self.optimizer = None
        self.lr_scheduler = None
        self.step_count = 0
        self._tiers = None
        self._dtypes = None

    def init(self):
        """初始化 DeepSpeed 引擎."""
        print("[DESLOCEngine] detecting GPU topology...")
        self._tiers = detect_gpu_tiers()

        if self.ds_config is None:
            ratios = compute_shard_ratios(self._tiers) if self.auto_detect_hetero else None
            self._dtypes = compute_dtype_per_gpu(self._tiers)
            use_bf16 = any(d == torch.bfloat16 for d in self._dtypes)
            self.ds_config = build_ds_config(
                hetero_shard_ratio=ratios,
                bf16=use_bf16,
                fp16=not use_bf16,
            )

        self.engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            config=self.ds_config,
            model_parameters=self.model.parameters(),
        )
        print(f"[DESLOCEngine] initialized, world_size={dist.get_world_size()}, "
              f"ZeRO stage={self.ds_config['zero_optimization']['stage']}")
        return self

    def train_step(self, input_ids, labels=None, attention_mask=None):
        """一步训练: forward + backward + optimizer step."""
        local_rank = dist.get_rank()

        # 异构精度: 根据当前 GPU 选 dtype
        if self._dtypes and local_rank < len(self._dtypes):
            target_dtype = self._dtypes[local_rank]
            input_ids = input_ids.to(self.engine.device)
            if labels is not None:
                labels = labels.to(self.engine.device)
        else:
            input_ids = input_ids.to(self.engine.device)
            if labels is not None:
                labels = labels.to(self.engine.device)

        outputs = self.engine(input_ids=input_ids, labels=labels,
                              attention_mask=attention_mask)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        self.engine.backward(loss)
        self.engine.step()
        self.step_count += 1

        return loss.item()

    def save_checkpoint(self, path: str, tag: Optional[str] = None):
        """保存 checkpoint (DeepSpeed 格式, 含 ZeRO states)."""
        tag = tag or f"step_{self.step_count}"
        self.engine.save_checkpoint(path, tag=tag)
        print(f"[DESLOCEngine] checkpoint saved: {path}/{tag}")

    def load_checkpoint(self, path: str, tag: Optional[str] = None):
        """加载 checkpoint, 用于阶段间传递."""
        _, client_state = self.engine.load_checkpoint(path, tag=tag)
        if client_state and "step_count" in client_state:
            self.step_count = client_state["step_count"]
        print(f"[DESLOCEngine] checkpoint loaded: {path}/{tag}, step={self.step_count}")
        return client_state

    def current_lr(self) -> float:
        if self.lr_scheduler:
            return self.lr_scheduler.get_last_lr()[0]
        return self.ds_config.get("optimizer", {}).get("params", {}).get("lr", 0)


# ── HeteroMIMO runtime bridge (used by train_three_stage.py) ──

def build_neuron_sp_runtime(
    model: torch.nn.Module,
    lr: float,
    cache_max_gb: float = 192.0,
    grad_clip: float = 1.0,
):
    """Build the per-stage hetero runtime (recompute config + MIMO loop).

    This wires the two already-existing hetero entry points so each stage of
    ``train_three_stage.py`` gets a freshly-initialised training loop with the
    correct learning rate and a Neuron_SP-targeted recompute policy.

    Returns
    -------
    (loop, recompute_config) : Tuple[HeteroMIMOTrainingLoop, HeteroRecomputeConfig]
    """
    from deepspeed.runtime.hetero_gdn_selective_recompute import build_neuron_sp_config
    from deepspeed.runtime.hetero_mimo_training_loop import (
        setup_hetero_mimo_training,
        PerModuleOptimizerConfig,
    )

    # 1) Recompute / activation-offload policy tuned for the target cluster.
    #    Auto-detects A6000 vs H100 indices when possible; falls back to defaults.
    try:
        tiers = detect_gpu_tiers()
        a6000_idx = tuple(t["index"] for t in tiers if t["is_ampere"])
        h100_idx_list = [t["index"] for t in tiers if t["is_hopper"]]
        if a6000_idx and h100_idx_list:
            recompute_config = build_neuron_sp_config(
                a6000_indices=a6000_idx, h100_index=h100_idx_list[0],
            )
        else:
            recompute_config = build_neuron_sp_config()
    except Exception as err:  # noqa: BLE001
        print(f"[bridge] GPU auto-detect failed ({err}); using default Neuron_SP config")
        recompute_config = build_neuron_sp_config()

    # 2) Hetero MIMO training loop with stage-specific LR.
    optimizer_config = PerModuleOptimizerConfig(lr=lr)
    loop = setup_hetero_mimo_training(
        model,
        optimizer_config=optimizer_config,
        cache_max_gb=cache_max_gb,
        grad_clip=grad_clip,
    )
    print(f"[bridge] HeteroMIMOTrainingLoop ready (lr={lr}, cache_max_gb={cache_max_gb})")
    return loop, recompute_config


def teardown_hetero_runtime(loop) -> None:
    """Tear down a HeteroMIMOTrainingLoop between training stages.

    Clears the shared locality cache, drops optimizer state references, and
    frees CUDA memory so the next stage's ``setup_hetero_mimo_training`` call
    starts from a clean slate (fresh optimizers + lr schedule, no stale cached
    activations from the previous stage).
    """
    if loop is None:
        return
    try:
        cache = getattr(loop, "locality_cache", None)
        if cache is not None and hasattr(cache, "clear"):
            cache.clear()
            print("[bridge] cleared SharedLocalityCache")
    except Exception as err:  # noqa: BLE001
        print(f"[bridge] locality_cache.clear() warning: {err}")

    # Drop per-module optimizers held by the router so their state is GC'd.
    try:
        router = getattr(loop, "optimizer_router", None)
        for attr in ("module_optimizers", "_module_optimizers", "optimizers"):
            if router is not None and hasattr(router, attr):
                store = getattr(router, attr)
                if hasattr(store, "clear"):
                    store.clear()
    except Exception as err:  # noqa: BLE001
        print(f"[bridge] optimizer router teardown warning: {err}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("[bridge] hetero runtime torn down")
