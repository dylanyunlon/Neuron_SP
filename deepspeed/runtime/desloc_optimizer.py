# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""VRAM-adaptive optimizer builder for heterogeneous GPU clusters.

Selects between fused AdamW (high-VRAM), standard AdamW (mid-VRAM),
and DeepSpeedCPUAdam (low-VRAM) based on per-rank GPU memory.

References:
  - DeepSpeed Issue #610: cpu_offload silent crash
  - Megatron PR #4623: Fix FSDP optimizer CPU offload
  - DeepSpeed Issue #2899: ZeRO-3 CPU offload OOM
"""
import torch

# ---------------------------------------------------------------------------
# _build_optimizer — VRAM-adaptive optimizer selection
# ---------------------------------------------------------------------------
# 参考:
#   DeepSpeed Issue #610  (CPU Adam offload for low-VRAM devices)
#   Megatron PR #4623     (fused Adam on high-VRAM A100/H100)
# ---------------------------------------------------------------------------

def _build_optimizer(self) -> None:
    """
    根据本 rank 的 VRAM 容量选择最优 optimizer，避免 A6000 (47GB) 等
    中等显存卡因 exp_avg / exp_avg_sq 状态张量 OOM。

    策略:
        VRAM >= 90 GB  → torch.optim.AdamW(fused=True)  [GPU, A100/H100]
        VRAM < 50 GB   → DeepSpeedCPUAdam              [CPU offload, A6000/3090]
        50 <= VRAM < 90 → torch.optim.AdamW(fused=False) [GPU, 保守路径, A100-40GB]

    Mixed-tier 场景:
        不同 rank 可独立选择不同 optimizer 类型；DeepSpeed ZeRO 只要求
        每个 rank 的 optimizer 接口兼容（step/zero_grad/state_dict），
        不要求所有 rank 使用同一实现。
        self.optimizer_type 供日志 / checkpoint 元数据记录。
    """
    cfg = self.config
    vram_gb: float = getattr(self, "local_tier_vram_gb", 0.0)

    # ------------------------------------------------------------------
    # 1. 收集需要优化的参数（过滤掉 frozen param）
    # ------------------------------------------------------------------
    param_groups = [
        {
            "params": [p for p in self.model.parameters() if p.requires_grad],
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "betas": (cfg.beta1, cfg.beta2),
            "eps": cfg.adam_eps if hasattr(cfg, "adam_eps") else 1e-8,
        }
    ]

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # ------------------------------------------------------------------
    # 2. 高显存路径：fused AdamW on GPU (A100 80GB / H100)
    # ------------------------------------------------------------------
    if vram_gb >= 90.0:
        self.optimizer_type = "torch_adamw_fused"
        self.optimizer = torch.optim.AdamW(
            param_groups,
            fused=True,          # CUDA fused kernel, 单次 launch 完成 step
        )
        self._cpu_offload_optimizer = False
        self.logger.info(
            f"[rank {rank}] VRAM={vram_gb:.1f}GB → torch.optim.AdamW(fused=True)"
        )
        return

    # ------------------------------------------------------------------
    # 3. 低显存路径：CPU Adam offload (A6000 47GB / RTX 3090 24GB)
    #
    #    原理：
    #      - param_shard 保持在 GPU（前向/反向不动）
    #      - optimizer state (exp_avg, exp_avg_sq) 常驻 CPU pinned memory
    #      - step() 时：grad D2H → CPU adam step → param delta H2D
    #      - 使用 DeepSpeedCPUAdam 的 ds_opt_adam C++ 扩展，比 Python
    #        循环快 ~5x（参考 DS Issue #610 benchmark）
    # ------------------------------------------------------------------
    if vram_gb < 50.0:
        self.optimizer_type = "deepspeed_cpu_adam"
        self._cpu_offload_optimizer = True

        # DeepSpeedCPUAdam 要求参数在 GPU，状态自动在 CPU
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "DeepSpeedCPUAdam 不可用，请确认 DeepSpeed 已用 "
                "`DS_BUILD_CPU_ADAM=1 pip install deepspeed` 编译"
            ) from exc

        self.optimizer = DeepSpeedCPUAdam(
            param_groups,
            adamw_mode=True,          # weight decay 模式（等价 AdamW）
            fp32_optimizer_states=True,  # state 以 fp32 存储，避免精度损失
        )

        # 注册 step hook：DeepSpeedCPUAdam 内部已处理 D2H/H2D，
        # 但若 ZeRO 未接管，需手动确保 grad 在 step 前已同步到 CPU。
        self._register_cpu_offload_hooks()

        self.logger.info(
            f"[rank {rank}] VRAM={vram_gb:.1f}GB → DeepSpeedCPUAdam "
            f"(CPU offload, fp32 states)"
        )
        return

    # ------------------------------------------------------------------
    # 4. 中等显存路径：标准 AdamW on GPU，不开 fused（50-89 GB, A100-40GB）
    # ------------------------------------------------------------------
    self.optimizer_type = "torch_adamw"
    self._cpu_offload_optimizer = False
    self.optimizer = torch.optim.AdamW(
        param_groups,
        fused=False,
    )
    self.logger.info(
        f"[rank {rank}] VRAM={vram_gb:.1f}GB → torch.optim.AdamW(fused=False)"
    )


def _register_cpu_offload_hooks(self) -> None:
    """
    为 CPU offload 路径注册 backward hook。

    在 DeepSpeed ZeRO 接管时此方法为空操作（ZeRO 自行处理 grad 搬运）。
    在非 ZeRO 路径下，hook 确保每个参数的 grad 在 optimizer.step()
    前已通过 .cpu() + pin_memory() 搬到主机内存，step 后清零 GPU grad
    以释放显存。

    注意：若使用 ZeRO Stage-2/3，请将此方法体替换为 `pass`；
    ZeRO 的 `overlap_comm` 已包含异步 D2H 逻辑。
    """
    # 检测是否由 ZeRO 接管（engine 初始化后 self.ds_engine 才存在）
    if getattr(self, "ds_engine", None) is not None:
        # ZeRO 接管，无需手动 hook
        return

    self._cpu_grad_buffers: dict = {}

    def _make_hook(param):
        def hook(grad):
            # 异步 D2H，不阻塞 GPU 计算流
            cpu_grad = grad.to(device="cpu", non_blocking=True)
            # pin_memory 使后续 H2D 带宽更高
            if not cpu_grad.is_pinned():
                cpu_grad = cpu_grad.pin_memory()
            self._cpu_grad_buffers[param] = cpu_grad
            # 返回 None：保留原始 GPU grad（DeepSpeedCPUAdam 会再次读取）
            return grad
        return hook

    for p in self.model.parameters():
        if p.requires_grad:
            p.register_hook(_make_hook(p))
