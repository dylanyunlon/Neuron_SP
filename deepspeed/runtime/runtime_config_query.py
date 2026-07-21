"""
runtime_config_query.py — Ask claude-hk-config for training parameters at runtime.

Instead of hardcoding memory budgets and heuristics for every GPU combination,
this module:
  1. Collects the actual runtime environment (GPU specs, free VRAM, model size, etc.)
  2. Sends it as a structured prompt to a sub-Claude via claude_hk_chat.sh
  3. Parses the JSON response into concrete numbers
  4. Returns a dict that the caller uses to override TrainingConfig fields

If the query fails (no network, rate limit, timeout), returns None and the
caller falls back to existing defaults. Training never blocks on this.

Usage:
    from deepspeed.runtime.runtime_config_query import query_runtime_config

    overrides = query_runtime_config(
        model_params=5_147_000_000,
        num_layers=32,
        hidden_size=4096,
        seq_len=2048,
        vocab_size=32000,
    )
    if overrides:
        tc.micro_batch_size = overrides["micro_batch_size"]
        tc.grad_accum_steps = overrides["grad_accum_steps"]
        # ... etc
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("runtime_config_query")


# ---------------------------------------------------------------------------
# Step 1: Collect environment factors
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Per-GPU hardware snapshot."""
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    used_vram_gb: float
    sm_version: int          # e.g. 86, 90, 120
    pcie_gen: int            # 4 or 5
    compute_capability: str  # e.g. "8.6"


@dataclass
class EnvironmentSnapshot:
    """Everything the decision-maker needs to know."""
    # Hardware
    gpus: List[GPUInfo]
    cpu_cores: int
    cpu_ram_gb: float
    numa_nodes: int

    # Model
    model_params: int        # total parameter count
    num_layers: int
    hidden_size: int
    num_heads: int
    seq_len: int
    vocab_size: int
    dtype: str               # "bf16" or "fp32"

    # Training intent
    target_batch_size: int
    target_steps: int
    gradient_checkpointing: bool

    # Software
    torch_version: str
    cuda_version: str
    nccl_p2p_disabled: bool


def collect_environment(
    model_params: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int = 32,
    seq_len: int = 2048,
    vocab_size: int = 32000,
    dtype: str = "bf16",
    target_batch_size: int = 1,
    target_steps: int = 100_000,
    gradient_checkpointing: bool = True,
) -> EnvironmentSnapshot:
    """Collect a full snapshot of the current runtime environment."""
    import torch

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_mem / (1 << 30)
            # Get current free/used memory
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1 << 30)
            used_gb = total_gb - free_gb
            sm = props.major * 10 + props.minor

            # Infer PCIe gen from SM version (heuristic)
            pcie_gen = 5 if sm >= 90 else 4

            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_vram_gb=round(total_gb, 1),
                free_vram_gb=round(free_gb, 1),
                used_vram_gb=round(used_gb, 1),
                sm_version=sm,
                pcie_gen=pcie_gen,
                compute_capability=f"{props.major}.{props.minor}",
            ))

    # CPU info
    cpu_cores = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    cpu_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                cpu_ram_gb = 0
    except Exception:
        cpu_ram_gb = 0

    # NUMA nodes
    try:
        numa_dirs = list(Path("/sys/devices/system/node/").glob("node*"))
        numa_nodes = len(numa_dirs) if numa_dirs else 1
    except Exception:
        numa_nodes = 1

    # Software versions
    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "unknown"
    nccl_p2p = os.environ.get("NCCL_P2P_DISABLE", "0") == "1"

    return EnvironmentSnapshot(
        gpus=gpus,
        cpu_cores=cpu_cores,
        cpu_ram_gb=round(cpu_ram_gb, 1),
        numa_nodes=numa_nodes,
        model_params=model_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dtype=dtype,
        target_batch_size=target_batch_size,
        target_steps=target_steps,
        gradient_checkpointing=gradient_checkpointing,
        torch_version=torch_version,
        cuda_version=cuda_version,
        nccl_p2p_disabled=nccl_p2p,
    )


# ---------------------------------------------------------------------------
# Step 2: Build the prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
你是 Neuron_SP 训练系统的运行时配置顾问。根据下面的硬件环境和模型参数，\
返回一组具体的训练参数，使得所有 GPU 都不会 OOM，同时吞吐量最大化。

## 硬件环境
{gpu_table}
CPU: {cpu_cores} cores, {cpu_ram_gb} GB RAM, {numa_nodes} NUMA nodes
NCCL P2P disabled: {nccl_p2p}
PyTorch: {torch_version}, CUDA: {cuda_version}

## 模型参数
- 总参数量: {model_params:,} ({model_params_b:.2f}B)
- 层数: {num_layers}, hidden: {hidden_size}, heads: {num_heads}
- 词表: {vocab_size}, 序列长度: {seq_len}
- dtype: {dtype}
- gradient_checkpointing: {gradient_checkpointing}

## 约束条件
1. ZeRO-3 异构分片：按 GPU 可用 VRAM 比例分配参数 shard
2. 每个 GPU 上，param_shard(FP32) + param_shard.grad(FP32) + 激活值 + 梯度临时缓冲 + clip_grad_norm 的 FP32 转换 必须 < 可用 VRAM
3. A6000 (48GB) 上 Adam m1/v1 必须 offload 到 CPU
4. clip_grad_norm 不能一次性把所有梯度转 FP32，必须逐参数累加（已修复）
5. 留至少 500MB headroom 给 CUDA allocator 碎片

## 请返回严格的 JSON（不要 markdown 代码块，不要解释）:
{{
  "micro_batch_size_per_gpu": [每个GPU的micro batch size, 按GPU index顺序],
  "grad_accum_steps": 整数,
  "activation_checkpointing": true/false,
  "checkpoint_activations_granularity": "full" 或 "selective",
  "use_activation_offload": true/false,
  "empty_unused_memory_level": 0/1/2,
  "shard_weights": [每个GPU的ZeRO-3 shard权重, 按可用VRAM比例],
  "cpu_offload_optimizer": [每个GPU是否CPU offload optimizer, true/false],
  "max_lr": float,
  "grad_clip": float
}}
"""


def _build_prompt(env: EnvironmentSnapshot) -> str:
    """Build the prompt string from an environment snapshot."""
    gpu_lines = []
    for g in env.gpus:
        gpu_lines.append(
            f"  GPU{g.index}: {g.name}  SM{g.compute_capability}  "
            f"total={g.total_vram_gb}GB  free={g.free_vram_gb}GB  "
            f"used={g.used_vram_gb}GB  PCIe_gen={g.pcie_gen}"
        )
    gpu_table = "\n".join(gpu_lines) if gpu_lines else "  (no GPUs detected)"

    return _PROMPT_TEMPLATE.format(
        gpu_table=gpu_table,
        cpu_cores=env.cpu_cores,
        cpu_ram_gb=env.cpu_ram_gb,
        numa_nodes=env.numa_nodes,
        nccl_p2p=env.nccl_p2p_disabled,
        torch_version=env.torch_version,
        cuda_version=env.cuda_version,
        model_params=env.model_params,
        model_params_b=env.model_params / 1e9,
        num_layers=env.num_layers,
        hidden_size=env.hidden_size,
        num_heads=env.num_heads,
        vocab_size=env.vocab_size,
        seq_len=env.seq_len,
        dtype=env.dtype,
        gradient_checkpointing=env.gradient_checkpointing,
    )


# ---------------------------------------------------------------------------
# Step 3: Call claude_hk_chat.sh and parse the response
# ---------------------------------------------------------------------------

def _find_chat_script() -> Optional[str]:
    """Locate claude_hk_chat.sh relative to the project root."""
    # Try project root (where run_pretrain.py lives)
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "claude_hk_chat.sh"),
        os.path.join(os.path.dirname(__file__), "..", "..", ".claude-hk-config", "dispatch.sh"),
    ]
    for c in candidates:
        p = os.path.realpath(c)
        if os.path.isfile(p):
            return p
    return None


def _call_claude_hk(prompt: str, timeout: int = 120, max_retries: int = 5) -> Optional[str]:
    """Send a prompt to claude_hk_chat.sh and return the raw response text.

    Handles the claude.hk.cn rate limit (HTTP 429 / permission_error /
    "频率过快" / "reached the limit") by retrying with exponential backoff.
    The backoff schedule is 10s, 20s, 40s, 60s, 90s — matching the typical
    rate-limit window recovery on shared-cookie deployments.

    A fresh conversation is used on each retry (claude_hk_chat.sh creates a
    new conv_id per invocation), which avoids the 404 stale-conversation trap
    that the proxy_v3.py retry loop already handles.
    """
    script = _find_chat_script()
    if not script:
        logger.warning("[runtime_config] claude_hk_chat.sh not found; skipping query")
        return None

    _RATE_LIMIT_MARKERS = (
        "permission_error",
        "频率过快",
        "reached the limit",
        "提问频率过快",
        '"type":"error"',
    )

    _BACKOFF_SCHEDULE = [10, 20, 40, 60, 90]  # seconds

    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["bash", script, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(script),
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "[runtime_config] attempt %d/%d timed out (%ds)",
                attempt + 1, max_retries, timeout,
            )
            # Timeout is not a rate limit — don't retry, just give up
            return None
        except Exception as e:
            logger.warning("[runtime_config] attempt %d/%d error: %s", attempt + 1, max_retries, e)
            return None

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + stderr

        # Check if this is a rate-limit / permission error
        is_rate_limited = any(marker in combined for marker in _RATE_LIMIT_MARKERS)

        if is_rate_limited:
            backoff = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
            logger.warning(
                "[runtime_config] 429/rate-limit on attempt %d/%d — "
                "backoff %ds before retry (new conversation)",
                attempt + 1, max_retries, backoff,
            )
            time.sleep(backoff)
            continue

        # Non-rate-limit failure
        if result.returncode != 0:
            logger.warning(
                "[runtime_config] claude_hk_chat.sh failed (rc=%d): %s",
                result.returncode, stderr[:300],
            )
            return None

        # Success
        if attempt > 0:
            logger.info(
                "[runtime_config] succeeded on attempt %d/%d",
                attempt + 1, max_retries,
            )
        return stdout

    # All retries exhausted
    logger.warning(
        "[runtime_config] all %d retries exhausted (rate-limited); using defaults",
        max_retries,
    )
    return None


def _parse_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from the claude_hk_chat.sh output.

    The output contains conversation metadata lines before/after the actual
    JSON response. We scan for the first '{' ... last '}' pair.

    Rejects error responses from claude.hk.cn that look like:
        {"error":{"message":"频率过快...","type":"permission_error"},"type":"error"}
    """
    if not raw:
        return None

    # Find the JSON object in the output
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("[runtime_config] no JSON found in response (%d chars)", len(raw))
        return None

    json_str = raw[start:end + 1]
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            logger.warning("[runtime_config] response is not a dict: %s", type(data))
            return None

        # Reject claude.hk.cn error envelopes
        if data.get("type") == "error" or "error" in data:
            err_msg = ""
            err_obj = data.get("error", {})
            if isinstance(err_obj, dict):
                err_msg = err_obj.get("message", str(err_obj))
            logger.warning("[runtime_config] response is an error object: %s", err_msg[:200])
            return None

        return data
    except json.JSONDecodeError as e:
        logger.warning("[runtime_config] JSON parse error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Step 4: Validate and return
# ---------------------------------------------------------------------------

# Fields we accept from the response and their expected types
_EXPECTED_FIELDS = {
    "micro_batch_size_per_gpu": list,
    "grad_accum_steps": int,
    "activation_checkpointing": bool,
    "checkpoint_activations_granularity": str,
    "use_activation_offload": bool,
    "empty_unused_memory_level": int,
    "shard_weights": list,
    "cpu_offload_optimizer": list,
    "max_lr": (int, float),
    "grad_clip": (int, float),
}


def _validate(data: Dict[str, Any], num_gpus: int) -> Dict[str, Any]:
    """Validate and sanitize the response. Drop invalid fields, keep valid ones."""
    result = {}
    for key, expected_type in _EXPECTED_FIELDS.items():
        if key not in data:
            continue
        val = data[key]
        if not isinstance(val, expected_type):
            logger.warning("[runtime_config] field '%s' has wrong type %s, skipping", key, type(val))
            continue

        # List fields must match GPU count
        if isinstance(val, list) and key in ("micro_batch_size_per_gpu", "shard_weights", "cpu_offload_optimizer"):
            if len(val) != num_gpus:
                logger.warning(
                    "[runtime_config] field '%s' has %d elements but %d GPUs, skipping",
                    key, len(val), num_gpus,
                )
                continue

        # Sanity bounds
        if key == "grad_accum_steps" and (val < 1 or val > 256):
            continue
        if key == "empty_unused_memory_level" and val not in (0, 1, 2):
            continue
        if key == "max_lr" and (val <= 0 or val > 1.0):
            continue
        if key == "grad_clip" and (val <= 0 or val > 100.0):
            continue
        if key == "micro_batch_size_per_gpu" and any(x < 1 for x in val):
            continue

        result[key] = val

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_runtime_config(
    model_params: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int = 32,
    seq_len: int = 2048,
    vocab_size: int = 32000,
    dtype: str = "bf16",
    target_batch_size: int = 1,
    target_steps: int = 100_000,
    gradient_checkpointing: bool = True,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """Query claude-hk-config for runtime training parameters.

    Returns a dict of validated overrides, or None if the query fails.
    The caller should apply the overrides to TrainingConfig fields.

    This function is safe to call from any rank, but should only be called
    from rank 0 to avoid redundant network calls. The result can be
    broadcast to other ranks via torch.distributed.

    Example:
        overrides = query_runtime_config(model_params=5_147_000_000, ...)
        if overrides:
            for key, val in overrides.items():
                setattr(training_config, key, val)
    """
    # Only rank 0 does the query
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return None

    # Check kill switch
    if os.environ.get("NEURON_SP_NO_RUNTIME_QUERY", "0") == "1":
        logger.info("[runtime_config] disabled via NEURON_SP_NO_RUNTIME_QUERY=1")
        return None

    logger.info("[runtime_config] collecting environment snapshot...")
    t0 = time.time()

    env = collect_environment(
        model_params=model_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_len=seq_len,
        vocab_size=vocab_size,
        dtype=dtype,
        target_batch_size=target_batch_size,
        target_steps=target_steps,
        gradient_checkpointing=gradient_checkpointing,
    )

    prompt = _build_prompt(env)
    logger.info("[runtime_config] querying claude-hk-config (%d chars, timeout=%ds)...", len(prompt), timeout)

    raw = _call_claude_hk(prompt, timeout=timeout)
    if raw is None:
        logger.info("[runtime_config] query returned None; using defaults")
        return None

    data = _parse_response(raw)
    if data is None:
        logger.info("[runtime_config] no valid JSON in response; using defaults")
        return None

    result = _validate(data, num_gpus=len(env.gpus))
    elapsed = time.time() - t0

    if result:
        logger.info("[runtime_config] got %d overrides in %.1fs: %s", len(result), elapsed, result)
    else:
        logger.info("[runtime_config] response had no valid fields; using defaults")
        return None

    return result


def apply_overrides(config: Any, overrides: Dict[str, Any]) -> None:
    """Apply runtime overrides to a TrainingConfig-like object.

    Only sets attributes that exist on the config or are in _EXPECTED_FIELDS.
    Logs each override for traceability.
    """
    for key, val in overrides.items():
        old = getattr(config, key, "<unset>")
        setattr(config, key, val)
        logger.info("[runtime_config] override: %s = %s (was %s)", key, val, old)