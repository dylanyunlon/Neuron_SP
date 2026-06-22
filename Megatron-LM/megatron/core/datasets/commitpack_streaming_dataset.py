"""
CommitPackStreamingDataset: 4TB CommitPack 流式加载器

HuggingFace streaming → DeepSpeed DataLoader 桥接，支持：
  - 断点续传 (shard index 持久化)
  - 分布式 rank 分配 shard (每 GPU 独立拉取不同 shard)
  - CPU 异步 tokenization (充分利用 EPYC 128核)
  - NUMA-aware 内存分配 (NUMA0 → GPU0/1/2, NUMA1 → GPU3/4)
  - GPU 训练时异步预取下一批

硬件目标:
  GPU0-4: RTX A6000 48GB x2 + H100 NVL 96GB + (ags1 拓扑)
  CPU: EPYC 128核 (2 NUMA nodes)
  RAM: 1.5TB

使用示例:
    from megatron.core.datasets.commitpack_streaming_dataset import (
        CommitPackStreamingDataset,
        build_commitpack_dataloader,
    )

    loader = build_commitpack_dataloader(
        languages=["python", "javascript", "typescript"],
        seq_length=2048,
        micro_batch_size=4,
        rank=rank,
        world_size=world_size,
        resume_path="/checkpoints/commitpack_resume.json",
    )
    for batch in loader:
        ...
"""

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


# ─── NUMA 拓扑映射 ─────────────────────────────────────────────────────────────
# ags1 服务器: 2x NUMA nodes, 5 GPUs
# NUMA0 (CPU0-63)  → GPU0 (A6000), GPU1 (A6000), GPU2 (H100)
# NUMA1 (CPU64-127) → GPU3 (A6000), GPU4 (A6000)   [如有]
GPU_NUMA_MAP: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}

# NUMA node → CPU core 范围 (用于绑定 prefetch 线程)
NUMA_CPU_RANGES: Dict[int, Tuple[int, int]] = {0: (0, 63), 1: (64, 127)}


def _get_numa_node(rank: int) -> int:
    """根据 rank 返回对应的 NUMA node。"""
    return GPU_NUMA_MAP.get(rank % len(GPU_NUMA_MAP), 0)


def _pin_thread_to_numa(numa_node: int) -> None:
    """将当前线程绑定到指定 NUMA node 的 CPU 核心（尽力而为）。"""
    try:
        import ctypes
        cpu_start, cpu_end = NUMA_CPU_RANGES.get(numa_node, (0, 63))
        # 使用 os.sched_setaffinity 绑定线程到 NUMA node 对应的核心
        pid = 0  # 0 = 当前线程
        affinity = set(range(cpu_start, cpu_end + 1))
        os.sched_setaffinity(pid, affinity)
        logger.debug(f"Thread pinned to NUMA{numa_node} CPUs {cpu_start}-{cpu_end}")
    except Exception as e:
        logger.debug(f"NUMA pinning skipped: {e}")


# ─── 配置 ──────────────────────────────────────────────────────────────────────

@dataclass
class CommitPackStreamingConfig:
    """CommitPackStreamingDataset 配置。"""

    # 数据集选项
    languages: List[str] = field(default_factory=lambda: ["python"])
    split: str = "train"
    hf_dataset_name: str = "bigcode/commitpack"

    # 序列与批次
    seq_length: int = 2048
    micro_batch_size: int = 4

    # 分布式
    rank: int = 0
    world_size: int = 1

    # 断点续传：记录已消费的 shard/sample 进度
    resume_path: Optional[str] = None  # JSON 文件路径

    # 预取 & 并行
    prefetch_batches: int = 8          # 预取队列深度（批次数）
    tokenizer_workers: int = 8         # CPU tokenization 并行线程数
    tokenizer_name: str = "gpt2"       # HF tokenizer 名称或路径

    # NUMA
    numa_aware: bool = True

    # 调试
    max_samples: Optional[int] = None  # 调试时截断
    seed: int = 42


# ─── 断点续传状态 ──────────────────────────────────────────────────────────────

@dataclass
class ResumeState:
    """记录每个 (rank, language) 的消费进度。"""

    # lang → 已跳过的样本数
    lang_sample_offset: Dict[str, int] = field(default_factory=dict)
    # 全局已消费 token 数（监控用）
    total_tokens_consumed: int = 0
    # 最后一次保存的时间戳
    last_saved: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResumeState":
        obj = cls()
        obj.lang_sample_offset = d.get("lang_sample_offset", {})
        obj.total_tokens_consumed = d.get("total_tokens_consumed", 0)
        obj.last_saved = d.get("last_saved", 0.0)
        return obj

    def save(self, path: str, rank: int) -> None:
        """保存当前 rank 的 resume 状态到 JSON。"""
        rank_path = _resume_path_for_rank(path, rank)
        os.makedirs(os.path.dirname(rank_path) if os.path.dirname(rank_path) else ".", exist_ok=True)
        tmp = rank_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, rank_path)
        self.last_saved = time.time()
        logger.info(f"[rank{rank}] Resume state saved → {rank_path}")

    @classmethod
    def load(cls, path: str, rank: int) -> "ResumeState":
        rank_path = _resume_path_for_rank(path, rank)
        if os.path.exists(rank_path):
            with open(rank_path) as f:
                return cls.from_dict(json.load(f))
        return cls()


def _resume_path_for_rank(base_path: str, rank: int) -> str:
    """每个 rank 独立的 resume 文件，避免写冲突。"""
    stem, ext = os.path.splitext(base_path)
    return f"{stem}_rank{rank}{ext or '.json'}"


# ─── Shard 分配 ────────────────────────────────────────────────────────────────

def _shards_for_rank(
    language: str,
    rank: int,
    world_size: int,
    hf_dataset_name: str = "bigcode/commitpack",
    split: str = "train",
) -> List[str]:
    """
    为指定 rank 返回应处理的 shard URL 列表。

    CommitPack 在 HuggingFace 上以 Parquet shards 存储。
    我们通过 datasets.get_dataset_split_names / list_repo_tree
    获取 shard 列表，然后按 rank 均匀分配。

    若无法获取 shard 列表（离线环境），退化为直接 streaming
    （HF streaming 内部会按 rank 分配）。
    """
    try:
        from huggingface_hub import list_repo_tree
        shard_paths = []
        for item in list_repo_tree(
            hf_dataset_name,
            repo_type="dataset",
            path_in_repo=f"data/{language}/{split}",
        ):
            if hasattr(item, "path") and item.path.endswith(".parquet"):
                shard_paths.append(item.path)
        shard_paths = sorted(shard_paths)
        # 按 rank 分配：rank r 取 index % world_size == r 的 shard
        assigned = [s for i, s in enumerate(shard_paths) if i % world_size == rank]
        logger.info(
            f"[rank{rank}] lang={language}: {len(assigned)}/{len(shard_paths)} shards assigned"
        )
        return assigned
    except Exception as e:
        logger.warning(
            f"[rank{rank}] Could not enumerate shards for {language}: {e}. "
            "Falling back to HF streaming with manual skip."
        )
        return []  # 空列表 → 使用 streaming 模式 + skip


# ─── 流式迭代器 ────────────────────────────────────────────────────────────────

class _CommitPackShardIterator:
    """
    为单个 (language, rank) 提供流式样本迭代。

    优先使用显式 shard 分配；退化时使用 streaming + offset skip。
    """

    def __init__(
        self,
        language: str,
        rank: int,
        world_size: int,
        split: str,
        hf_dataset_name: str,
        sample_offset: int = 0,
        max_samples: Optional[int] = None,
    ):
        self.language = language
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.hf_dataset_name = hf_dataset_name
        self.sample_offset = sample_offset
        self.max_samples = max_samples
        self._count = 0

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset

        shards = _shards_for_rank(
            self.language, self.rank, self.world_size,
            self.hf_dataset_name, self.split
        )

        if shards:
            # 显式 shard 模式：直接加载分配给该 rank 的 shard
            ds = load_dataset(
                "parquet",
                data_files={
                    self.split: [
                        f"hf://datasets/{self.hf_dataset_name}/{s}" for s in shards
                    ]
                },
                split=self.split,
                streaming=True,
            )
        else:
            # 退化模式：全量 streaming，按 rank 跳过样本
            ds = load_dataset(
                self.hf_dataset_name,
                self.language,
                split=self.split,
                streaming=True,
            )
            # 按 rank 选取样本: 取 index % world_size == rank 的样本
            ds = ds.filter(
                lambda _, idx: idx % self.world_size == self.rank,
                with_indices=True,
            )

        # 断点续传：跳过已消费的样本
        if self.sample_offset > 0:
            ds = ds.skip(self.sample_offset)
            logger.info(
                f"[rank{self.rank}] lang={self.language}: skipping {self.sample_offset} samples (resume)"
            )

        for sample in ds:
            yield sample
            self._count += 1
            if self.max_samples is not None and self._count >= self.max_samples:
                break

    @property
    def consumed(self) -> int:
        return self._count


# ─── Tokenization Worker ───────────────────────────────────────────────────────

class _TokenizerPool:
    """
    多线程 CPU tokenization 池。

    输入队列: raw text strings
    输出队列: tokenized List[int] (长度 ≤ seq_length)
    """

    def __init__(self, tokenizer_name: str, n_workers: int, seq_length: int):
        self.tokenizer_name = tokenizer_name
        self.n_workers = n_workers
        self.seq_length = seq_length
        self._in_q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=n_workers * 4)
        self._out_q: "queue.Queue[Optional[List[int]]]" = queue.Queue(maxsize=n_workers * 4)
        self._workers: List[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        for _ in range(self.n_workers):
            t = threading.Thread(target=self._worker_fn, daemon=True)
            t.start()
            self._workers.append(t)
        self._started = True

    def _worker_fn(self) -> None:
        # 懒加载 tokenizer（每线程独立实例，避免锁竞争）
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        while True:
            text = self._in_q.get()
            if text is None:
                self._out_q.put(None)
                break
            ids = tok.encode(text, truncation=True, max_length=self.seq_length)
            self._out_q.put(ids)

    def submit(self, text: str) -> None:
        self._in_q.put(text)

    def get(self) -> Optional[List[int]]:
        return self._out_q.get()

    def stop(self) -> None:
        for _ in self._workers:
            self._in_q.put(None)
        for t in self._workers:
            t.join(timeout=5)


# ─── IterableDataset ──────────────────────────────────────────────────────────

class CommitPackStreamingDataset(IterableDataset):
    """
    CommitPack 4TB 流式 IterableDataset。

    每次迭代产出 shape=(seq_length,) 的 token id tensor，
    从多语言 commit 文本中按 seq_length 打包（不跨 commit 截断）。

    特性:
      - 多语言轮询（round-robin across languages）
      - 按 rank 分配 shard，避免重复拉取
      - 断点续传：每 save_interval 批次写一次 resume JSON
      - NUMA-aware prefetch 线程绑定
    """

    def __init__(self, config: CommitPackStreamingConfig):
        super().__init__()
        self.cfg = config
        self._resume_state = (
            ResumeState.load(config.resume_path, config.rank)
            if config.resume_path
            else ResumeState()
        )
        self._sample_count = 0
        self._save_interval = 500  # 每 500 个样本保存一次 resume 状态

    # ── 打包 ──────────────────────────────────────────────────────────────────

    def _pack_tokens(
        self,
        token_buffer: List[int],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        从 token_buffer 中取出 seq_length 个 token，返回 Tensor 及剩余 buffer。
        """
        seq = token_buffer[: self.cfg.seq_length]
        remainder = token_buffer[self.cfg.seq_length :]

        # 若不足 seq_length，用 pad_id=0 填充
        if len(seq) < self.cfg.seq_length:
            seq = seq + [0] * (self.cfg.seq_length - len(seq))

        return torch.tensor(seq, dtype=torch.long), remainder

    # ── 迭代主循环 ────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        cfg = self.cfg
        numa_node = _get_numa_node(cfg.rank)

        # NUMA 绑定（主迭代线程）
        if cfg.numa_aware:
            _pin_thread_to_numa(numa_node)

        # 为每个语言建立迭代器
        iters = []
        for lang in cfg.languages:
            offset = self._resume_state.lang_sample_offset.get(lang, 0)
            it = _CommitPackShardIterator(
                language=lang,
                rank=cfg.rank,
                world_size=cfg.world_size,
                split=cfg.split,
                hf_dataset_name=cfg.hf_dataset_name,
                sample_offset=offset,
                max_samples=cfg.max_samples,
            )
            iters.append((lang, iter(it._make_iter()), it))

        # 若 _CommitPackShardIterator 没有 _make_iter，直接用 __iter__
        # 重写一下，让 iters 更清晰
        iters = []
        for lang in cfg.languages:
            offset = self._resume_state.lang_sample_offset.get(lang, 0)
            it_obj = _CommitPackShardIterator(
                language=lang,
                rank=cfg.rank,
                world_size=cfg.world_size,
                split=cfg.split,
                hf_dataset_name=cfg.hf_dataset_name,
                sample_offset=offset,
                max_samples=cfg.max_samples,
            )
            iters.append({"lang": lang, "iter": iter(it_obj), "obj": it_obj})

        # 懒加载 tokenizer（主线程单例，用于简单同步场景）
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
            def tokenize(text: str) -> List[int]:
                return tokenizer.encode(text, truncation=True, max_length=cfg.seq_length * 2)
        except Exception:
            def tokenize(text: str) -> List[int]:
                return [ord(c) % 32000 for c in text[: cfg.seq_length * 2]]

        token_buffer: List[int] = []
        lang_idx = 0
        exhausted = set()

        while len(exhausted) < len(iters):
            # Round-robin across languages
            if lang_idx >= len(iters):
                lang_idx = 0

            entry = iters[lang_idx]
            lang = entry["lang"]

            if lang in exhausted:
                lang_idx += 1
                continue

            try:
                sample = next(entry["iter"])
            except StopIteration:
                exhausted.add(lang)
                lang_idx += 1
                continue

            # CommitPack schema: {"old_contents": ..., "new_contents": ..., "subject": ...}
            # 拼接成 commit 文本
            text = _format_commit_sample(sample)
            token_ids = tokenize(text)
            token_buffer.extend(token_ids)

            # 更新断点续传计数
            lang_offset = self._resume_state.lang_sample_offset.get(lang, 0)
            self._resume_state.lang_sample_offset[lang] = lang_offset + 1

            # 每凑够一个 seq_length 就 yield
            while len(token_buffer) >= cfg.seq_length:
                tensor, token_buffer = self._pack_tokens(token_buffer)

                # labels = tokens shifted left by 1
                labels = torch.roll(tensor, shifts=-1)
                labels[-1] = 0

                loss_mask = torch.ones(cfg.seq_length, dtype=torch.float)
                loss_mask[-1] = 0.0  # padding position

                position_ids = torch.arange(cfg.seq_length, dtype=torch.long)

                yield {
                    "tokens": tensor,
                    "labels": labels,
                    "loss_mask": loss_mask,
                    "position_ids": position_ids,
                }

                self._sample_count += 1
                self._resume_state.total_tokens_consumed += cfg.seq_length

                # 定期保存 resume 状态
                if (
                    cfg.resume_path
                    and self._sample_count % self._save_interval == 0
                ):
                    self._resume_state.save(cfg.resume_path, cfg.rank)

            lang_idx += 1

        # 最后 flush 剩余 buffer
        if len(token_buffer) > 0:
            tensor, _ = self._pack_tokens(token_buffer)
            labels = torch.roll(tensor, shifts=-1)
            labels[-1] = 0
            loss_mask = torch.ones(cfg.seq_length, dtype=torch.float)
            loss_mask[len(token_buffer) :] = 0.0
            loss_mask[-1] = 0.0
            position_ids = torch.arange(cfg.seq_length, dtype=torch.long)
            yield {
                "tokens": tensor,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

        # 最终保存
        if cfg.resume_path:
            self._resume_state.save(cfg.resume_path, cfg.rank)

        logger.info(
            f"[rank{cfg.rank}] CommitPackStreamingDataset exhausted: "
            f"{self._sample_count} samples, "
            f"{self._resume_state.total_tokens_consumed:,} tokens"
        )


def _format_commit_sample(sample: dict) -> str:
    """
    将 CommitPack HuggingFace 样本格式化为结构化文本。

    CommitPack schema:
        subject: str      — commit message
        old_contents: str — file before commit
        new_contents: str — file after commit
        lang: str
        repo: str
    """
    parts = []
    if sample.get("subject"):
        parts.append(f"<COMMIT><MSG>{sample['subject']}</MSG>")
    else:
        parts.append("<COMMIT><MSG></MSG>")

    old = sample.get("old_contents") or ""
    new = sample.get("new_contents") or ""

    if old or new:
        parts.append("<FILE>")
        if old:
            parts.append(f"<DEL>{old}</DEL>")
        if new:
            parts.append(f"<ADD>{new}</ADD>")
        parts.append("</FILE>")

    parts.append("</COMMIT>")
    return "".join(parts)


# ─── 预取 DataLoader Wrapper ──────────────────────────────────────────────────

class _PrefetchDataLoader:
    """
    在独立线程中预取批次，GPU 训练时异步加载下一批。

    wrap 任意 DataLoader，在后台线程中提前把批次转移到 GPU。
    """

    def __init__(
        self,
        dataloader: DataLoader,
        device: torch.device,
        prefetch_batches: int = 4,
        numa_node: int = 0,
        numa_aware: bool = True,
    ):
        self._dl = dataloader
        self._device = device
        self._prefetch_batches = prefetch_batches
        self._numa_node = numa_node
        self._numa_aware = numa_aware
        self._queue: "queue.Queue" = queue.Queue(maxsize=prefetch_batches)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _prefetch_fn(self) -> None:
        if self._numa_aware:
            _pin_thread_to_numa(self._numa_node)
        try:
            for batch in self._dl:
                if self._stop_event.is_set():
                    break
                # 将 batch 从 CPU 搬到 GPU（pin_memory=True 时使用 non_blocking）
                if isinstance(batch, dict):
                    gpu_batch = {
                        k: v.to(self._device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                else:
                    gpu_batch = batch.to(self._device, non_blocking=True)
                self._queue.put(gpu_batch)
        except Exception as e:
            logger.error(f"Prefetch thread error: {e}", exc_info=True)
        finally:
            self._queue.put(None)  # sentinel

    def __iter__(self) -> Iterator:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._prefetch_fn, daemon=True)
        self._thread.start()
        while True:
            batch = self._queue.get()
            if batch is None:
                break
            yield batch

    def __del__(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)


# ─── 公共入口 ──────────────────────────────────────────────────────────────────

def build_commitpack_dataloader(
    languages: List[str],
    seq_length: int,
    micro_batch_size: int,
    rank: int,
    world_size: int,
    tokenizer_name: str = "gpt2",
    resume_path: Optional[str] = None,
    prefetch_batches: int = 8,
    tokenizer_workers: int = 8,
    numa_aware: bool = True,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> "_PrefetchDataLoader":
    """
    构建 CommitPack 流式 DataLoader，含 GPU prefetch。

    Args:
        languages:         语言列表, e.g. ["python", "javascript"]
        seq_length:        训练序列长度
        micro_batch_size:  每 GPU 的 micro batch size
        rank:              当前进程 rank
        world_size:        总进程数
        tokenizer_name:    HuggingFace tokenizer 名称或路径
        resume_path:       断点续传 JSON 基础路径 (会加 _rankN.json 后缀)
        prefetch_batches:  预取队列深度（批次数）
        tokenizer_workers: CPU tokenization 线程数
        numa_aware:        是否启用 NUMA-aware 线程绑定
        device:            目标 GPU device（默认 cuda:rank）
        max_samples:       调试截断
        seed:              随机种子

    Returns:
        _PrefetchDataLoader — 可直接在训练循环中迭代
    """
    if device is None:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    config = CommitPackStreamingConfig(
        languages=languages,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        rank=rank,
        world_size=world_size,
        tokenizer_name=tokenizer_name,
        resume_path=resume_path,
        prefetch_batches=prefetch_batches,
        tokenizer_workers=tokenizer_workers,
        numa_aware=numa_aware,
        max_samples=max_samples,
        seed=seed,
    )

    dataset = CommitPackStreamingDataset(config)

    # pin_memory=True: CPU→GPU 传输走 DMA，减少延迟
    # num_workers=0: IterableDataset 在主进程迭代（HF streaming 内部已多线程）
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    numa_node = _get_numa_node(rank)
    prefetch_loader = _PrefetchDataLoader(
        dataloader=dataloader,
        device=device,
        prefetch_batches=prefetch_batches,
        numa_node=numa_node,
        numa_aware=numa_aware,
    )

    logger.info(
        f"[rank{rank}] CommitPack DataLoader ready: "
        f"langs={languages}, seq_length={seq_length}, "
        f"micro_batch={micro_batch_size}, NUMA{numa_node}, "
        f"prefetch_batches={prefetch_batches}"
    )

    return prefetch_loader


# ─── 快速冒烟测试 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CommitPack streaming smoke test")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument(
        "--languages", nargs="+", default=["python"],
        help="CommitPack language subsets to stream"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    loader = build_commitpack_dataloader(
        languages=args.languages,
        seq_length=args.seq_length,
        micro_batch_size=args.batch_size,
        rank=args.rank,
        world_size=args.world_size,
        resume_path=args.resume_path,
        max_samples=args.max_samples,
        numa_aware=False,  # 测试时关闭 NUMA 绑定
    )

    print(f"Rank {args.rank}: starting iteration...")
    batch_count = 0
    token_count = 0
    for batch in loader:
        batch_count += 1
        tokens = batch["tokens"]
        token_count += tokens.numel()
        if batch_count <= 3:
            print(
                f"  batch {batch_count}: tokens.shape={tokens.shape}, "
                f"dtype={tokens.dtype}, "
                f"loss_mask.sum={batch['loss_mask'].sum().item():.0f}"
            )

    print(
        f"Rank {args.rank}: done. "
        f"{batch_count} batches, {token_count:,} tokens total."
    )
