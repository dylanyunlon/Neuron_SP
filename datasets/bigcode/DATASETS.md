# BigCode Commit 数据集注册表

## 拉取方式
```bash
cd datasets/bigcode
bash pull_all_datasets.sh    # 在 ags1 上执行
```

## 四大数据集

### 1. CommitPackFT (`bigcode/commitpackft`)
- **规模**: ~2 GB
- **来源**: GPT-4 从 CommitPack 中筛选的高质量子集
- **格式**: `{old_contents, new_contents, subject, message, lang}`
- **用途**: 指令微调 (instruction tuning)，commit message 像自然语言指令
- **HF**: https://huggingface.co/datasets/bigcode/commitpackft
- **论文**: OctoPack (arXiv:2308.07124)

### 2. StarCoder Commits (`bigcode/starcoderdata`, data_dir="git-commits" / "git-commits-cleaned")
- **规模**: ~32 GB (git-commits, raw) / ~64 GB (git-commits-cleaned, HuggingFace 来源)
- **来源**: Google BigQuery 公开 GitHub 数据，单文件 commit
- **格式**: `{old_contents, new_contents, subject, message, repos, old_file, new_file, ...}`
- **处理**: 80% 样本只取变更行 ±32 行窗口，20% 保留全文件
- **过滤**: commit message 黑名单 + ≤2行变更 50% 丢弃 + 100K 字符上限
- **git-commits-cleaned split**: 在 git-commits 基础上追加 near-dedup（MinHash LSH）+ exact-dedup 过滤；
  保留更高质量样本，token 密度更高，推荐用于 DES-LOC 预训练主语料
- **HF**: https://huggingface.co/datasets/bigcode/starcoderdata
- **论文**: StarCoder (arXiv:2305.06161)

### 3. CommitPack (`bigcode/commitpack`)
- **规模**: ~4 TB
- **来源**: GHArchive 元数据 + GitHub API 逐个爬取代码变更
- **格式**: 同 CommitPackFT，但未经质量筛选
- **覆盖**: 截至 2016 年的 1.45 亿个唯一 commit
- **加载模式**: `streaming=True` 强制 — 4TB 语料不可全量落盘，每次拉取一个 Arrow shard (~128 MB)
- **DATASET_REGISTRY 条目**: `load_commits.DATASET_REGISTRY["commitpack"]`，`config_field="lang"` 按语言分片
- **HF**: https://huggingface.co/datasets/bigcode/commitpack
- **论文**: OctoPack (arXiv:2308.07124)

### 4. The Stack v2 (`bigcode/the-stack-v2`)
- **规模**: ~900B tokens (文件级)，PR/commit 子集未公开总量
- **来源**: Software Heritage + GHArchive
- **处理**: 只取 main/master 分支最新 revision，按 directory_id hash 去重
- **许可**: 需在 HuggingFace 上同意使用协议
- **HF**: https://huggingface.co/datasets/bigcode/the-stack-v2
- **论文**: StarCoder2 (arXiv:2402.19173)

## 在 DES-LOC 训练中的使用建议

| 阶段 | 推荐数据集 | 理由 |
|------|-----------|------|
| 预训练 (code) | StarCoder commits (32GB) | 规模适中，已过滤，直接可用 |
| 指令微调 | CommitPackFT (2GB) | 高质量，GPT-4 筛选，格式规整 |
| 大规模预训练 | CommitPack (4TB, streaming) | `DATASET_REGISTRY["commitpack"]` 强制 streaming=True，按语言分片迭代 |
| 全量预训练语料 | The Stack v2 | 需要 HF 协议，但覆盖最全 |
| 消融实验 | CommitPackFT Python subset | 小而精，快速迭代 |

---

## Stack v2 PR/commit 适配器 (M761-M775)

新增模块: `datasets/bigcode/the_stack_v2/`

| 文件 | 功能 |
|------|------|
| `stackv2_commits.py` | Stack v2 → DES-LOC 格式化、过滤、PII 清理 |
| `megatron_indexed.py` | 写出 Megatron `.bin`/`.idx` indexed dataset |

### 输出格式 (DES-LOC diff tokens)
```
<|diff_start|>
<|lang|>python
<|file_path|> src/foo.py
<|old|>
<old content>
<|new|>
<new content>
<|msg|> fix: handle edge case
<|diff_end|>
```

### 过滤规则
- 丢弃 merge commit (`Merge pull request / branch`)
- 丢弃 changed lines < 10 的无意义 diff
- 丢弃超过 100K 字符的样本
- `directory_id` hash 去重 (Stack v2 论文策略)
- PII 清除: email / IPv4 / hex secret / AWS key → `<REDACTED>`

### 使用示例
```python
from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter
from datasets.bigcode.the_stack_v2.megatron_indexed import MegatronIndexedWriter

adapter = StackV2CommitAdapter()

# HF Hub (需先 huggingface-cli login 并接受协议)
for sample in adapter.stream_hf(max_samples=10000):
    print(sample["text"][:200])

# 本地 parquet
for sample in adapter.stream_parquet("/data/stackv2_commits/*.parquet"):
    ...

# 写 Megatron indexed dataset
writer = MegatronIndexedWriter("/data/stackv2_megatron", tokenizer)
writer.write_from_adapter(adapter.stream_parquet("/data/*.parquet"))
writer.finalize()
# → /data/stackv2_megatron.bin  +  /data/stackv2_megatron.idx
```

### Smoke test
```bash
python datasets/bigcode/the_stack_v2/stackv2_commits.py --samples 5 --source dummy
python datasets/bigcode/the_stack_v2/megatron_indexed.py --dummy --output /tmp/test_sv2
```

---

## 完整数据集矩阵 (更新: Phase 8 / M1166-M1180)

| 数据集 | 规模 | 来源 | HuggingFace ID | 用途 |
|--------|------|------|---------------|------|
| StarCoder commits (raw) | 32 GB | BigQuery | `bigcode/starcoderdata` (data_dir="git-commits") | 预训练 code diff |
| StarCoder commits (cleaned) | 64 GB | BigQuery + near-dedup | `bigcode/starcoderdata` (data_dir="git-commits-cleaned") | 预训练主语料（推荐）|
| CommitPack | 4 TB | GHArchive + GitHub API 爬取 | `bigcode/commitpack` | 大规模预训练 (streaming=True, DATASET_REGISTRY 注册) |
| CommitPackFT | 2 GB (高质量子集) | GPT-4 筛选 | `bigcode/commitpackft` | 指令微调 |
| The Stack v2 (PR/commit) | 未公开总量 | GHArchive + Software Heritage | `bigcode/the-stack-v2` | 全量预训练语料 (M761-M775 适配完成) |

---

## DATASET_REGISTRY (M1166-M1180 新增)

`load_commits.py` 中新增 `DATASET_REGISTRY` 字典，统一管理所有数据集的加载配置，消除散落在各处的硬编码路径和 loader 逻辑。

### 注册条目

#### `commitpackft`
```python
DATASET_REGISTRY["commitpackft"] = {
    "hf_id":        "bigcode/commitpackft",
    "local_subdir": "commitpackft",
    "loader":       "jsonl",          # → load_commit_dataset()
    "langs": ["python", "javascript", "typescript", "java", "go", "rust",
              "c", "cpp", "ruby", "php", "swift", "kotlin", "scala", "r",
              "julia", "lua", "haskell", "perl", "shell", "powershell"],
    "description": "GPT-4-filtered high-quality commit instructions (~2 GB)",
    "paper":       "OctoPack (arXiv:2308.07124)",
    "size_hint":   "~2 GB",
    "requires_agreement": False,
}
```

#### `the_stack_v2`
```python
DATASET_REGISTRY["the_stack_v2"] = {
    "hf_id":        "bigcode/the-stack-v2",
    "local_subdir": "the_stack_v2",
    "loader":       "stackv2",        # → StackV2CommitAdapter
    "langs":        None,             # adapter handles lang normalisation
    "description":  "Stack v2 PR/commit subset; ~900 B tokens; dedup + PII",
    "paper":        "StarCoder2 (arXiv:2402.19173)",
    "size_hint":    "~900 B tokens (gated)",
    "requires_agreement": True,
}
```

### 使用方式
```python
from datasets.bigcode.load_commits import (
    DATASET_REGISTRY, get_registry_entry, load_commit_dataset, MixedCommitDataset
)

# 列出所有已注册数据集
for name, cfg in DATASET_REGISTRY.items():
    print(name, "→", cfg["description"])

# CommitPackFT (JSONL loader)
ds = load_commit_dataset("commitpackft", lang="python")

# The Stack v2 (registry 校验后交给 StackV2CommitAdapter)
# load_commit_dataset 会检测 loader="stackv2" 并抛出说明性 ValueError
from datasets.bigcode.the_stack_v2.stackv2_commits import StackV2CommitAdapter
entry = get_registry_entry("the-stack-v2")   # 接受连字符别名
adapter = StackV2CommitAdapter()
for sample in adapter.stream_hf(max_samples=1000):
    print(sample["text"][:200])
```

### The Stack v2 PR/commit 流式下载 (M1166-M1180 新增)

`pull_all_datasets.sh` 的第 4 节现在额外生成 `the_stack_v2/stream_pr_commits.py`，
支持将 PR/commit 子集按语言分片写出为 JSONL，与 `StackV2CommitAdapter` 直接集成：

```bash
# 需先 huggingface-cli login 并接受 bigcode/the-stack-v2 协议
python datasets/bigcode/the_stack_v2/stream_pr_commits.py \
    --max-samples 50000 \
    --out-dir /data/stackv2_commits
```
