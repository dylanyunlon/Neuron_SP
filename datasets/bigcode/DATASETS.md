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

### 2. StarCoder Commits (`bigcode/starcoderdata`, data_dir="git-commits")
- **规模**: ~32 GB (starcoderdata 总量 783 GB 的子集)
- **来源**: Google BigQuery 公开 GitHub 数据，单文件 commit
- **格式**: `{old_contents, new_contents, subject, message, repos, old_file, new_file, ...}`
- **处理**: 80% 样本只取变更行 ±32 行窗口，20% 保留全文件
- **过滤**: commit message 黑名单 + ≤2行变更 50% 丢弃 + 100K 字符上限
- **HF**: https://huggingface.co/datasets/bigcode/starcoderdata
- **论文**: StarCoder (arXiv:2305.06161)

### 3. CommitPack (`bigcode/commitpack`)
- **规模**: ~4 TB
- **来源**: GHArchive 元数据 + GitHub API 逐个爬取代码变更
- **格式**: 同 CommitPackFT，但未经质量筛选
- **覆盖**: 截至 2016 年的 1.45 亿个唯一 commit
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
| 大规模预训练 | The Stack v2 | 需要 HF 协议，但覆盖最全 |
| 消融实验 | CommitPackFT Python subset | 小而精，快速迭代 |
