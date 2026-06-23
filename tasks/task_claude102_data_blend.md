# Claude-102: 数据混合比例配置

## 任务
完善 `data/blend_datasets.py` — 配置四大 commit 数据集的混合比例。

## 具体工作
1. `cat data/blend_datasets.py` 先读
2. 添加 DES-LOC 推荐的混合比例:
   - CommitPackFT: 40% (高质量, GPT-4 筛选)
   - StarCoder git-commits-cleaned: 30% (去重后)
   - CommitPack (streaming): 20% (量大但质量参差)
   - The Stack v2 PR/commit: 10% (补充 PR 合并风格)
3. 添加 `build_blended_dataloader()` 函数，按比例采样
4. 支持 `--blend-config` YAML 覆盖默认比例

## 铁律
- MODIFY EXISTING FILES ONLY
- 作者: dylanyunlon <dogechat@163.com>
- git commit --signoff
