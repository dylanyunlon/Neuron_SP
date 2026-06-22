# Claude-12 任务: M611-M625 — GHArchive Event 提取器 (Stack v2 参考)

Session: Claude-12 (M611-M625) | Base: latest main

## 目标
参考 Stack v2 的 GHArchive pipeline，实现独立的 GitHub 事件提取和过滤工具。

## 步骤

### 1. 研读 Stack v2 pipeline
```bash
cat references/data_pipelines/stackv2_pr_commits/README.md
cat references/data_pipelines/stackv2_pr_commits/0_get_gharchive_events.py
cat references/data_pipelines/stackv2_pr_commits/cfg.py
```

### 2. 实现精简版
- 从 GHArchive 下载指定时间段的 events
- 过滤 PushEvent (包含 commit) 和 PullRequestEvent
- 按编程语言和 license 过滤
- 输出: JSONL 格式，每行一个 commit diff

### 3. 与 BigCode v1 pipeline 对接
- 输出格式兼容 filtering_git_commits.ipynb 的输入

## 交付物
- 工具脚本写入 tools/ 目录 + commit push
- 1 小时 GHArchive 数据的提取样例
