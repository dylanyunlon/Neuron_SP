#!/usr/bin/env bash
# dispatch_megatron_review.sh — 派发Megatron commit审阅给子Claude
# 用法: bash dispatch_megatron_review.sh <commit_hash> <commit_number>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMMIT="$1"
NUM="$2"
MEGATRON_BARE="/home/claude/megatron_bare"

# 获取commit信息
MSG=$(cd "$MEGATRON_BARE" && git log --format="%s" -1 "$COMMIT")
DIFF=$(cd "$MEGATRON_BARE" && git show "$COMMIT" --stat --format="" | tail -1)
# 获取diff内容(截断到8000字符防prompt过长)
DIFF_CONTENT=$(cd "$MEGATRON_BARE" && git show "$COMMIT" --format="" | head -300)

PROMPT="你是Neuron_SP项目的算法审阅员。任务：审阅Megatron-LM的一个commit,提取可融入我们DES-LOC预训练系统的算法改动。

Megatron commit #${NUM}: ${COMMIT:0:12}
Message: ${MSG}
Stats: ${DIFF}

== DIFF内容(截断) ==
${DIFF_CONTENT}
== END DIFF ==

请执行:
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP
2. 阅读上面的diff,找出算法核心(不是字符串/docstring/格式改动)
3. 如果有可融入的算法逻辑,修改Neuron_SP中对应的现有文件(MODIFY EXISTING FILES ONLY)
4. 修改20%内容——鲁迅式化用,不照搬
5. 加入print诊断:每个关键数据结构状态、每个算法决策点
6. 不用v2/port/alt等后缀。不开新分支。
7. 如果commit是纯格式/文档/trivial修改,回复: SKIP-TRIVIAL
8. 如果有改动,输出git diff格式的patch

作者: dylanyunlon <dogechat@163.com>
Signed-off-by: dylanyunlon <dogechat@163.com>"

echo "=== Dispatching Megatron commit #${NUM}: ${COMMIT:0:12} — ${MSG:0:60} ==="
echo "$PROMPT" | timeout 120 bash claude_hk_chat.sh "$PROMPT" 2>&1
