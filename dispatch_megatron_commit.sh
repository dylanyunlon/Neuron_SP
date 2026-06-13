#!/usr/bin/env bash
# dispatch_megatron_commit.sh — 给每个 Megatron commit 派一位 sub-Claude
# 用法: bash dispatch_megatron_commit.sh <megatron_hash> <M_number>
# 例: bash dispatch_megatron_commit.sh 13c96dc08 M1445
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_DIR="/home/claude/Megatron-LM"
HASH="$1"
MNUM="$2"

# 提取 commit 信息
MSG=$(cd "$MEGATRON_DIR" && git log --oneline -1 "$HASH" 2>/dev/null | cut -d' ' -f2-)
DIFF=$(cd "$MEGATRON_DIR" && git diff "${HASH}~1" "$HASH" 2>/dev/null | head -800)
DIFF_LINES=$(cd "$MEGATRON_DIR" && git diff "${HASH}~1" "$HASH" 2>/dev/null | wc -l)
FILES_CHANGED=$(cd "$MEGATRON_DIR" && git diff --name-only "${HASH}~1" "$HASH" 2>/dev/null | tr '\n' ', ')

echo "=== Dispatch: $MNUM — $HASH ===" 
echo "Message: $MSG"
echo "Diff: ${DIFF_LINES} lines, files: $FILES_CHANGED"

# 构造精简 prompt (鲁迅式: clone + 看diff + 改20%)
PROMPT="Clone github.com/dylanyunlon/Neuron_SP, tree -L 2 看架构.
这是 Megatron-LM commit ${HASH} (${DIFF_LINES}行diff):
Message: ${MSG}
Files: ${FILES_CHANGED}

任务: 将此commit的改动迁移到 Neuron_SP. 规则:
1. 鲁迅式: 不是复制, 修改20%的算法内容让它适配DES-LOC
2. 每处改动加 print() 诊断 (数据状态/结构体/当前值)
3. git config user.name dylanyunlon && git config user.email dogechat@163.com
4. commit message: ${MNUM}: Megatron ${HASH} — ${MSG}
5. git format-patch -1 输出patch
6. 不开新分支, 不加v2/port后缀, 改现有文件

Diff内容:
\`\`\`
${DIFF}
\`\`\`"

# 写入临时 prompt 文件
PROMPT_FILE="/tmp/dispatch_${MNUM}.txt"
echo "$PROMPT" > "$PROMPT_FILE"

echo "Prompt: $(wc -c < "$PROMPT_FILE") bytes"
echo "Dispatching via claude_hk_chat.sh..."

# 调用 sub-Claude
TASK_FILE="$PROMPT_FILE" bash "$SCRIPT_DIR/claude_hk_chat.sh" 2>&1 | tee "/home/claude/dispatch_logs/${MNUM}.log"

echo "=== ${MNUM} dispatch complete ==="
