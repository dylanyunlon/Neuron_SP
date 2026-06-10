#!/usr/bin/env bash
# dispatch_all.sh — Neuron_SP 子Claude任务派发器
# 用法: bash dispatch_all.sh [task_number]
# 示例: bash dispatch_all.sh 2   # 派发第2号子Claude任务
#       bash dispatch_all.sh all  # 顺序派发所有任务
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

dispatch_one() {
    local num=$1
    local pattern="tasks/task_claude${num}_M*.md"
    local files=( $pattern )

    if [ ! -f "${files[0]}" ]; then
        echo "ERROR: No task file matching $pattern"
        return 1
    fi

    local task_file="${files[0]}"
    echo "=== Dispatching Claude-${num} ==="
    echo "Task: $task_file ($(wc -l < "$task_file") lines)"
    echo ""

    TASK_FILE="$task_file" bash claude_hk_chat.sh
    echo ""
    echo "=== Claude-${num} dispatch complete ==="
    echo ""
}

if [ $# -lt 1 ]; then
    echo "Usage: bash dispatch_all.sh <task_number|all>"
    echo ""
    echo "Available tasks:"
    ls tasks/task_claude*_M*.md 2>/dev/null | while read f; do
        num=$(echo "$f" | grep -oP 'claude\K\d+')
        miles=$(echo "$f" | grep -oP 'M\d+_M\d+')
        echo "  $num  →  $f  ($miles)"
    done
    exit 1
fi

TASK_NUM=${1}

if [ "$TASK_NUM" = "all" ]; then
    for f in tasks/task_claude*_M*.md; do
        num=$(echo "$f" | grep -oP 'claude\K\d+')
        dispatch_one "$num"
        echo "--- Sleeping 10s before next dispatch ---"
        sleep 10
    done
else
    dispatch_one "$TASK_NUM"
fi
