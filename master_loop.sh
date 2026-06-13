#!/usr/bin/env bash
# master_loop.sh — 循环派发 Megatron commits 给 sub-Claude 小弟们
# 用法: bash master_loop.sh [start_index] [count]
# 例:   bash master_loop.sh 1 10   # 从第1个待迁移commit开始, 派10个
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MEGATRON_DIR="/home/claude/Megatron-LM"
COMMIT_LIST="/home/claude/megatron_pending_commits.txt"
LOG_DIR="/home/claude/dispatch_logs"
mkdir -p "$LOG_DIR"

START=${1:-1}
COUNT=${2:-5}
BASE_M=1444  # Neuron_SP 当前最高 M 编号

echo "============================================"
echo " Megatron → Neuron_SP Migration Loop"
echo " Start: commit #$START"
echo " Count: $COUNT commits"
echo " Base M: M${BASE_M}"
echo " $(date)"
echo "============================================"

COMPLETED=0
FAILED=0

for i in $(seq $START $((START + COUNT - 1))); do
    LINE=$(sed -n "${i}p" "$COMMIT_LIST")
    if [ -z "$LINE" ]; then
        echo "[LOOP] No more commits at index $i"
        break
    fi
    
    HASH=$(echo "$LINE" | cut -d' ' -f1)
    MNUM="M$((BASE_M + i))"
    MSG=$(echo "$LINE" | cut -d' ' -f2-)
    
    # 检查 diff 大小, 跳过 merge commits (diff=0)
    DIFF_SIZE=$(cd "$MEGATRON_DIR" && git diff "${HASH}~1" "$HASH" 2>/dev/null | wc -l)
    if [ "$DIFF_SIZE" -eq 0 ]; then
        echo "[LOOP] [$MNUM] $HASH — SKIP (merge commit, 0 diff)"
        echo "SKIP:MERGE" > "$LOG_DIR/${MNUM}.status"
        continue
    fi
    
    echo ""
    echo "[LOOP] [$MNUM] $HASH — $MSG (${DIFF_SIZE} lines)"
    echo "[LOOP] Dispatching sub-Claude..."
    
    # 派发任务
    if bash "$SCRIPT_DIR/dispatch_megatron_commit.sh" "$HASH" "$MNUM" 2>&1; then
        echo "OK" > "$LOG_DIR/${MNUM}.status"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "FAILED" > "$LOG_DIR/${MNUM}.status"
        FAILED=$((FAILED + 1))
    fi
    
    echo "[LOOP] [$MNUM] Done. Completed=$COMPLETED Failed=$FAILED"
    
    # 限速: 每个 commit 之间等 5 秒
    sleep 5
done

echo ""
echo "============================================"
echo " Loop Complete"
echo " Completed: $COMPLETED"
echo " Failed: $FAILED"
echo " $(date)"
echo "============================================"
