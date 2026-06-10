#!/usr/bin/env bash
# master_loop.sh — 主调度循环: 逐commit派发给子Claude
# 用法: bash master_loop.sh [start_idx] [end_idx]
# 默认: 从commit 1开始
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MEGATRON_BARE="/home/claude/megatron_bare"
RESULTS_DIR="$SCRIPT_DIR/megatron_review_results"
mkdir -p "$RESULTS_DIR"

START=${1:-1}
END=${2:-20}

# 生成全部commit列表(正序: 从first commit开始)
cd "$MEGATRON_BARE"
git log --reverse --format="%H" > /tmp/megatron_all_commits.txt
TOTAL=$(wc -l < /tmp/megatron_all_commits.txt)
echo "Megatron-LM total commits: $TOTAL"
echo "Processing: #${START} to #${END}"
cd "$SCRIPT_DIR"

IDX=0
while IFS= read -r COMMIT; do
    IDX=$((IDX + 1))
    if [ "$IDX" -lt "$START" ]; then continue; fi
    if [ "$IDX" -gt "$END" ]; then break; fi

    # 获取commit消息
    MSG=$(cd "$MEGATRON_BARE" && git log --format="%s" -1 "$COMMIT")
    STAT=$(cd "$MEGATRON_BARE" && git show "$COMMIT" --stat --format="" 2>/dev/null | tail -1 || echo "?")

    echo ""
    echo "================================================================"
    echo " [${IDX}/${END}] Commit: ${COMMIT:0:12} — ${MSG:0:70}"
    echo " Stats: $STAT"
    echo "================================================================"

    # 获取diff(截断到6000字符)
    DIFF_CONTENT=$(cd "$MEGATRON_BARE" && git show "$COMMIT" --format="" 2>/dev/null | head -250 || echo "empty")

    PROMPT="你是Neuron_SP算法审阅员。审阅Megatron-LM commit,提取算法精华融入DES-LOC。

Megatron commit #${IDX}: ${COMMIT:0:12}
Message: ${MSG}
Stats: ${STAT}

== DIFF ==
${DIFF_CONTENT}
== END ==

执行:
1. git clone https://github.com/dylanyunlon/Neuron_SP.git && cd Neuron_SP && git log --oneline -3
2. 阅读diff,找算法核心(跳过纯文档/格式)
3. 有算法改动→修改Neuron_SP现有文件,20%化用+print诊断
4. 无算法改动→回复SKIP-TRIVIAL
5. 有改动时: git add -A && git commit --signoff --author='dylanyunlon <dogechat@163.com>' -m 'M${IDX}: integrate Megatron ${COMMIT:0:9} — ${MSG:0:50}'
6. 铁律: 不用v2/port后缀,不开新分支,MODIFY EXISTING FILES ONLY"

    OUTFILE="${RESULTS_DIR}/commit_${IDX}_${COMMIT:0:12}.txt"

    # 派发给子Claude
    echo "$PROMPT" > /tmp/dispatch_prompt.txt
    TASK_FILE=/tmp/dispatch_prompt.txt timeout 180 bash claude_hk_chat.sh 2>&1 | tee "$OUTFILE" || true

    # 检查结果
    if grep -qi "SKIP-TRIVIAL" "$OUTFILE" 2>/dev/null; then
        echo "  → SKIPPED (trivial)"
    elif [ -s "$OUTFILE" ]; then
        BYTES=$(wc -c < "$OUTFILE")
        echo "  → Response: ${BYTES} bytes"
    else
        echo "  → EMPTY response (retry needed)"
    fi

    # 短暂间隔避免rate limit
    sleep 3

done < /tmp/megatron_all_commits.txt

echo ""
echo "================================================================"
echo " Master loop complete: processed commits #${START}-#${END}"
echo " Results in: $RESULTS_DIR"
echo "================================================================"
