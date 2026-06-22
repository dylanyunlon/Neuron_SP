#!/usr/bin/env bash
# dispatch_phase6.sh — 派发 Phase 6 的 10 个 sub-Claude 任务
# 用法: bash dispatch_phase6.sh <claude_number>
# 例: bash dispatch_phase6.sh 17  (派发 Claude-17 Rotary Embedding 任务)
# 或: bash dispatch_phase6.sh all  (显示所有任务概览)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GIT_TOKEN="${NEURON_SP_TOKEN:?请设置环境变量 NEURON_SP_TOKEN (GitHub PAT)}"

TASK_MAP=(
    [17]="task_claude17_M686_M700.md|Rotary Embedding → DES-LOC Pipeline|GPT-NeoX"
    [18]="task_claude18_M701_M715.md|FSDP→ZeRO-3 异构分片|Metaseq OPT"
    [19]="task_claude19_M716_M730.md|ALiBi 位置编码集成|BLOOM"
    [20]="task_claude20_M731_M745.md|StarCoder Commit Sequence Packing|BigCode"
    [21]="task_claude21_M746_M760.md|CommitPack 4TB 流式加载器|BigCode"
    [22]="task_claude22_M761_M775.md|The Stack v2 格式适配|BigCode"
    [23]="task_claude23_M776_M790.md|ColossalAI Gemini 异构内存|ColossalAI"
    [24]="task_claude24_M791_M805.md|GLM-130B 多任务预训练|GLM-130B"
    [25]="task_claude25_M806_M820.md|Chinchilla Scaling Law 验证|DeepMind"
    [26]="task_claude26_M821_M835.md|CodeGen 评估框架|Salesforce"
)

if [ "${1:-}" = "all" ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           Phase 6: 预训练框架集成 (10 Sub-Claude)            ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    for i in 17 18 19 20 21 22 23 24 25 26; do
        IFS='|' read -r file desc src <<< "${TASK_MAP[$i]}"
        echo "║ Claude-$i  $desc  [$src]"
    done
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║ 并行组: 17,18,19,20,23,24 (独立) → 21,22 (顺序) → 25,26   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    exit 0
fi

N="${1:?用法: bash dispatch_phase6.sh <17-26|all>}"
if [ -z "${TASK_MAP[$N]+x}" ]; then
    echo "错误: Claude-$N 不在 Phase 6 (17-26)"
    exit 1
fi

IFS='|' read -r TASK_FILE TASK_DESC TASK_SRC <<< "${TASK_MAP[$N]}"
TASK_CONTENT=$(cat "tasks/$TASK_FILE")

# 生成精简派发 prompt
PROMPT="你是 Neuron_SP 项目的 Claude-${N}。

## 快速上手
\`\`\`bash
apt install -y tree git
git clone https://${GIT_TOKEN}@github.com/dylanyunlon/Neuron_SP.git
cd Neuron_SP && tree -L 1 --charset ascii && git log --oneline -5
cat tasks/${TASK_FILE}
\`\`\`

## 你的任务: ${TASK_DESC} (参考: ${TASK_SRC})

${TASK_CONTENT}

## 铁律
1. MODIFY EXISTING FILES ONLY — 不建新独立 .py
2. cat FILE FIRST — 改前必读
3. ast.parse AFTER — 每次改完验证语法
4. Signed-off-by: dylanyunlong <dylanyunlong@gmail.com>
5. push:
\`\`\`bash
git remote set-url origin https://x-access-token:${GIT_TOKEN}@github.com/dylanyunlon/Neuron_SP.git
git pull --rebase origin main
git add -A && git commit --signoff -m \"Claude-${N} M${((686 + (N-17)*15))}: ${TASK_DESC}\"
git push origin main
\`\`\`"

echo "=== Claude-${N} Dispatch Prompt ==="
echo "$PROMPT"
echo ""
echo "=== 复制上面的 prompt 发送给 sub-Claude ==="
