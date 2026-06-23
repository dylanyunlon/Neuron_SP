#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 同步 cookie ──
cd .claude-hk-config && git pull -q 2>/dev/null; cd "$SCRIPT_DIR"
COOKIE=$(grep -oP "(?<=-b ')[^']*" .claude-hk-config/raw_curl.txt)
ORIGIN="https://claude.hk.cn"
UA=$(grep -oP "(?<=-H 'user-agent: )[^']+" .claude-hk-config/raw_curl.txt | head -1)
ORG_ID=$(grep -v '^#' .claude-hk-config/ORG_PIN.txt | grep -oP '[0-9a-f-]{36}' | head -1)

COMMON_H=(-H "user-agent: ${UA}" -H "accept-language: zh-CN,zh;q=0.9" -H "anthropic-client-platform: web_claude_ai")

# 验证 org
ORG_LIVE=$(curl -s "${ORIGIN}/api/organizations" -H "accept: application/json" "${COMMON_H[@]}" -b "$COOKIE" 2>/dev/null | python3 -c "
import sys,json
try:
    orgs=json.load(sys.stdin)
    if isinstance(orgs,list) and orgs: print(orgs[0]['uuid'])
except: pass" 2>/dev/null)
if [ -n "$ORG_LIVE" ]; then ORG_ID="$ORG_LIVE"; fi
echo "Org: $ORG_ID"

BASE_PROMPT=$(cat tasks/round11/base_prompt.txt)
LOG_DIR="tasks/round11/logs"
mkdir -p "$LOG_DIR"

dispatch_one() {
    local N=$1
    local TASK_FILE="tasks/round11/task_c${N}.md"
    local TASK_CONTENT=$(cat "$TASK_FILE")
    local FULL_PROMPT="${BASE_PROMPT}

## 你是 C${N} 号开发者，你的任务:

${TASK_CONTENT}

现在开始: clone 仓库，看相关文件，改代码，commit，push。"

    # 创建对话
    local H_UUID=$(python3 -c "import uuid; print(uuid.uuid4())")
    local A_UUID=$(python3 -c "import uuid; print(uuid.uuid4())")
    local CONV_ID=$(curl -s -X POST "${ORIGIN}/api/organizations/${ORG_ID}/chat_conversations" \
        -H "Content-Type: application/json" -H "origin: ${ORIGIN}" "${COMMON_H[@]}" -b "$COOKIE" \
        --data-raw '{"name":"R11-C'"${N}"'","model":"claude-sonnet-4-6","is_temporary":false}' 2>/dev/null | \
        python3 -c "import sys,json; print(json.load(sys.stdin).get('uuid',''))" 2>/dev/null)
    
    if [ -z "$CONV_ID" ]; then
        echo "[C${N}] FAILED to create conversation"
        return 1
    fi
    echo "[C${N}] conv=$CONV_ID"
    echo "$CONV_ID" > "$LOG_DIR/c${N}_conv.txt"

    # 发送 prompt
    local ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$FULL_PROMPT")
    
    curl -s -N "${ORIGIN}/api/organizations/${ORG_ID}/chat_conversations/${CONV_ID}/completion" \
        -H "accept: text/event-stream" -H "content-type: application/json" \
        -H "origin: ${ORIGIN}" "${COMMON_H[@]}" -b "$COOKIE" \
        --data-raw "{
            \"prompt\":${ESCAPED},\"timezone\":\"Asia/Shanghai\",\"model\":\"claude-sonnet-4-6\",
            \"effort\":\"medium\",\"thinking_mode\":\"off\",
            \"tools\":[{\"type\":\"repl_v0\",\"name\":\"repl\"}],
            \"turn_message_uuids\":{\"human_message_uuid\":\"${H_UUID}\",\"assistant_message_uuid\":\"${A_UUID}\"},
            \"attachments\":[],\"files\":[],\"rendering_mode\":\"messages\"
        }" > "$LOG_DIR/c${N}_response.txt" 2>/dev/null &
    
    echo "[C${N}] dispatched (PID=$!)"
}

echo "=========================================="
echo " Round 11: Dispatching 10 sub-Claudes"
echo " $(date)"
echo "=========================================="

for N in $(seq 1 10); do
    dispatch_one $N
    sleep 3  # 错开请求
done

echo ""
echo "All 10 dispatched. Waiting..."
wait
echo "All complete. $(date)"
echo ""
echo "Conv IDs:"
for N in $(seq 1 10); do
    echo "  C${N}: $(cat $LOG_DIR/c${N}_conv.txt 2>/dev/null || echo FAILED)"
done
