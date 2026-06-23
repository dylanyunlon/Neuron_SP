#!/usr/bin/env bash
# claude_hk_chat.sh — 通过 claude.hk.cn 派发任务给子 Claude (Sonnet 4.6 medium)
# 用法: bash claude_hk_chat.sh "任务prompt"
# 或:   TASK_FILE=task.md bash claude_hk_chat.sh
# 续传: CONV_ID=xxx bash claude_hk_chat.sh "Continue"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 从 claude-hk-config 同步 cookie ──
CONFIG_REPO="${SCRIPT_DIR}/.claude-hk-config"
if [ ! -d "$CONFIG_REPO" ]; then
    git clone https://github.com/dylanyunlon/claude-hk-config.git "$CONFIG_REPO" 2>/dev/null || true
else
    cd "$CONFIG_REPO" && git pull -q 2>/dev/null || true
    cd "$SCRIPT_DIR"
fi

RAW_CURL="${CONFIG_REPO}/raw_curl.txt"
if [ ! -f "$RAW_CURL" ]; then
    echo "ERROR: $RAW_CURL not found"
    exit 1
fi

COOKIE=$(grep -oP "(?<=-b 'share-session=1hflpc6f3gsw90djg8exc61e9g5jr95q; ajs_anonymous_id=claudeai.v1.fd38ff3a-6fa9-4bb4-8b45-3d502873bd1f; user-sidebar-visible-on-load=true; CH-prefers-color-scheme=light; user-sidebar-pinned=false; lastActiveOrg=2ec8e60d-f0c6-456e-971d-735ec9e61e70; _dd_s=aid=1f0fa0af-44f1-4b2f-8054-15dc64a41176&rum=2&id=152ec09d-ba3a-437f-920f-cbbeaff40d6e&created=1782197873615&expire=1782199218823']*" "$RAW_CURL" || echo "")
ORG_ID=$(grep -oP 'organizations/\K[^/]+' "$RAW_CURL" | head -1)
ORIGIN=$(grep -oP "(?<=-H 'origin: )[^']+" "$RAW_CURL" | head -1 || echo "https://claude.hk.cn")
UA=$(grep -oP "(?<=-H 'user-agent: )[^']+" "$RAW_CURL" | head -1 || echo "Mozilla/5.0")
COMMON_H=(-H "user-agent: ${UA}" -H "referer: ${ORIGIN}/" -H "accept-language: zh-CN,zh;q=0.9" -H "anthropic-client-platform: web_claude_ai")

if [ -z "$COOKIE" ] || [ -z "$ORG_ID" ]; then
    echo "ERROR: Cannot extract cookie/org"; exit 1
fi

# ── org 动态解析: raw_curl.txt里的orgid可能过期 ──
ORG_LIVE=$(curl -s "${ORIGIN}/api/organizations" \
    -H "accept: application/json" "${COMMON_H[@]}" -b "$COOKIE" 2>/dev/null | python3 -c "
import sys, json
try:
    orgs = json.load(sys.stdin)
    if isinstance(orgs, list) and orgs:
        print(orgs[0].get('uuid', ''))
except Exception:
    pass" 2>/dev/null || echo "")
if [ -n "$ORG_LIVE" ]; then
    if [ "$ORG_LIVE" != "$ORG_ID" ]; then
        echo "Org updated: raw_curl=$ORG_ID -> live=$ORG_LIVE"
    fi
    ORG_ID="$ORG_LIVE"
fi
echo "Org: $ORG_ID"

if [ -n "${TASK_FILE:-}" ] && [ -f "$TASK_FILE" ]; then
    PROMPT=$(cat "$TASK_FILE")
elif [ $# -gt 0 ]; then
    PROMPT="$*"
else
    echo "Usage: bash claude_hk_chat.sh \"prompt\""; exit 1
fi

echo "=== Claude HK → Sonnet 4.6 (medium) ==="
echo "Prompt: ${#PROMPT} chars"

ESCAPED_PROMPT=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$PROMPT")
HUMAN_UUID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
ASST_UUID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")

# 创建或复用对话
if [ -n "${CONV_ID:-}" ]; then
    echo "Conv (reuse): $CONV_ID"
else
    CREATE_RESP=$(curl -s -X POST "${ORIGIN}/api/organizations/${ORG_ID}/chat_conversations" \
        -H "Content-Type: application/json" -H "origin: ${ORIGIN}" "${COMMON_H[@]}" -b "$COOKIE" \
        --data-raw '{"name":"","model":"claude-opus-4-6","is_temporary":false}' 2>/dev/null)
    CONV_ID=$(echo "$CREATE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('uuid',''))" 2>/dev/null || echo "")
    if [ -z "$CONV_ID" ]; then
        echo "ERROR: Create conversation failed: ${CREATE_RESP:0:300}"; exit 1
    fi
    echo "Conv (new): $CONV_ID  — 续传: CONV_ID=$CONV_ID bash claude_hk_chat.sh Continue"
fi

OUTPUT_FILE="${SCRIPT_DIR}/submodel_response_$(date +%Y%m%d_%H%M%S).txt"
> "$OUTPUT_FILE"

curl -s -N "${ORIGIN}/api/organizations/${ORG_ID}/chat_conversations/${CONV_ID}/completion" \
    -H "accept: text/event-stream" -H "content-type: application/json" \
    -H "origin: ${ORIGIN}" "${COMMON_H[@]}" -b "$COOKIE" \
    --data-raw "{
        \"prompt\":${ESCAPED_PROMPT},\"timezone\":\"Asia/Shanghai\",\"model\":\"claude-opus-4-6\",
        \"effort\":\"medium\",\"thinking_mode\":\"off\",
        \"tools\":[{\"type\":\"repl_v0\",\"name\":\"repl\"}],
        \"turn_message_uuids\":{\"human_message_uuid\":\"${HUMAN_UUID}\",\"assistant_message_uuid\":\"${ASST_UUID}\"},
        \"attachments\":[],\"files\":[],\"rendering_mode\":\"messages\"
    }" 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        echo "${line#data: }" | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.read())
    if d.get('type')=='content_block_delta':
        t=d.get('delta',{}).get('text','')
        if t: print(t,end='',flush=True)
except: pass
" 2>/dev/null | tee -a "$OUTPUT_FILE"
    fi
done

echo ""
echo "=== Saved: $OUTPUT_FILE ($(wc -c < "$OUTPUT_FILE") bytes) ==="
