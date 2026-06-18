#!/usr/bin/env bash
set -euo pipefail

TASK_ID="$1"
PROMPT_FILE="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RAW_CURL=".claude-hk-config/raw_curl.txt"
COOKIE=$(grep -oP "(?<=-b ')[^']*" "$RAW_CURL")
ORIGIN=$(grep -oP "(?<=-H 'origin: )[^']+" "$RAW_CURL" | head -1)
UA=$(grep -oP "(?<=-H 'user-agent: )[^']+" "$RAW_CURL" | head -1)

PIN_FILE=".claude-hk-config/ORG_PIN.txt"
if [ -f "$PIN_FILE" ]; then
    ORG_ID=$(grep -v '^#' "$PIN_FILE" | grep -oP '[0-9a-f-]{36}' | head -1)
fi
if [ -z "$ORG_ID" ]; then
    ORG_ID=$(curl -s "${ORIGIN}/api/organizations" \
        -H "accept: application/json" -H "user-agent: ${UA}" \
        -H "accept-language: zh-CN,zh;q=0.9" -H "anthropic-client-platform: web_claude_ai" \
        -b "$COOKIE" 2>/dev/null | python3 -c "
import sys,json
try:
    orgs=json.load(sys.stdin)
    if isinstance(orgs,list) and orgs: print(orgs[0]['uuid'])
except: pass" 2>/dev/null)
fi

H_UUID=$(python3 -c "import uuid; print(uuid.uuid4())")
A_UUID=$(python3 -c "import uuid; print(uuid.uuid4())")

CONV_ID=$(curl -s -X POST "${ORIGIN}/api/organizations/${ORG_ID}/chat_conversations" \
    -H "Content-Type: application/json" -H "origin: ${ORIGIN}" \
    -H "user-agent: ${UA}" -H "accept-language: zh-CN,zh;q=0.9" \
    -H "anthropic-client-platform: web_claude_ai" -b "$COOKIE" \
    --data-raw '{"name":"","model":"claude-sonnet-4-6","is_temporary":false}' 2>/dev/null | \
    python3 -c "import sys,json; print(json.load(sys.stdin)['uuid'])" 2>/dev/null)

echo "TASK_${TASK_ID}: conv=$CONV_ID"

BODY_FILE="/tmp/body_task${TASK_ID}.json"
OUTPUT="/home/claude/tasks/result_task${TASK_ID}.txt"

# 用python3脚本（不是heredoc）构建body，避免shell转义问题
python3 -c "
import json, sys, os
pf = sys.argv[1]
bf = sys.argv[2]
hu = sys.argv[3]
au = sys.argv[4]
with open(pf) as f:
    prompt = f.read()
body = {
    'prompt': prompt,
    'timezone': 'Asia/Shanghai',
    'model': 'claude-sonnet-4-6',
    'effort': 'medium',
    'thinking_mode': 'off',
    'tools': [],
    'turn_message_uuids': {'human_message_uuid': hu, 'assistant_message_uuid': au},
    'attachments': [], 'files': [], 'rendering_mode': 'messages'
}
with open(bf, 'w') as f:
    json.dump(body, f, ensure_ascii=False)
" "$PROMPT_FILE" "$BODY_FILE" "$H_UUID" "$A_UUID"

curl -s -N "${ORIGIN}/api/organizations/${ORG_ID}/chat_conversations/${CONV_ID}/completion" \
    -H "accept: text/event-stream" -H "content-type: application/json" \
    -H "origin: ${ORIGIN}" -H "user-agent: ${UA}" \
    -H "accept-language: zh-CN,zh;q=0.9" \
    -H "anthropic-client-platform: web_claude_ai" \
    -b "$COOKIE" \
    -d @"$BODY_FILE" 2>/dev/null | tr -d '\r' | python3 -u -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: '):
        try:
            d = json.loads(line[6:])
            if d.get('type') == 'content_block_delta':
                delta = d.get('delta', {})
                dt = delta.get('type', '')
                if dt == 'text_delta':
                    txt = delta.get('text', '')
                    if txt: print(txt, end='', flush=True)
        except: pass
" > "$OUTPUT" 2>/dev/null

rm -f "$BODY_FILE"
SIZE=$(wc -c < "$OUTPUT")
echo ""
echo "TASK_${TASK_ID}: DONE (${SIZE} bytes)"
