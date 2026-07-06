#!/usr/bin/env python3
"""Dispatch a task to a sub-Claude using the EXACT raw_curl.txt format."""
import json, sys, uuid, subprocess, os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.claude-hk-config')

# Read cookie from raw_curl.txt
with open(os.path.join(CONFIG_DIR, 'raw_curl.txt')) as f:
    raw = f.read()

import re
cookie_match = re.search(r"-b '([^']+)'", raw)
cookie = cookie_match.group(1) if cookie_match else ''

# Read org from ORG_PIN.txt
with open(os.path.join(CONFIG_DIR, 'ORG_PIN.txt')) as f:
    org = f.read().strip()

# Args: model, conv_id, prompt_file
if len(sys.argv) < 4:
    print("Usage: python dispatch_worker.py <model> <conv_id> <prompt_file>")
    print("  model: claude-sonnet-4-6 | claude-opus-4-6")
    sys.exit(1)

model = sys.argv[1]
conv_id = sys.argv[2]
prompt_file = sys.argv[3]

with open(prompt_file) as f:
    prompt = f.read()

h_uuid = str(uuid.uuid4())
a_uuid = str(uuid.uuid4())

# Build the EXACT body matching raw_curl.txt format
body = {
    "prompt": prompt,
    "timezone": "Asia/Shanghai",
    "locale": "en-US",
    "model": model,
    "effort": "medium",
    "thinking_mode": "extended",
    "tools": [
        {"name":"show_widget","description":"Show visual content","input_schema":{"type":"object","properties":{"loading_messages":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":4},"title":{"type":"string"},"widget_code":{"type":"string"}},"required":["loading_messages","title","widget_code"]},"integration_name":"visualize","is_mcp_app":True},
        {"name":"read_me","description":"Returns required context for show_widget","input_schema":{"type":"object","properties":{"modules":{"type":"array","items":{"type":"string","enum":["diagram","mockup","interactive","data_viz","art","chart","elicitation"]}},"platform":{"type":"string","enum":["mobile","desktop","unknown"]}}},"integration_name":"visualize","is_mcp_app":False},
        {"type":"web_search_v0","name":"web_search"},
        {"type":"artifacts_v0","name":"artifacts"},
        {"type":"repl_v0","name":"repl"},
        {"type":"widget","name":"weather_fetch"},
        {"type":"widget","name":"recipe_display_v0"},
        {"type":"widget","name":"places_map_display_v0"},
        {"type":"widget","name":"message_compose_v1"},
        {"type":"widget","name":"ask_user_input_v0"},
        {"type":"widget","name":"recommend_claude_apps"},
        {"type":"widget","name":"schedule_cowork_task_v0"},
        {"type":"widget","name":"places_search"},
        {"type":"widget","name":"fetch_sports_data"}
    ],
    "turn_message_uuids": {
        "human_message_uuid": h_uuid,
        "assistant_message_uuid": a_uuid
    },
    "attachments": [],
    "files": [],
    "sync_sources": [],
    "rendering_mode": "messages",
    "create_conversation_params": {
        "name": "",
        "model": model,
        "include_conversation_preferences": True,
        "paprika_mode": None,
        "compass_mode": None,
        "tool_search_mode": "off",
        "is_temporary": False,
        "enabled_imagine": True
    }
}

url = f"https://claude.hk.cn/api/organizations/{org}/chat_conversations/{conv_id}/completion"

# Build curl command with ALL headers from raw_curl.txt
cmd = [
    'curl', '-s', '-N', url,
    '-H', 'accept: text/event-stream',
    '-H', 'accept-language: zh-CN,zh;q=0.9',
    '-H', 'anthropic-client-platform: web_claude_ai',
    '-H', 'content-type: application/json',
    '-b', cookie,
    '-H', 'origin: https://claude.hk.cn',
    '-H', 'priority: u=1, i',
    '-H', 'referer: https://claude.hk.cn/new',
    '-H', 'sec-ch-ua: "Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    '-H', 'sec-ch-ua-mobile: ?0',
    '-H', 'sec-ch-ua-platform: "Windows"',
    '-H', 'sec-fetch-dest: empty',
    '-H', 'sec-fetch-mode: cors',
    '-H', 'sec-fetch-site: same-origin',
    '-H', 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
    '--data-raw', json.dumps(body),
    '--max-time', '600',
]

print(f"Dispatching to {model} conv={conv_id[:8]}...")
print(f"Prompt: {len(prompt)} chars")
print(f"URL: {url}")

# Run and stream output
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

output_file = f"/tmp/worker_{conv_id[:8]}.txt"
texts = []
tools = []
errors = []

with open(output_file, 'w') as out:
    for line in proc.stdout:
        out.write(line)
        out.flush()
        line = line.strip()
        if line.startswith('data: '):
            try:
                d = json.loads(line[6:])
                t = d.get('type','')
                if t == 'content_block_delta':
                    txt = d.get('delta',{}).get('text','')
                    if txt:
                        texts.append(txt)
                        print(txt, end='', flush=True)
                elif t == 'content_block_start':
                    cb = d.get('content_block',{})
                    if cb.get('type') == 'tool_use':
                        tools.append(cb.get('name',''))
                        print(f'\n[TOOL:{cb.get("name","")}]', end='', flush=True)
                elif t == 'message_stop':
                    print('\n[DONE]', flush=True)
                elif t == 'error':
                    err = d.get('error',{}).get('message','')
                    errors.append(err)
                    print(f'\n[ERROR: {err}]', flush=True)
            except:
                pass
        elif line.startswith('{') and '"error"' in line:
            try:
                d = json.loads(line)
                err = d.get('error',{}).get('message','')
                errors.append(err)
                print(f'[ERROR: {err}]')
            except:
                pass

proc.wait()
print(f"\n\n=== Summary ===")
print(f"Tools used: {tools}")
print(f"Errors: {errors}")
print(f"Text length: {len(''.join(texts))} chars")
print(f"Output saved: {output_file} ({os.path.getsize(output_file)} bytes)")
