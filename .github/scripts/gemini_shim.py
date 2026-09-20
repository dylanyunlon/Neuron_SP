"""Local Gemini API shim — translates Gemini generateContent → c.aimonkey.plus Claude."""
import http.server, json, os, sys, threading, requests

COOKIE = os.environ["AIMONKEY_COOKIE"]
ORG = "80008ad8-c3c8-4f47-b1a1-ca77c382bff2"
BASE = f"https://c.aimonkey.plus/api/organizations/{ORG}"

def parse_cookies(raw):
    return {c.split("=",1)[0].strip(): c.split("=",1)[1].strip() for c in raw.split(";") if "=" in c}

def create_conversation():
    r = requests.post(f"{BASE}/chat_conversations",
        json={"name":"","model":"claude-sonnet-4-20250514","include_conversation_preferences":True},
        headers={"content-type":"application/json","anthropic-client-platform":"web_claude_ai","origin":"https://c.aimonkey.plus"},
        cookies=parse_cookies(COOKIE))
    return r.json()["uuid"]

def send_message(conv_id, text):
    r = requests.post(f"{BASE}/chat_conversations/{conv_id}/completion",
        json={"prompt":text,"timezone":"UTC","locale":"en-US","model":"claude-sonnet-4-20250514",
              "tools":[],"attachments":[],"files":[],"sync_sources":[],"rendering_mode":"messages"},
        headers={"content-type":"application/json","accept":"text/event-stream",
                 "anthropic-client-platform":"web_claude_ai","origin":"https://c.aimonkey.plus"},
        cookies=parse_cookies(COOKIE), stream=True, timeout=180)
    full = ""
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]": break
            try:
                ev = json.loads(d)
                if ev.get("type") == "content_block_delta":
                    full += ev.get("delta",{}).get("text","")
            except: pass
    return full

class GeminiShimHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # Extract prompt from Gemini generateContent format
        contents = body.get("contents", "")
        if isinstance(contents, list):
            text_parts = []
            for c in contents:
                for p in c.get("parts", []):
                    if isinstance(p, dict) and "text" in p:
                        text_parts.append(p["text"])
                    elif isinstance(p, str):
                        text_parts.append(p)
            contents = "\n".join(text_parts)
        system = body.get("systemInstruction", "")
        if isinstance(system, dict):
            parts = system.get("parts", [])
            system = "\n".join(p.get("text","") if isinstance(p,dict) else str(p) for p in parts)
        prompt = (system + "\n\n" + contents).strip() if system else contents

        print(f"[shim] Received {len(prompt)} chars, routing to c.aimonkey.plus...")
        conv_id = create_conversation()
        response_text = send_message(conv_id, prompt)
        print(f"[shim] Got {len(response_text)} chars back")

        # Return in Gemini API format
        result = {
            "candidates": [{"content": {"parts": [{"text": response_text}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": len(prompt)//4, "candidatesTokenCount": len(response_text)//4, "totalTokenCount": (len(prompt)+len(response_text))//4},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, fmt, *args):
        print(f"[shim] {fmt % args}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = http.server.HTTPServer(("127.0.0.1", port), GeminiShimHandler)
    print(f"[shim] Gemini API shim on http://127.0.0.1:{port}")
    server.serve_forever()
