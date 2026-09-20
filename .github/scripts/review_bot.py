"""PR Review Bot — routes through c.aimonkey.plus to get Claude review."""
import json, os, sys, requests

def main():
    cookie = os.environ["AIMONKEY_COOKIE"]
    gh_token = os.environ["GH_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    org = "80008ad8-c3c8-4f47-b1a1-ca77c382bff2"
    base = f"https://c.aimonkey.plus/api/organizations/{org}"

    # 1. Get PR diff
    print(f"[1/4] Fetching PR #{pr_number} diff...")
    diff_resp = requests.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3.diff"},
    )
    diff = diff_resp.text[:10000]
    print(f"  Diff: {len(diff)} chars")

    # 2. Get PR metadata
    meta_resp = requests.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"},
    )
    meta = meta_resp.json()
    title = meta.get("title", "")
    body = (meta.get("body", "") or "")[:1000]
    print(f"  Title: {title}")

    # 3. Create conversation and send review
    print("[2/4] Creating conversation...")
    conv = requests.post(f"{base}/chat_conversations",
        json={"name": "", "model": "claude-sonnet-4-20250514", "include_conversation_preferences": True},
        headers={"content-type": "application/json", "anthropic-client-platform": "web_claude_ai", "origin": "https://c.aimonkey.plus"},
        cookies={c.split("=", 1)[0].strip(): c.split("=", 1)[1].strip() for c in cookie.split(";") if "=" in c},
    )
    conv_id = conv.json()["uuid"]
    print(f"  Conv: {conv_id}")

    prompt = f"""You are Gemini Code Assist, a senior code reviewer. Review this pull request.

**PR #{pr_number}: {title}**

{body}

Focus on: correctness, race conditions, deadlock risks, test coverage, style consistency.

Format your review EXACTLY like this:

## Summary
(2-3 sentences)

## Findings

### 🔴 Critical
### 🟠 High  
### 🟡 Medium
### 🟢 Suggestions

(omit empty sections)

## Verdict
✅ LGTM / ⚠️ Needs Changes / 🚫 Request Changes

```diff
{diff}
```"""

    print("[3/4] Sending review request (streaming)...")
    resp = requests.post(f"{base}/chat_conversations/{conv_id}/completion",
        json={"prompt": prompt, "timezone": "UTC", "locale": "en-US", "model": "claude-sonnet-4-20250514",
              "tools": [], "attachments": [], "files": [], "sync_sources": [], "rendering_mode": "messages"},
        headers={"content-type": "application/json", "accept": "text/event-stream",
                 "anthropic-client-platform": "web_claude_ai", "origin": "https://c.aimonkey.plus"},
        cookies={c.split("=", 1)[0].strip(): c.split("=", 1)[1].strip() for c in cookie.split(";") if "=" in c},
        stream=True, timeout=120,
    )

    review = ""
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]":
                break
            try:
                ev = json.loads(d)
                if ev.get("type") == "content_block_delta":
                    review += ev.get("delta", {}).get("text", "")
            except json.JSONDecodeError:
                pass
    print(f"  Review: {len(review)} chars")

    # 4. Post as PR comment
    print("[4/4] Posting review to PR...")
    comment = f"🤖 **Gemini Code Assist Review**\n\n{review}\n\n---\n_Powered by Claude via c.aimonkey.plus routing · triggered automatically on PR_"
    requests.post(
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
        json={"body": comment},
        headers={"Authorization": f"token {gh_token}", "Accept": "application/vnd.github.v3+json"},
    )
    print(f"✅ Review posted to PR #{pr_number}")

if __name__ == "__main__":
    main()
