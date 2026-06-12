# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M835: Megatron a5bfc2966 — added new inference to the server
# ===========================================================================
#
# Upstream source:
#   tools/run_text_generation_server.py
#   (NVIDIA/Megatron-LM commit a5bfc296648b8c77374d7df0176d304b4d5ea421)
#   Author: mshoeybi <mshoeybi@nvidia.com>  Date: 2021-10-10
#
# Mapping: tools/run_text_generation_server.py
#          → deepspeed/tools/run_cli.py
#
# Changes ported from upstream:
#   - Replace `from megatron.text_generation_utils import generate` with
#     `from megatron.inference.api import generate_and_post_process`
#     (mapped here to deepspeed.compile.inference_api).
#   - In the __main__ while-True worker loop, replace
#     `generate(model)` with `generate_and_post_process(model)`.
#   - Interactive CLI section: update request body to use "prompts" key
#     (was "sentences") and "tokens_to_generate" (was "max_len") to match
#     the new server API introduced in M835.
#   - Update response key from "sentences" to "text".
#
# ===========================================================================
#
# ===========================================================================
# M729: Megatron 7a9c4a03f — Removing bug possibilities and adding timing info
# ===========================================================================
#
# Upstream source:
#   tools/run_cli.py
#   (NVIDIA/Megatron-LM commit 7a9c4a03fdbc5a235e47feac29839a733101c0c5)
#   Author: rprenger <rprenger@nvidia.com>  Date: 2021-07-19
#
# Mapping: tools/run_cli.py
#          → deepspeed/tools/run_cli.py
#
# Changes ported from upstream:
#   - Added `import sys` (was missing; caused NameError when accessing argv).
#   - Replaced hardcoded URL string "http://sc-sdgx2-484:5000/generate" with
#     `url = sys.argv[1]` so the endpoint is configurable at runtime.
#     The hardcoded hostname was a hostname-specific bug; any deployment on a
#     different host would silently hit the wrong server.
#
# Usage:
#   python deepspeed/tools/run_cli.py http://<host>:<port>/api
#
# ===========================================================================

import json
import sys  # M729: added — required for sys.argv[1] URL argument
import urllib.request  # Python-3 equivalent of Python-2 urllib2

# M835: import generate_and_post_process (replaces legacy generate)
from deepspeed.compile.inference_api import generate_and_post_process

print('[M835]')


class PutRequest(urllib.request.Request):
    """HTTP PUT request wrapper (mirrors Megatron tools/run_text_generation_server.py)."""

    def get_method(self, *args, **kwargs):
        return 'PUT'


if __name__ == "__main__":
    # M729: use sys.argv[1] instead of hardcoded hostname
    url = sys.argv[1]
    while True:
        # M835: use "prompts" and "tokens_to_generate" (new server API)
        sentence = input("Enter prompt: ")
        tokens_to_generate = int(input("Enter number tokens to generate: "))
        data = json.dumps({"prompts": [sentence],
                           "tokens_to_generate": tokens_to_generate}).encode("utf-8")
        req = PutRequest(url, data,
                         {'Content-Type': 'application/json'})
        response = urllib.request.urlopen(req)
        resp = json.load(response)
        print("Megatron Response: ")
        # M835: response key is now "text" (was "sentences")
        print(resp["text"][0])
