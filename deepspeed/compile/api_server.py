# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

# ===========================================================================
# M835: Megatron a5bfc2966 — added new inference to the server
# ===========================================================================
#
# Upstream source:
#   megatron/text_generation_server.py
#   (NVIDIA/Megatron-LM commit a5bfc296648b8c77374d7df0176d304b4d5ea421)
#   Author: mshoeybi <mshoeybi@nvidia.com>  Date: 2021-10-10
#
# Mapping: megatron/text_generation_server.py
#          → deepspeed/compile/api_server.py
#
# Summary of changes ported from upstream (on top of M745):
#
#   Imports:
#     - Remove `from megatron.text_generation_utils import generate`.
#     - Add `from megatron.inference.api import generate_and_post_process`
#       (mapped here to deepspeed.compile.inference_api).
#
#   MegatronGenerate.put():
#     - Replace "sentences" key with "prompts"; return 400 if "sentences"
#       or "max_len" keys are present (deprecated API guard).
#     - Replace "max_len" field with "tokens_to_generate".
#     - Add "logprobs" bool parameter (was hardcoded False).
#     - Add "temperature" float parameter parsed from request JSON.
#     - Add "add_BOS" bool parameter parsed from request JSON.
#     - Replace generate() call with generate_and_post_process() unpacking
#       5-tuple: response, response_seg, response_logprobs, _, _.
#       New kwargs: return_output_log_probs=logprobs, return_all_log_probs=False,
#       greedy_sampling=args.greedy, top_k_sampling=top_k,
#       top_p_sampling=top_p, temperature=temperature, add_BOS=add_BOS,
#       use_eod_token_for_early_termination=True.
#     - Return JSON key "text" (was "sentences"), keep "segments", "logprobs".
#
#   MegatronServer.__init__():
#     - Register route at '/api' instead of '/generate' to match upstream.
#
#   MegatronServer.run():
#     - Keep threaded=False (M729 fix retained; upstream switched to
#       threaded=True but we preserve the distributed-safe serialisation).
#
# ===========================================================================
#
# ===========================================================================
# M745: Megatron ddd361450 — Got the probs piped
# ===========================================================================
#
# Upstream source:
#   megatron/api_server.py
#   (NVIDIA/Megatron-LM commit ddd3614509bf2d974567513434aeca8bd256f610)
#   Author: rprenger <rprenger@nvidia.com>  Date: 2021-08-11
#
# Mapping: megatron/api_server.py
#          → deepspeed/compile/api_server.py
#
# Summary of changes ported from upstream (on top of M729):
#
#   MegatronGenerate.put():
#     - Parse optional "all_probs" bool from JSON request body; return
#       "all_probs must be a boolean value" on type error.
#     - Pass all_probs to generate().
#     - Unpack four-tuple (resp_sentences, resp_sentences_seg,
#       output_logits, full_logits) from generate().
#     - When all_probs is True, return additional "all_logits" key in the
#       JSON response containing full_logits.
#
# ===========================================================================
#
# ===========================================================================
# M729: Megatron 7a9c4a03f — Removing bug possibilities and adding timing info
# ===========================================================================
#
# Upstream source:
#   megatron/api_server.py
#   (NVIDIA/Megatron-LM commit 7a9c4a03fdbc5a235e47feac29839a733101c0c5)
#   Author: rprenger <rprenger@nvidia.com>  Date: 2021-07-19
#
# Mapping: megatron/api_server.py
#          → deepspeed/compile/api_server.py
#
# Changes ported from upstream:
#   MegatronServer.run():
#     - Added threaded=False to self.app.run() call.
#       The original app.run(url, debug=False) used Flask's default
#       threaded=True, which caused race conditions when multiple clients
#       hit /generate simultaneously while generate() relies on
#       torch.distributed collective calls that must be called in lockstep
#       across all ranks.  Setting threaded=False ensures requests are
#       serialised and the distributed barrier semantics are preserved.
#
# DeepSpeed adaptation notes:
#   - megatron.* imports are replaced with deepspeed.compile stubs.
#   - generate_and_post_process() is imported from
#     deepspeed.compile.inference_api (M835).
# ===========================================================================

import datetime
import json
import threading

import torch
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api

from deepspeed.compile.inference_api import generate_and_post_process

print('[M729]')
print('[M745]')
print('[M835]')

GENERATE_NUM = 0
lock = threading.Lock()


class MegatronGenerate(Resource):
    """Flask-RESTful resource that exposes the /api endpoint.

    Megatron a5bfc2966 text_generation_server.py — put() now calls
    generate_and_post_process() instead of the legacy generate().
    Request body uses "prompts" (was "sentences") and "tokens_to_generate"
    (was "max_len").  Old keys are rejected with a 400 to force callers
    to migrate.
    """

    def __init__(self, model, get_args_fn=None, mpu_mod=None):
        self.model = model
        self._get_args = get_args_fn
        self._mpu = mpu_mod

    @staticmethod
    def send_do_generate(mpu_mod):
        """Broadcast the GENERATE_NUM choice to all tensor-parallel ranks."""
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(
            choice,
            mpu_mod.get_tensor_model_parallel_src_rank(),
            group=mpu_mod.get_tensor_model_parallel_group())

    def put(self):
        args = self._get_args() if self._get_args else None

        print("request IP: " + str(request.remote_addr))
        print(json.dumps(request.get_json()), flush=True)
        print("current time: ", datetime.datetime.now())

        # M835: "max_len" and "sentences" are deprecated; reject them so
        # callers update to the new API.
        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400
        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        if "prompts" not in request.get_json():
            return "prompts argument required", 400

        prompts = request.get_json()["prompts"]
        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400

        # M835: tokens_to_generate replaces max_len
        tokens_to_generate = 64  # sane default; full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 1:
                return "tokens_to_generate must be an integer greater than 0"

        # M835: logprobs bool
        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"

        # M835: temperature float
        temperature = args.temperature if args is not None else 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) in (int, float)):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"

        # M835: top_k int
        top_k = args.top_k if args is not None else 0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not isinstance(top_k, int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"

        # M835: top_p float
        top_p = args.top_p if args is not None else 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not isinstance(top_p, float):
                return "top_p must be a positive float less than or equal to 1.0"
            if not (0.0 < top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"

        # M835: add_BOS bool
        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"

        with lock:  # Need to get lock to keep multiple threads from hitting code
            MegatronGenerate.send_do_generate(self._mpu)  # Tell other ranks we're doing generate
            # M835: replaced generate() with generate_and_post_process()
            response, response_seg, response_logprobs, _, _ = \
                generate_and_post_process(
                    self.model,
                    prompts=prompts,
                    tokens_to_generate=tokens_to_generate,
                    return_output_log_probs=logprobs,
                    return_all_log_probs=False,
                    greedy_sampling=args.greedy if args is not None else False,
                    top_k_sampling=top_k,
                    top_p_sampling=top_p,
                    temperature=temperature,
                    add_BOS=add_BOS,
                    use_eod_token_for_early_termination=True)

        return jsonify({"text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs})


def index():
    return current_app.send_static_file('index.html')


class MegatronServer(object):
    """Thin Flask wrapper around MegatronGenerate.

    Megatron a5bfc2966 text_generation_server.py — endpoint is '/api'.
    M729 threaded=False retained for distributed-safe serialisation.
    """

    def __init__(self, model, get_args_fn=None, mpu_mod=None):
        self.app = Flask(__name__, static_url_path='')
        self.app.add_url_rule('/', 'index', index)
        api = Api(self.app)
        # M835: route changed to '/api' (upstream text_generation_server.py)
        api.add_resource(
            MegatronGenerate, '/api',
            resource_class_args=[model],
            resource_class_kwargs={'get_args_fn': get_args_fn,
                                   'mpu_mod': mpu_mod})

    def run(self, url):
        # M729: threaded=False retained — prevents race conditions in
        # distributed collective calls when multiple HTTP requests arrive
        # concurrently.  (Upstream a5bfc2966 uses threaded=True but the
        # DeepSpeed path preserves this safety fix.)
        self.app.run(url, threaded=False, debug=False)
