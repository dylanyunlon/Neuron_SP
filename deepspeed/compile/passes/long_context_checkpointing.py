# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import inspect
import logging
import textwrap
import torch._functorch.partitioners as _partitioners

logger = logging.getLogger(__name__)

_CUSTOM_SHOULD_BAN = """\
def should_ban_recomputation(node):
    if node.op != "call_function":
        return None
    if node.target == operator.getitem:
        return None
    if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
        return "autosp: MUST_SAVE policy"
    if config.recompute_views and op_types.is_view(node):
        return None
    if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
        return None

    must_save_set = [
        aten.convolution,
        aten.convolution_backward,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._flash_attention_forward,
        aten._efficient_attention_forward,
        aten.upsample_bilinear2d,
        aten.native_dropout,
        aten.rand_like,
        aten.randn_like,
    ]

    if get_aten_target(node) in must_save_set:
        return "autosp: attention/stochastic op"

    if hasattr(node.target, '__module__') and 'autosp' in str(node.target):
        return "autosp: collective op"

    def heuristic(node):
        if "val" in node.meta:
            if isinstance(node.meta["val"], torch.Tensor) and node.meta["val"].dim() >= 2:
                return node.meta["val"].shape[1] >= 4096
        return False

    if min_cut_options.ban_if_not_in_allowlist:
        if not op_types.is_recomputable(node):
            return None

    if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(node):
        if heuristic(node):
            return None
        return "autosp: materialized backward (small tensor)"

    if node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw:
        return None

    if min_cut_options.ban_if_reduction:
        input_tensors_size = sum(
            _size_of(i) for i in node.args if isinstance(i, fx.Node)
        )
        output_size = _size_of(node)
        if output_size * 4 < input_tensors_size:
            return "autosp: reduction op"
    return None
"""

_NEEDLE = '    def should_ban_recomputation('

_ORIGINAL_SOLVE_MIN_CUT = _partitioners.solve_min_cut


def restore_default_checkpointing():
    _partitioners.solve_min_cut = _ORIGINAL_SOLVE_MIN_CUT


def register_long_context_checkpointing():
    try:
        src = inspect.getsource(_partitioners.solve_min_cut)
    except (OSError, TypeError):
        logger.warning("AutoSP: could not retrieve source for solve_min_cut; "
                       "selective activation checkpointing disabled.")
        return

    if 'def should_ban_recomputation(' not in src:
        logger.warning(
            f"AutoSP: PyTorch {__import__('torch').__version__} changed "
            f"solve_min_cut signature. Selective activation checkpointing disabled.")
        return

    lines = src.split('\n')

    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith(_NEEDLE) or line.lstrip().startswith('def should_ban_recomputation('):
            if start is None:
                start = i
        elif start is not None and end is None and line.startswith('    def '):
            end = i

    if start is None or end is None:
        logger.warning(
            "AutoSP: solve_min_cut structure does not match expected pattern; "
            "selective activation checkpointing disabled.")
        return

    replacement = textwrap.indent(_CUSTOM_SHOULD_BAN, '    ')

    new_src = '\n'.join(lines[:start]) + '\n' + replacement + '\n'.join(lines[end:])

    original_solve_min_cut = _partitioners.solve_min_cut
    try:
        exec(new_src, _partitioners.__dict__)
    except Exception as e:
        _partitioners.solve_min_cut = original_solve_min_cut
        logger.warning(f"AutoSP: failed to inject custom checkpointing policy: {e}. "
                       "Falling back to default PyTorch checkpointing.")
