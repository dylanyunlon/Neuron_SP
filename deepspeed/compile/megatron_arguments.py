# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M343: Megatron 0403b8081 — added gpu initialization and option to avoid
#       master values
# Source: megatron/arguments.py (NVIDIA/Megatron-LM commit 0403b8081)
# Author: Mohammad Shoeybi <mshoeybi@nvidia.com>  Date: 2020-08-03
#
# Mapping: megatron/arguments.py → deepspeed/compile/megatron_arguments.py
#          (project convention: megatron top-level → deepspeed/compile/)
#
# Changes ported from arguments.py:
#   1. import torch added at module level.
#   2. parse_args(): after dynamic_loss_scale block, add params_dtype
#      assignment:
#        args.params_dtype = torch.float
#        if args.fp16: args.params_dtype = torch.half
#        if args.rank == 0: print('using {} for parameters ...')
#
# 20% adaptation: deepspeed uses ds_config.fp16.enabled rather than
# argparse args.fp16; _GLOBAL_ARGS singleton pattern used for get_args();
# adds print('[M343]') marker.
# ---------------------------------------------------------------------------

print('[M343]')

import torch

_GLOBAL_ARGS = None


def get_args():
    """Return the global args object.

    Megatron 0403b8081 arguments.py — global accessor used by mpu/layers.py
    to retrieve params_dtype without passing args through every call site.
    """
    return _GLOBAL_ARGS


def set_args(args):
    """Set the global args object.

    Called once during initialize_megatron so that downstream modules
    (mpu/layers.py) can access params_dtype via get_args().
    """
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    print(f'[M343] set_args: params_dtype={getattr(args, "params_dtype", None)}')


def set_params_dtype(args):
    """Set args.params_dtype based on fp16 flag.

    Megatron 0403b8081 arguments.py parse_args():
      args.params_dtype = torch.float
      if args.fp16:
          args.params_dtype = torch.half
      if args.rank == 0:
          print('using {} for parameters ...'.format(args.params_dtype), flush=True)

    Called after dynamic_loss_scale is resolved.
    """
    args.params_dtype = torch.float
    if getattr(args, 'fp16', False):
        args.params_dtype = torch.half
    rank = getattr(args, 'rank', 0)
    if rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype), flush=True)
    print(f'[M343] set_params_dtype: params_dtype={args.params_dtype}')
    return args
