import logging
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import deepspeed.comm as dist

from .sp_dp_registry import (
    is_setup, sp_size, dp_size, get_group,
    is_loc_enabled, get_loc_sp_group_ids,
    fence_all_sp_handles, track_a2a_handle,
)

logger = logging.getLogger(__name__)


class LOCSPShardingAdapter(nn.Module):

    def __init__(self, sp_world_size, loc_group=None):
        super().__init__()
        self._sp_world_size = sp_world_size
        self._loc_group = loc_group
        self._scatter_cache = {}

    def scatter_to_sp_region(self, hidden_states, seq_dim=1):
        total_seq = hidden_states.shape[seq_dim]
        assert total_seq % self._sp_world_size == 0, (
            f"seq_len={total_seq} not divisible by sp_size={self._sp_world_size}")

        local_seq = total_seq // self._sp_world_size
        rank = dist.get_rank()
        sp_rank = rank % self._sp_world_size
        start = sp_rank * local_seq
        end = start + local_seq

        return hidden_states.narrow(seq_dim, start, local_seq).contiguous()

    def gather_from_sp_region(self, hidden_states, seq_dim=1):
        local_seq = hidden_states.shape[seq_dim]
        gathered = [torch.empty_like(hidden_states) for _ in range(self._sp_world_size)]

        rank = dist.get_rank()
        gid = rank // self._sp_world_size
        group = get_group(gid) if is_setup() else None

        dist.all_gather(gathered, hidden_states.contiguous(), group=group)

        return torch.cat(gathered, dim=seq_dim)

    def forward(self, hidden_states, mode="scatter", seq_dim=1):
        if mode == "scatter":
            return self.scatter_to_sp_region(hidden_states, seq_dim)
        return self.gather_from_sp_region(hidden_states, seq_dim)


class LOCRemoteSequentialSPWrapper(nn.Module):

    def __init__(self, remote_sequential, sp_world_size, loc_group=None):
        super().__init__()
        self.remote_sequential = remote_sequential
        self.adapter = LOCSPShardingAdapter(sp_world_size, loc_group)
        self._sp_world_size = sp_world_size

    def forward(self, inputs, prompts=None, **kwargs):
        assert inputs.ndim == 3, (
            f"inputs must be [batch_size, seq_length, hidden_size], got {inputs.shape}")

        sharded = self.adapter.scatter_to_sp_region(inputs, seq_dim=1)

        output = self.remote_sequential(sharded, prompts=prompts, **kwargs)

        gathered = self.adapter.gather_from_sp_region(output, seq_dim=1)

        return gathered


class LOCSPGradientBridge:

    def __init__(self, sp_world_size, kx=1, warmup_steps=512):
        self._sp_world_size = sp_world_size
        self._kx = kx
        self._warmup_steps = warmup_steps
        self._step = 0
        self._synced_count = 0
        self._skipped_count = 0

    def should_sync(self):
        if self._kx <= 1:
            return True
        if self._step < self._warmup_steps:
            return True
        return (self._step % self._kx) == 0

    def pre_backward_fence(self):
        if is_setup():
            fence_all_sp_handles()

    def post_backward_reduce(self, model):
        self._step += 1
        if not self.should_sync():
            self._skipped_count += 1
            return False

        if is_setup():
            fence_all_sp_handles()

        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        self._synced_count += 1
        return True

    def state_dict(self):
        return {
            'step': self._step,
            'kx': self._kx,
            'warmup_steps': self._warmup_steps,
            'synced': self._synced_count,
            'skipped': self._skipped_count,
        }

    def load_state_dict(self, sd):
        self._step = sd.get('step', 0)
        self._kx = sd.get('kx', self._kx)
        self._warmup_steps = sd.get('warmup_steps', self._warmup_steps)
        self._synced_count = sd.get('synced', 0)
        self._skipped_count = sd.get('skipped', 0)


def create_loc_sp_wrapper(remote_sequential, config_param_dict):
    sp = config_param_dict.get('sequence_parallel_size', 1)
    loc_cfg = config_param_dict.get('loc', {})
    desloc_cfg = config_param_dict.get('desloc', {})
    kx = desloc_cfg.get('Kx', 1)
    warmup = desloc_cfg.get('warmup', 512)

    n_heads = config_param_dict.get('n_heads',
                config_param_dict.get('num_attention_heads', 0))
    if n_heads > 0 and sp > 1 and n_heads % sp != 0:
        raise ValueError(
            f"[LOC+SP] n_heads={n_heads} not divisible by sp_size={sp}. "
            f"A2A scatter on head dim will produce incorrect shapes.")

    loc_group = None
    if is_loc_enabled():
        gids = get_loc_sp_group_ids()
        if gids:
            loc_group = get_group(gids[0])

    wrapper = LOCRemoteSequentialSPWrapper(
        remote_sequential, sp, loc_group)

    grad_bridge = LOCSPGradientBridge(sp, kx=kx, warmup_steps=warmup)

    logger.info(
        f"[LOC+SP] wrapper sp={sp} kx={kx} warmup={warmup} "
        f"loc={is_loc_enabled()} "
        f"model={loc_cfg.get('model_size', '7B')} n_heads={n_heads}")

    return wrapper, grad_bridge
