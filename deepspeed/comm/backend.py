# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
DeepSpeed Communication Backend.

In the future, directly use NCCL/MPI/Gloo/etc without requiring torch.distributed. Simply wrap torch.distributed for now.

# Custom DS Backends -- Direct C/Ops
 - NCCL -- [EXPERIMENTAL]
 - MPI -- [EXPERIMENTAL]
 - RCCL -- [EXPERIMENTAL]
 - GLOO -- [EXPERIMENTAL]
# DS backend wrapper for torch.distributed [DEFAULT]
 - T-NCCL -- [DEFAULT]
 - T-GLOO
 - T-MPI

# Backend is the base class
     -- NcclBackend, MpiBackend, and TorchBackend are the main subclasses. TorchBackend is the only officially supported backend for now.
"""


class Backend(object):

    def __init__(self, name='backend', rank=0, size=1):
        self.name = name
        # The world size and rank of the world process group
        self.world_group = None
        self.world_size = size
        self.world_rank = rank
        # Single process group (pg) implementation for now but keep a list for future
        self.process_groups = []
        self.initialized = False

    def is_initialized(self):
        return self.initialized

    def new_group(self):
        # create a new pg and add it to pg list
        pass

    def init_process_group(self):
        # subclasses will initialize them fully
        # - initialize a default world process group and add it to pg list
        self.initialized = True


# =========================================================================
# DES-LOC Backend Mixin (Algorithm 1 communication tier awareness)
# =========================================================================

class DeslocBackendMixin:
    """Mixin adding DES-LOC tier awareness to any comm backend.
    Ref: Section 4.1 - Ring-AllReduce with independent sync periods."""

    def init_desloc(self, Kx=1, Ku=3, Kv=6):
        self._desloc_Kx = Kx
        self._desloc_Ku = Ku
        self._desloc_Kv = Kv
        self._desloc_step = 0
        self._desloc_bytes_sent = 0
        self._desloc_bytes_skipped = 0

    def desloc_should_comm(self, tier):
        """Check if communication should happen for given tier at current step."""
        period = {0: self._desloc_Kx, 1: self._desloc_Ku,
                  2: self._desloc_Kv}.get(tier, 1)
        if period <= 1:
            return True
        return self._desloc_step % period == 0

    def desloc_advance(self):
        self._desloc_step += 1

    def desloc_stats(self):
        return {
            'step': self._desloc_step,
            'bytes_sent': self._desloc_bytes_sent,
            'bytes_skipped': self._desloc_bytes_skipped,
        }


# =========================================================================
# M212: DES-LOC Communication Scheduler (Algorithm 1, lines 9-21)
# =========================================================================

import math
import time
from collections import defaultdict, deque


