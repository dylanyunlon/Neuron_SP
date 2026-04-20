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



# M292 — Claude-19: HealthMon + FaultTolerantReducer
class DeslocHealthMon:
    __slots__=('gh','fl','cf','mcf','ns','st','lc')
    def __init__(s,ci=100,mcf=5):s.gh={};s.fl=[];s.cf=0;s.mcf=mcf;s.ns=[];s.st=s.lc=0
    def ok(s,st,g=0,ms=0,b=0):s.st=st;s.cf=0;s.gh.setdefault(g,{'ok':0,'fail':0,'t':0,'b':0})['ok']+=1;h=s.gh[g];h['t']+=ms;h['b']+=b
    def fail(s,st,g=0,et='',em=''):s.st=st;s.cf+=1;s.gh.setdefault(g,{'ok':0,'fail':0,'t':0,'b':0})['fail']+=1;s.fl.append({'st':st,'g':g,'et':et});s.fl=s.fl[-500:]if len(s.fl)>1000 else s.fl
    def nan(s,st,n=''):s.ns.append((st,n));s.ns=s.ns[-500:]if len(s.ns)>1000 else s.ns
    def check(s,st):
        r={'status':'healthy','issues':[]}
        if s.cf>=s.mcf:r['status']='failed';r['issues'].append(f"{s.cf} consecutive failures")
        for g,h in s.gh.items():
            t=h['ok']+h['fail']
            if t>10 and h['fail']/t>.1:r['status']='degraded';r['issues'].append(f"GPU{g}: {h['fail']}/{t} fail")
        rn=[x for x,_ in s.ns if x>st-100]
        if len(rn)>5:r['status']='critical';r['issues'].append(f"{len(rn)} NaN")
        return r
class DeslocFTReducer:
    __slots__=('mr','bt','tm','sb','sk','hm','st')
    def __init__(s,mr=3,bt=5000,tm=2.,sb=3):s.mr=mr;s.bt=bt;s.tm=tm;s.sb=sb;s.sk=0;s.hm=DeslocHealthMon();s.st={'t':0,'ok':0,'rt':0,'sk':0,'f':0}
    def reduce(s,t,st,g=0):
        import time;s.st['t']+=1
        for a in range(s.mr+1):
            try:
                import torch.distributed as dist
                if not dist.is_initialized():return True
                t0=time.monotonic();dist.all_reduce(t);w=dist.get_world_size();t.div_(w)
                s.hm.ok(st,g,(time.monotonic()-t0)*1000,t.numel()*t.element_size());s.st['ok']+=1;s.sk=0;return True
            except RuntimeError as e:
                s.hm.fail(st,g,'RT',str(e))
                if a<s.mr:time.sleep(s.bt*(s.tm**a)/1000);s.st['rt']+=1
        if s.sk<s.sb:s.sk+=1;s.st['sk']+=1;return False
        s.st['f']+=1;return False
# M292: end
