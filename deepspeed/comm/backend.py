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



# M307: Health + NaN + watchdog
class DeslocHealthMon:
    def __init__(self, w=100): self._w = w; self._l = []; self._nan = 0; self._ops = 0; self._ok = True
    def record(self, lat_us, nan=False):
        self._ops += 1; self._l.append(lat_us)
        if len(self._l) > self._w: self._l.pop(0)
        if nan: self._nan += 1
        if self._ops > 10: self._ok = self._nan / self._ops < 0.05
    def avg(self): return sum(self._l) / max(1, len(self._l)) if self._l else 0
    def p99(self):
        if not self._l: return 0
        s = sorted(self._l); return s[min(int(len(s) * 0.99), len(s) - 1)]
    def ok(self): return self._ok
    def stats(self): return {'ok': self._ok, 'ops': self._ops, 'nan': self._nan, 'avg': round(self.avg(), 2), 'p99': round(self.p99(), 2)}

class DeslocNaNDet:
    def __init__(self, freq=1): self._f = freq; self._s = 0; self._ev = []
    def check(self, t, lbl='?'):
        self._s += 1
        if self._s % self._f != 0: return True
        try:
            if bool(t.isnan().any()) or bool(t.isinf().any()): self._ev.append({'s': self._s, 'l': lbl}); return False
        except: pass
        return True
    def events(self): return list(self._ev)

class DeslocWatchdog:
    def __init__(self, base=300): self._b = base; self._t = []; self._on = False; self._ns = 0
    def start(self): import time; self._on = True; self._ns = time.perf_counter_ns()
    def stop(self):
        import time
        if not self._on: return 0
        e = (time.perf_counter_ns() - self._ns) / 1e9; self._on = False
        self._t.append(e)
        if len(self._t) > 50: self._t.pop(0)
        return e
    def timeout(self): return max(self._b, sum(self._t) / max(1, len(self._t)) * 10) if len(self._t) >= 5 else self._b
    def expired(self):
        import time
        return self._on and (time.perf_counter_ns() - self._ns) / 1e9 > self.timeout()
# --- End M307 ---
