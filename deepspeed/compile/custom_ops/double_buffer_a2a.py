import threading
from typing import Optional, Tuple

import torch
from .sp_dp_registry import track_buffer_event


class DoubleBuffer:

    def __init__(self, dtype=torch.bfloat16, index_dtype=torch.long, device=None):
        self._dtype = dtype
        self._index_dtype = index_dtype
        self._device = device or (torch.device(f"cuda:{torch.cuda.current_device()}")
                                  if torch.cuda.is_available() else torch.device("cpu"))
        self.selector = 0
        self._data = [None, None]
        self._index = [None, None]
        self._valid = [False, False]
        self._allocated = False
        self._lock = threading.Lock()
        self._swap_count = 0

    def allocate(self, shape, dtype=None):
        with self._lock:
            if dtype is not None and dtype != self._dtype:
                self._dtype = dtype
                if self._allocated:
                    self._free_unlocked()
            if self._allocated:
                if self._data[0] is not None and self._data[0].shape == shape:
                    return
                self._free_unlocked()
            numel = 1
            for s in shape:
                numel *= s
            elem_bytes = torch.tensor([], dtype=self._dtype).element_size()
            buf_bytes = numel * elem_bytes * 2
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info(self._device)[0]
                if buf_bytes > free_mem * 0.3:
                    raise RuntimeError(
                        f"[DoubleBuffer] buffer {buf_bytes/(1024**2):.0f}MB exceeds "
                        f"30% of free GPU memory {free_mem/(1024**2):.0f}MB. "
                        f"Reduce batch_size or enable cpu_offload.")
            for i in range(2):
                self._data[i] = torch.empty(shape, dtype=self._dtype, device=self._device)
                self._index[i] = torch.empty(shape[0], dtype=self._index_dtype, device=self._device)
            self._allocated = True
            track_buffer_event("created")

    def current(self):
        return self._data[self.selector]

    def alternate(self):
        return self._data[self.selector ^ 1]

    def current_index(self):
        return self._index[self.selector]

    def alternate_index(self):
        return self._index[self.selector ^ 1]

    def swap(self):
        with self._lock:
            self.selector ^= 1
            self._swap_count += 1
            track_buffer_event("swapped")

    def swap_count(self):
        return self._swap_count

    def mark_valid(self, slot=-1):
        if slot < 0:
            slot = self.selector
        self._valid[slot] = True

    def is_valid(self, slot=-1):
        if slot < 0:
            slot = self.selector
        return self._valid[slot]

    def invalidate(self, slot=-1):
        if slot < 0:
            slot = self.selector
        self._valid[slot] = False

    def _free_unlocked(self):
        for i in range(2):
            self._data[i] = None
            self._index[i] = None
            self._valid[i] = False
        self._allocated = False
        self.selector = 0
        self._swap_count = 0
        track_buffer_event("freed")

    def free(self):
        with self._lock:
            self._free_unlocked()

    @property
    def allocated(self):
        return self._allocated


class BufferPool:

    def __init__(self):
        self._buffers = {}
        self._lock = threading.Lock()

    def get_or_create(self, key, dtype=torch.bfloat16, device=None):
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = DoubleBuffer(dtype=dtype, device=device)
            return self._buffers[key]

    def swap_all(self):
        with self._lock:
            for buf in self._buffers.values():
                buf.swap()

    def free_all(self):
        with self._lock:
            for buf in self._buffers.values():
                buf.free()
            self._buffers.clear()

    def keys(self):
        return list(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def __len__(self):
        return len(self._buffers)


_GLOBAL_POOL = None


def get_buffer_pool():
    global _GLOBAL_POOL
    if _GLOBAL_POOL is None:
        _GLOBAL_POOL = BufferPool()
    return _GLOBAL_POOL


def _raw_a2a(input_tensor, scatter_idx, sp_size_val, group):
    import deepspeed.comm as comm
    from .all_to_all import _SCATTER_HEADS, _SCATTER_SEQ

    B, dim1, dim2, H = input_tensor.shape
    plan = _SCATTER_HEADS if scatter_idx == 1 else _SCATTER_SEQ
    P = sp_size_val

    input_t = input_tensor.reshape(*plan["pre_reshape"](B, P, dim1, dim2, H))
    input_t = input_t.permute(*plan["pre_permute"]).contiguous()
    output = torch.empty_like(input_t)
    comm.all_to_all_single(output, input_t, group=group)
    output = output.permute(*plan["post_permute"]).contiguous()
    return output.reshape(*plan["post_reshape"](B, P, dim1, dim2, H))


def execute_double_buffered_a2a(input_tensor, scatter_idx, gather_idx,
                                 sp_size_val, group, pass_index,
                                 is_last_pass, pool=None):
    pool = pool or get_buffer_pool()

    buf_key = f"a2a_pass_{pass_index % 2}"
    buf = pool.get_or_create(buf_key, dtype=input_tensor.dtype)

    if not buf.allocated or buf.current().shape != input_tensor.shape:
        buf.free()
        buf.allocate(input_tensor.shape, dtype=input_tensor.dtype)

    result = _raw_a2a(input_tensor, scatter_idx, sp_size_val, group)

    write_slot = buf.alternate()
    if write_slot is not None and write_slot.shape == result.shape:
        write_slot.copy_(result)
    buf.mark_valid(buf.selector ^ 1)
    buf.swap()

    if is_last_pass:
        idx_buf = pool.get_or_create(f"idx_{buf_key}", dtype=torch.long)
        if idx_buf.allocated:
            idx_buf.swap()

    return result
