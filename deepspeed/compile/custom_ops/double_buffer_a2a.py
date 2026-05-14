import threading
from typing import Optional, Tuple
from dataclasses import dataclass

import torch


@dataclass
class BufferSlot:
    data: Optional[torch.Tensor] = None
    index: Optional[torch.Tensor] = None
    valid: bool = False
    numel: int = 0


class DoubleBufferA2A:

    def __init__(self, max_elements: int, dtype: torch.dtype = torch.bfloat16,
                 index_dtype: torch.dtype = torch.long,
                 device: Optional[torch.device] = None):
        self._max_elements = max_elements
        self._dtype = dtype
        self._index_dtype = index_dtype
        self._device = device or (torch.device(f"cuda:{torch.cuda.current_device()}")
                                  if torch.cuda.is_available() else torch.device("cpu"))
        self._selector = 0
        self._slots = [BufferSlot(), BufferSlot()]
        self._lock = threading.Lock()
        self._allocated = False

    def allocate(self, shape: Tuple[int, ...]):
        with self._lock:
            if self._allocated:
                return
            for i in range(2):
                self._slots[i].data = torch.empty(shape, dtype=self._dtype, device=self._device)
                self._slots[i].index = torch.empty(shape[0], dtype=self._index_dtype, device=self._device)
                self._slots[i].numel = shape[0]
            self._allocated = True

    def current(self) -> BufferSlot:
        return self._slots[self._selector]

    def alternate(self) -> BufferSlot:
        return self._slots[self._selector ^ 1]

    def swap(self):
        with self._lock:
            self._selector ^= 1

    def current_data(self) -> Optional[torch.Tensor]:
        return self._slots[self._selector].data

    def alternate_data(self) -> Optional[torch.Tensor]:
        return self._slots[self._selector ^ 1].data

    def current_index(self) -> Optional[torch.Tensor]:
        return self._slots[self._selector].index

    def alternate_index(self) -> Optional[torch.Tensor]:
        return self._slots[self._selector ^ 1].index

    def mark_valid(self, slot_id: int = -1):
        if slot_id < 0:
            slot_id = self._selector
        self._slots[slot_id].valid = True

    def is_valid(self, slot_id: int = -1) -> bool:
        if slot_id < 0:
            slot_id = self._selector
        return self._slots[slot_id].valid

    def invalidate(self, slot_id: int = -1):
        if slot_id < 0:
            slot_id = self._selector
        self._slots[slot_id].valid = False

    def free(self):
        with self._lock:
            for s in self._slots:
                s.data = None
                s.index = None
                s.valid = False
                s.numel = 0
            self._allocated = False
            self._selector = 0

    @property
    def selector(self) -> int:
        return self._selector

    @property
    def allocated(self) -> bool:
        return self._allocated


class A2ABufferPool:

    def __init__(self):
        self._buffers = {}
        self._lock = threading.Lock()

    def get_or_create(self, key: str, max_elements: int,
                      dtype: torch.dtype = torch.bfloat16,
                      device: Optional[torch.device] = None) -> DoubleBufferA2A:
        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = DoubleBufferA2A(
                    max_elements=max_elements,
                    dtype=dtype,
                    device=device,
                )
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

    def __contains__(self, key: str) -> bool:
        return key in self._buffers

    def __len__(self) -> int:
        return len(self._buffers)


_GLOBAL_BUFFER_POOL: Optional[A2ABufferPool] = None


def get_buffer_pool() -> A2ABufferPool:
    global _GLOBAL_BUFFER_POOL
    if _GLOBAL_BUFFER_POOL is None:
        _GLOBAL_BUFFER_POOL = A2ABufferPool()
    return _GLOBAL_BUFFER_POOL


def execute_double_buffered_a2a(input_tensor: torch.Tensor,
                                scatter_idx: int,
                                gather_idx: int,
                                sp_size: int,
                                group,
                                pass_index: int,
                                pool: Optional[A2ABufferPool] = None) -> torch.Tensor:
    import deepspeed.comm as comm

    pool = pool or get_buffer_pool()
    B, dim1, dim2, H = input_tensor.shape

    buf_key = f"a2a_pass_{pass_index % 2}"
    buf = pool.get_or_create(buf_key, max_elements=input_tensor.numel(),
                             dtype=input_tensor.dtype)

    if not buf.allocated:
        buf.allocate(input_tensor.shape)

    write_slot = buf.alternate_data()
    if write_slot is None or write_slot.shape != input_tensor.shape:
        buf.free()
        buf.allocate(input_tensor.shape)
        write_slot = buf.alternate_data()

    if scatter_idx == 1:
        N, local_S = dim1, dim2
        input_t = input_tensor.reshape(B, sp_size, N // sp_size, local_S, H)
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        output = torch.empty_like(input_t)
        comm.all_to_all_single(output, input_t, group=group)
        result = output.permute(1, 2, 0, 3, 4).contiguous()
        result = result.reshape(B, N // sp_size, sp_size * local_S, H)
    else:
        local_N, S = dim1, dim2
        input_t = input_tensor.reshape(B, local_N, sp_size, S // sp_size, H)
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
        output = torch.empty_like(input_t)
        comm.all_to_all_single(output, input_t, group=group)
        result = output.permute(1, 0, 2, 3, 4).contiguous()
        result = result.reshape(B, sp_size * local_N, S // sp_size, H)

    write_slot.copy_(result.reshape(write_slot.shape) if write_slot.shape == result.shape
                     else result.view(-1)[:write_slot.numel()].reshape(write_slot.shape))
    buf.mark_valid(buf.selector ^ 1)
    buf.swap()

    return result
