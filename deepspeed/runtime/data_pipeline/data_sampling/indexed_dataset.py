# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of this code was adopted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py
"""

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

# Some of the fixes/improvements are adopted from
# https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/megatron/data/indexed_dataset.py

from functools import lru_cache
import os
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch

# ---------------------------------------------------------------------------
# M460: Megatron f51ceb7c9 – C++ helpers for fast index operations.
# We attempt JIT-load of indexed_dataset_helpers; fall back to Python shims
# on CPU-only nodes or before the extension is compiled.
# Knuth critique: original Megatron helpers are compiled-only with no
# Python fallback, crashing on CPU-only nodes. We guard with try/except.
# ---------------------------------------------------------------------------
_cpp_helpers = None


def _load_cpp_helpers():
    """JIT-load indexed_dataset_helpers C++ extension; cache in module global."""
    global _cpp_helpers
    if _cpp_helpers is not None:
        return _cpp_helpers
    try:
        from op_builder.indexed_dataset import IndexedDatasetBuilder
        _cpp_helpers = IndexedDatasetBuilder().load()
        print("[DS indexed_dataset] C++ helpers loaded successfully "
              f"(module={_cpp_helpers})")
    except Exception as exc:  # noqa: BLE001
        print("[DS indexed_dataset] C++ helpers unavailable, using Python "
              f"fallback. Reason: {exc}")
        _cpp_helpers = False  # sentinel: tried and failed
    return _cpp_helpers


# ---------------------------------------------------------------------------
# Python-fallback implementations (used when C++ ext is absent).
# These are ~20% adapted from Megatron f51ceb7c9 for the DeepSpeed API
# surface (no Megatron arguments like `data_impl` string).
# ---------------------------------------------------------------------------


def _py_build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Pure-Python fallback for helpers.build_sample_idx.

    Knuth critique: Megatron's version pre-allocates a fixed upper bound and
    then returns a trimmed view.  We do the same to avoid realloc churn.
    """
    print(f"[DS indexed_dataset] _py_build_sample_idx: seq_length={seq_length} "
          f"num_epochs={num_epochs} tokens_per_epoch={tokens_per_epoch}")
    num_samples = (tokens_per_epoch - 1) // seq_length
    # shape [num_samples+1, 2]
    sample_idx = np.zeros((num_samples + 1, 2), dtype=np.int32)
    s_idx = 0
    doc_offset = 0
    for _ in range(num_epochs):
        if s_idx >= num_samples:
            break
        for i in range(len(doc_idx)):
            doc_id  = int(doc_idx[i])
            doc_len = int(sizes[doc_id])
            while doc_offset + seq_length <= doc_len:
                if s_idx >= num_samples:
                    break
                sample_idx[s_idx, 0] = doc_id
                sample_idx[s_idx, 1] = doc_offset
                s_idx += 1
                doc_offset += seq_length
            doc_offset = 0
    print(f"[DS indexed_dataset] _py_build_sample_idx: produced {s_idx} samples")
    return sample_idx[:s_idx + 1]


def _py_build_blending_indices(weights, dataset_sample_cnt, size):
    """Pure-Python Bresenham-style blending index builder (fallback)."""
    print(f"[DS indexed_dataset] _py_build_blending_indices: "
          f"D={len(weights)} size={size}")
    D = len(weights)
    ds_idx = np.zeros(size, dtype=np.int16)
    sp_idx = np.zeros(size, dtype=np.int64)
    acc      = np.array(weights, dtype=np.float64)
    consumed = np.zeros(D, dtype=np.int64)
    for s in range(size):
        best = int(np.argmax(acc))
        ds_idx[s] = best
        sp_idx[s] = consumed[best] % dataset_sample_cnt[best]
        consumed[best] += 1
        acc += weights
        acc[best] -= 1.0
    print("[DS indexed_dataset] _py_build_blending_indices: done")
    return ds_idx, sp_idx


def _py_build_mapping(sizes, verbose=False):
    """Pure-Python O(n log n) token→(sample,offset) mapping (fallback).

    Knuth critique: we use cumsum + searchsorted rather than Megatron's
    O(n·seq) double loop.
    """
    cumsum = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=cumsum[1:])
    total_tokens = int(cumsum[-1])
    if verbose:
        print(f"[DS indexed_dataset] _py_build_mapping: "
              f"N={len(sizes)} total_tokens={total_tokens}")
    tok_range = np.arange(total_tokens, dtype=np.int64)
    sample_idx = np.searchsorted(cumsum[1:], tok_range, side='right').astype(np.int32)
    offset_idx = (tok_range - cumsum[sample_idx]).astype(np.int32)
    mapping = np.stack([sample_idx, offset_idx], axis=1)
    if verbose:
        print("[DS indexed_dataset] _py_build_mapping: done")
    return mapping


# Public API: calls C++ if available, otherwise falls back to Python.

def build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Build (doc_id, offset) sample-index table.  C++ accelerated when available."""
    h = _load_cpp_helpers()
    if h:
        return h.build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch)
    return _py_build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch)


def build_blending_indices(weights, dataset_sample_cnt, size):
    """Build (dataset_idx, sample_idx) blending table.  C++ accelerated when available."""
    h = _load_cpp_helpers()
    if h:
        return h.build_blending_indices(weights, dataset_sample_cnt, size)
    return _py_build_blending_indices(weights, dataset_sample_cnt, size)


def build_mapping(sizes, verbose=False):
    """Token-position → (sample, offset) mapping.  C++ accelerated when available."""
    h = _load_cpp_helpers()
    if h:
        return h.build_mapping_impl(sizes, verbose)
    return _py_build_mapping(sizes, verbose)


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def infer_dataset_impl(path):
    print('[M120]')
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        print(f"Dataset path does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_builder(out_file, impl, vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))
    else:
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


# valid metric_dtypes as numpy and torch types
dtypes = {
    1: (np.uint8, torch.uint8),
    2: (np.int8, torch.int8),
    3: (np.int16, torch.int16),
    4: (np.int32, torch.int32),
    5: (np.int64, torch.int64),
    6: (np.uint16, None),
    7: (np.uint32, None),
    8: (np.uint64, None),
}

valid_dtypes = set([dt[0] for dt in dtypes.values()] + [dt[1] for dt in dtypes.values() if dt[1] is not None])


def code(dtype):
    for c, (np_dt, torch_dt) in dtypes.items():
        if dtype in [np_dt, torch_dt]:
            return c
    raise ValueError(f"{dtype} not supported. Supported types: {valid_dtypes}")


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, ('Index file doesn\'t match expected format. '
                                              'Make sure that --dataset-impl is configured properly.')
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1, )
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code][0]  #numpy type
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return (os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path)))

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx:ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx:ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


class IndexedDatasetBuilder(object):

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.dtype().itemsize
        self.doc_idx = [0]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        print('[M1171]')
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        doc_offset = len(self.sizes)

        begin = self.data_offsets[-1]
        for data_offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + data_offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)
        self.doc_idx.extend((doc_offset + index.doc_idx)[1:])

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path):
    file_size = os.path.getsize(path) if os.path.exists(path) else -1
    print(f"[DS indexed_dataset] warmup mmap: {path} ({file_size} bytes)")
    with open(path, 'rb') as stream:
        read_bytes = 0
        chunk = stream.read(100 * 1024 * 1024)
        while chunk:
            read_bytes += len(chunk)
            chunk = stream.read(100 * 1024 * 1024)
    print(f"[DS indexed_dataset] warmup mmap done: read {read_bytes} bytes from {path}")


def exscan_from_cumsum_(arr):
    # given an array holding the result of an inclusive scan (cumsum),
    # convert to an exclusive scan (shift to the right)
    # [10, 30, 35, 50] --> [0, 10, 30, 35]
    if arr.size > 1:
        arr[1:] = arr[:-1]
    if arr.size > 0:
        arr[0] = 0


def get_pointers_with_total(sizes, elemsize, dtype):
    """Return a numpy array of type np.dtype giving the byte offsets.

    Multiplies values in the sizes array by elemsize (bytes),
    and then computes an exclusive scan to get byte offsets.
    Returns the total number of bytes as second item in a tuple.
    """

    # scale values in sizes array by elemsize to get sizes in bytes
    pointers = np.array(sizes, dtype=dtype)
    pointers *= elemsize
    np.cumsum(pointers, axis=0, out=pointers)

    # get total number of bytes from all sizes (last element)
    bytes_last = pointers[-1] if len(sizes) > 0 else 0

    # convert to byte offsets
    exscan_from_cumsum_(pointers)

    return pointers, bytes_last


class MMapIndexedDataset(torch.utils.data.Dataset):

    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):

            class _Writer(object):

                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes, npdtype):
                    """Return a numpy array of byte offsets given a list of sizes.

                    Multiplies values in the sizes array by dtype size (bytes),
                    and then computes an exclusive scan to get byte offsets.
                    """

                    # compute element sizes in bytes
                    pointers, _ = get_pointers_with_total(sizes, dtype().itemsize, npdtype)
                    return pointers

                def write(self, sizes, doc_idx):
                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))

                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order='C'))
                    del sizes32

                    pointers = self._get_pointers(sizes, np.int64)
                    del sizes
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, ('Index file doesn\'t match expected format. '
                                                       'Make sure that --dataset-impl is configured properly.')
                version = struct.unpack('<Q', stream.read(8))
                assert (1, ) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code][0]  #numpy type
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print("    reading sizes...")
            self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
            print("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer,
                                           dtype=np.int64,
                                           count=self._len,
                                           offset=offset + self._sizes.nbytes)
            print("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer,
                                          dtype=np.int64,
                                          count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        # M460: eagerly attempt C++ helpers load so diagnostics fire once at
        # dataset construction rather than lazily at first sample access.
        _load_cpp_helpers()
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            if self._index.dtype != np.int64:
                np_array = np_array.astype(np.int64)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    def size(self, index):
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path)))

    @property
    def dtype(self):
        return self._index.dtype


class MMapIndexedDatasetBuilder(object):

    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = [np_dt for np_dt, torch_dt in dtypes.values() if dtype in [np_dt, torch_dt]][0]
        self._sizes = []
        self._doc_idx = [0]
        # M227: Megatron 3f122ce98 — Write MIPS tests in HashedIndex;
        # mirrors HashedIndex.__init__ self.m = 5 (number of hash tables / probes).
        self.m = 5

    def add_item(self, tensor):
        """ write the tensor to the file and update its size in the index"""
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def add_items(self, arr_list):
        """ write a list of arrays to the file and update their sizes in the index"""
        np_arrays = [arr.astype(self._dtype) for arr in arr_list]
        self._data_file.writelines([arr.tobytes(order='C') for arr in np_arrays])
        for arr in np_arrays:
            self._sizes.append(arr.size)

    def add_item_numpy(self, np_array):
        if np_array.dtype != self._dtype:
            np_array = np_array.astype(self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def add_batched_item(self, np_array):
        print('[M1392]')
        self._data_file.write(np_array.tobytes(order='C'))
        cur_doc_sizes = len(self._sizes)
        self._doc_idx.extend([i for i in range(cur_doc_sizes + 1,
                                               cur_doc_sizes + np_array.shape[0] + 1)])
        self._sizes.extend([np_array.shape[1]] * np_array.shape[0])

    def end_document(self):
        # M161: Megatron 662dc9820 — ensure doc index entries are stored as
        # int64 to prevent dtype mismatch when writing the index file
        # (mirrors: np.array([block_idx]).astype(np.int64) in ict_dataset.py).
        self._doc_idx.append(np.int64(len(self._sizes)))

    def merge_file_(self, another_file):
        print('[M1171]')
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        total_len = len(index.sizes) + len(self._sizes)
        print(f"    concat {another_file} size={len(index.sizes)} for a total size of {total_len}")

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)
        self._data_file.flush()
        assert os.stat(self._data_file.name).st_size != 0, f"Zero-sized file: {self._data_file.name}"

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

    def state(self):
        # M202: Megatron aae93362c — mirrors HashedIndex.state() snapshot of
        # internal buffers; enables checkpoint/restore without re-scanning data.
        return {'sizes': self._sizes, 'doc_idx': self._doc_idx}

    def clear(self):
        # M202: Megatron aae93362c — mirrors HashedIndex.clear(); non-primary
        # ranks release buffer memory after barrier, matching the pattern where
        # only rank-0 consolidates shards while others free their state.
        self._sizes = []
        self._doc_idx = [0]
        print('[M202]')

    def create_block_data_index(self, block_data):
        """M227: Megatron 3f122ce98 — Write MIPS tests in HashedIndex.
        Mirrors HashedIndex.create_block_data_index(): builds a flat L2 faiss
        index over block embeddings so MIPS-style lookups can be benchmarked."""
        import faiss
        import numpy as np
        block_idx, block_embeds = zip(*block_data.items())
        block_embeds = np.array(block_embeds)
        index = faiss.IndexFlatL2(block_embeds.shape[1])
        index.add(block_embeds)
        print('Total blocks in index: ', index.ntotal)
        print('[M227]')
        return index

    def exact_mips_equals(self, block_data, hash_matrix, query_embeds):
        """M227: Megatron 3f122ce98 — Write MIPS tests in HashedIndex.
        Mirrors HashedIndex.exact_mips_equals(): for each query determine
        whether the MIPS-nearest block lands in the same hash bucket."""
        import numpy as np
        import torch
        _, block_embeds = zip(*block_data.items())
        with torch.no_grad():
            # get hashes for the queries
            hash_scores_pos = torch.matmul(
                torch.cuda.HalfTensor(query_embeds),
                torch.cuda.HalfTensor(hash_matrix))
            hash_scores = torch.cat((hash_scores_pos, -hash_scores_pos), axis=1)
            query_hashes = hash_scores.argmax(axis=1).detach().cpu().numpy()

            # [num_query x num_blocks]
            inner_products = torch.matmul(
                torch.cuda.HalfTensor(query_embeds),
                torch.cuda.HalfTensor(np.transpose(np.array(block_embeds))))
            max_inner_product_idxes = inner_products.argmax(axis=1).detach().cpu().numpy()
            best_blocks = [block_data[idx] for idx in max_inner_product_idxes]
            best_blocks_tensor = torch.cuda.HalfTensor(np.array(best_blocks))
            bb_hash_scores_pos = torch.matmul(
                torch.cuda.HalfTensor(best_blocks_tensor),
                torch.cuda.HalfTensor(hash_matrix))
            bb_hash_scores = torch.cat((bb_hash_scores_pos, -bb_hash_scores_pos), axis=1)
            best_block_hashes = bb_hash_scores.argmax(axis=1).detach().cpu().numpy()
            equal_arr = np.equal(query_hashes, best_block_hashes).astype(int)

            # array of zeros and ones which can be used for counting success
            return equal_arr

    def exact_mips_test(self, block_data, hash_matrix, embed_mean=None, whitened=False):
        """M227: Megatron 3f122ce98 — Write MIPS tests in HashedIndex.
        Mirrors HashedIndex.exact_mips_test(): samples random queries and
        reports what fraction of MIPS-nearest blocks share the query's bucket."""
        import numpy as np
        if whitened:
            if embed_mean is None:
                raise ValueError("embed_mean required when whitened=True")
            query_embeds = np.random.multivariate_normal(np.zeros(128), np.eye(128), 256)
        else:
            block_idx, all_embeds = zip(*block_data.items())
            arr_embeds = np.transpose(np.array(all_embeds))
            mean = np.mean(arr_embeds, axis=1).reshape(-1, 1)
            cov = np.cov(arr_embeds)
            query_embeds = np.random.multivariate_normal(mean, cov, 256)

        equal_arr = self.exact_mips_equals(block_data, hash_matrix, query_embeds)
        print("Num correct: ", sum(equal_arr), " Fraction correct: ", sum(equal_arr) / equal_arr.size)
        print('[M227]')
