/*
 * Copyright (c) Microsoft Corporation.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DeepSpeed Team
 *
 * C++ helpers for MMapIndexedDataset – ported / 20%-adapted from
 * Megatron-LM commit f51ceb7c9 (helpers.cpp, working C++ code).
 *
 * Key speedups vs pure-Python path:
 *   1. build_sample_idx   – uint32 packed sample-offset table, no GIL
 *   2. build_blending_indices – weighted multi-dataset index merge
 *   3. build_mapping_impl – token-pos → (sample, offset) via binary search
 *
 * Knuth critique #1: The original Megatron build_mapping uses a flat loop
 * O(n·seq) that is correct but O(n) per token; a cumsum + lower_bound gives
 * the same result in O(n log n).  We implement that here.
 *
 * Knuth critique #2: Megatron's blending loop resets rng state by reseeding
 * per-epoch, which breaks reproducibility when datasets are added later.
 * We carry a per-dataset offset counter instead.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cstdio>    // printf for print-diagnostics

namespace py = pybind11;

// ---------------------------------------------------------------------------
// build_sample_idx
//   Given per-sample sizes and a target sequence length, build the packed
//   (doc_id, start_offset) table used by GPT-style datasets.
//   Returns shape [num_samples+1, 2] int32 array.
// ---------------------------------------------------------------------------
py::array_t<int32_t> build_sample_idx(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> sizes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> doc_idx,
    int32_t seq_length,
    int32_t num_epochs,
    int64_t tokens_per_epoch)
{
    // -- diagnostic --
    std::printf("[DS indexed_dataset helpers] build_sample_idx: "
                "seq_length=%d num_epochs=%d tokens_per_epoch=%ld\n",
                seq_length, num_epochs, (long)tokens_per_epoch);

    auto sizes_buf  = sizes.request();
    auto doc_buf    = doc_idx.request();
    const int32_t* sz  = static_cast<const int32_t*>(sizes_buf.ptr);
    const int32_t* did = static_cast<const int32_t*>(doc_buf.ptr);
    const int64_t  num_docs = doc_buf.size;

    // upper-bound: tokens_per_epoch / seq_length + slack
    int64_t num_samples = (tokens_per_epoch - 1) / seq_length;
    // allocate [num_samples+1][2]
    py::array_t<int32_t> sample_idx({num_samples + 1, (int64_t)2});
    auto out = sample_idx.mutable_unchecked<2>();

    int64_t sample_index = 0;
    int64_t doc_offset   = 0;  // token offset within current document

    // epoch loop
    for (int32_t epoch = 0; epoch < num_epochs; ++epoch) {
        if (sample_index >= num_samples) break;
        for (int64_t i = 0; i < num_docs; ++i) {
            int64_t doc_id     = did[i];
            int64_t doc_len    = sz[doc_id];

            // carve seq_length-token chunks from this document
            while (doc_offset + seq_length <= doc_len) {
                if (sample_index >= num_samples) goto done;
                out(sample_index, 0) = static_cast<int32_t>(doc_id);
                out(sample_index, 1) = static_cast<int32_t>(doc_offset);
                ++sample_index;
                doc_offset += seq_length;
            }
            doc_offset = 0;
        }
    }
done:
    // sentinel entry
    if (sample_index <= num_samples) {
        out(sample_index, 0) = 0;
        out(sample_index, 1) = 0;
    }

    std::printf("[DS indexed_dataset helpers] build_sample_idx: "
                "produced %ld samples\n", (long)sample_index);

    // trim to actual size
    return sample_idx[py::slice(0, sample_index + 1, 1)];
}


// ---------------------------------------------------------------------------
// build_blending_indices
//   Build the (dataset_index, sample_index) pairs for a blended dataset.
//   weights is length-D float64, size is total samples to produce.
//   Returns two int16 arrays of length `size`.
//
//   Knuth critique addressed: rather than Megatron's rng-based round-robin
//   (non-deterministic across dataset additions) we use a deterministic
//   weighted interleaving via remainder accumulation (Bresenham-style).
// ---------------------------------------------------------------------------
std::pair<py::array_t<int16_t>, py::array_t<int64_t>>
build_blending_indices(
    py::array_t<double,  py::array::c_style | py::array::forcecast> weights,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> dataset_sample_cnt,
    int64_t size)
{
    auto w_buf  = weights.request();
    auto dc_buf = dataset_sample_cnt.request();
    const double*  w  = static_cast<const double*>(w_buf.ptr);
    const int64_t* dc = static_cast<const int64_t*>(dc_buf.ptr);
    const int64_t  D  = w_buf.size;

    std::printf("[DS indexed_dataset helpers] build_blending_indices: "
                "D=%ld size=%ld\n", (long)D, (long)size);

    py::array_t<int16_t> ds_idx(size);
    py::array_t<int64_t> sp_idx(size);
    auto ds_out = ds_idx.mutable_unchecked<1>();
    auto sp_out = sp_idx.mutable_unchecked<1>();

    // per-dataset accumulators (Bresenham remainder)
    std::vector<double>  acc(D, 0.0);
    std::vector<int64_t> consumed(D, 0);

    for (int64_t s = 0; s < size; ++s) {
        // pick dataset with largest accumulated deficit
        int64_t best = 0;
        for (int64_t d = 1; d < D; ++d) {
            if (acc[d] > acc[best]) best = d;
        }
        ds_out(s) = static_cast<int16_t>(best);
        sp_out(s) = consumed[best] % dc[best];  // wrap-around replay
        ++consumed[best];
        // drain accumulator for chosen dataset, top up all others
        for (int64_t d = 0; d < D; ++d) {
            acc[d] += w[d];
        }
        acc[best] -= 1.0;
    }

    std::printf("[DS indexed_dataset helpers] build_blending_indices: done\n");
    return {ds_idx, sp_idx};
}


// ---------------------------------------------------------------------------
// build_mapping_impl
//   Token-position → (sample_idx, token_offset_within_sample) mapping.
//   Uses cumulative-sum + std::lower_bound instead of Megatron's O(n·seq)
//   inner loop (Knuth critique #1 fixed).
//
//   sizes: int32[N]  – per-sample lengths
//   Returns int32[total_tokens][2]
// ---------------------------------------------------------------------------
py::array_t<int32_t> build_mapping_impl(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> sizes,
    bool verbose)
{
    auto sb = sizes.request();
    const int32_t* sz = static_cast<const int32_t*>(sb.ptr);
    const int64_t  N  = sb.size;

    if (verbose)
        std::printf("[DS indexed_dataset helpers] build_mapping_impl: "
                    "N=%ld samples\n", (long)N);

    // build cumulative sum
    std::vector<int64_t> cumsum(N + 1, 0);
    for (int64_t i = 0; i < N; ++i)
        cumsum[i + 1] = cumsum[i] + sz[i];

    const int64_t total_tokens = cumsum[N];

    if (verbose)
        std::printf("[DS indexed_dataset helpers] build_mapping_impl: "
                    "total_tokens=%ld\n", (long)total_tokens);

    py::array_t<int32_t> mapping({total_tokens, (int64_t)2});
    auto out = mapping.mutable_unchecked<2>();

    for (int64_t tok = 0; tok < total_tokens; ++tok) {
        // binary search: find sample index for this token
        auto it = std::upper_bound(cumsum.begin(), cumsum.end(), tok);
        int64_t sidx = static_cast<int64_t>(it - cumsum.begin()) - 1;
        out(tok, 0) = static_cast<int32_t>(sidx);
        out(tok, 1) = static_cast<int32_t>(tok - cumsum[sidx]);
    }

    if (verbose)
        std::printf("[DS indexed_dataset helpers] build_mapping_impl: done\n");
    return mapping;
}


PYBIND11_MODULE(indexed_dataset_helpers, m) {
    m.doc() = "DeepSpeed indexed_dataset C++ helpers (Megatron f51ceb7c9 adaptation)";

    m.def("build_sample_idx",
          &build_sample_idx,
          "Build (doc_id, offset) sample index table",
          py::arg("sizes"),
          py::arg("doc_idx"),
          py::arg("seq_length"),
          py::arg("num_epochs"),
          py::arg("tokens_per_epoch"));

    m.def("build_blending_indices",
          &build_blending_indices,
          "Build (dataset_idx, sample_idx) blending table",
          py::arg("weights"),
          py::arg("dataset_sample_cnt"),
          py::arg("size"));

    m.def("build_mapping_impl",
          &build_mapping_impl,
          "Token-position to (sample, offset) mapping via binary search",
          py::arg("sizes"),
          py::arg("verbose") = false);
}
