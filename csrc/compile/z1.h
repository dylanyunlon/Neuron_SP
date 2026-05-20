// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "deepcompile.h"

#pragma once

namespace dc {

void register_graph_z1(long graph_id, const std::vector<long>& ds_ids);
void register_graph_z1_sp(long graph_id,
                          const std::vector<long>& ds_ids,
                          int sp_size,
                          int kx,
                          int warmup_steps);
void register_param(long ds_id,
                    const std::vector<int64_t>& ds_shape,
                    at::Tensor ds_tensor,
                    at::Tensor grad_buffer,
                    int64_t offset);
void register_param_sp(long ds_id,
                       const std::vector<int64_t>& ds_shape,
                       at::Tensor ds_tensor,
                       at::Tensor grad_buffer,
                       int64_t offset,
                       int sp_group_id);
}  // namespace dc
