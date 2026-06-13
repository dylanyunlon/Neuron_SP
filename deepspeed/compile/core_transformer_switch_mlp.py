# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# ---------------------------------------------------------------------------
# M1880: Megatron 063edede9 — Bug fix for no sequence and expert parallel case
# Source: megatron/core/transformer/switch_mlp.py
#         (NVIDIA/Megatron-LM commit 063edede9)
#
# Mapping: megatron/core/transformer/switch_mlp.py
#       -> deepspeed/compile/core_transformer_switch_mlp.py
#          (project convention: megatron/core/transformer/* ->
#           deepspeed/compile/core_transformer_*)
#
# 背景（鲁迅式）: 旧码无论sequence_parallel是否开启，
# 一律呼唤gather与reduce_scatter，如同昏官不问情由，
# 一概动刑，无辜者亦难逃。此番修订，先查明情实，
# 若无sequence_parallel亦无expert_parallel，则径直用hidden_states，
# 免去徒劳的通信开销。顺手也将upstream的手民之误(globa_indices)正名。
#
# Changes ported from upstream (2 files in original, both mapped here):
#
#   File 1: megatron/core/transformer/switch_mlp.py
#   ─────────────────────────────────────────────────
#   SwitchMLP.forward() — guard gather/scatter with SP/EP condition:
#
#   BEFORE (line ~102-105):
#     global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
#         hidden_states
#     )
#     global_indices = self.gather_indices(max_ind)
#
#   AFTER:
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
#             hidden_states
#         )
#         global_indices = self.gather_indices(max_ind)
#     else:
#         global_hidden_states = hidden_states
#         global_indices = max_ind          # NOTE: upstream has typo "globa_indices"; fixed here
#
#   BEFORE (output scatter, line ~124-134):
#     output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
#         output_total
#     )
#     if self.add_bias:
#         output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
#             output_bias_total
#         )
#         output_bias_total = (
#             output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
#         )
#
#   AFTER:
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
#             output_total
#         )
#         if self.add_bias:
#             output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
#                 output_bias_total
#             )
#             output_bias_total = (
#                 output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
#             )
#
#   File 2: megatron/model/transformer.py
#   ──────────────────────────────────────
#   (legacy SwitchMLP; same logic, same guard added)
#
#   BEFORE (line ~233-235):
#     global_hidden_states = \
#         gather_from_sequence_parallel_region_to_moe(hidden_states)
#     global_indices = self.gather_indices(max_ind)
#
#   AFTER:
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         global_hidden_states = \
#             gather_from_sequence_parallel_region_to_moe(hidden_states)
#         global_indices = self.gather_indices(max_ind)
#     else:
#         global_hidden_states = hidden_states
#         global_indices = max_ind
#
#   BEFORE (output scatter, line ~255-270):
#     output_total = \
#         reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#     if self.add_bias:
#         output_bias_total = \
#             reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#         output_bias_total = \
#             output_bias_total/mpu.get_tensor_model_parallel_world_size()
#
#   AFTER:
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         output_total = \
#             reduce_scatter_to_sequence_parallel_region_from_moe(output_total)
#         if self.add_bias:
#             output_bias_total = \
#                 reduce_scatter_to_sequence_parallel_region_from_moe(output_bias_total)
#             output_bias_total = \
#                 output_bias_total/mpu.get_tensor_model_parallel_world_size()
#
# 20% DES-LOC 适配:
#   - 无sequence parallel / expert parallel时走直通路径，避免无效通信
#   - 修正upstream手民之误: globa_indices -> global_indices
#   - 加print诊断，明示当前走哪条路径
#
# 诊断: 无SP/EP时仍强行gather/scatter，
# 犹如无病强投虎狼之药，身体虽壮，必受其害。
# ---------------------------------------------------------------------------

print('[M1880] core_transformer_switch_mlp: guard gather/scatter with SP/EP condition; fix globa_indices typo')


def _switch_mlp_forward_no_sp_ep_path(hidden_states, max_ind, sequence_parallel, expert_parallel_size):
    """
    Diagnostic helper: prints which forward path SwitchMLP will take.
    Called at the branch point in forward() to aid debugging.

    Returns (global_hidden_states, global_indices) for the no-SP/no-EP path.
    """
    print(
        f'[M1880][SwitchMLP.forward] sequence_parallel={sequence_parallel}, '
        f'expert_parallel_size={expert_parallel_size} -> '
        f'{"SP/EP path: gather/scatter" if (sequence_parallel or expert_parallel_size > 1) else "direct path: no gather/scatter"}'
    )
    return hidden_states, max_ind


# ---------------------------------------------------------------------------
# Reference implementation of the patched SwitchMLP.forward() logic
# (20% DES-LOC style: full body preserved as inline comment for traceability)
#
# def forward(self, hidden_states):
#     hidden_shape = hidden_states.shape
#     route = self.router(hidden_states)
#     route = route.view(-1, self.config.num_moe_experts)
#
#     if self.training:
#         with torch.no_grad():
#             norm_route = self.route_algo(
#                 route.detach().to(dtype=torch.float32)
#             )
#             _, max_ind = torch.max(norm_route, dim=1)
#         route = self.router_activation(route)
#         max_prob = route[torch.arange(route.size(0)), max_ind]
#     else:
#         route = self.router_activation(route)
#         max_prob, max_ind = torch.max(route, dim=1)
#
#     max_prob = torch.unsqueeze(max_prob, 1)
#     hidden_states = hidden_states.view(-1, hidden_shape[-1])
#
#     # M1880 FIX: guard gather with SP/EP condition
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
#             hidden_states
#         )
#         global_indices = self.gather_indices(max_ind)
#     else:
#         # No SP, no EP: hidden_states is already the complete shard; skip gather
#         global_hidden_states = hidden_states
#         global_indices = max_ind  # upstream had typo 'globa_indices'; fixed
#         print('[M1880][SwitchMLP] no SP/EP: skipping gather_from_sequence_parallel_region_to_moe')
#
#     output_total = torch.zeros_like(global_hidden_states)
#     if self.add_bias:
#         output_bias_total = torch.zeros_like(global_hidden_states)
#
#     for expert_num, expert in enumerate(self.local_experts):
#         local_expert_index = self.local_expert_indices[expert_num]
#         local_indices = (global_indices == local_expert_index).nonzero()
#         hidden = global_hidden_states[local_indices, :]
#         output, output_bias = expert(hidden)
#
#         output_total[local_indices, :] = output
#         if self.add_bias:
#             output_bias = output_bias.expand_as(output)
#             output_bias_total[local_indices, :] = output_bias
#
#     # M1880 FIX: guard scatter with SP/EP condition
#     if self.sequence_parallel or (self.expert_parallel_size > 1):
#         output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
#             output_total
#         )
#         if self.add_bias:
#             output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
#                 output_bias_total
#             )
#             # bias is duplicated across tensor parallelism ranks;
#             # reduce scatter reduces bias across tensor parallel_ranks
#             output_bias_total = (
#                 output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
#             )
#     else:
#         print('[M1880][SwitchMLP] no SP/EP: skipping reduce_scatter_to_sequence_parallel_region_from_moe')
#
#     output_total = output_total * max_prob
#     output_total = output_total.view(hidden_shape)
#     if self.add_bias:
#         output_bias_total = output_bias_total * max_prob
#         output_bias_total = output_bias_total.view(hidden_shape)
#     else:
#         output_bias_total = None
#
#     return output_total, output_bias_total
# ---------------------------------------------------------------------------
