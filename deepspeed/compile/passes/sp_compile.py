import logging
import operator
from collections import deque
from typing import Optional, List, Callable, NamedTuple

import torch
import deepspeed.comm as dist

logger = logging.getLogger(__name__)
from torch._subclasses.fake_tensor import FakeTensorMode, maybe_get_fake_mode
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from deepspeed.compile import constants

from ..custom_ops import all_to_all, sp_dp_registry  # noqa: F401
from ..fx import find_node_by_name, get_node_shape_meta
from ..util import get_input_id_node, get_label_id_node, get_position_id_node, shard_tensor_node, get_sdpa_nodes


class AutoSPInputs(NamedTuple):
    input_id: torch.Tensor
    label_id: torch.Tensor
    position_id: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]


def prepare_autosp_inputs(input_id: torch.Tensor,
                          label_id: torch.Tensor,
                          position_id: torch.Tensor = None,
                          attention_mask: torch.Tensor = None,
                          seq_dim: int = 1) -> AutoSPInputs:

    if input_id is None:
        raise ValueError("input_id is required")
    if label_id is None:
        raise ValueError("label_id is required")

    if seq_dim < 0 or seq_dim >= input_id.ndim:
        raise ValueError(f"seq_dim {seq_dim} must be a valid index for input_id with shape {input_id.shape}")

    if position_id is not None:
        if seq_dim >= position_id.ndim:
            raise ValueError(f"seq_dim {seq_dim} is out of bounds for position_id with shape {position_id.shape}")

    if attention_mask is not None:
        if seq_dim >= attention_mask.ndim:
            raise ValueError(
                f"seq_dim {seq_dim} is out of bounds for attention_mask with shape {attention_mask.shape}")

    torch._dynamo.decorators.mark_dynamic(input_id, seq_dim)
    torch._dynamo.decorators.mark_dynamic(label_id, seq_dim)
    if position_id is not None:
        torch._dynamo.decorators.mark_dynamic(position_id, seq_dim)
    if attention_mask is not None:
        torch._dynamo.decorators.mark_dynamic(attention_mask, seq_dim)

    input_id.tag = constants.AUTOSP_INPUT_ID_KEY
    label_id.tag = constants.AUTOSP_LABEL_ID_KEY
    if position_id is not None:
        position_id.tag = constants.AUTOSP_POSITION_ID_KEY

    return AutoSPInputs(input_id, label_id, position_id, attention_mask)


def pass_shard_seq_dim(gm: GraphModule, example_inputs):
    sp_size = sp_dp_registry.sp_size()

    input_ids_node = get_input_id_node(gm)
    val = get_node_shape_meta(input_ids_node)
    seq_symint = val.shape[1]
    assert isinstance(
        seq_symint,
        torch.SymInt), f"expected sequence dimension to be of type {torch.SymInt!r} but found {type(seq_symint)!r}"

    sym_seq_dim_node = find_node_by_name(gm, str(seq_symint))
    if sym_seq_dim_node is None:
        logger.warning("Could not find the symbolic node for the sequence dimension")
        return

    with gm.graph.inserting_after(sym_seq_dim_node):
        sharded_node = gm.graph.call_function(operator.floordiv, args=(sym_seq_dim_node, sp_size))

    sharded_input_nodes = set()
    label_ids_node = get_label_id_node(gm)
    position_ids_node = get_position_id_node(gm)

    if input_ids_node is not None:
        sharded_input_nodes.add(input_ids_node)
    if label_ids_node is not None:
        sharded_input_nodes.add(label_ids_node)
    if position_ids_node is not None:
        sharded_input_nodes.add(position_ids_node)

    consumer_nodes = set()
    worklist = deque(sharded_input_nodes)
    visited = set()

    while worklist:
        node = worklist.popleft()
        if node in visited:
            continue
        visited.add(node)
        consumer_nodes.add(node)

        for user in node.users:
            if user not in visited:
                worklist.append(user)

    to_replace = []
    for node in consumer_nodes:
        if sym_seq_dim_node in node.all_input_nodes:
            to_replace.append(node)

    for user in to_replace:
        user.replace_input_with(sym_seq_dim_node, sharded_node)


def pass_shard_input_ids(gm: GraphModule, example_inputs):
    input_ids_node = get_input_id_node(gm)
    shard_tensor_node(gm, input_ids_node)


def pass_shard_label_ids(gm: GraphModule, example_inputs):
    label_ids_node = get_label_id_node(gm)
    shard_tensor_node(gm, label_ids_node)


def pass_shard_position_ids(gm: GraphModule, example_inputs):
    position_ids_node = get_position_id_node(gm)
    if position_ids_node is None:
        logger.warning("position id node not found. Skipping sharding of position ids.")
        return
    shard_tensor_node(gm, position_ids_node)


def pass_insert_attention_all_to_all(gm: GraphModule, real_inputs):

    def insert_a2a(node: Node, scatter_idx: int, gather_idx: int, name: str) -> Node:
        with gm.graph.inserting_after(node):
            a2a_node = gm.graph.call_function(
                torch.ops.autosp.all_to_all.default,
                args=(node, scatter_idx, gather_idx, name),
            )
            a2a_node.name = f"a2a_{name}"
            node.replace_all_uses_with(a2a_node)
            a2a_node.update_arg(0, node)
        return a2a_node

    attention_nodes = get_sdpa_nodes(gm)
    if len(attention_nodes) == 0:
        raise RuntimeError("AutoSP currently supports torch.nn.functional.scaled_dot_product_attention as the "
                           "attention backend. No SDPA attention operations were found in the compiled graph. "
                           "Please ensure your model uses torch.nn.functional.scaled_dot_product_attention "
                           "for AutoSP to work as expected.")

    for idx, attn_node in enumerate(attention_nodes):
        q, k, v = attn_node.args[:3]
        suffix = f"_{idx}" if len(attention_nodes) > 1 else ""

        insert_a2a(q, scatter_idx=1, gather_idx=2, name=f"q{suffix}")
        insert_a2a(k, scatter_idx=1, gather_idx=2, name=f"k{suffix}")
        insert_a2a(v, scatter_idx=1, gather_idx=2, name=f"v{suffix}")

        insert_a2a(attn_node, scatter_idx=2, gather_idx=1, name=f"o{suffix}")


def pass_canonicalize(gm: GraphModule, real_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def pass_propagate_shapes(gm: torch.fx.GraphModule, real_inputs):
    fake_mode = None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_mode = maybe_get_fake_mode(fake_val)
        elif fake_mode is None:
            fake_val = node.meta.get("example_value", node.meta.get("val"))
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_mode = maybe_get_fake_mode(fake_val)
        if fake_mode is not None:
            break

    if fake_mode is None:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    _placeholder_dtypes = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val") or node.meta.get("example_value")
            if val is not None and isinstance(val, torch.Tensor) and val.is_floating_point():
                _placeholder_dtypes.append(val.dtype)
            else:
                _placeholder_dtypes.append(None)

    fake_inputs = []
    for idx, t in enumerate(real_inputs):
        if isinstance(t, torch.Tensor):
            target_dtype = _placeholder_dtypes[idx] if idx < len(_placeholder_dtypes) else None
            if (target_dtype is not None
                    and t.is_floating_point()
                    and t.dtype != target_dtype):
                t = t.to(target_dtype)
            fake_inputs.append(fake_mode.from_tensor(t))
        else:
            fake_inputs.append(t)

    saved_sdpa_masks = []
    for attn_node in get_sdpa_nodes(gm):
        attn_mask = attn_node.kwargs.get("attn_mask")
        if attn_mask is not None:
            saved_sdpa_masks.append((attn_node, attn_mask))
            attn_node.update_kwarg("attn_mask", None)

    _snapshot_ops = {"call_function", "call_method"}
    meta_snapshot = {node.name: dict(node.meta) for node in gm.graph.nodes
                     if node.op in _snapshot_ops}

    prop = FakeTensorProp(gm, mode=fake_mode)
    try:
        prop.propagate_dont_convert_inputs(*fake_inputs)
    except Exception:
        for node in gm.graph.nodes:
            if node.name in meta_snapshot:
                node.meta = meta_snapshot[node.name]
        raise
    finally:
        for attn_node, attn_mask in saved_sdpa_masks:
            attn_node.update_kwarg("attn_mask", attn_mask)


def apply_autosp(gm: GraphModule,
                 real_inputs,
                 debug: bool = False,
                 passes: Optional[List[Callable]] = None,
                 sp_size: int = 2,
                 dp_size: int = 1):

    assert sp_size * dp_size <= dist.get_world_size(), 'Insufficient device count for mesh size'

    if not sp_dp_registry.is_setup():
        sp_dp_registry.populate_registry(sp_size, dp_size)

    AUTOSP_PASSES = [
        pass_shard_seq_dim,
        pass_shard_input_ids,
        pass_shard_label_ids,
        pass_shard_position_ids,
        pass_insert_attention_all_to_all,
        pass_propagate_shapes,
        pass_canonicalize,
    ]

    passes = passes or AUTOSP_PASSES
    rank = dist.get_rank()

    _n_sdpa = len(get_sdpa_nodes(gm))
    if _n_sdpa == 0:
        raise RuntimeError(
            "[AutoSP] No SDPA nodes in graph. Model must use "
            "F.scaled_dot_product_attention (set _attn_implementation='sdpa').")
    if rank == 0:
        first_sdpa = get_sdpa_nodes(gm)[0]
        for arg_idx, arg_name in [(0, "Q"), (1, "K"), (2, "V")]:
            if arg_idx < len(first_sdpa.args):
                node = first_sdpa.args[arg_idx]
                meta = node.meta.get("val") or node.meta.get("example_value")
                if meta is not None and hasattr(meta, 'shape') and len(meta.shape) >= 2:
                    n_heads = meta.shape[1]
                    if isinstance(n_heads, int) and n_heads % sp_size != 0:
                        raise RuntimeError(
                            f"[AutoSP] {arg_name} n_heads={n_heads} not divisible "
                            f"by sp_size={sp_size}. For GQA models, "
                            f"set sp_size to a divisor of num_kv_heads={n_heads}.")

    for p in passes:
        if debug and rank == 0:
            print(f"\n{'='*60}")
            print(f" BEFORE: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))

        p(gm, real_inputs)

        if debug and rank == 0:
            print(f"\n{'='*60}")
            print(f" AFTER: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))

    a2a_count = sum(1 for n in gm.graph.nodes
                    if n.op == "call_function"
                    and n.target is torch.ops.autosp.all_to_all.default)
    expected = _n_sdpa * 4
    if a2a_count != expected:
        raise RuntimeError(
            f"[AutoSP] A2A count mismatch: got {a2a_count}, expected {expected} "
            f"(4 x {_n_sdpa} SDPA). NCCL deadlock will occur.")