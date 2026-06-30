import logging
import operator
import threading
from collections import deque
from typing import Optional, List, Callable, NamedTuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
from torch._subclasses.fake_tensor import FakeTensorMode, maybe_get_fake_mode
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from deepspeed.compile import constants

from ..custom_ops import all_to_all, sp_dp_registry
from ..fx import find_node_by_name, get_node_shape_meta
from ..util import get_input_id_node, get_label_id_node, get_position_id_node, shard_tensor_node, get_sdpa_nodes


class AutoSPInputs(NamedTuple):
    input_id: torch.Tensor
    label_id: torch.Tensor
    position_id: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]


def prepare_autosp_inputs(input_id: torch.Tensor,
                          label_id: torch.Tensor = None,
                          position_id: torch.Tensor = None,
                          attention_mask: torch.Tensor = None,
                          seq_dim: int = 1) -> AutoSPInputs:
    if input_id is None:
        raise ValueError("input_id is required")
    if seq_dim < 0 or seq_dim >= input_id.ndim:
        raise ValueError(f"seq_dim {seq_dim} must be a valid index for input_id with shape {input_id.shape}")
    if position_id is not None and seq_dim >= position_id.ndim:
        raise ValueError(f"seq_dim {seq_dim} is out of bounds for position_id with shape {position_id.shape}")
    if attention_mask is not None and seq_dim >= attention_mask.ndim:
        raise ValueError(f"seq_dim {seq_dim} is out of bounds for attention_mask with shape {attention_mask.shape}")

    for tensor in (input_id, label_id, position_id, attention_mask):
        if tensor is not None:
            torch._dynamo.decorators.mark_dynamic(tensor, seq_dim)

    input_id.tag = constants.AUTOSP_INPUT_ID_KEY
    if label_id is not None:
        label_id.tag = constants.AUTOSP_LABEL_ID_KEY
    if position_id is not None:
        position_id.tag = constants.AUTOSP_POSITION_ID_KEY
    if attention_mask is not None:
        attention_mask.tag = constants.AUTOSP_ATTENTION_MASK_KEY

    return AutoSPInputs(input_id, label_id, position_id, attention_mask)


def _collect_sharded_consumers(seed_nodes, sym_seq_dim_node):
    consumer_nodes = set()
    worklist = deque(seed_nodes)
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
    return to_replace


def pass_shard_seq_dim(gm: GraphModule, example_inputs):
    sp_sz = sp_dp_registry.sp_size()
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
        sharded_node = gm.graph.call_function(operator.floordiv, args=(sym_seq_dim_node, sp_sz))

    # Global replacement: every node in the graph that references the original
    # s0 symbolic dim should use s0//sp_size instead. This covers not just
    # input_ids consumers but also torch.arange(T) where T = input_ids.shape[1],
    # pos_embedding lookups, and any other op that derived a size from the
    # unsharded sequence length.
    #
    # We skip the sharded_node itself (it needs s0 as its own input) and
    # the SDPA attention nodes (A2A pass handles those separately).
    sdpa_nodes = set(get_sdpa_nodes(gm))
    for node in list(gm.graph.nodes):
        if node is sharded_node:
            continue
        if node in sdpa_nodes:
            continue
        if sym_seq_dim_node in node.all_input_nodes:
            node.replace_input_with(sym_seq_dim_node, sharded_node)


_SHARD_TARGETS = [
    (get_input_id_node, True),
    (get_label_id_node, False),   # optional: custom models may not pass labels through graph
    (get_position_id_node, False),
]


def pass_shard_tagged_inputs(gm: GraphModule, example_inputs):
    for getter, required in _SHARD_TARGETS:
        node = getter(gm)
        if node is None:
            if required:
                raise RuntimeError(f"[AutoSP] Required node not found via {getter.__name__}")
            continue
        shard_tensor_node(gm, node)

    from ..fx import find_node_by_tag
    mask_node = find_node_by_tag(gm, constants.AUTOSP_ATTENTION_MASK_KEY)
    if mask_node is not None:
        shard_tensor_node(gm, mask_node)


def _insert_a2a(gm, node, scatter_idx, gather_idx, name):
    with gm.graph.inserting_after(node):
        a2a_node = gm.graph.call_function(
            torch.ops.autosp.all_to_all.default,
            args=(node, scatter_idx, gather_idx, name),
        )
        a2a_node.name = f"a2a_{name}"
        node.replace_all_uses_with(a2a_node)
        a2a_node.update_arg(0, node)
    return a2a_node


_A2A_QKV_PLAN = {"scatter_idx": 1, "gather_idx": 2}
_A2A_O_PLAN   = {"scatter_idx": 2, "gather_idx": 1}


def _wire_sdpa_a2a(gm, attn_node, idx, n_total):
    q, k, v = attn_node.args[:3]
    suffix = f"_{idx}" if n_total > 1 else ""
    _insert_a2a(gm, q, **_A2A_QKV_PLAN, name=f"q{suffix}")
    _insert_a2a(gm, k, **_A2A_QKV_PLAN, name=f"k{suffix}")
    _insert_a2a(gm, v, **_A2A_QKV_PLAN, name=f"v{suffix}")
    o_a2a = _insert_a2a(gm, attn_node, **_A2A_O_PLAN, name=f"o{suffix}")

    # After reverse A2A, dim1 may be padded (e.g. 33 instead of 32).
    # Insert a slice to trim back to the original n_heads if needed.
    q_meta = q.meta.get("val") or q.meta.get("example_value")
    if q_meta is not None and hasattr(q_meta, 'shape') and len(q_meta.shape) >= 2:
        orig_n_heads = q_meta.shape[1]  # original n_heads before padding
        _sp = sp_dp_registry.sp_size()
        if isinstance(orig_n_heads, int) and orig_n_heads % _sp != 0:
            # o_a2a output is [B, padded_heads, S_local, H] — trim dim1
            with gm.graph.inserting_after(o_a2a):
                trim_node = gm.graph.call_function(
                    torch.ops.aten.narrow.default,
                    args=(o_a2a, 1, 0, orig_n_heads),
                )
                trim_node.name = f"trim_heads{suffix}"
                o_a2a.replace_all_uses_with(trim_node)
                trim_node.update_arg(0, o_a2a)


def pass_insert_attention_all_to_all(gm: GraphModule, real_inputs):
    attention_nodes = get_sdpa_nodes(gm)
    if len(attention_nodes) == 0:
        raise RuntimeError(
            "AutoSP currently supports torch.nn.functional.scaled_dot_product_attention as the "
            "attention backend. No SDPA attention operations were found in the compiled graph. "
            "Please ensure your model uses torch.nn.functional.scaled_dot_product_attention "
            "for AutoSP to work as expected.")

    for idx, attn_node in enumerate(attention_nodes):
        _wire_sdpa_a2a(gm, attn_node, idx, len(attention_nodes))


def pass_canonicalize(gm: GraphModule, real_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def _discover_fake_mode(gm):
    for node in gm.graph.nodes:
        for key in ("val", "example_value"):
            fake_val = node.meta.get(key)
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                mode = maybe_get_fake_mode(fake_val)
                if mode is not None:
                    return mode
    return FakeTensorMode(shape_env=ShapeEnv())


def _extract_placeholder_dtypes(gm):
    dtypes = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val") or node.meta.get("example_value")
            if val is not None and isinstance(val, torch.Tensor) and val.is_floating_point():
                dtypes.append(val.dtype)
            else:
                dtypes.append(None)
    return dtypes


def _build_fake_inputs(real_inputs, fake_mode, placeholder_dtypes):
    fake_inputs = []
    for idx, t in enumerate(real_inputs):
        if isinstance(t, torch.Tensor):
            target_dtype = placeholder_dtypes[idx] if idx < len(placeholder_dtypes) else None
            if target_dtype is not None and t.is_floating_point() and t.dtype != target_dtype:
                logger.debug(
                    f"[AutoSP] placeholder[{idx}] dtype mismatch: "
                    f"real={t.dtype} vs graph={target_dtype}, casting")
                t = t.to(target_dtype)
            fake_inputs.append(fake_mode.from_tensor(t))
        else:
            fake_inputs.append(t)
    return fake_inputs


def _snapshot_meta(gm):
    return {node.name: dict(node.meta) for node in gm.graph.nodes
            if node.op in {"call_function", "call_method"}}


def _restore_meta(gm, snapshot):
    for node in gm.graph.nodes:
        if node.name in snapshot:
            node.meta = snapshot[node.name]


def _strip_sdpa_masks(gm):
    saved = []
    for attn_node in get_sdpa_nodes(gm):
        attn_mask = attn_node.kwargs.get("attn_mask")
        if attn_mask is not None:
            saved.append((attn_node, attn_mask))
            attn_node.update_kwarg("attn_mask", None)
    return saved


def _restore_sdpa_masks(saved):
    for attn_node, attn_mask in saved:
        attn_node.update_kwarg("attn_mask", attn_mask)


def pass_propagate_shapes(gm: torch.fx.GraphModule, real_inputs):
    fake_mode = _discover_fake_mode(gm)
    placeholder_dtypes = _extract_placeholder_dtypes(gm)
    fake_inputs = _build_fake_inputs(real_inputs, fake_mode, placeholder_dtypes)

    saved_masks = _strip_sdpa_masks(gm)
    snapshot = _snapshot_meta(gm)
    try:
        FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)
    except Exception:
        _restore_meta(gm, snapshot)
        raise
    finally:
        _restore_sdpa_masks(saved_masks)


def _validate_sdpa_nodes(gm, sp_sz, rank):
    sdpa_nodes = get_sdpa_nodes(gm)
    n_sdpa = len(sdpa_nodes)
    if n_sdpa == 0:
        return 0
    if rank == 0:
        first_sdpa = sdpa_nodes[0]
        for arg_idx, arg_name in [(0, "Q"), (1, "K"), (2, "V")]:
            if arg_idx < len(first_sdpa.args):
                node = first_sdpa.args[arg_idx]
                meta = node.meta.get("val") or node.meta.get("example_value")
                if meta is not None and hasattr(meta, 'shape') and len(meta.shape) >= 2:
                    n_heads = meta.shape[1]
                    if isinstance(n_heads, int) and n_heads % sp_sz != 0:
                        logger.warning(
                            "[AutoSP] %s n_heads=%d not divisible by sp_size=%d. "
                            "A2A will use head-padding for uneven split.",
                            arg_name, n_heads, sp_sz)
    return n_sdpa


def _validate_a2a_count(gm, expected_sdpa_count):
    a2a_count = sum(1 for n in gm.graph.nodes
                    if n.op == "call_function"
                    and n.target is torch.ops.autosp.all_to_all.default)
    expected = expected_sdpa_count * 4
    if a2a_count != expected:
        raise RuntimeError(
            f"[AutoSP] A2A count mismatch: got {a2a_count}, expected {expected} "
            f"(4 x {expected_sdpa_count} SDPA). NCCL deadlock will occur.")


def pass_validate_loc_shapes(gm: GraphModule, real_inputs):
    if not sp_dp_registry.is_loc_enabled():
        return
    for node in gm.graph.nodes:
        if (node.op == "call_function"
                and node.target is torch.ops.autosp.all_to_all.default):
            meta = node.meta.get("val") or node.meta.get("example_value")
            if meta is not None and hasattr(meta, 'shape'):
                B, N, S, H = meta.shape
                _sp = sp_dp_registry.sp_size()
                if N > 0 and isinstance(N, int) and N % _sp != 0 and S > 0 and isinstance(S, int) and S % _sp != 0:
                    logger.warning(
                        f"[LOC+SP] A2A node {node.name} shape ({B},{N},{S},{H}) "
                        f"neither N nor S divisible by sp_size={_sp}")


_APPLY_LOCK = threading.Lock()


AUTOSP_PASSES = [
    pass_shard_seq_dim,
    pass_shard_tagged_inputs,
    pass_insert_attention_all_to_all,
    pass_propagate_shapes,
    pass_validate_loc_shapes,
    pass_canonicalize,
]


def apply_autosp(gm: GraphModule,
                 real_inputs,
                 debug: bool = False,
                 passes: Optional[List[Callable]] = None,
                 sp_size: int = 2,
                 dp_size: int = 1):
    assert sp_size * dp_size <= dist.get_world_size(), 'Insufficient device count for mesh size'

    with _APPLY_LOCK:
        if not sp_dp_registry.is_setup():
            sp_dp_registry.populate_registry(sp_size, dp_size)

    passes = passes or AUTOSP_PASSES
    rank = dist.get_rank()
    n_sdpa = _validate_sdpa_nodes(gm, sp_size, rank)
    if n_sdpa == 0:
        # No SDPA nodes found — PyTorch may have decomposed them.
        # Skip AutoSP passes; training proceeds without SP.
        if rank == 0:
            logger.warning(
                "[AutoSP] No SDPA nodes in FX graph (PyTorch %s may decompose "
                "F.scaled_dot_product_attention). Skipping SP passes.",
                torch.__version__,
            )
        return gm.forward

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

    _validate_a2a_count(gm, n_sdpa)
