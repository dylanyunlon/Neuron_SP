import torch
import torch.distributed as dist
from torch.utils._sympy.functions import FloorDiv
from .sp_dp_registry import get_group, is_setup, sp_size, track_a2a_handle, finalize_a2a_pass, is_loc_enabled

_FP32_SP_GRAD = False


def set_fp32_sp_grad(enabled):
    global _FP32_SP_GRAD
    _FP32_SP_GRAD = enabled


_SCATTER_HEADS = {
    "pre_reshape": lambda B, P, d1, d2, H: (B, P, d1 // P, d2, H),
    "pre_permute": (1, 0, 2, 3, 4),
    "post_permute": (1, 2, 0, 3, 4),
    "post_reshape": lambda B, P, d1, d2, H: (B, d1 // P, P * d2, H),
}

_SCATTER_SEQ = {
    "pre_reshape": lambda B, P, d1, d2, H: (B, d1, P, d2 // P, H),
    "pre_permute": (2, 0, 1, 3, 4),
    "post_permute": (1, 0, 2, 3, 4),
    "post_reshape": lambda B, P, d1, d2, H: (B, P * d1, d2 // P, H),
}


def _execute_a2a(input, B, dim1, dim2, H, group, plan):
    P = sp_size()
    input_t = input.reshape(*plan["pre_reshape"](B, P, dim1, dim2, H))
    input_t = input_t.permute(*plan["pre_permute"]).contiguous()
    output = torch.empty_like(input_t)
    handle = dist.all_to_all_single(output, input_t, group=group, async_op=False)
    track_a2a_handle(handle)
    output = output.permute(*plan["post_permute"]).contiguous()
    return output.reshape(*plan["post_reshape"](B, P, dim1, dim2, H))


def _resolve_group():
    _sp = sp_size()
    _rank = dist.get_rank()
    gid = _rank // _sp
    grp = get_group(gid)
    if is_loc_enabled():
        from .sp_dp_registry import get_loc_sp_group_ids
        loc_gids = get_loc_sp_group_ids()
        if loc_gids:
            grp = get_group(loc_gids[0])
    return grp


@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    name: str,
) -> torch.Tensor:
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
    assert input.ndim == 4, (
        f"[AutoSP] all_to_all expects 4D [B,N,S,H], got {input.ndim}D shape {input.shape}. "
        f"For 13B (n_head=40) verify sp_size divides n_heads.")
    if not input.is_contiguous():
        input = input.contiguous()
    assert input.shape[-1] > 0, (
        f"[AutoSP] head_dim=0 in shape {input.shape}. Model config error.")
    B, dim1, dim2, H = input.shape
    _sp = sp_size()
    if scatter_idx == 1:
        assert dim1 % _sp == 0, (
            f"[AutoSP] all_to_all forward: N={dim1} not divisible by sp_size={_sp}. "
            f"For GQA models, set sequence_parallel_size to a divisor of num_kv_heads.")
    else:
        assert dim2 % _sp == 0, (
            f"[AutoSP] all_to_all forward: S={dim2} not divisible by sp_size={_sp}. "
            f"Sequence length must be divisible by sequence_parallel_size.")

    group = _resolve_group()
    plan = _SCATTER_HEADS if scatter_idx == 1 else _SCATTER_SEQ
    return _execute_a2a(input, B, dim1, dim2, H, group, plan)


@torch.library.register_fake("autosp::all_to_all")
def all_to_all_fake(input, scatter_idx, gather_idx, name):

    def maybe_restore_sharded_dim(dim, factor):
        node = getattr(dim, "node", None)
        if node is None:
            return dim * factor

        expr = node.expr
        if isinstance(expr, FloorDiv) and expr.args[1] == factor:
            hint = node.hint * factor if node.has_hint() else None
            return node.shape_env.create_symintnode(expr.args[0], hint=hint)

        return dim * factor

    B, dim1, dim2, H = input.shape
    if scatter_idx == 1:
        return input.new_empty(B, dim1 // sp_size(), maybe_restore_sharded_dim(dim2, sp_size()), H)
    return input.new_empty(B, dim1 * sp_size(), dim2 // sp_size(), H)


def _backward_setup(ctx, inputs, output):
    _, scatter_idx, gather_idx, name = inputs
    ctx.scatter_idx = gather_idx
    ctx.gather_idx = scatter_idx
    ctx.name = name + "_grad"
    ctx.orig_dtype = inputs[0].dtype


def _backward(ctx, grad):
    if not grad.is_contiguous():
        grad = grad.contiguous()

    use_fp32 = (
        _FP32_SP_GRAD
        and ctx.orig_dtype in (torch.bfloat16, torch.float16)
        and grad.dtype in (torch.bfloat16, torch.float16))

    if use_fp32:
        grad_fp32 = grad.float()
        out_fp32 = all_to_all(grad_fp32, ctx.scatter_idx, ctx.gather_idx, ctx.name)
        return (out_fp32.to(ctx.orig_dtype), None, None, None)

    return (all_to_all(grad, ctx.scatter_idx, ctx.gather_idx, ctx.name), None, None, None)


torch.library.register_autograd("autosp::all_to_all", _backward, setup_context=_backward_setup)
