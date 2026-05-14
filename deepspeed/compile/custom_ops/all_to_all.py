import torch
import deepspeed.comm as dist
from torch.utils._sympy.functions import FloorDiv
from .sp_dp_registry import get_group, is_setup, sp_size, track_a2a_handle, finalize_a2a_pass

_FP32_SP_GRAD = False


def set_fp32_sp_grad(enabled):
    global _FP32_SP_GRAD
    _FP32_SP_GRAD = enabled


def _scatter_heads_gather_seq(input, B, N, local_S, H, group):
    P = sp_size()
    input_t = input.reshape(B, P, N // P, local_S, H)
    input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    handle = dist.all_to_all_single(output, input_t, group=group, async_op=False)
    track_a2a_handle(handle)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.reshape(B, N // P, P * local_S, H)


def _scatter_seq_gather_heads(input, B, local_N, S, H, group):
    P = sp_size()
    input_t = input.reshape(B, local_N, P, S // P, H)
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    handle = dist.all_to_all_single(output, input_t, group=group, async_op=False)
    track_a2a_handle(handle)
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    return output.reshape(B, P * local_N, S // P, H)


@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    name: str,
) -> torch.Tensor:
    assert is_setup(), 'Incorrect initialization of SP/DP mesh.'
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
    gid = dist.get_rank() // _sp
    group = get_group(gid)

    if scatter_idx == 1:
        return _scatter_heads_gather_seq(input, B, dim1, dim2, H, group)
    return _scatter_seq_gather_heads(input, B, dim1, dim2, H, group)


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
