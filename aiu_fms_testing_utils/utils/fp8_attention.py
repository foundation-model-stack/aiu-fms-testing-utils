import math
from typing import (
    Any,
    Callable,
    Mapping,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Unpack,
)
import torch
from torch import Tensor

from fms.modules.attention import AttentionKwargs, register_attention_op

class MathFP8AttentionKwargs(AttentionKwargs):
    mask: NotRequired[Tensor]
    is_causal_mask: bool


def _math_fp8_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[torch.Tensor],
    value_cache: Optional[torch.Tensor],
    **attn_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # keys comes from rope, so not yet fp8 (assume scale=1)
    keys = keys.transpose(2, 1).to(torch.float8_e4m3fn)
    # values should come in fp8 already
    values = values.transpose(2, 1)

    if key_cache is not None and value_cache is not None and value_cache.numel() > 0:
        key_cache_result = torch.cat((key_cache, keys), dim=2)
        value_cache_result = torch.cat((value_cache, values), dim=2)
        return (
            key_cache_result,
            value_cache_result,
            key_cache_result,
            value_cache_result,
        )
    else:
        return (keys, values, keys, values)
    

########
## scaled_bmm - A batched version of _scaled_mm
########
@torch.library.custom_op("sendnn::scaled_bmm", mutates_args=())
def sendnn_scaled_bmm(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    bias: Optional[Tensor] = None,
    scale_result: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    assert mat1.shape[:-2] == mat2.shape[:-2], "batch dimensions must match for mat1 and mat2"
    assert mat1.shape[:-2] == scale1.shape[:-2], "batch dimensions must match for mat1 and scale1"
    assert mat2.shape[:-2] == scale2.shape[:-2], "batch dimensions must match for mat2 and scale2"
    if bias:
        assert mat1.shape[:-2] == bias.shape[:-2], "batch dimensions must match for mat1 and bias"
    orig_batch = mat2.shape[:-2]
    mat1 = mat1.view(-1, *mat1.shape[-2:])
    mat2 = mat2.view(-1, *mat2.shape[-2:])
    scale1 = scale1.view(-1, *scale1.shape[-2:])
    scale2 = scale2.view(-1, *scale2.shape[-2:])
    if bias:
        bias = bias.view(-1, *bias.shape[-2:])
    if scale_result:
        scale_result = scale_result.view(-1, *scale_result.shape[-2:])
    out = torch.empty((mat1.shape[0], mat1.shape[1], mat2.shape[2]), dtype=out_dtype, device=mat1.device)
    for b_idx in range(mat1.shape[0]):
        out[b_idx] = torch._scaled_mm(
            mat1[b_idx],
            mat2[b_idx],
            scale1[b_idx],
            scale2[b_idx],
            bias[b_idx] if bias else None,
            scale_result[b_idx] if scale_result else None,
            out_dtype,
            use_fast_accum
        )
    return out

# All that's needed for torch.compile support
@sendnn_scaled_bmm.register_fake
def _(
    mat1: Tensor,
    mat2: Tensor,
    scale1: Tensor,
    scale2: Tensor,
    bias: Optional[Tensor] = None,
    scale_result: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> Tensor:
    return torch.empty((mat1.shape[0], mat1.shape[1], mat2.shape[2]), dtype=out_dtype, device=mat1.device)


# TODO: Doens't quite work yet, more discussion needed
def _math_fp8_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    # query comes from rope, so not yet fp8 (assume scale=1)
    orig_dtype = query.dtype
    query = query.transpose(2, 1).to(torch.float8_e4m3fn)

    # no longer transposing prior to store, so need to check this in case of no cache
    if key_cache.shape[1] != kvheads and key_cache.shape[2] == kvheads:
        key_cache = key_cache.transpose(2, 1).to(
            torch.float8_e4m3fn
        )  # might not have been converted
        value_cache = value_cache.transpose(2, 1)

    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
        # we need to create the nheads dimension
        while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
            mask = mask.unsqueeze(1)

    L, S = query.size(-2), key_cache.size(-2)
    scale_factor = (
        1 / math.sqrt(query.size(-1)) if scale_factor is None else scale_factor
    )
    attn_bias = torch.zeros(L, S, dtype=orig_dtype, device=query.device)
    if attn_kwargs.get("is_causal_mask", False):
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(torch.float32)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias = mask + attn_bias

    expansion = nheads // kvheads
    if expansion > 1:
        key_cache = key_cache.repeat_interleave(
            query.size(-3) // key_cache.size(-3), -3
        )
        value_cache = value_cache.repeat_interleave(
            query.size(-3) // key_cache.size(-3), -3
        )

    scale = torch.ones((1,), dtype=torch.float32, device=query.device)
    attn_weight = (
        torch.ops.sendnn.scaled_bmm(
            query,
            key_cache.transpose(-2, -1),
            scale,
            scale,
            out_dtype=orig_dtype,
            use_fast_accum=True,
        )
        * scale_factor
    )
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, p_dropout, train=True)
    # Do matmul in orig_dtype
    attn = attn_weight @ value_cache.to(orig_dtype)

    attn = attn.to(orig_dtype).transpose(2, 1).contiguous()
    return attn


register_attention_op(
    "math_fp8",
    _math_fp8_store_op,
    _math_fp8_compute_op,
)