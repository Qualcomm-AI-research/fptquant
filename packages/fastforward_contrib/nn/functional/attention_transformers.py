# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math
from typing import Optional

import torch
from fastforward import get_strict_quantization, set_strict_quantization
from fastforward.dispatcher import dispatch, register
from fastforward.nn import Quantizer
from fastforward.nn import functional as FFF
from fastforward.quantized_tensor import QuantizedTensor
from torch import Tensor


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask: Tensor | None = None,
    dropout_p=0.0,
    is_causal=False,
    scale: Optional[float] = None,
    *,
    matmul_k_quantizer: Optional[Quantizer] = None,
    softmax_quantizer: Optional[Quantizer] = None,
    dropout_quantizer: Optional[Quantizer] = None,
    output_quantizer: Optional[Quantizer] = None,
    strict_quantization: Optional[bool] = None,
) -> torch.Tensor:
    if strict_quantization is None:
        strict_quantization = get_strict_quantization()

    dispatch_op = dispatch(
        "scaled_dot_product_attention",
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        matmul_k_quantizer=matmul_k_quantizer,
        softmax_quantizer=softmax_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )
    selected_op = dispatch_op
    assert selected_op is not None

    return selected_op(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        matmul_k_quantizer=matmul_k_quantizer,
        softmax_quantizer=softmax_quantizer,
        dropout_quantizer=dropout_quantizer,
        output_quantizer=output_quantizer,
        strict_quantization=strict_quantization,
    )


@register("scaled_dot_product_attention", None)  # type: ignore[arg-type]
def _scaled_dot_product_attention(
    query: QuantizedTensor,
    key: QuantizedTensor,
    value: QuantizedTensor,
    attn_mask: QuantizedTensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    matmul_k_quantizer: Optional[Quantizer] = None,
    softmax_quantizer: Optional[Quantizer] = None,
    dropout_quantizer: Optional[Quantizer] = None,
    output_quantizer: Optional[Quantizer] = None,
    strict_quantization: Optional[bool] = None,
) -> QuantizedTensor | Tensor:
    """Quantized version of scaled_dot_product_attention from torch documentation

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias_shape = (1, 1, L, S) if attn_mask is not None and attn_mask.dim() == 4 else (L, S)
    attn_bias = torch.zeros(attn_bias_shape, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=attn_bias.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask.to(attn_bias.device)

    attn_scores = FFF.matmul(
        query, key.transpose(-2, -1), output_quantizer=matmul_k_quantizer, strict_quantization=strict_quantization
    )
    attn_scores *= scale_factor  # *= 1/sqrt(d_model)

    # NOTE: attn_bias is a causal attention mask, which is an upper triangular matrix containing -inf.
    # Because we do not want to quantize it, the following add and softmax operations are not
    # strictly quantized.
    with set_strict_quantization(False):
        attn_scores += attn_bias.to(attn_scores.device)

    attn_weight = FFF.softmax(attn_scores, dim=-1, output_quantizer=softmax_quantizer, strict_quantization=False)

    attn_weight = FFF.dropout(
        attn_weight,
        dropout_p,
        training=True,
        output_quantizer=dropout_quantizer,
        strict_quantization=strict_quantization,
    )

    attn_out = FFF.matmul(
        attn_weight, value, output_quantizer=output_quantizer, strict_quantization=strict_quantization
    )
    return attn_out
