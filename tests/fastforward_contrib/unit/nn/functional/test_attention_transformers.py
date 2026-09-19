# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch
from fastforward import estimate_ranges, range_setting
from fastforward.nn import LinearQuantizer
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastforward_contrib.nn.functional.attention_transformers import scaled_dot_product_attention


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=lambda x: str(x))
def test_attention_unquantized(device, dtype):
    torch.backends.cudnn.deterministic = True

    q = (torch.rand(10, 2048) * 2 - 1).to(device=device).to(dtype)
    k = (torch.rand(10, 2048) * 2 - 1).to(device=device).to(dtype)
    v = (torch.rand(10, 2048) * 2 - 1).to(device=device).to(dtype)

    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_f = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    with torch.no_grad():
        out_fff = scaled_dot_product_attention(
            q, k, value=v, dropout_p=0.0, is_causal=False, strict_quantization=False
        )

    res = torch.finfo(dtype).resolution
    assert torch.allclose(out_fff, other=out_f, atol=res, rtol=res)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bits, tol", [(16, 1e-4), (8, 1e-2)])
def test_quantized_attention(bits, tol, device):
    # Arrange:
    q = (torch.rand(10, 2048) * 2 - 1).to(device=device)
    k = (torch.rand(10, 2048) * 2 - 1).to(device=device)
    v = (torch.rand(10, 2048) * 2 - 1).to(device=device)
    q_matmul = LinearQuantizer(bits, symmetric=False).to(device)
    q_softmax = LinearQuantizer(bits, symmetric=False).to(device)
    q_output = LinearQuantizer(bits, symmetric=False).to(device)

    with sdpa_kernel(backends=[SDPBackend.MATH]), torch.no_grad():
        out_f = F.scaled_dot_product_attention(q.dequantize(), k.dequantize(), v.dequantize())

    # Act:
    with torch.no_grad():
        with estimate_ranges([q_matmul, q_softmax, q_output], range_setting.running_minmax):
            _ = scaled_dot_product_attention(
                q,
                k,
                v,
                strict_quantization=False,
                matmul_k_quantizer=q_matmul,
                softmax_quantizer=q_softmax,
                output_quantizer=q_output,
            )
        out_fff = scaled_dot_product_attention(
            q,
            k,
            v,
            strict_quantization=False,
            matmul_k_quantizer=q_matmul,
            softmax_quantizer=q_softmax,
            output_quantizer=q_output,
        ).dequantize()

    # Assert:
    assert torch.allclose(out_fff, other=out_f, atol=tol, rtol=tol)
