# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import fastforward.quantization.granularity as granularities
import pytest
import torch
from fastforward import QuantizedTensor
from fastforward.nn.dynamic_linear_quantizer import (
    DynamicLinearQuantizer as _FFDynamicLinearQuantizer,
)

from fastforward_contrib.nn import (
    CLIP_PARAMETRIZATION_REGISTRY,
    DynamicLinearClipQuantizer,
    DynamicLinearQuantizer,
    dynamic_scale_offset_with_grad,
)


@pytest.mark.parametrize("nbits", [2, 4, 8], ids=lambda x: f"bits={x}")
def test_dynamic_linear_quantizer_forward(nbits):
    """Dynamic quantizer computes scale/offset per forward; output is dequantizable."""
    torch.manual_seed(0)
    data = torch.randn(4, 16)
    quantizer = DynamicLinearQuantizer(num_bits=nbits, granularity=granularities.PerChannel(channel_dim=0))
    with ff.strict_quantization(False):
        out = quantizer(data)
    assert isinstance(out, QuantizedTensor)
    assert out.dequantize().shape == data.shape


def test_dynamic_linear_quantizer_enable_grad_minmax_flows_to_input():
    """enable_grad_minmax=True installs the grad-enabled inference fn and routes
    gradients back to the input data through the dynamic range."""
    quantizer = DynamicLinearQuantizer(
        num_bits=4,
        granularity=granularities.PerChannel(channel_dim=0),
        enable_grad_minmax=True,
    )
    assert quantizer.parameter_inference_fn is dynamic_scale_offset_with_grad

    data = torch.randn(4, 16, requires_grad=True)
    with ff.strict_quantization(False):
        quantizer(data).dequantize().sum().backward()
    assert data.grad is not None
    assert data.grad.abs().sum() > 0


def test_dynamic_linear_quantizer_no_grad_minmax_by_default():
    """Without enable_grad_minmax no inference fn is installed and the default
    (non-grad) min/max path is used."""
    quantizer = DynamicLinearQuantizer(num_bits=4)
    assert quantizer.parameter_inference_fn is None
    assert quantizer.enable_grad_minmax is False


def _grad_through_quantizer(quantizer, data):
    """Backprop ``sum(dequantize(quantize(data)))`` and return the input gradient."""
    data = data.clone().detach().requires_grad_(True)
    with ff.strict_quantization(False):
        quantizer(data).dequantize().sum().backward()
    assert data.grad is not None
    return data.grad


def test_dynamic_linear_quantizer_grad_matches_base_when_no_grad_minmax():
    """enable_grad_minmax=False must be gradient-identical to the FF base quantizer."""
    torch.manual_seed(0)
    data = torch.randn(4, 16)
    granularity = granularities.PerChannel(channel_dim=0)

    base_grad = _grad_through_quantizer(_FFDynamicLinearQuantizer(num_bits=4, granularity=granularity), data)
    ours_grad = _grad_through_quantizer(
        DynamicLinearQuantizer(num_bits=4, granularity=granularity, enable_grad_minmax=False), data
    )
    assert torch.equal(ours_grad, base_grad)


def test_dynamic_linear_quantizer_grad_minmax_differs_only_at_min_and_max():
    """enable_grad_minmax=True matches the base everywhere except the per-row
    argmin/argmax positions (asymmetric quant => both min and max carry grad)."""
    torch.manual_seed(0)
    data = torch.randn(4, 16)
    granularity = granularities.PerChannel(channel_dim=0)

    base_grad = _grad_through_quantizer(
        _FFDynamicLinearQuantizer(num_bits=4, granularity=granularity, symmetric=False), data
    )
    ours_grad = _grad_through_quantizer(
        DynamicLinearQuantizer(num_bits=4, granularity=granularity, symmetric=False, enable_grad_minmax=True),
        data,
    )

    # Positions where the extra (min/max) gradient is expected: per-row argmin/argmax.
    extra = torch.zeros_like(data, dtype=torch.bool)
    extra[torch.arange(data.shape[0]), data.argmin(dim=1)] = True
    extra[torch.arange(data.shape[0]), data.argmax(dim=1)] = True

    # Everywhere else the gradient must be identical to the base.
    assert torch.equal(ours_grad[~extra], base_grad[~extra])
    # At the min/max positions the grad path actually contributes (differs from base).
    assert (ours_grad[extra] != base_grad[extra]).all()


def test_dynamic_linear_quantizer_rejects_explicit_fn_with_grad_minmax():
    """enable_grad_minmax cannot be combined with an explicit parameter_inference_fn."""
    with pytest.raises(ValueError):
        DynamicLinearQuantizer(
            num_bits=4,
            parameter_inference_fn=lambda params, data: (None, None),
            enable_grad_minmax=True,
        )


@pytest.mark.parametrize("clip_granularity", ["tensor", "follow_quantizer"])
def test_dynamic_linear_clip_quantizer_materializes_and_trains(clip_granularity):
    """Clip factors are lazily materialized on first forward and receive gradient."""
    quantizer = DynamicLinearClipQuantizer(
        num_bits=4,
        granularity=granularities.PerChannel(channel_dim=0),
        clip_granularity=clip_granularity,
    )
    assert quantizer.has_uninitialized_params

    data = torch.randn(4, 16, requires_grad=True)
    with ff.strict_quantization(False):
        out = quantizer(data)
    assert not quantizer.has_uninitialized_params

    expected_numel = 1 if clip_granularity == "tensor" else 4
    assert quantizer.clip_factor_max.numel() == expected_numel
    # Reading the (parametrized) clip factor returns the effective gamma, seeded
    # to clip_init (default 0.982 == sigmoid(4.0), the FlatQuant raw-4.0 seed).
    assert torch.allclose(
        quantizer.clip_factor_max,
        torch.full_like(quantizer.clip_factor_max, 0.982),
        atol=1e-4,
    )

    out.dequantize().pow(2).sum().backward()
    # The effective clip factor is a parametrized (non-leaf) view; gradient
    # accrues on the underlying raw parameters.
    assert quantizer.parametrizations.clip_factor_max.original.grad is not None
    assert quantizer.parametrizations.clip_factor_min.original.grad is not None


@pytest.mark.parametrize("clip_parametrization", sorted(CLIP_PARAMETRIZATION_REGISTRY))
def test_dynamic_linear_clip_quantizer_parametrizations(clip_parametrization):
    """Each registered clip parametrization seeds clip_init and produces a
    finite, trainable forward."""
    clip_init = 1.5 if clip_parametrization in ("identity", "abs", "softplus", "exp") else 0.7
    quantizer = DynamicLinearClipQuantizer(
        num_bits=4,
        granularity=granularities.PerChannel(channel_dim=0),
        clip_granularity="follow_quantizer",
        clip_parametrization=clip_parametrization,
        clip_init=clip_init,
    )
    data = torch.randn(4, 16, requires_grad=True)
    with ff.strict_quantization(False):
        out = quantizer(data)

    assert torch.allclose(
        quantizer.clip_factor_max,
        torch.full_like(quantizer.clip_factor_max, clip_init),
        atol=1e-4,
    )
    dq = out.dequantize()
    assert torch.isfinite(dq).all()
    dq.pow(2).sum().backward()
    assert quantizer.parametrizations.clip_factor_max.original.grad is not None


def test_dynamic_linear_clip_quantizer_rejects_unknown_parametrization():
    with pytest.raises(ValueError):
        DynamicLinearClipQuantizer(num_bits=4, clip_parametrization="not_a_real_one")


def test_dynamic_linear_clip_quantizer_rejects_unknown_clip_granularity():
    with pytest.raises(ValueError):
        DynamicLinearClipQuantizer(num_bits=4, clip_granularity="not_a_real_one")


def test_dynamic_linear_clip_quantizer_reset_parameters():
    """reset_parameters returns clip factors to the uninitialized state."""
    quantizer = DynamicLinearClipQuantizer(num_bits=4, granularity=granularities.PerChannel(channel_dim=0))
    with ff.strict_quantization(False):
        quantizer(torch.randn(4, 16))
    assert not quantizer.has_uninitialized_params
    quantizer.reset_parameters()
    assert quantizer.has_uninitialized_params
    # A subsequent forward re-materializes and re-registers the parametrization.
    with ff.strict_quantization(False):
        quantizer(torch.randn(4, 16))
    assert not quantizer.has_uninitialized_params


def test_dynamic_linear_clip_quantizer_with_grad_minmax_flows_to_input():
    quantizer = DynamicLinearClipQuantizer(
        num_bits=4,
        granularity=granularities.PerChannel(channel_dim=0),
        clip_granularity="follow_quantizer",
        enable_grad_minmax=True,
    )
    data = torch.randn(4, 16, requires_grad=True)
    with ff.strict_quantization(False):
        quantizer(data).dequantize().sum().backward()
    assert data.grad is not None
