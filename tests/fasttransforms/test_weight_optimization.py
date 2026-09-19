# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

import fasttransforms as ft
import fasttransforms.nn.transforms as T
from fasttransforms.parameter_setting import WeightLpEstimator
from fasttransforms.parameter_setting.weight_optimization import _lp_norm


class NoForwardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        raise AssertionError("WeightLpEstimator must not run a model forward")


class HookOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = torch.nn.Identity()

    def forward(self, x):
        raise AssertionError("WeightLpEstimator must not run a model forward")


def test_weight_lp_estimator_optimizes_weight_parametrization_without_model_forward():
    model = NoForwardModel()
    transform_config = {
        "scale": ft.TransformConfig(
            transform=T.ScaleTransform(),
            operation_configs=[
                ft.LinearParametrizationConfig(pattern="linear", input=False, inv=False),
            ],
            per_layer=False,
            priority=0,
        )
    }
    transforms, transform_ops = ft.transforms_from_config_dict(transform_config, model=model)

    WeightLpEstimator(p=2.0, lr=1e-1, num_steps=3).estimate(model=model)

    scale = transforms["scale"].scale
    assert scale is not None
    assert not torch.allclose(scale, torch.ones_like(scale))
    assert model.linear.parametrizations.weight.original.grad is None


def test_lp_norm_chunked_matches_full_float32_formula_and_preserves_gradients():
    tensor = torch.randn(5, 7, dtype=torch.bfloat16).requires_grad_()

    actual = _lp_norm(tensor, p=3.0, chunk_size=4)
    expected = tensor.float().abs().pow(3.0).mean().pow(1.0 / 3.0)

    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()


def test_weight_lp_estimator_skips_hook_only_transforms():
    model = HookOnlyModel()
    transform = T.ScaleTransform()
    transform.__init_parameters__(4, dtype=torch.float32, device=torch.device("cpu"))
    before = transform.scale.detach().clone()
    transform_config = {
        "scale": ft.TransformConfig(
            transform=transform,
            operation_configs=[
                ft.HookConfig(pattern="input", dim=-1, pre_hook=True, inv=False),
            ],
            per_layer=False,
            priority=0,
        )
    }
    transforms, transform_ops = ft.transforms_from_config_dict(transform_config, model=model)

    WeightLpEstimator(p=2.0, lr=1e-1, num_steps=3).estimate(model=model)

    torch.testing.assert_close(transforms["scale"].scale, before)


def test_weight_lp_estimator_skips_inverse_only_weight_transforms():
    """An inverse-only weight leg makes the Lp objective unbounded.

    The counterbalancing forward leg lives on an activation hook that this
    estimator never sees, so optimizing would shrink the weight purely by
    inflating the transform. Such transforms must be left untouched.
    """
    model = NoForwardModel()
    transform_config = {
        "scale": ft.TransformConfig(
            transform=T.ScaleTransform(),
            operation_configs=[
                ft.LinearParametrizationConfig(pattern="linear", input=True, inv=True),
            ],
            per_layer=False,
            priority=0,
        )
    }
    transforms, transform_ops = ft.transforms_from_config_dict(transform_config, model=model)
    weight_before = model.linear.weight.detach().clone()

    WeightLpEstimator(p=2.0, lr=1e-1, num_steps=3).estimate(model=model)

    scale = transforms["scale"].scale
    assert scale is not None
    # Left at identity init, and the effective weight is unchanged.
    torch.testing.assert_close(scale, torch.ones_like(scale))
    torch.testing.assert_close(model.linear.weight, weight_before)


def test_weight_lp_estimator_does_not_retain_grads_on_frozen_weight_leaves():
    """The `.original` leaves are only backpropagated through, never optimized.

    They must not be left holding gradient buffers (which would persist for the
    whole run, one full-size buffer per parametrized weight), and their original
    `requires_grad` state must be restored.
    """
    model = NoForwardModel()
    transform_config = {
        "scale": ft.TransformConfig(
            transform=T.ScaleTransform(),
            operation_configs=[
                ft.LinearParametrizationConfig(pattern="linear", input=False, inv=False),
            ],
            per_layer=False,
            priority=0,
        )
    }
    ft.transforms_from_config_dict(transform_config, model=model)
    leaf = model.linear.parametrizations.weight.original
    assert leaf.requires_grad

    WeightLpEstimator(p=2.0, lr=1e-1, num_steps=5).estimate(model=model)

    assert leaf.grad is None
    assert leaf.requires_grad


def test_lp_norm_gradient_is_finite_for_all_zero_tensor():
    """`pow(1/p)` has an infinite derivative at zero without an epsilon.

    A NaN here would silently poison the transform parameters and propagate into
    range setting.
    """
    tensor = torch.zeros(4, 4, requires_grad=True)

    loss = _lp_norm(tensor, p=4.0)
    loss.backward()

    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all()
