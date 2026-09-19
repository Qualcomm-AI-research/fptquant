# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
import pytest
import torch

import fasttransforms as ft
import fasttransforms.nn.transforms as T

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-4),
    torch.float16: dict(atol=2e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


# ------------------------- Fixtures & helpers -------------------------
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(1234)


def build_model(in_dim=64, hidden_dim=64 * 3, n_layers=2):
    ff.set_strict_quantization(False)

    class SimpleBlock(torch.nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.up = ff.nn.QuantizedLinear(in_dim, hidden_dim)
            self.gate = torch.nn.ReLU(True)
            self.down = ff.nn.QuantizedLinear(hidden_dim, in_dim)

        def forward(self, x):
            y = self.up(x)
            y = self.gate(y)
            y = self.down(y)
            return y + x

    class SimpleModel(torch.nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, n_layers: int):
            super().__init__()
            self.input_quantizer = ff.nn.QuantizerStub()
            self.layers = torch.nn.Sequential()
            for _ in range(n_layers):
                self.layers.append(SimpleBlock(in_dim=in_dim, hidden_dim=hidden_dim))
            self.output_quantizer = ff.nn.QuantizerStub()

        def forward(self, x):
            y = self.input_quantizer(x)
            y = self.layers(y)
            y = self.output_quantizer(y)
            return y

    model = SimpleModel(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers)

    # Overwrite quantizers for weight & input paths to stubs (consistent with your snippet)
    for layer in model.layers:
        layer.up.input_quantizer = ff.nn.QuantizerStub()
        layer.up.weight_quantizer = ff.nn.QuantizerStub()
        layer.down.input_quantizer = ff.nn.QuantizerStub()
        layer.down.weight_quantizer = ff.nn.QuantizerStub()
        layer.down.bias_quantizer = ff.nn.QuantizerStub()

    return model


def make_transform_configs():
    """
    Returns the transform configuration dictionary (kwargs-only).
    """
    return {
        # H_up: rotation on input of `up`, inverse fused in `up.weight` (transpose)
        "H_up": ft.TransformConfig(
            transform=T.RotationTransform(),
            operation_configs=[
                ft.HookConfig(pattern="**/up/input_quantizer", dim=-1, pre_hook=True, inv=False),
                ft.ParametrizationConfig(pattern="**/up", tensor_name="weight", dim=-1, transpose=True, inv=True),
            ],
            priority=0,
            per_layer=True,
        ),
        # H_down: Regular Hadamard on input of `down`, inverse fused into linear input path
        "H_down": ft.TransformConfig(
            transform=T.HadamardTransform(hadamard_type="regular"),
            operation_configs=[
                ft.HookConfig(pattern="**/down/input_quantizer", dim=-1, pre_hook=True, inv=False),
                ft.LinearParametrizationConfig(pattern="**/down", input=True, inv=True),
            ],
            priority=0,
            per_layer=True,
        ),
        # R: Global rotation applied on model input/residuals; inverse fused as needed
        "R": ft.TransformConfig(
            transform=T.RotationTransform("random"),
            operation_configs=[
                ft.HookConfig(pattern="input_quantizer", dim=-1, pre_hook=True, inv=False),
                ft.LinearParametrizationConfig(pattern="**/up", input=True, inv=True),
                ft.LinearParametrizationConfig(pattern="**/down", input=False, inv=False),
                ft.HookConfig(pattern="output_quantizer", dim=-1, pre_hook=True, inv=True),
            ],
            per_layer=False,
            priority=1,  # applied first
        ),
    }


def make_input(batch=100, in_dim=64, device=torch.device("cpu"), dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(1234)
    return torch.randn(batch, in_dim, generator=g, device=device, dtype=dtype)


def mse_sum(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a - b).float()
    return torch.pow(diff, 2).sum().item()


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    diff_norm = torch.norm((a - b).float())
    base_norm = torch.norm(a.float()).clamp_min(1e-8)
    return (diff_norm / base_norm).item()


# ------------------------- Tests -------------------------
@pytest.mark.parametrize("dtype", DTYPES)  # keep tests stable; can add fp16/bf16 if supported
def test_transform_attachment_and_structure(dtype, device):
    """
    Verifies that transforms attach hooks/parametrizations to the intended modules.

    - Each 'up.input_quantizer' and 'down.input_quantizer' gets a hook submodule.
    - Global rotation attaches to input_quantizer and output_quantizer.
    - Parametrizations are fused on 'up' (weight) and both 'up'/'down' linear paths.
    """
    model = build_model().to(device=device, dtype=dtype)
    x = make_input(device=device, dtype=dtype)
    orig_output = model(x)

    transform_configs = make_transform_configs()
    transforms, transform_ops = ft.transforms_from_config_dict(transform_configs, model=model)

    # Base sanity checks
    assert isinstance(transforms, torch.nn.ModuleDict), "Expected 'transforms' to be a dict keyed by config names."
    assert set(transforms.keys()) == {"H_up", "H_down", "R"}
    assert transform_ops, "Transform operations list/dict should not be empty."

    # Check the TransformOperations
    for idx, layer in enumerate(model.layers):
        assert len(layer.up.input_quantizer.hooks) == 1, f"Missing hook on layers[{idx}].up.input_quantizer"
        assert layer.up.input_quantizer.hooks[0].transform == transforms["H_up"][str(idx)]
        assert len(layer.up.parametrizations["weight"]) == 2, f"Incorrect parametrization on layers[{idx}].up"
        assert layer.up.parametrizations["weight"][0].transform == transforms["R"]
        assert layer.up.parametrizations["weight"][1].transform == transforms["H_up"][str(idx)]

        assert len(layer.down.input_quantizer.hooks) == 1, f"Missing hook on layers[{idx}].down.input_quantizer"
        assert layer.down.input_quantizer.hooks[0].transform == transforms["H_down"][str(idx)]
        assert len(layer.down.parametrizations["weight"]) == 2, (
            f"Incorrect weight parametrization on layers[{idx}].down"
        )
        assert len(layer.down.parametrizations["bias"]) == 1, f"Incorrect bias parametrization on layers[{idx}].down"
        assert layer.down.parametrizations["weight"][0].transform == transforms["R"]
        assert layer.down.parametrizations["weight"][1].transform == transforms["H_down"][str(idx)]
        assert layer.down.parametrizations["bias"][0].transform == transforms["R"]

    transformed_output = model(x)

    # Temporarily disable transforms: output should match original behavior
    with ft.disable_transforms(model):
        all_disabled_output = model(x)

    torch.testing.assert_close(
        orig_output.float(),
        transformed_output.float(),
        **TOL[dtype],
        msg=f"Consistency check failed for dtype={dtype}, device={device}",
    )

    torch.testing.assert_close(
        orig_output.float(),
        all_disabled_output.float(),
        atol=0,
        rtol=0,
        msg=f"Consistency check failed for dtype={dtype}, device={device}",
    )


def test_fuse_transforms_preserves_output_and_removes_parametrizations(device):
    dtype = torch.float32
    in_dim = 8
    hidden_dim = 6
    model = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_dim),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_dim, 3),
    ).to(device=device, dtype=dtype)
    x = make_input(batch=16, in_dim=in_dim, device=device, dtype=dtype)

    original_output = model(x)

    transforms = [
        T.InvertibleMatrixTransform(init_type="random_orthogonal"),
        T.InvertibleMatrixTransform(init_type="random_orthogonal"),
    ]
    for transform in transforms:
        forward_op = ft.TransformOperation(transform=transform, dim=-1, inv=False)
        inverse_weight_op = ft.TransformOperation(transform=transform, dim=-1, inv=True, transpose=True)

        forward_op.attach_hook(model[0], pre_hook=True)
        inverse_weight_op.parametrize(model[0], tensor_name="weight")

    transformed_output = model(x)
    materialized_weight = model[0].weight.detach().clone()

    torch.testing.assert_close(transformed_output, original_output, atol=1e-5, rtol=1e-4)
    assert len(model[0].hooks) == 2
    assert len(model[0].parametrizations["weight"]) == 2

    fused_ops = ft.fuse_transforms(model)
    fused_output = model(x)
    remaining_ops = list(ft.get_transform_operations(model))

    torch.testing.assert_close(fused_output, transformed_output, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(model[0].weight, materialized_weight, atol=0, rtol=0)
    assert len(fused_ops) == 2
    assert all(not op.is_attached for op in fused_ops)
    assert not hasattr(model[0], "parametrizations")
    assert len(remaining_ops) == 2
    assert all(op.is_hook for op in remaining_ops)
