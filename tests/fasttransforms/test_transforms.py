# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

import fasttransforms.nn.transforms as T

# ---------- Configuration ----------
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-4),
    torch.float16: dict(atol=2e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}

# Unified shapes list (no separation of image vs axis)
SHAPES = [
    (16,),  # 1D
    (16, 16),  # 2D
    (16, 16, 16),  # 3D
    (16, 16, 16, 16),  # 4D
]

# Unified transform set (no separation)

TRANSFORM_SPECS = [
    (T.RotationTransform, {"init_type": "random"}),
    (T.HadamardTransform, {"hadamard_type": "sylvester", "permutation": "walsh"}),
    (T.HadamardTransform, {"hadamard_type": "sylvester", "permutation": "identity"}),
    (T.HadamardTransform, {"hadamard_type": "regular"}),
    (T.InvertibleSVDMatrixTransform, {}),
    (T.InvertibleMatrixTransform, {}),
    (T.ZigZag2DPermutation, {}),
]

# Transforms for which we allow a larger slack (numerically less stable)
# the tolerances are multiplied by a factor >1
LOW_PRECISION_TRANSFORMS = {
    T.InvertibleSVDMatrixTransform: 1.2,
    T.InvertibleMatrixTransform: 1.8,
}


# ---------- Fixtures ----------
@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(1234)


def rand_like(shape, dtype, device):
    # Generate in float32 for numerical stability, then cast
    g = torch.Generator(device=device).manual_seed(1234)
    x = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
    return x.to(dtype)


def forward_inverse_roundtrip(transform, x, dim):
    """Attempt y = fwd(x), x_rec = inv(y) at given dim; return x_rec or raise."""
    y = transform(x, inv=False, dim=dim)
    x_rec = transform(y, inv=True, dim=dim)
    return x_rec


def specs_to_id(x):
    if isinstance(x, dict):
        return ", ".join([f"{name}={value}" for name, value in x.items()])
    return x.__name__


# ---------- Consistency of forward and inverse test ----------
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"shape={s}")
@pytest.mark.parametrize("transform_cls, init_kwargs", TRANSFORM_SPECS, ids=specs_to_id)
def test_transforms_roundtrip_unified(dtype, shape, device, transform_cls, init_kwargs):
    x = rand_like(shape, dtype, device)
    tol = TOL[dtype]

    # For some transforms we allow larger tolerance
    if transform_cls in LOW_PRECISION_TRANSFORMS:
        tol["atol"] *= LOW_PRECISION_TRANSFORMS[transform_cls]
        tol["rtol"] *= LOW_PRECISION_TRANSFORMS[transform_cls]

    for dim_choice in range(len(shape)):
        transform = transform_cls(**init_kwargs)

        x_rec = forward_inverse_roundtrip(transform, x, dim_choice)
        # Compare in float32 to keep tolerances straightforward per dtype
        torch.testing.assert_close(
            x_rec.float(),
            x.float(),
            **tol,
            msg=f"Failed for {transform.__class__.__name__}, shape={shape}, apply_on={dim_choice}, dtype={dtype}, device={device}",
        )
