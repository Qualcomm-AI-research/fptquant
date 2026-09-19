# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math
from functools import partial
from typing import Callable

import torch

from .common import MatmulTransform, Transform


def _move_to_dim(data: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    if dim < 0:
        dim = data.ndim + dim
    return torch.movedim(data, dim, -1), dim


def _move_from_last(data: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.movedim(data, -1, dim)


def _rotate_half(data: torch.Tensor) -> torch.Tensor:
    first, second = data.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class PreRopeTransform(Transform):
    required_parameters = ["angles", "scales"]

    def __init__(
        self,
        num_heads: int,
        num_key_value_heads: int | None = None,
        num_key_value_groups: int | None = None,
        init_type: str = "identity",
    ):
        if (num_key_value_heads is None) == (num_key_value_groups is None):
            raise ValueError("Specify exactly one of num_key_value_heads or num_key_value_groups.")
        if num_key_value_heads is None:
            if num_key_value_groups is None or num_heads % num_key_value_groups != 0:
                raise ValueError("num_heads must be divisible by num_key_value_groups.")
            num_key_value_heads = num_heads // num_key_value_groups
        if num_key_value_groups is None:
            if num_heads % num_key_value_heads != 0:
                raise ValueError("num_heads must be divisible by num_key_value_heads.")
            num_key_value_groups = num_heads // num_key_value_heads

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.init_type = init_type
        super().__init__()

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        if parameter_dim % self.num_key_value_heads != 0:
            raise ValueError(
                f"PreRopeTransform expected a dimension divisible by {self.num_key_value_heads}, "
                f"but got {parameter_dim}."
            )

        head_dim = parameter_dim // self.num_key_value_heads
        if head_dim % 2 != 0:
            raise ValueError(f"PreRopeTransform requires an even head dimension, but got {head_dim}.")

        self.head_dim = head_dim
        scale_shape = (self.num_key_value_heads, head_dim // 2)
        angle_shape = (self.num_key_value_heads, head_dim // 2)

        if self.init_type == "identity":
            angles = torch.zeros(angle_shape, dtype=dtype, device=device)
            scales = torch.ones(scale_shape, dtype=dtype, device=device)
        elif self.init_type == "random":
            angles = torch.randn(angle_shape, dtype=dtype, device=device) * 0.1
            scales = torch.rand(scale_shape, dtype=dtype, device=device) + 0.5
        elif self.init_type == "quarter":
            angles = torch.full(angle_shape, math.pi / 4, dtype=dtype, device=device)
            scales = torch.ones(scale_shape, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown PreRopeTransform init_type: {self.init_type}")

        self.angles = torch.nn.Parameter(angles)
        self.scales = torch.nn.Parameter(scales)

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_apply_prerope, transform=self, inv=False)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_apply_prerope, transform=self, inv=True)


def _apply_prerope(data: torch.Tensor, dim: int, transform: PreRopeTransform, inv: bool = False) -> torch.Tensor:
    data_dtype = data.dtype
    data, original_dim = _move_to_dim(data, dim)

    head_dim = transform.head_dim
    if inv:
        num_heads = transform.num_heads
        if data.shape[-1] != num_heads * head_dim:
            raise ValueError(f"Expected query dimension {num_heads * head_dim}, got {data.shape[-1]}.")
        data = data.reshape(*data.shape[:-1], num_heads, head_dim)
        scales = transform.scales.repeat_interleave(transform.num_key_value_groups, dim=0).repeat(1, 2)
        angles = transform.angles.repeat_interleave(transform.num_key_value_groups, dim=0)
        data = data / (scales + 1e-8)
    else:
        num_heads = transform.num_key_value_heads
        if data.shape[-1] != num_heads * head_dim:
            raise ValueError(f"Expected key dimension {num_heads * head_dim}, got {data.shape[-1]}.")
        data = data.reshape(*data.shape[:-1], num_heads, head_dim)
        data = data * transform.scales.repeat(1, 2)
        angles = transform.angles

    cos = torch.cos(angles).repeat(1, 2)
    sin = torch.sin(angles).repeat(1, 2)
    data = data * cos + _rotate_half(data) * sin
    data = data.reshape(*data.shape[:-2], -1).to(data_dtype)
    return _move_from_last(data, original_dim)


class ValueBatchedTransform(MatmulTransform):
    required_parameters = ["matrix"]

    def __init__(
        self,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        init_type: str = "identity",
        shared_across_heads: bool = False,
        eps: float = 1e-6,
        dtype: torch.dtype | str = "float32",
    ):
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads.")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype: {dtype.__class__.__name__}")

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = head_dim
        self.init_type = init_type
        self.shared_across_heads = shared_across_heads
        self.eps = eps
        self._dtype = dtype
        super().__init__()

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        valid_dims = {
            self.num_key_value_heads * self.head_dim,
            self.num_heads * self.head_dim,
        }
        if parameter_dim not in valid_dims:
            raise ValueError(
                f"ValueBatchedTransform expected dimension in {sorted(valid_dims)}, but got {parameter_dim}."
            )

        if self.init_type == "identity":
            matrix = torch.eye(self.head_dim, device=device, dtype=self._dtype)
            if not self.shared_across_heads:
                matrix = matrix.repeat(self.num_key_value_heads, 1, 1)
        elif self.init_type == "random":
            shape = (
                (self.head_dim, self.head_dim)
                if self.shared_across_heads
                else (self.num_key_value_heads, self.head_dim, self.head_dim)
            )
            matrix = torch.randn(*shape, device=device, dtype=self._dtype)
        else:
            raise ValueError(f"Unknown ValueBatchedTransform init_type: {self.init_type}")

        self.matrix = torch.nn.Parameter(matrix)

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_apply_value_batched, transform=self, inv=False)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_apply_value_batched, transform=self, inv=True)


def _apply_value_batched(
    data: torch.Tensor,
    dim: int,
    transform: ValueBatchedTransform,
    inv: bool = False,
    transpose: bool = False,
) -> torch.Tensor:
    data_dtype = data.dtype
    data, original_dim = _move_to_dim(data, dim)

    matrix = transform.matrix
    if inv:
        eye = torch.eye(transform.head_dim, device=matrix.device, dtype=matrix.dtype)
        if matrix.ndim == 3:
            eye = eye.unsqueeze(0)
        matrix = torch.linalg.inv((matrix + transform.eps * eye).float()).to(matrix.dtype)
    if transpose:
        matrix = matrix.mT if matrix.ndim == 3 else matrix.T

    kv_dim = transform.num_key_value_heads * transform.head_dim
    full_dim = transform.num_heads * transform.head_dim
    if data.shape[-1] == kv_dim and not inv:
        reshaped = data.reshape(*data.shape[:-1], transform.num_key_value_heads, transform.head_dim)
        if matrix.ndim == 2:
            output = torch.einsum("... h j, j k -> ... h k", reshaped, matrix.to(data.dtype))
        else:
            output = torch.einsum("... h j, h j k -> ... h k", reshaped, matrix.to(data.dtype))
    elif data.shape[-1] == full_dim:
        reshaped = data.reshape(
            *data.shape[:-1],
            transform.num_key_value_heads,
            transform.num_key_value_groups,
            transform.head_dim,
        )
        if matrix.ndim == 2:
            output = torch.einsum("... h g j, j k -> ... h g k", reshaped, matrix.to(data.dtype))
        else:
            output = torch.einsum("... h g j, h j k -> ... h g k", reshaped, matrix.to(data.dtype))
    elif data.shape[-1] == kv_dim:
        reshaped = data.reshape(*data.shape[:-1], transform.num_key_value_heads, transform.head_dim)
        if matrix.ndim == 2:
            output = torch.einsum("... h j, j k -> ... h k", reshaped, matrix.to(data.dtype))
        else:
            output = torch.einsum("... h j, h j k -> ... h k", reshaped, matrix.to(data.dtype))
    else:
        raise ValueError(f"Unexpected value transform dimension {data.shape[-1]}; expected {kv_dim} or {full_dim}.")

    output = output.reshape(*data.shape[:-1], -1).to(data_dtype)
    return _move_from_last(output, original_dim)


class HeadMixingTransform(MatmulTransform):
    required_parameters = ["matrix"]

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        init_type: str = "hadamard",
        eps: float = 1e-6,
        dtype: torch.dtype | str = "float32",
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype: {dtype.__class__.__name__}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.init_type = init_type
        self.eps = eps
        self._dtype = dtype
        super().__init__()

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        valid_dims = {self.num_heads, self.num_heads * self.head_dim}
        if parameter_dim not in valid_dims:
            raise ValueError(f"HeadMixingTransform expected dimension in {sorted(valid_dims)}, got {parameter_dim}.")

        if self.init_type == "identity":
            matrix = torch.eye(self.num_heads, device=device, dtype=self._dtype)
        elif self.init_type == "hadamard":
            from .hadamard import HadamardTransform

            matrix = HadamardTransform()(torch.eye(self.num_heads, device=device, dtype=dtype)).to(dtype=self._dtype)
        else:
            raise ValueError(f"Unknown HeadMixingTransform init_type: {self.init_type}")

        self.matrix = torch.nn.Parameter(matrix)

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_apply_head_mixing, transform=self, inv=False)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_apply_head_mixing, transform=self, inv=True)


def _apply_head_mixing(
    data: torch.Tensor,
    dim: int,
    transform: HeadMixingTransform,
    inv: bool = False,
    transpose: bool = False,
) -> torch.Tensor:
    data_dtype = data.dtype
    data, original_dim = _move_to_dim(data, dim)

    matrix = transform.matrix
    if inv:
        eye = torch.eye(transform.num_heads, device=matrix.device, dtype=matrix.dtype)
        matrix = torch.linalg.inv((matrix + transform.eps * eye).float()).to(matrix.dtype)
    if transpose:
        matrix = matrix.T

    if data.shape[-1] == transform.num_heads:
        output = torch.einsum("... h, h k -> ... k", data, matrix.to(data.dtype))
    elif data.shape[-1] == transform.num_heads * transform.head_dim:
        reshaped = data.reshape(*data.shape[:-1], transform.num_heads, transform.head_dim)
        output = torch.einsum("... h d, h k -> ... k d", reshaped, matrix.to(data.dtype))
        output = output.reshape(*data.shape[:-1], -1)
    else:
        raise ValueError(
            f"Unexpected head-mixing dimension {data.shape[-1]}; expected "
            f"{transform.num_heads} or {transform.num_heads * transform.head_dim}."
        )

    output = output.to(data_dtype)
    return _move_from_last(output, original_dim)


class DynamicScalingTransform(Transform):
    def __init__(
        self,
        scale_type: str = "rmsnorm",
        detach_norm: bool = False,
        variance_epsilon: float = 1e-8,
    ):
        self.scale_type = scale_type
        self.detach_norm = detach_norm
        self.variance_epsilon = variance_epsilon
        self.tmp_norm: torch.Tensor | None = None
        self.count = 0
        super().__init__()

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        del parameter_dim, dtype, device

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_dynamic_scale, transform=self)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(_dynamic_unscale, transform=self)

    def reset(self) -> None:
        self.tmp_norm = None
        self.count = 0


def _dynamic_rescale(
    data: torch.Tensor,
    transform: DynamicScalingTransform,
    expand_dim: int | None = None,
    reset: bool = False,
) -> torch.Tensor:
    """Reapply the already-recorded norm without recomputing or accumulating it."""
    if transform.tmp_norm is None:
        raise RuntimeError("DynamicScalingTransform rescale called before a scale was recorded.")

    data_dtype = data.dtype
    norm = transform.tmp_norm
    if expand_dim is not None:
        norm = norm.unsqueeze(expand_dim)

    output = (data.float() * norm).to(data_dtype)
    if reset:
        transform.reset()
    return output


def _dynamic_scale(
    data: torch.Tensor,
    dim: int,
    transform: DynamicScalingTransform,
    first: bool = False,
    rescale: bool = False,
    expand_dim: int | None = None,
    reset: bool = False,
) -> torch.Tensor:
    if rescale:
        del dim, first
        return _dynamic_rescale(data, transform=transform, expand_dim=expand_dim, reset=reset)

    data_dtype = data.dtype
    compute_data = data.float() if data.dtype != torch.float64 else data
    norm = _dynamic_norm(compute_data, dim=dim, variance_epsilon=transform.variance_epsilon)
    if first or transform.tmp_norm is None:
        transform.tmp_norm = norm
        transform.count = 1
    else:
        transform.tmp_norm = transform.tmp_norm * norm
        transform.count += 1
    if transform.detach_norm:
        transform.tmp_norm = transform.tmp_norm.detach()
    return (compute_data * norm).to(data_dtype)


def _dynamic_unscale(data: torch.Tensor, dim: int, transform: DynamicScalingTransform) -> torch.Tensor:
    del dim
    if transform.tmp_norm is None:
        raise RuntimeError("DynamicScalingTransform inverse called before a scale was recorded.")
    data_dtype = data.dtype
    norm = transform.tmp_norm
    transform.reset()
    return (data.float() / (norm + transform.variance_epsilon)).to(data_dtype)


def _dynamic_norm(data: torch.Tensor, dim: int, variance_epsilon: float) -> torch.Tensor:
    return torch.rsqrt(data.pow(2).mean(dim=dim, keepdim=True) + variance_epsilon)
