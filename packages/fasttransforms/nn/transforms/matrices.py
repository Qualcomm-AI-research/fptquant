# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from enum import auto
from functools import partial
from typing import Callable, List

import torch

from fasttransforms.nn.functional.dtype import cast_and_apply
from fasttransforms.nn.functional.matrices import apply_invmatmul_on_dim, apply_matmul_on_dim
from fasttransforms.types import StrEnum

from .common import MatmulTransform, OrthogonalTransform, transpose
from .composition import SequentialMatmulTransform
from .scale import ScaleInitType, ScaleParametrizationType, ScaleTransform


class RotationInitType(StrEnum):
    random = auto()
    hadamard = auto()


class RotationTransform(OrthogonalTransform):
    required_parameters: List[str] = ["R"]

    def __init__(
        self,
        init_type: RotationInitType = RotationInitType.hadamard,
        orthogonal_map: str = "cayley",
        use_trivialization: bool = True,
        dtype: torch.dtype | str = "float32",
    ):
        super().__init__()

        self._orthogonal_map = orthogonal_map
        self._use_trivialization = use_trivialization
        self._init_type = init_type
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype: {dtype.__class__.__name__}")
        self._dtype = dtype

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        if self._init_type == RotationInitType.random:
            R = torch.randn(parameter_dim, parameter_dim, dtype=self._dtype, device=device)
        elif self._init_type == RotationInitType.hadamard:
            from .hadamard import HadamardTransform

            R = HadamardTransform().forward(torch.eye(parameter_dim, dtype=self._dtype, device=device))
        else:
            raise ValueError(f"Unsupported init_type: {self._init_type}")

        self.R = torch.nn.Parameter(R)
        torch.nn.utils.parametrizations.orthogonal(
            self, "R", orthogonal_map=self._orthogonal_map, use_trivialization=self._use_trivialization
        )

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return cast_and_apply(partial(apply_matmul_on_dim, M=self.R), dtype=self._dtype)


class InvertibleSVDMatrixTransform(SequentialMatmulTransform):
    def __init__(
        self,
        rotation_init_type: RotationInitType = RotationInitType.random,
        orthogonal_map: str = "cayley",
        use_trivialization: bool = True,
        scale_parametrization: ScaleParametrizationType | None = ScaleParametrizationType.softplus,
        scale_init_type: ScaleInitType = ScaleInitType.random,
        dtype: torch.dtype | str = "float32",
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        rotation = RotationTransform(
            init_type=rotation_init_type,
            orthogonal_map=orthogonal_map,
            use_trivialization=use_trivialization,
        )
        scale = ScaleTransform(init_type=scale_init_type, parametrization=scale_parametrization)
        super().__init__(
            transforms=[rotation, scale, transpose(rotation)],
        )
        self._dtype = dtype

    def _extra_repr(self):
        return ""

    @property
    def forward_fn(self):
        return cast_and_apply(super().forward_fn, dtype=self._dtype)

    @property
    def inverse_fn(self):
        return cast_and_apply(super().inverse_fn, dtype=self._dtype)


class InvertibleMatrixInitType(StrEnum):
    identity = auto()
    random = auto()
    hadamard = auto()
    random_orthogonal = auto()


class InvertibleMatrixTransform(MatmulTransform):
    required_parameters: List[str] = ["M"]

    def __init__(
        self,
        dtype: torch.dtype | str = "float32",
        init_type: InvertibleMatrixInitType = InvertibleMatrixInitType.hadamard,
        inverse_eps: float = 0.0,
        inverse_compute_dtype: torch.dtype | str = "float64",
    ):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype: {dtype.__class__.__name__}")
        if isinstance(inverse_compute_dtype, str):
            inverse_compute_dtype = getattr(torch, inverse_compute_dtype)
        if not isinstance(inverse_compute_dtype, torch.dtype):
            raise ValueError(f"Invalid inverse_compute_dtype: {inverse_compute_dtype.__class__.__name__}")
        self._dtype = dtype
        self._init_type = init_type
        self._inverse_eps = inverse_eps
        self._inverse_compute_dtype = inverse_compute_dtype

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        if self._init_type == InvertibleMatrixInitType.identity:
            M = torch.eye(parameter_dim).to(dtype=self._dtype, device=device)
        elif self._init_type == InvertibleMatrixInitType.random:
            M = torch.randn(parameter_dim, parameter_dim).to(dtype=self._dtype, device=device)
        else:
            if self._init_type == InvertibleMatrixInitType.hadamard:
                from .hadamard import HadamardTransform

                transform: MatmulTransform = HadamardTransform()
            elif self._init_type == InvertibleMatrixInitType.random_orthogonal:
                transform = RotationTransform(init_type=RotationInitType.random)
            M = transform(torch.eye(parameter_dim)).to(dtype=self._dtype, device=device)
        self._M = torch.nn.Parameter(M)

    @property
    def M(self) -> torch.Tensor:
        return self._M

    @M.setter
    def M(self, new_value: torch.Tensor):
        if not hasattr(self, "_M"):
            if torch.is_tensor(new_value):
                new_value = torch.nn.Parameter(new_value)
            self._M = new_value
        else:
            self._M.data = new_value

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return cast_and_apply(partial(apply_matmul_on_dim, M=self.M), dtype=self._dtype)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return cast_and_apply(
            partial(
                apply_invmatmul_on_dim,
                M=self.M,
                eps=self._inverse_eps,
                inverse_compute_dtype=self._inverse_compute_dtype,
            ),
            dtype=self._dtype,
        )
