# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from enum import auto
from functools import partial
from typing import Callable, List, Optional

import torch

from fasttransforms.nn.functional.dtype import cast_and_apply
from fasttransforms.nn.functional.scale import apply_scale_on_dim
from fasttransforms.types import StrEnum

from .common import MatmulTransform


class ScaleInitType(StrEnum):
    ones = auto()
    random = auto()


class ScaleParametrizationType(StrEnum):
    exponential = auto()
    softplus = auto()


class ExponentialParametrization(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)


class SoftPlusParametrization(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)


class ScaleTransform(MatmulTransform):
    required_parameters: List[str] = ["scale"]

    def __init__(
        self,
        init_type: ScaleInitType = ScaleInitType.ones,
        parametrization: ScaleParametrizationType | None = None,
        dtype: Optional[torch.dtype] | str = None,
        inverse_eps: float = 0.0,
    ):
        super().__init__()
        self._init_type = init_type
        self._parametrization = parametrization
        self._inverse_eps = inverse_eps
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError(f"Expected torch.dtype or None, got {type(dtype).__name__}")
        self._dtype: Optional[torch.dtype] = dtype
        self._scale: Optional[torch.Tensor] = None

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        if self._init_type == ScaleInitType.ones:
            scale = torch.ones(parameter_dim)
        elif self._init_type == ScaleInitType.random:
            scale = torch.randn(parameter_dim)
        else:
            raise ValueError(f"Invalid initialization {self._init_type.name}.")

        self._scale = torch.nn.Parameter(scale.to(device=device, dtype=dtype if self._dtype is None else self._dtype))

        if self._parametrization is not None:
            parametrization: Callable[[torch.Tensor], torch.Tensor]
            if self._parametrization == ScaleParametrizationType.exponential:
                parametrization = ExponentialParametrization()
            elif self._parametrization == ScaleParametrizationType.softplus:
                parametrization = SoftPlusParametrization()
            else:
                raise ValueError(f"Invalid parametrization {self._parametrization}.")
            torch.nn.utils.parametrize.register_parametrization(self, "_scale", parametrization)

    @property
    def scale(self) -> Optional[torch.Tensor]:
        return self._scale

    @scale.setter
    def scale(self, new_value: Optional[torch.Tensor]):
        if new_value is None:
            self._scale = None
        elif not hasattr(self, "_scale"):
            self._scale = torch.nn.Parameter(new_value)
        else:
            assert isinstance(self._scale, torch.Tensor)
            self._scale.data.copy_(new_value)

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        assert self.scale is not None
        return cast_and_apply(partial(apply_scale_on_dim, inv=False, scale=self.scale), dtype=self._dtype)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        assert self.scale is not None
        return cast_and_apply(
            partial(apply_scale_on_dim, inv=True, scale=self.scale, inverse_eps=self._inverse_eps),
            dtype=self._dtype,
        )
