# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import math
from enum import auto
from typing import Callable, List

import torch

from fasttransforms.nn.functional.dtype import cast_and_apply
from fasttransforms.nn.functional.hadamard import (
    apply_regular_hadamard_on_dim,
    apply_sylvester_hadamard_on_dim,
    get_max_pow_divisor,
)
from fasttransforms.types import StrEnum

from .common import IdentityTransform, OrthogonalTransform
from .composition import KroneckerPowerTransform, KroneckerProductTransform, SequentialMatmulTransform, SplitSpec
from .permutation import RandomPermutation, Walsh2DPermutation, WalshPermutation

fht: Callable[[torch.Tensor, int, bool], torch.Tensor] | None

try:
    from fast_hadamard_transform import hadamard_transform  # type: ignore[import-not-found]

    def _fast_hadamard_transform(x: torch.Tensor, dim: int, transpose: bool = False) -> torch.Tensor:
        """
        Hadamard transform applied to dim dim. transpose included
        for compatibility, but H=H^T, so it's ignored.
        """
        x = x.transpose(dim, -1)
        assert (math.log2(x.shape[-1]) % 1) == 0
        scale = 1.0 / math.sqrt(x.shape[-1])
        return hadamard_transform(x, scale).transpose(dim, -1)

    fht = _fast_hadamard_transform

except ImportError:
    logger = logging.getLogger("fasttransforms")
    logger.warning(
        "The package fast_hadamard_transform is not installed. The performance of online (sylvester) hadamard will be impacted"
    )
    fht = None

    def apply_and_scale(fn):
        def wrapper(data, dim, *args, **kwargs):
            data = fn(data, dim=dim, *args, **kwargs)
            data = data / math.sqrt(data.shape[dim])
            return data

        return wrapper


class HadamardType(StrEnum):
    sylvester = auto()
    regular = auto()


class HadamardPermutationType(StrEnum):
    identity = auto()
    random = auto()
    walsh = auto()
    walsh2d = auto()


class SylvesterHadamard2Transform(OrthogonalTransform):
    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        if parameter_dim != 2:
            raise RuntimeError(f"The transformation can only be applied to a size of 2, {parameter_dim} was found.")

    @property
    def forward_fn(self):
        return apply_sylvester_hadamard_on_dim


class SylvesterHadamardTransform(KroneckerPowerTransform):
    def __init__(self, dtype: torch.dtype | str = "float32"):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        super().__init__(SylvesterHadamard2Transform(), block_size=2)
        self._dtype = dtype

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        # If possible use the custom fht kernel by casting to float32 and back
        if fht:
            return cast_and_apply(fht, dtype=torch.float32)
        # Otherwise, use the default Kronecker product implementation
        else:
            return cast_and_apply(apply_and_scale(super().forward_fn), dtype=self._dtype)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return self.forward_fn

    def _extra_repr(self) -> str:
        return ""


class RegularHadamard4Transform(OrthogonalTransform):
    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        if parameter_dim != 4:
            raise RuntimeError(f"The transformation can only be applied to a size of 4, {parameter_dim} was found.")

    @property
    def forward_fn(self):
        return apply_regular_hadamard_on_dim


class RegularHadamardTransform(KroneckerPowerTransform):
    def __init__(self, dtype: torch.dtype | str = "float32"):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        super().__init__(RegularHadamard4Transform(), block_size=4)
        self._dtype = dtype

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return cast_and_apply(super().forward_fn, dtype=self._dtype)

    def _extra_repr(self) -> str:
        return ""


class HadamardTransform(SequentialMatmulTransform):
    def __init__(
        self,
        size: int | None = None,
        hadamard_type: HadamardType = HadamardType.sylvester,
        permutation: HadamardPermutationType = HadamardPermutationType.identity,
        dtype: torch.dtype | str = "float32",
    ):
        self._hadamard_type = hadamard_type
        self._permutation = permutation

        # Initialze the type of hadamard
        split: SplitSpec
        if hadamard_type == HadamardType.sylvester:
            transforms = [IdentityTransform(), SylvesterHadamardTransform(dtype=dtype)]
            if size is None:

                def split_pow_2(parameter_dim: int) -> List[int]:
                    return [-1, get_max_pow_divisor(parameter_dim, base=2)]

                split = split_pow_2
            else:
                split = [-1, size]
        elif hadamard_type == HadamardType.regular:
            transforms = [IdentityTransform(), RegularHadamardTransform(dtype=dtype)]
            if size is None:

                def split_pow_4(parameter_dim: int) -> List[int]:
                    return [-1, get_max_pow_divisor(parameter_dim, base=4)]

                split = split_pow_4
            else:
                split = [-1, size]
        else:
            raise ValueError(f"Invalid value for 'hadamard_type': {hadamard_type}.")

        transforms = [
            KroneckerProductTransform(
                transforms=transforms,
                split=split,
            )
        ]

        # Add a permutation if specified
        if permutation == HadamardPermutationType.identity:
            pass
        elif permutation == HadamardPermutationType.random:
            transforms.append(RandomPermutation())
        elif permutation == HadamardPermutationType.walsh:
            transforms.append(WalshPermutation())
        elif permutation == HadamardPermutationType.walsh2d:
            transforms.append(Walsh2DPermutation())
        else:
            raise ValueError(f"Unrecognized Permutation {permutation}.")

        super().__init__(transforms=transforms)

    def _extra_repr(self) -> str:
        tmp = []
        if self._hadamard_type != HadamardType.sylvester:
            tmp += [f"hadamard_type={self._hadamard_type}"]
        if self._permutation != HadamardPermutationType.identity:
            tmp += [f"permutation={self._permutation}"]

        return ", ".join(tmp)
