# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from abc import ABC
from functools import partial
from typing import Callable, List, Sequence

import torch

from fasttransforms.nn.functional.composition import (
    apply_block_transforms,
    apply_kronecker,
    compose_transforms,
    equal_split,
    exp_split,
    resolve_sizes_block,
    validate_sizes,
)

from .common import IdentityTransform, MatmulTransform, Transform

SplitSpec = List[int] | Callable[[int], List[int]]


class TransformList(Transform, ABC):
    def __init__(
        self,
        transforms: Sequence[Transform] | None = None,
    ):
        if transforms is None:
            transforms = []
        super().__init__()
        self._transforms = torch.nn.ModuleList(transforms)

        for i, transform in enumerate(transforms):
            if transform.metadata._name is None:
                transform.metadata._name = str(i)
                transform.metadata._parent = self._metadata

    @property
    def transforms(self):
        return self._transforms

    def _extra_repr(self):
        tmp = ",\n  ".join([transform.__repr__().replace("\n", "\n  ") for transform in self.transforms])
        if len(tmp) > 0:
            tmp = "transforms=[\n  " + tmp + "\n]"
        return tmp

    @property
    def initialized(self) -> bool:
        for param_name in self.required_parameters:
            if getattr(self, param_name) is None:
                return False
        for transform in self.transforms:
            if not transform.initialized:
                return False
        return True


class SequentialMatmulTransform(MatmulTransform, TransformList, ABC):
    def __init__(
        self,
        transforms: List[MatmulTransform] | None = None,
    ):
        if transforms is None:
            transforms = []

        for transform in transforms:
            if not isinstance(transform, MatmulTransform):
                raise ValueError(f"Transform {transform} must be an instance of MatmulTransform")

        TransformList.__init__(self, transforms=transforms)

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        pass

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(compose_transforms, inv=False, transforms=self.transforms)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(compose_transforms, inv=True, transforms=self.transforms)

    def _extra_repr(self):
        return TransformList._extra_repr(self)


class KroneckerProductTransform(MatmulTransform, TransformList):
    required_parameters: List[str] = ["sizes"]

    def __init__(
        self,
        transforms: List[MatmulTransform],
        split: List[int] | Callable[[int], List[int]] | int,
    ):
        if isinstance(split, list):
            validate_sizes(split)
        elif isinstance(split, int):
            split = partial(exp_split, base=split)

        TransformList.__init__(self, transforms=transforms)
        self._split = split

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        # Determine the factorization
        if isinstance(self._split, list):
            sizes = self._split
        else:
            sizes = self._split(parameter_dim)

        self.sizes = sizes

        if (len(self.transforms) != len(sizes)) and len(self.transforms) > 1:
            raise ValueError(
                f"The number of transforms {len(self.transforms)} does not match the number of splits {len(sizes)}."
            )

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(apply_kronecker, transforms=self.transforms, inv=False, sizes=self.sizes)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(apply_kronecker, transforms=self.transforms, inv=True, sizes=self.sizes)

    def _extra_repr(self):
        return TransformList._extra_repr(self) + f",\nsizes={self.sizes}"


class KroneckerPowerTransform(KroneckerProductTransform):
    def __init__(
        self,
        transform: MatmulTransform,
        exponent: int | None = None,
        block_size: int | None = None,
    ):
        self._exponent = exponent
        self._block_size = block_size
        if exponent is None and block_size is None:
            raise ValueError("Please specify either exponent or block_size.")
        if block_size is None:

            def split_in_n(parameter_dim: int, exponent: int) -> List[int]:
                block_size = round(parameter_dim ** (1.0 / exponent))

                if block_size**exponent != parameter_dim:
                    raise RuntimeError(
                        f"The size {parameter_dim} can't be factorized as a kronecker exponential of {exponent} steps."
                    )
                return [block_size] * exponent

            assert isinstance(exponent, int)
            split = partial(split_in_n, exponent=exponent)
        else:

            def split_in_size(parameter_dim: int, block_size: int) -> List[int]:
                n_blocks = 1
                d = parameter_dim
                while d > block_size:
                    d //= block_size
                    n_blocks += 1

                if block_size**n_blocks != parameter_dim:
                    raise RuntimeError(
                        f"The size {parameter_dim} can't be factorized as a Kronecker exponential with block of size {block_size}."
                    )
                return [block_size] * n_blocks

            split = partial(split_in_size, block_size=block_size)
        super().__init__(transforms=[transform], split=split)

    def _extra_repr(self):
        tmp = "transform=" + self.transforms[0].__repr__().replace("\n", "\n  ") + ","
        if self._exponent is not None:
            tmp += f"\nexponent={self._exponent}"
        else:
            tmp += f"\nblock_size={self._block_size}"
        return tmp


class BlockDiagonalTransform(MatmulTransform, TransformList):
    required_parameters: List[str] = ["sizes"]

    def __init__(
        self,
        transforms: List[Transform],
        split: List[int] | Callable[[int], List[int]],
    ):
        if isinstance(split, list):
            validate_sizes(split)
        elif isinstance(split, int):
            split = partial(equal_split, base=split)

        for transform in transforms:
            if not isinstance(transform, MatmulTransform):
                raise ValueError(
                    f"All transforms must be instances of MatmulTransform: {transform.__class__.__name__} was found."
                )

        TransformList.__init__(self, transforms=transforms)
        self._split = split

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        # Determine the factorization
        if isinstance(self._split, list):
            sizes = resolve_sizes_block(parameter_dim, sizes=self._split)
        else:
            sizes = self._split(parameter_dim)

        self.sizes = sizes

        if len(self.transforms) != len(sizes):
            raise ValueError(
                f"The number of transforms {len(self.transforms)} does not match the number of splits {len(sizes)}."
            )

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return partial(apply_block_transforms, inv=False, transforms=self.transforms, sizes=self.sizes)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return partial(apply_block_transforms, inv=True, transforms=self.transforms, sizes=self.sizes)

    def _extra_repr(self):
        return SequentialMatmulTransform._extra_repr(self).replace("\n", "\n  ") + f",\n  sizes={self.sizes}"


class RepeatedBlockDiagonalTransform(KroneckerProductTransform):
    def __init__(
        self,
        base_transform: MatmulTransform,
        block_size: int | Callable[[int], int] | None = None,
        n_blocks: int | None = None,
    ):
        if not ((block_size is None) ^ (n_blocks is None)):
            raise ValueError("Please specify either block_size or n_blocks.")
        self._block_size = block_size
        self._n_blocks = n_blocks
        split: SplitSpec
        if block_size is not None:
            if isinstance(block_size, int):
                split = [-1, block_size]
            else:

                def split(dim: int):
                    return [-1, block_size(dim)]
        else:
            assert isinstance(n_blocks, int)
            split = [n_blocks, -1]
        super().__init__(transforms=[IdentityTransform(), base_transform], split=split)

    def _extra_repr(self):
        tmp = "transform=" + self.transforms[1].__repr__().replace("\n", "\n  ") + ","
        if self._block_size is not None:
            tmp += f"\nblock_size={self._block_size}"
        else:
            tmp += f"\nn_blocks={self._n_blocks}"
        return tmp


class CrossBlockDiagonalTransform(KroneckerProductTransform):
    def __init__(
        self,
        base_transform: MatmulTransform,
        block_size: int | Callable[[int], int] | None = None,
        n_blocks: int | None = None,
    ):
        if not ((block_size is None) ^ (n_blocks is None)):
            raise ValueError("Please specify either block_size or n_blocks.")
        self._block_size = block_size
        self._n_blocks = n_blocks
        split: SplitSpec

        if block_size is not None:
            if isinstance(block_size, int):
                split = [-1, block_size]
            else:

                def split_block(dim: int):
                    return [-1, block_size(dim)]

                split = split_block
        else:
            assert isinstance(n_blocks, int)
            split = [n_blocks, -1]
        super().__init__(transforms=[base_transform, IdentityTransform()], split=split)

    def _extra_repr(self):
        tmp = "transform=" + self.transforms[0].__repr__().replace("\n", "\n  ") + ","
        if self._block_size is not None:
            tmp += f"\nblock_size={self._block_size}"
        else:
            tmp += f"\nn_blocks={self._n_blocks}"
        return tmp
