# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from functools import partial
from typing import Any, Callable

import torch

from fasttransforms.nn.functional.blockmatmul import apply_blockmatmul_on_dim
from fasttransforms.nn.functional.dtype import cast_and_apply
from fasttransforms.nn.functional.per_head import apply_per_head
from fasttransforms.nn.transforms.matrices import InvertibleMatrixInitType

from .common import MatmulTransform, Transform
from .composition import CrossBlockDiagonalTransform, SequentialMatmulTransform
from .hadamard import HadamardTransform
from .matrices import InvertibleMatrixTransform


class InvertibleBlockMatmulTransform(MatmulTransform):
    required_parameters = ["M_block"]

    def __init__(
        self,
        block_size: int,
        dtype: torch.dtype | str = "float32",
        init_type: InvertibleMatrixInitType = InvertibleMatrixInitType.hadamard,
    ):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype: {dtype.__class__.__name__}")
        self._dtype = dtype
        self._block_size = block_size
        self._init_type = init_type

    @property
    def block_size(self) -> int:
        return self._block_size

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        block_size = self._block_size
        n_blocks = parameter_dim // self._block_size

        if n_blocks * block_size != parameter_dim:
            raise ValueError(f"parameter_dim {parameter_dim} is not divisible by block_size {block_size}")

        blocks = []
        for _ in range(n_blocks):
            transform: Transform = InvertibleMatrixTransform(init_type=self._init_type)
            M = transform(torch.eye(block_size)).unsqueeze(0)
            blocks.append(M)
        M_block = torch.cat(blocks, 0)

        self._M_block = torch.nn.Parameter(M_block.to(dtype=self._dtype, device=device))

    @property
    def M_block(self) -> torch.Tensor:
        return self._M_block

    @M_block.setter
    def M_block(self, new_value: torch.Tensor):
        if not hasattr(self, "_M_block"):
            if torch.is_tensor(new_value):
                new_value = torch.nn.Parameter(new_value)
            self._M_block = new_value
        else:
            self._M_block.data = new_value

    @property
    def forward_fn(self) -> Callable[[Any], torch.Tensor]:
        return cast_and_apply(partial(apply_blockmatmul_on_dim, M_block=self.M_block, inv=False), dtype=self._dtype)

    @property
    def inverse_fn(self) -> Callable[[Any], torch.Tensor]:
        return cast_and_apply(partial(apply_blockmatmul_on_dim, M_block=self.M_block, inv=True), dtype=self._dtype)

    def _extra_repr(self) -> str:
        return f"block_size={self._block_size}"


class InvertibleBlockMatmulHadTransform(SequentialMatmulTransform):
    def __init__(
        self,
        block_size: int,
        dtype: torch.dtype | str = "float32",
        init_type: InvertibleMatrixInitType = InvertibleMatrixInitType.hadamard,
    ):
        super().__init__(
            transforms=[
                InvertibleBlockMatmulTransform(block_size=block_size, dtype=dtype, init_type=init_type),
                CrossBlockDiagonalTransform(base_transform=HadamardTransform(), block_size=block_size),
            ]
        )


class PerHeadInvertibleBlockMatmulTransform(SequentialMatmulTransform):
    def __init__(
        self,
        head_dim: int,
        num_key_value_heads: int,
        init_type: Any = "hadamard",
    ):
        super().__init__(transforms=[InvertibleBlockMatmulTransform(block_size=head_dim, init_type=init_type)])

        self._head_dim = head_dim
        self._num_key_value_heads = num_key_value_heads

    @property
    def forward_fn(self) -> Callable[[Any], torch.Tensor]:
        return apply_per_head(
            self.transforms[0], inv=False, head_dim=self._head_dim, num_key_value_heads=self._num_key_value_heads
        )

    @property
    def inverse_fn(self) -> Callable[[Any], torch.Tensor]:
        return apply_per_head(
            self.transforms[0], inv=True, head_dim=self._head_dim, num_key_value_heads=self._num_key_value_heads
        )

    def _extra_repr(self) -> str:
        return f"head_dim={self._head_dim}, num_key_value_heads={self._num_key_value_heads}"
