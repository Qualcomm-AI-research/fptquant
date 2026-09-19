# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from functools import partial
from heapq import heapify, heappop, heappush
from typing import List, Literal, Optional

import torch

from fasttransforms.nn.transforms.common import MatmulTransform
from fasttransforms.nn.transforms.composition import KroneckerProductTransform, SequentialMatmulTransform
from fasttransforms.nn.transforms.matrices import (
    InvertibleMatrixInitType,
    InvertibleMatrixTransform,
    InvertibleSVDMatrixTransform,
    RotationInitType,
)
from fasttransforms.nn.transforms.scale import ScaleInitType, ScaleParametrizationType, ScaleTransform


def _prime_factors(m: int) -> list:
    factors = []
    while m % 2 == 0:
        factors.append(2)
        m //= 2
    p = 3
    while p * p <= m:
        while m % p == 0:
            factors.append(p)
            m //= p
        p += 2
    if m > 1:
        factors.append(m)
    return factors


KroneckerFactorOrder = Literal["balanced", "old"]


def factorize_in_n(n: int, k: int, factor_order: KroneckerFactorOrder = "balanced") -> list:
    """Return k integers whose product is n, as balanced (close) as possible."""
    if factor_order not in ("balanced", "old"):
        raise ValueError(f"Unknown factor_order: {factor_order}")
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if n == 0:
        return [0] + [1] * (k - 1)

    sign = -1 if n < 0 else 1
    m = abs(n)

    if m == 1:
        res = [1] * k
        res[0] *= sign
        return res

    pf = _prime_factors(m)
    pf.sort(reverse=True)

    heap = [(1, i) for i in range(k)]
    heapify(heap)
    res = [1] * k

    for p in pf:
        val, i = heappop(heap)
        val *= p
        res[i] = val
        heappush(heap, (val, i))

    if sign < 0:
        idx = min(range(k), key=lambda i: abs(res[i]))
        res[idx] = -res[idx]

    if factor_order == "old":
        return list(reversed(res))
    return res


class InvertibleKroneckerMatrixTransform(KroneckerProductTransform):
    required_parameters: List[str] = ["sizes"]

    def __init__(
        self,
        n_blocks: int,
        init_type: InvertibleMatrixInitType = InvertibleMatrixInitType.hadamard,
        dtype: torch.dtype | str = "float32",
        inverse_eps: float = 0.0,
        inverse_compute_dtype: torch.dtype | str = "float64",
        factor_order: KroneckerFactorOrder = "balanced",
    ):
        self._n_blocks = n_blocks
        transforms: List[MatmulTransform] = [
            InvertibleMatrixTransform(
                init_type=init_type,
                dtype=dtype,
                inverse_eps=inverse_eps,
                inverse_compute_dtype=inverse_compute_dtype,
            )
            for _ in range(n_blocks)
        ]
        super().__init__(transforms=transforms, split=partial(factorize_in_n, k=n_blocks, factor_order=factor_order))

    def _extra_repr(self) -> str:
        return f"sizes={self.sizes}"


class SVDInvertibleKroneckerMatrixTransform(KroneckerProductTransform):
    required_parameters: List[str] = ["sizes"]

    def __init__(
        self,
        n_blocks: int,
        rotation_init_type: RotationInitType = RotationInitType.random,
        scale_init_type: ScaleInitType = ScaleInitType.ones,
        dtype: torch.dtype | str = "float32",
    ):
        self._n_blocks = n_blocks
        transforms: List[MatmulTransform] = [
            InvertibleSVDMatrixTransform(
                rotation_init_type=rotation_init_type,
                scale_init_type=scale_init_type,
                scale_parametrization=None,
                use_trivialization=False,
                dtype=dtype,
            )
            for _ in range(n_blocks)
        ]
        super().__init__(transforms=transforms, split=partial(factorize_in_n, k=n_blocks))

    def _extra_repr(self) -> str:
        return f"sizes={self.sizes}"


class InvertibleKroneckerMatrixTransformWithDiagonal(SequentialMatmulTransform):
    def __init__(
        self,
        n_blocks: int,
        scale_init_type: ScaleInitType = ScaleInitType.ones,
        kron_init_type: InvertibleMatrixInitType = InvertibleMatrixInitType.hadamard,
        scale_parametrization: Optional[ScaleParametrizationType] = None,
        dtype: str = "float32",
        inverse_eps: float = 0.0,
        inverse_compute_dtype: torch.dtype | str = "float64",
        diagonal_inverse_eps: float = 0.0,
        factor_order: KroneckerFactorOrder = "balanced",
    ):
        transforms = [
            ScaleTransform(
                init_type=scale_init_type,
                parametrization=scale_parametrization,
                dtype=dtype,
                inverse_eps=diagonal_inverse_eps,
            ),
            InvertibleKroneckerMatrixTransform(
                n_blocks,
                kron_init_type,
                dtype=dtype,
                inverse_eps=inverse_eps,
                inverse_compute_dtype=inverse_compute_dtype,
                factor_order=factor_order,
            ),
        ]
        super().__init__(transforms=transforms)


class SVDInvertibleKroneckerMatrixTransformWithDiagonal(SequentialMatmulTransform):
    def __init__(
        self,
        n_blocks: int,
        scale_init_type: ScaleInitType = ScaleInitType.ones,
        rotation_init_type: RotationInitType = RotationInitType.random,
        scale_parametrization: Optional[ScaleParametrizationType] = None,
        dtype: str = "float32",
    ):
        transforms = [
            ScaleTransform(init_type=scale_init_type, parametrization=scale_parametrization, dtype=dtype),
            SVDInvertibleKroneckerMatrixTransform(
                n_blocks,
                rotation_init_type=rotation_init_type,
                scale_init_type=scale_init_type,
                dtype=dtype,
            ),
        ]
        super().__init__(transforms=transforms)
