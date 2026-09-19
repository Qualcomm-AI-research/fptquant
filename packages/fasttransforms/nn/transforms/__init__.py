# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import List

from .blockmatmul import (
    InvertibleBlockMatmulHadTransform,
    InvertibleBlockMatmulTransform,
    PerHeadInvertibleBlockMatmulTransform,
)
from .common import IdentityTransform, MatmulTransform, OrthogonalTransform, Transform, invert, transpose
from .composition import (
    BlockDiagonalTransform,
    CrossBlockDiagonalTransform,
    KroneckerPowerTransform,
    KroneckerProductTransform,
    RepeatedBlockDiagonalTransform,
    SequentialMatmulTransform,
)
from .flatquant import (
    InvertibleKroneckerMatrixTransform,
    InvertibleKroneckerMatrixTransformWithDiagonal,
    SVDInvertibleKroneckerMatrixTransform,
    SVDInvertibleKroneckerMatrixTransformWithDiagonal,
)
from .fptquant import DynamicScalingTransform, HeadMixingTransform, PreRopeTransform, ValueBatchedTransform
from .hadamard import HadamardTransform
from .matrices import (
    InvertibleMatrixInitType,
    InvertibleMatrixTransform,
    InvertibleSVDMatrixTransform,
    RotationTransform,
)
from .permutation import Permutation, RandomPermutation, Walsh2DPermutation, WalshPermutation, ZigZag2DPermutation
from .scale import ScaleTransform
from .utils import find_transforms, get_transforms

__all__: List[str] = [
    "Transform",
    "IdentityTransform",
    "MatmulTransform",
    "OrthogonalTransform",
    "transpose",
    "invert",
    "SequentialMatmulTransform",
    "Permutation",
    "ScaleTransform",
    "RandomPermutation",
    "WalshPermutation",
    "Walsh2DPermutation",
    "ZigZag2DPermutation",
    "HadamardTransform",
    "InvertibleBlockMatmulTransform",
    "InvertibleBlockMatmulHadTransform",
    "PerHeadInvertibleBlockMatmulTransform",
    "InvertibleMatrixInitType",
    "PreRopeTransform",
    "ValueBatchedTransform",
    "HeadMixingTransform",
    "DynamicScalingTransform",
    "InvertibleMatrixTransform",
    "InvertibleSVDMatrixTransform",
    "RotationTransform",
    "KroneckerProductTransform",
    "KroneckerPowerTransform",
    "BlockDiagonalTransform",
    "RepeatedBlockDiagonalTransform",
    "CrossBlockDiagonalTransform",
    "InvertibleKroneckerMatrixTransform",
    "SVDInvertibleKroneckerMatrixTransform",
    "InvertibleKroneckerMatrixTransformWithDiagonal",
    "SVDInvertibleKroneckerMatrixTransformWithDiagonal",
    "get_transforms",
    "find_transforms",
]
