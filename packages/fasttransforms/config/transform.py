# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from fasttransforms.nn.transforms import (
    CrossBlockDiagonalTransform,
    HadamardTransform,
    IdentityTransform,
    InvertibleBlockMatmulHadTransform,
    InvertibleKroneckerMatrixTransform,
    InvertibleKroneckerMatrixTransformWithDiagonal,
    InvertibleMatrixTransform,
    InvertibleSVDMatrixTransform,
    PerHeadInvertibleBlockMatmulTransform,
    RandomPermutation,
    RepeatedBlockDiagonalTransform,
    RotationTransform,
    ScaleTransform,
    SVDInvertibleKroneckerMatrixTransformWithDiagonal,
    Walsh2DPermutation,
    WalshPermutation,
    ZigZag2DPermutation,
)
from fasttransforms.nn.transforms.fptquant import (
    DynamicScalingTransform,
    HeadMixingTransform,
    PreRopeTransform,
    ValueBatchedTransform,
)

transform_func_store = store(group="transform", provider=__name__)

# Identity
transform_func_store(
    builds(IdentityTransform, populate_full_signature=True),
    name="identity",
)

# Scale transform (diagonal)
transform_func_store(
    builds(ScaleTransform, populate_full_signature=True),
    name="scale",
)

transform_func_store(
    builds(
        PreRopeTransform,
        num_heads="${llm_config:num_attention_heads}",
        num_key_value_heads="${llm_config:num_key_value_heads}",
        populate_full_signature=True,
    ),
    name="prerope",
)

transform_func_store(
    builds(DynamicScalingTransform, populate_full_signature=True),
    name="dynamic_scale",
)

transform_func_store(
    builds(
        ValueBatchedTransform,
        num_heads="${llm_config:num_attention_heads}",
        num_key_value_heads="${llm_config:num_key_value_heads}",
        head_dim="${llm_config:head_dim}",
        populate_full_signature=True,
    ),
    name="values_batched",
)

transform_func_store(
    builds(
        HeadMixingTransform,
        num_heads="${llm_config:num_attention_heads}",
        head_dim="${llm_config:head_dim}",
        populate_full_signature=True,
    ),
    name="head_mixing",
)

# Random permutation
transform_func_store(
    builds(RandomPermutation, populate_full_signature=True),
    name="random_permutation",
)


# Walsh permutation
transform_func_store(
    builds(WalshPermutation, populate_full_signature=True),
    name="walsh_permutation",
)
transform_func_store(
    builds(Walsh2DPermutation, populate_full_signature=True),
    name="walsh2d_permutation",
)

# ZigZag 2d permutation (ordering coefficient from 2D DCT)
transform_func_store(
    builds(ZigZag2DPermutation, populate_full_signature=True),
    name="zigzag2d_permutation",
)

# Hadamard Transform
transform_func_store(
    builds(HadamardTransform, populate_full_signature=True),
    name="hadamard",
)

# Regular Hadamard
transform_func_store(
    builds(HadamardTransform, hadamard_type="regular", populate_full_signature=True),
    name="regular_hadamard",
)

# Walsh Hadamard
transform_func_store(
    builds(HadamardTransform, permutation="walsh", populate_full_signature=True),
    name="walsh_hadamard",
)

# Randomized Hadamard
transform_func_store(
    builds(HadamardTransform, permutation="random", populate_full_signature=True),
    name="random_hadamard",
)

# Randomized Regular Hadamard
transform_func_store(
    builds(HadamardTransform, permutation="random", hadamard_type="regular", populate_full_signature=True),
    name="random_regular_hadamard",
)

# Invertible matrix
transform_func_store(
    builds(InvertibleMatrixTransform, populate_full_signature=True),
    name="invertible_matrix",
)

# Invertible matrix (SVD parametrization)
transform_func_store(
    builds(InvertibleSVDMatrixTransform, populate_full_signature=True),
    name="invertible_svd_matrix",
)

# Rotation matrix
transform_func_store(
    builds(RotationTransform, populate_full_signature=True),
    name="rotation",
)

transform_func_store(
    builds(RotationTransform, init_type="random", populate_full_signature=True),
    name="random_rotation",
)

# per-head-transform
transform_func_store(
    builds(
        RepeatedBlockDiagonalTransform,
        block_size="${llm_config:head_dim}",
        hydra_defaults=[{"/transform@base_transform": "hadamard"}],
        populate_full_signature=True,
    ),
    name="per_head",
)

# across-head-transform
transform_func_store(
    builds(
        CrossBlockDiagonalTransform,
        block_size="${llm_config:head_dim}",
        hydra_defaults=[{"/transform@base_transform": "hadamard"}],
        populate_full_signature=True,
    ),
    name="cross_head",
)

# FlatQuant Kronecker transforms
transform_func_store(
    builds(InvertibleKroneckerMatrixTransform, n_blocks=2, populate_full_signature=True),
    name="invertible_kronecker_matrix",
)

transform_func_store(
    builds(InvertibleKroneckerMatrixTransformWithDiagonal, n_blocks=2, populate_full_signature=True),
    name="invertible_kronecker_matrix_with_diagonal",
)

transform_func_store(
    builds(SVDInvertibleKroneckerMatrixTransformWithDiagonal, n_blocks=2, populate_full_signature=True),
    name="svd_invertible_kronecker_matrix_with_diagonal",
)


# Invertible block matmul followed by cross-block Hadamard
transform_func_store(
    builds(InvertibleBlockMatmulHadTransform, block_size=128, populate_full_signature=True),
    name="invertible_blockmatmul_had",
)

transform_func_store(
    builds(
        PerHeadInvertibleBlockMatmulTransform,
        head_dim="${llm_config:head_dim}",
        num_key_value_heads="${llm_config:num_key_value_heads}",
    ),
    name="per_head_invertible_blockmatmul",
)
