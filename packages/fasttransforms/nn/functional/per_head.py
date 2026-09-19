# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable


def apply_per_head(transform: Callable, inv: bool, head_dim: int, num_key_value_heads: int):
    # This is applied to flattened head dimension.
    # The operation is used in Pv and uses the repeat interleaved operation required to broadcast the multi-head attentions
    # Note that head_dim refers to the v_proj head dimensions and num_key_value_heads to the number of key-value heads.
    # The input dimension of o_proj will be a multiple of head_dim * num_key_value_heads for which the operation "repeat" will be applied.
    def wrapper(data, dim, **kwargs):
        # If transforming the values or key heads
        if data.shape[dim] == head_dim * num_key_value_heads:
            # Apply the transform directly
            return transform(data=data, inv=inv, dim=dim, **kwargs)
        # Otherwise check the heads are repeated and use the repeat interleaved
        else:
            assert data.shape[dim] % (head_dim * num_key_value_heads) == 0
            data = data.transpose(dim, -1)
            base_shape = list(data.shape[:-1])
            data = data.view(base_shape + [num_key_value_heads, -1, head_dim]).transpose(-2, -3)
            data = data.reshape(base_shape + [num_key_value_heads, -1])
            data = data.view(base_shape + [-1] + [head_dim * num_key_value_heads])

            transformed_data = transform(data, inv=inv, dim=-1, **kwargs)
            transformed_data = transformed_data.view(base_shape + [-1, num_key_value_heads, head_dim]).transpose(
                -2, -3
            )
            transformed_data = transformed_data.reshape(base_shape + [-1])
            transformed_data = transformed_data.transpose(dim, -1)
            return transformed_data

    return wrapper
