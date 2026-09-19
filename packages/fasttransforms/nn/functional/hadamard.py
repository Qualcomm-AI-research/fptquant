# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


def get_max_pow_divisor(dim: int, base: int) -> int:
    n = 1
    while dim % base == 0:
        dim //= base
        n *= base
    return n


def is_pow(dim: int, base: int):
    while dim > 1:
        if dim % base != 0:
            return False
        dim = dim // base

    return dim == 1


def apply_sylvester_hadamard_on_dim(data: torch.Tensor, dim: int = -1, transpose: bool = False):
    out = data.transpose(dim, -1) + 0

    out[..., 0] = out[..., 0] + out[..., 1]
    out[..., 1] = out[..., 0] - 2 * out[..., 1]

    out = out.transpose(dim, -1)
    return out


def apply_regular_hadamard_on_dim(data: torch.Tensor, dim: int = -1, transpose: bool = False):
    if data.shape[dim] != 4:
        raise RuntimeError(f"The transformation can only be applied to a size of 4, {data.shape[dim]} was found.")
    data = data.sum(dim, keepdim=True) / 2.0 - data

    return data
