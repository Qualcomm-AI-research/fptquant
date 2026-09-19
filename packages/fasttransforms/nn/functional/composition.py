# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import List

import torch


def compose_transforms(data: torch.Tensor, dim: int, transpose: bool, inv: bool, transforms: List):
    # If transpose or inverse (not both), reverse the order
    if transpose ^ inv:
        transforms = list(reversed(transforms))

    for transform in transforms:
        data = transform(data=data, dim=dim, transpose=transpose, inv=inv)

    return data


def validate_sizes(sizes):
    n_placeholders = 0
    for s in sizes:
        if s == -1:
            n_placeholders += 1
        elif s <= -1:
            raise ValueError("Dimensions can't be negative")

    if n_placeholders > 1:
        raise ValueError(f"Cannot have more than one placeholder dimension: {sizes}.")


def resolve_sizes(total_size, sizes) -> List[int]:
    active_size = 1
    placeholder_index = None

    for i, s in enumerate(sizes):
        if s == -1:
            if placeholder_index is not None:
                raise ValueError("Only one dimension can be -1")
            placeholder_index = i
        else:
            active_size *= s

    if placeholder_index is not None:
        if total_size % active_size != 0:
            raise RuntimeError(f"Size {total_size} is not compatible with the sizes {sizes}")
        inferred_size = total_size // active_size
        sizes[placeholder_index] = inferred_size
    elif active_size != total_size:
        raise RuntimeError(f"Size {total_size} is not compatible with the sizes {sizes}")

    return sizes


def exp_split(parameter_dim: int, base: int) -> List[int]:
    exp = 1
    s = parameter_dim
    while s > base:
        s //= base
        exp += 1

    if base**exp != parameter_dim:
        raise ValueError(f"{parameter_dim} can't be written as an exponential with base {base}")
    return [base] * exp


def apply_kronecker(data: torch.Tensor, dim: int, sizes: List[int], inv: bool, transpose: bool, transforms: List):
    orig_shape = data.shape
    if dim < 0:
        dim = dim + data.ndim

    new_shape = list(data.shape[:dim]) + sizes + list(data.shape[dim + 1 :])
    data = data.reshape(new_shape)

    # Determine if we want to broadcast (use one transform everywhere)
    if len(transforms) == 1 and len(sizes) > 1:
        transform = transforms[0]
        for i in range(len(sizes)):
            data = transform(data=data, dim=dim + i, transpose=transpose, inv=inv)
    else:
        for i, transform in enumerate(transforms):
            data = transform(data=data, dim=dim + i, transpose=transpose, inv=inv)

    return data.reshape(orig_shape)


def resolve_sizes_block(total_size: int, sizes: List[int]):
    active_size = 0
    placeholder_index = None

    for i, s in enumerate(sizes):
        if s == -1:
            if placeholder_index is not None:
                raise ValueError("Only one dimension can be -1")
            placeholder_index = i
        else:
            active_size += s

    if placeholder_index is not None:
        if total_size == active_size:
            raise RuntimeError(f"Size {total_size} is not compatible with the sizes {sizes}")
        inferred_size = total_size - active_size
        sizes[placeholder_index] = inferred_size
    return sizes


def equal_split(parameter_dim: int, base: int) -> List[int]:
    n_splits = parameter_dim // base

    if base * n_splits != parameter_dim:
        raise ValueError(f"{parameter_dim} can't be written a block diagonal matrix with of blocks of size {base}")
    return [base] * n_splits


def apply_block_transforms(
    data: torch.Tensor, dim: int, sizes: List[int], inv: bool, transpose: bool, transforms: List
):
    data_chunks = list(torch.split(data, sizes, dim=dim))

    for i, transform in enumerate(transforms):
        data_chunks[i] = transform(data_chunks[i], dim=dim, transpose=transpose, inv=inv)

    return torch.cat(data_chunks, dim=dim)
