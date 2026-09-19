# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


def apply_blockmatmul_on_dim(data: torch.Tensor, dim: int, inv: bool, transpose: bool, M_block: torch.Tensor):
    n_blocks = M_block.shape[0]
    block_size = M_block.shape[1]
    data = data.transpose(dim, -1)
    prev_shapes = list(data.shape[:-1])
    data = data.view(prev_shapes + [n_blocks, block_size])
    if inv:
        M_block = torch.linalg.inv(M_block.to(torch.float64)).to(data.dtype)
    if transpose:
        M_block = M_block.transpose(-1, -2)
    data = torch.einsum("...nb,nbc->...nc", data, M_block)
    data = data.reshape(prev_shapes + [-1])
    data = data.transpose(dim, -1)
    return data


def apply_cross_blockmatmul_on_dim(data: torch.Tensor, dim: int, inv: bool, transpose: bool, M_cross: torch.Tensor):
    n_blocks = M_cross.shape[0]
    block_size = M_cross.shape[1]
    data = data.transpose(dim, -1)
    prev_shapes = list(data.shape[:-1])
    data = data.view(prev_shapes + [n_blocks, block_size])
    if inv:
        # Invert the first two dimensions for M_cross
        M_cross = torch.linalg.inv(M_cross.transpose(0, 2).to(torch.float64)).transpose(0, 2).to(data.dtype)
    if transpose:
        M_cross = M_cross.transpose(0, 1)

    data = torch.einsum("...nb,nmb->...mb", data, M_cross)
    data = data.reshape(prev_shapes + [-1])
    data = data.transpose(dim, -1)
    return data
