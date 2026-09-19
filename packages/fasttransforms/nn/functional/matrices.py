# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


def apply_matmul_on_dim(data: torch.Tensor, M: torch.Tensor, dim: int = -1, transpose: bool = False):
    data = data.transpose(dim, -1)
    data = torch.einsum("...a,ab->...b", data, M.T if transpose else M)
    data = data.transpose(dim, -1)
    return data


def apply_invmatmul_on_dim(
    data: torch.Tensor,
    M: torch.Tensor,
    dim: int = -1,
    transpose: bool = False,
    eps: float = 0.0,
    inverse_compute_dtype: torch.dtype = torch.float64,
):
    data = data.transpose(dim, -1)
    M_ = M.to(inverse_compute_dtype)
    if eps:
        M_ = M_ + eps * torch.eye(M_.shape[-1], device=M_.device, dtype=M_.dtype)
    M_inv = torch.linalg.inv(M_).to(data.dtype)
    data = torch.einsum("...a,ab->...b", data, M_inv.T if transpose else M_inv)
    data = data.transpose(dim, -1)
    return data
