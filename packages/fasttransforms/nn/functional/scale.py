# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


def apply_scale_on_dim(
    data: torch.Tensor,
    scale: torch.Tensor,
    inv: bool = False,
    dim: int = -1,
    transpose: bool = False,
    inverse_eps: float = 0.0,
):
    shape = [1] * data.ndim
    shape[dim] = -1
    if inv:
        return data / (scale.view(shape) + inverse_eps)
    else:
        return data * scale.view(shape)
