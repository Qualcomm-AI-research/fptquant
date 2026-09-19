# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


def permute(data: torch.Tensor, order: torch.LongTensor, dim: int, transpose: bool) -> torch.Tensor:
    if transpose:
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), dtype=torch.long, device=order.device)
        return torch.index_select(data, dim, inv_order)
    return torch.index_select(data, dim, order)
