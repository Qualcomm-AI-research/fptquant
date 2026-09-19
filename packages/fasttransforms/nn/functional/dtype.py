# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Optional

import torch


def cast_and_apply(func, dtype: Optional[torch.dtype | str]):
    if dtype is None:
        return func

    dtype = get_torch_dtype(dtype)

    def cast_wrapper(data: torch.Tensor, *args, **kwargs):
        original_dtype = data.dtype
        output = func(data.to(dtype=dtype), *args, **kwargs)
        return output.to(dtype=original_dtype)

    return cast_wrapper


def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Expected torch.dtype, got {type(dtype).__name__}")
    return dtype
