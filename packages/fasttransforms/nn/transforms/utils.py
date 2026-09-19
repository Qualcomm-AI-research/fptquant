# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Iterator

import torch
from fastforward import mpath

from .common import Transform


def get_transforms(model: torch.nn.Module, recurse: bool = False) -> Iterator[Transform]:
    from ..operation import TransformOperation

    modules = [module for _, module in model.named_modules()]
    transforms = []
    if not recurse:
        for module in modules:
            if isinstance(module, TransformOperation) and module.transform not in transforms:
                transforms.append(module.transform)
                yield module.transform
    else:
        for module in modules:
            if isinstance(module, Transform) and module not in transforms:
                transforms.append(module)
                yield module


def find_transforms(root_module: torch.nn.Module, query: str, **kwargs) -> torch.nn.ModuleList:
    transforms = []
    for res in mpath.search(query, root_module, **kwargs):
        module = res.module
        if isinstance(module, Transform) and module not in transforms:
            transforms.append(module)

    return torch.nn.ModuleList(transforms)
