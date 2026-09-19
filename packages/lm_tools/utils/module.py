# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import contextmanager

from torch import nn


@contextmanager
def gradient_enabled(module: "nn.Module"):
    original_requires_grad = {}
    for name, parameter in module.named_parameters():
        original_requires_grad[name] = parameter.requires_grad
        parameter.requires_grad = True

    yield
    for name, parameter in module.named_parameters():
        parameter.requires_grad = original_requires_grad[name]


@contextmanager
def gradient_disabled(module: "nn.Module"):
    original_requires_grad = {}
    for name, parameter in module.named_parameters():
        original_requires_grad[name] = parameter.requires_grad
        parameter.requires_grad = False

    yield
    for name, parameter in module.named_parameters():
        parameter.requires_grad = original_requires_grad[name]


def remove_submodule(module: nn.Module, key: str) -> None:
    subkeys = key.split(".")
    if len(subkeys) == 1:
        delattr(module, subkeys[0])
    else:
        if subkeys[0].isnumeric():
            if not isinstance(module, nn.ModuleList):
                raise TypeError(f"Expected nn.ModuleList for numeric key {subkeys[0]!r}, got {type(module).__name__}")
            remove_submodule(module[int(subkeys[0])], ".".join(subkeys[1:]))
        else:
            if not hasattr(module, subkeys[0]):
                raise KeyError(f"The module {type(module).__name__} has no attribute {subkeys[0]}")
            remove_submodule(module.get_submodule(subkeys[0]), ".".join(subkeys[1:]))


def has_submodule(module: nn.Module, key: str) -> bool:
    subkeys = key.split(".")
    if len(subkeys) == 1:
        return hasattr(module, subkeys[0])
    else:
        if subkeys[0].isnumeric():
            if not isinstance(module, nn.ModuleList):
                raise TypeError(f"Expected nn.ModuleList for numeric key {subkeys[0]!r}, got {type(module).__name__}")
            return has_submodule(module[int(subkeys[0])], ".".join(subkeys[1:]))
        else:
            if not hasattr(module, subkeys[0]):
                raise KeyError(f"The module {type(module).__name__} has no attribute {subkeys[0]}")
            return has_submodule(module.get_submodule(subkeys[0]), ".".join(subkeys[1:]))
