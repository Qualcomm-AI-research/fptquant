# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import gc

import torch
from fastforward.nn.linear import QuantizedLinear
from fastforward.nn.quantizer import QuantizerStub
from fastforward.quantized_tensor import QuantizedTensor


def replace_module(model: torch.nn.Module, target_name: str, new_module: torch.nn.Module):
    names = target_name.split(".")
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    setattr(parent, names[-1], new_module)


def dequantize(model: torch.nn.Module, device: torch.device | str | None = None):
    if device is None:
        device = next(iter(model.parameters())).device
    replacements = {}

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            if isinstance(module.weight_quantizer, QuantizerStub):
                weight = module.weight
            else:
                weight = module.weight_quantizer(module.weight)
                if isinstance(weight, QuantizedTensor):
                    weight = weight.dequantize()

            replacement = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=isinstance(module.bias, torch.Tensor),
                dtype=getattr(model, "dtype", None) or next(iter(model.parameters())).dtype,
            )

            replacement.weight.data.copy_(weight)
            if isinstance(module.bias, torch.Tensor):
                replacement.bias.data.copy_(module.bias)

            replacements[name] = replacement

    for name in replacements:
        replace_module(model, name, replacements[name])

    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device)
    return model
