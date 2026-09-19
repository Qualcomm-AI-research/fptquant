# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from fastforward.nn import LinearQuantizer
from omegaconf import DictConfig, OmegaConf
from torch.nn import Module

from .enums import TorchDtype


def convert_transformer_float_input(batch: dict, dtype: TorchDtype) -> dict:
    if dtype.value == TorchDtype.auto.value:
        return batch

    assert dtype.name is not None
    torch_dtype = getattr(torch, dtype.name)

    new_batch = {}
    for k, v in batch.items():
        if torch.is_floating_point(v):
            new_batch[k] = v.to(torch_dtype)
        else:
            new_batch[k] = v
    return new_batch


def try_move_to_device(tensor: torch.Tensor, device: Optional[torch.device]):
    """Move input tensor to the target device if the device is valid.

    Args:
        tensor: input tensor.
        device: target device, if the device is a 'meta' device from accelerate or None, tensor will not be moved.

    Returns: the tensor, possibly moved to the target device.
    """
    if device is not None and device.type != "meta":
        return tensor.to(device)
    return tensor


def pprint(config: DictConfig, logger: Optional[logging.Logger] = None, indent: int = 4) -> None:
    """
    Pretty print for the configuration object.
    :param config: The configuration object to pretty-print.
    :param logger: An optional logger object passed to the instantiation function.
    :param indent: The indentation of the pretty-printed output.
    :return:
    """
    s = yaml.dump(OmegaConf.to_container(config, enum_to_str=True, resolve=True), indent=indent)
    if logger is None:
        print(s)
    else:
        logger.info(s)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def maybe_run_deterministically():
    """
    This function allows to run the code deterministically if the `CUBLAS_WORKSPACE_CONFIG` is set.

    Option :4096:8
        - Allows more efficient deterministic algorithms that require a larger workspace.
        - Preferred when performance is important and memory is not a constraint.

    Option :16:8
        - Uses less memory, but may force cuBLAS to use slower deterministic algorithms due to the smaller workspace.
        - Useful in memory-constrained environments or for debugging.

    For example, use `CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run main.py` to run your app.
    Note that it will slow down computations.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") in {":4096:8", ":16:8"}:
        print("************************************")
        print("*   USING DETERMINISTIC CALCULUS   *")
        print("************************************")
        torch.use_deterministic_algorithms(True)


def save_config(cfg):
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved_cfg, "hydra_config.yaml")


def match_patterns(string: str, patterns: List[str]):
    """Checks if a string matches any of the given patterns.

    A pattern can either be a substring to check for, or if it ends with "$",
    it checks if the string ends with the pattern (excluding the "$").

    Args:
        string: The string to check against the patterns.
        patterns: A list of pattern strings.

    Returns:
        True if the string matches any of the patterns, False otherwise.
    """
    for pattern in patterns:
        if pattern.endswith("$"):
            matching = string.endswith(pattern[:-1])
        else:
            matching = pattern in string
        if matching:
            return True
    return False


def ensure_scales_are_identical(model: Module, accelerator: Accelerator) -> None:
    """Ensures that quantization scales are identical across all GPUs.

    This function iterates through all LinearQuantizer modules in the model,
    gathers their scales across all distributed processes using the accelerator,
    and raises a RuntimeError if the scales are not identical.

    Args:
        model (Module): The PyTorch model containing LinearQuantizer modules.
        accelerator (Accelerator): The Hugging Face Accelerate Accelerator object.

    Raises:
        RuntimeError: If quantization scales are not identical across all GPUs.
    """
    scales = []
    for module in model.modules():
        if isinstance(module, LinearQuantizer):
            scales.append(module.scale.detach().clone())
    local_scales = torch.cat(scales).view(1, -1)
    all_scales = accelerator.gather(local_scales)
    is_equal = torch.all(all_scales == all_scales[0])
    if not is_equal:
        raise RuntimeError("Scales are not equal on each GPU, which is unexpected.")
