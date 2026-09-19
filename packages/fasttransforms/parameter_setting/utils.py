# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from contextlib import ExitStack
from typing import (
    Any,
    Callable,
    Optional,
)

import torch
from accelerate import Accelerator
from fastforward import disable_quantization, mpath
from torch.utils.data import DataLoader
from tqdm import tqdm

from fasttransforms.nn.operation import get_transform_operations
from fasttransforms.nn.transforms.common import Transform
from fasttransforms.parameter_setting.common import ParameterEstimator, estimate_parameters


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


def estimate_transform_parameters(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
    parameter_estimator: ParameterEstimator,
    forward_function: Optional[Callable] = None,
    aliases: Any = None,
    filter: str = "**",
    use_gradient: bool = False,
    cached: bool = True,
    remove_parameter_setting_ops: bool = True,
    logger: Optional[logging.Logger | logging.LoggerAdapter] = None,
):
    if logger is None:
        logger = logging.getLogger("fasttransforms")

    if forward_function is None:

        def forward_function(batch):
            return model(**batch)

    # Determine the transform ops used only for parameter setting and inference
    inference_transform_ops = [
        transform_op for transform_op in get_transform_operations(model) if transform_op._inference
    ]
    parameter_setting_transform_ops = [
        transform_op for transform_op in get_transform_operations(model) if transform_op._inference is False
    ]

    transforms_to_estimate = torch.nn.ModuleList()
    seen_transform_ids: set[int] = set()

    for res in mpath.search(filter, model, aliases=aliases):
        module = res.module
        if isinstance(module, Transform) and id(module) not in seen_transform_ids:
            transforms_to_estimate.append(module)
            seen_transform_ids.add(id(module))

    logger.info(f"Estimating the parameters of {len(transforms_to_estimate)} transforms")

    accelerator = Accelerator()
    model, calibration_loader = accelerator.prepare(model, calibration_loader)

    # Disable all inference only transforms
    for transform_op in inference_transform_ops:
        transform_op.disable()

    # Enable all parameter setting only transform operations
    for transform_op in parameter_setting_transform_ops:
        transform_op.enable()

    with ExitStack() as stack:
        if not use_gradient:
            stack.enter_context(torch.no_grad())
        if cached:
            stack.enter_context(torch.nn.utils.parametrize.cached())
        stack.enter_context(disable_quantization(model))
        stack.enter_context(estimate_parameters(transforms_to_estimate, parameter_estimator))

        for batch in tqdm(calibration_loader):
            forward_function(batch)

    # Enable all inference only transform operations
    for transform_op in inference_transform_ops:
        transform_op.enable()
    # Remove/disable all parameter setting only transform operations
    for transform_op in parameter_setting_transform_ops:
        if remove_parameter_setting_ops:
            transform_op.remove()
        else:
            transform_op.disable()
