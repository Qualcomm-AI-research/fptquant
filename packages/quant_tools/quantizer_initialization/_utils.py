# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import torch
from fastforward.nn.quantized_module import named_quantizers


def _move_quantizers_to_device(model: torch.nn.Module) -> torch.nn.Module:
    """Move all quantizers in ``model`` to the same device as the model's parameters.

    This is called after quantizer initialization to ensure newly created
    quantizer parameters (e.g., scale, zero-point) reside on the correct device.

    If the model has no parameters, this is a no-op.

    Args:
        model: The model whose quantizers should be moved.

    Returns:
        The same ``model`` (modified in-place), for convenience.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # Model has no parameters — nothing to move.
        return model

    for _name, quantizer in named_quantizers(model):
        quantizer.to(device)

    return model
