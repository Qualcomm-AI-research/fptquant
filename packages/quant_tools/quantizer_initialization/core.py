# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import logging
from typing import Mapping

import fastforward.nn.quantized_module as ffqm

from ._rule import QuantizationRule
from ._utils import _move_quantizers_to_device


def initialize_quantizers(
    model: ffqm.QuantizedModule,
    rules: Mapping[str, QuantizationRule],
    logger: logging.LoggerAdapter | logging.Logger,
) -> ffqm.QuantizedModule:
    """Apply a collection of quantization rules to a quantized model.

    Each rule in ``rules`` is applied in iteration order. After all rules have
    been applied, every newly created quantizer is moved to the same device as
    the model's parameters.

    Args:
        model: A quantized model (produced by e.g. :func:`get_quant_model`).
        rules: An ordered mapping of ``{rule_name: rule}`` pairs. The name is
            used only for logging.
        logger: Logger used to report how many quantizers each rule initialized.

    Returns:
        The same ``model`` (modified in-place), for convenience.
    """
    for name, rule in rules.items():
        quantizers = rule.initialize(model)
        if len(quantizers) == 0:
            logger.warning(f"Rule '{name}' applied to 0 quantizers — check your rule pattern.")
        else:
            logger.info(f"Rule '{name}' applied to {len(quantizers)} quantizers:")
        logger.info("\t" + str(rule))

    # move introduced quantizers to model device
    _move_quantizers_to_device(model)

    return model
