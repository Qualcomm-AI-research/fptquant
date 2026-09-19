# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Quantization summary logging utilities."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Optional, Set, Type

import fastforward.nn.quantized_module as ffqm
import torch

from ._types import FakeQuantizedModule


def _log_quantization_summary(
    model: torch.nn.Module,
    original_types: Set[Type[torch.nn.Module]],
    model_type: Optional[str],
    logger: logging.Logger | logging.LoggerAdapter,
) -> None:
    """
    Log a detailed, user-friendly summary of quantization results.

    Args:
        model: The quantized model (modified in-place by FastForward)
        original_types: Module types present before quantization
        model_type: The model_type string (e.g., 'llama')
        logger: Logger instance
    """
    # Get current module types and counts after quantization
    type_counts: Dict[Type, int] = defaultdict(int)
    for m in model.modules():
        type_counts[type(m)] += 1

    quantized_types = set(type_counts.keys())

    # Categorize modules: quantized vs skipped
    quantized_modules = []
    skipped_modules = []

    for orig_type in original_types:
        if orig_type is torch.nn.Module:
            continue

        count = type_counts.get(orig_type, 0)

        # Check if this type was replaced with a quantized version
        replaced = False
        for quant_type in quantized_types:
            if (
                quant_type != orig_type
                and issubclass(quant_type, ffqm.QuantizedModule)
                and issubclass(quant_type, orig_type)
            ):
                # If it has been replaced with a FakeQuantizedModule, we count it as skipped
                if issubclass(quant_type, FakeQuantizedModule):
                    skipped_modules.append((orig_type, count))
                else:
                    quantized_modules.append((orig_type, quant_type, type_counts[quant_type]))
                replaced = True
                break

        if not replaced:
            # Check if it's a QuantizedModule or not
            if issubclass(orig_type, ffqm.QuantizedModule):
                quantized_modules.append((orig_type, orig_type, count))
            else:
                skipped_modules.append((orig_type, count))

    # Calculate statistics
    total_quantized = sum(count for _, _, count in quantized_modules)
    total_skipped = sum(count for _, count in skipped_modules)
    total_modules = total_quantized + total_skipped

    # Format output
    lines = [
        f"\nQuantization Summary for '{model_type or 'model'}':",
        f"  Total modules: {total_modules}",
        f"  Quantized: {total_quantized}",
        f"  Skipped (FP): {total_skipped}",
        "",
        "Module Replacements:",
    ]

    if quantized_modules:
        for base_cls, quant_cls, count in sorted(quantized_modules, key=lambda x: x[0].__name__):
            base_path = f"{base_cls.__module__}.{base_cls.__name__}"
            quant_path = f"{quant_cls.__module__}.{quant_cls.__name__}"
            count_str = f" (x{count})" if count > 1 else ""

            if base_cls == quant_cls:
                # Already quantized, no replacement
                lines.append(f"  ✓ {base_path}{count_str}")
            else:
                # Replaced
                lines.append(f"  ✓ {base_path}{count_str}")
                lines.append(f"    → {quant_path}")

    if skipped_modules:
        lines.append("")
        lines.append("Skipped (FP):")
        for cls, count in sorted(skipped_modules, key=lambda x: x[0].__name__):
            cls_path = f"{cls.__module__}.{cls.__name__}"
            count_str = f" (x{count})" if count > 1 else ""
            lines.append(f"  ○ {cls_path}{count_str}")

    logger.info("\n".join(lines))
