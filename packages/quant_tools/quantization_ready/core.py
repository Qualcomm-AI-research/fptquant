# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Model-agnostic quantization orchestrator for FastForward (FF).

Key features:
- Plugins register a provider per `model_type` using callables and module objects (no string module paths).
- Per-call registry whitelisting ensures correct implementations even across mixed sequences.
- Supports strict and non-strict modes.
- Provides a default fallback that uses only already-registered quantized classes.

Concurrency note:
- The whitelist context manager mutates FF's global registry temporarily. If you
  quantize models concurrently in the same process, guard calls with a process-wide
  lock or run quantization in separate processes.
"""

from __future__ import annotations

import logging
import types
from typing import Dict, List, Type, cast

import fastforward.nn.quantized_module as ffqm
import torch

from ._logging import _log_quantization_summary
from ._provider import _REGISTRY
from ._registry import _allowed_prefixes, _whitelist_ff_registry
from ._types import FakeQuantizedModule
from ._utils import _model_type_of, _present_module_types


def get_quant_model(
    model: torch.nn.Module,
    logger: logging.LoggerAdapter | logging.Logger,
    *,
    default_fallback: bool = False,
) -> ffqm.QuantizedModule:
    """
    Quantize `model` in a model-agnostic way using a registered provider determined by
    `model.config.model_type`.

    Steps:
      1) Pick provider by model_type (highest priority wins).
      2) Run its initializers to import only its quant modules (auto-register classes).
      3) Whitelist FF registry for this call to the provider's prefixes.

    Args:
        model: a torch.nn.Module with model.config.model_type
        logger: a logger
        default_fallback: If True and no provider is registered for this model type,
                use `quantize_with_registry_fallback` instead of raising.

    Returns:
        ffqm.QuantizedModule (the quantized model)

    Raises:
        NotImplementedError if no provider is registered and default_fallback=False,
        or if model_type cannot be determined.
    """
    mtype = _model_type_of(model)

    # Capture original module types before quantization (FF modifies model in-place)
    original_types = _present_module_types(model)

    if default_fallback:
        quant = quantize_with_registry_fallback(model=model, logger=logger)
    else:
        if not mtype:
            raise NotImplementedError(
                "Cannot determine model_type from model.config.model_type."
                " You can quantize using default_fallback=True"
            )

        providers = sorted(_REGISTRY.get(mtype, []), key=lambda p: p.priority, reverse=True)
        if not providers:
            raise NotImplementedError(f"No quantization provider registered for model_type='{mtype}'")

        provider = providers[0]

        # 1) Import only the provider's modules via callables; collect module objects
        imported: List[types.ModuleType] = []
        for init in provider.initializers:
            mods = init() or ()
            imported.extend(mods)

        # 2) Compute allowed prefixes from actual module objects (refactor-safe)
        prefixes = _allowed_prefixes(provider, imported)
        prefixes_str = "\n  - ".join(prefixes)
        logger.info(f"Quantizing model of type '{mtype}' using quantized classes defined in:\n  - {prefixes_str}")

        # 3) Whitelist + build extra_conversion + quantize
        with _whitelist_ff_registry(prefixes):
            quant = cast(ffqm.QuantizedModule, ffqm.quantize_model(model, skip_quantized_modules=True))

        if not isinstance(quant, ffqm.QuantizedModule):
            raise RuntimeError(f"Quantization failed; got {type(quant).__name__}")

    # Log detailed quantization summary (works for both provider and fallback paths)
    _log_quantization_summary(quant, original_types, mtype, logger)

    return quant


def quantize_with_registry_fallback(
    model: torch.nn.Module,
    logger: logging.LoggerAdapter | logging.Logger,
) -> ffqm.QuantizedModule:
    """
    Fallback quantization using ONLY classes already registered in FastForward.
    - No provider is imported.
    - For each present module type, use the existing quantized class if available,
      otherwise explicitly SKIP_QUANTIZATION (leave FP).
    """
    # Consider all the quantized module defined in fastforward only
    with _whitelist_ff_registry(["fastforward", "fastforward_contrib"]):
        default_map = ffqm.quantized_module_map()  # {BaseModule: QuantizedSubclass}
    present = _present_module_types(model)

    extra: Dict[Type[torch.nn.Module], Type[ffqm.QuantizedModule] | ffqm.SkipQuantization] = {}
    # Create a "fake" QuantizedModule for the classes we don't know how to quantize
    for base in present:
        if base in default_map or issubclass(base, ffqm.QuantizedModule):
            continue
        else:
            logger.info(f"Skipping quantization of {base}.")
            extra[base] = type(f"FakeQuantized{base.__name__}", (FakeQuantizedModule, base), {})
    quant = ffqm.quantize_model(model, extra_conversion=extra, skip_quantized_modules=True)
    return cast(ffqm.QuantizedModule, quant)
