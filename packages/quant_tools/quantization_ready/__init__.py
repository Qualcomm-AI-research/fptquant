# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
quant_tools.quantization
========================
Model-agnostic quantization orchestrator for FastForward.

Public API
----------
- get_quant_model
- quantize_with_registry_fallback
- register_quantization_provider
- register_quantization_providers_from_package
- Provider
- FakeQuantizedModule

Internal helpers (prefixed with ``_``) are not part of the public API but are
re-exported here for convenience and backward compatibility.
"""

from __future__ import annotations

from ._logging import _log_quantization_summary
from ._provider import (
    _REGISTRY,
    Provider,
    register_quantization_provider,
    register_quantization_providers_from_package,
)
from ._registry import _allowed_prefixes, _whitelist_ff_registry
from ._types import FakeQuantizedModule
from ._utils import _model_type_of, _present_module_types
from .core import get_quant_model, quantize_with_registry_fallback

__all__ = [
    # Public API
    "get_quant_model",
    "quantize_with_registry_fallback",
    "register_quantization_provider",
    "register_quantization_providers_from_package",
    "Provider",
    "FakeQuantizedModule",
    # Registry (exposed for testing / advanced use)
    "_REGISTRY",
    # Internal helpers
    "_log_quantization_summary",
    "_allowed_prefixes",
    "_whitelist_ff_registry",
    "_model_type_of",
    "_present_module_types",
]
