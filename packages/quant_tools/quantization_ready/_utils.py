# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Utility helpers for quantization."""

from __future__ import annotations

from typing import Any, Optional, Set, Type

import torch


def _model_type_of(model: Any) -> Optional[str]:
    """Extract model_type from model.config.model_type, or None if not present."""
    return getattr(getattr(model, "config", object()), "model_type", None)


def _present_module_types(model: torch.nn.Module) -> Set[Type[torch.nn.Module]]:
    """Return the set of all module types present in the model graph."""
    return {type(model), *(type(m) for m in model.modules())}
