# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""FastForward registry manipulation: whitelisting and prefix helpers."""

from __future__ import annotations

import copy
import types
from contextlib import contextmanager
from typing import Sequence, Tuple

import fastforward.nn.quantized_module as ffqm

from ._provider import Provider


def _allowed_prefixes(provider: Provider, imported: Sequence[types.ModuleType]) -> Tuple[str, ...]:
    """
    Compute robust whitelist prefixes from actual module objects:
    - module.__name__ of prefix_modules (e.g., 'lm_tools.quantized_models')
    - module.__name__ of any modules returned by initializers (e.g., 'lm_tools.quantized_models.llama')
    """
    names = [m.__name__ for m in provider.prefix_modules] + [m.__name__ for m in imported]
    return tuple(sorted(set(names)))


@contextmanager
def _whitelist_ff_registry(allowed_prefixes: Sequence[str]):
    """
    Temporarily restrict FF's global registry so only QuantizedModule classes whose
    __module__ starts with one of the allowed prefixes remain visible. Restores afterward.

    This ensures that for a given call, overlapping base-class implementations from other
    providers cannot be selected by `quantized_module_map()` (which is "last created wins").
    """
    allowed = tuple(allowed_prefixes)
    backup = copy.deepcopy(ffqm._QUANTIZED_MODULE_MAP)  # type: ignore[attr-defined]
    try:
        for base, lst in list(ffqm._QUANTIZED_MODULE_MAP.items()):  # type: ignore[attr-defined]
            lst[:] = [q for q in lst if q.__module__.startswith(allowed)]
            if not lst:
                del ffqm._QUANTIZED_MODULE_MAP[base]  # type: ignore[attr-defined]
        yield
    finally:
        ffqm._QUANTIZED_MODULE_MAP.clear()  # type: ignore[attr-defined]
        ffqm._QUANTIZED_MODULE_MAP.update(backup)  # type: ignore[attr-defined]
