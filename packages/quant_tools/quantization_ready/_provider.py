# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Provider registry for model-type-specific quantization implementations."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import types
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

# model_type -> list of providers
_REGISTRY: Dict[str, List["Provider"]] = {}


@dataclass(frozen=True)
class Provider:
    """
    Describes how to enable quantization for a given model_type.

    - initializers: callables that import the provider's quant modules and return
      the actual module object(s) (or None). Importing triggers FF's auto-registration
      of QuantizedModule subclasses.
    - prefix_modules: module objects whose `__name__` are used as *stable* prefixes
      for whitelisting (e.g., the provider's package module). This is robust to refactors.
    - priority: if multiple providers register for the same model_type, the highest wins.
    """

    initializers: Sequence[Callable[[], Optional[Sequence[types.ModuleType]]]]
    prefix_modules: Sequence[types.ModuleType]
    priority: int = 100


def register_quantization_provider(
    model_type: str,
    *,
    initializers: Sequence[Callable[[], Optional[Sequence[types.ModuleType]]]],
    prefix_modules: Sequence[types.ModuleType],
    priority: int = 100,
) -> None:
    """
    Register a quantization provider for a given model_type.

    Plugins call this once at import time. No string paths required—only callables and module objects.
    """
    prov = Provider(
        initializers=tuple(initializers),
        prefix_modules=tuple(prefix_modules),
        priority=priority,
    )
    _REGISTRY.setdefault(model_type, []).append(prov)


def register_quantization_providers_from_package(
    package: types.ModuleType,
    *,
    generic_modules: Sequence[types.ModuleType] = (),
    skip: Sequence[str] = ("generic", "common", "__pycache__"),
    default_priority: int = 200,
) -> None:
    """
    Auto-register one provider per family subpackage under `package`.

    Conventions:
      - Each family lives in `package/<family>/` (e.g., quantized_models/llama/).
      - Optional: `package/<family>/meta.py` defines MODEL_TYPES = ("llama", "llama2", ...)
      - Aggregator is the family package itself (importing it registers classes).
    """
    base_pkg = package.__name__

    # Discover direct subpackages
    for _, family, is_pkg in pkgutil.iter_modules(package.__path__):  # type: ignore[attr-defined]
        if not is_pkg or family in skip:
            continue

        # Read metadata IF present (lightweight import)
        model_types = None
        meta_name = f"{base_pkg}.{family}.meta"
        if importlib.util.find_spec(meta_name):
            meta = importlib.import_module(meta_name)
            model_types = getattr(meta, "MODEL_TYPES", None)

        # Default: model_type == folder name if no meta present
        model_types = tuple(model_types) if model_types else (family,)

        # Initializer that imports the aggregator (family package) and returns module objects
        def make_init(fam: str):
            def _init():
                fam_mod = importlib.import_module(f"{base_pkg}.{fam}")
                # include any generic modules explicitly
                return [fam_mod, *generic_modules]

            return _init

        init = make_init(family)

        # Register for each declared model_type
        for mt in model_types:
            register_quantization_provider(
                model_type=mt,
                initializers=[init],
                prefix_modules=[*generic_modules],  # module objects (refactor-safe)
                priority=default_priority,
            )
