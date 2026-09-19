# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

import fasttransforms as ft

_PARAMETRIZATION_INDEX_KEY_RE = re.compile(
    r"^(?P<prefix>.+\.parametrizations\.[^.]+)\.(?P<index>\d+)(?P<suffix>\._transform\..+)$"
)


def _select(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def apply_debug_layer_slice(
    model: torch.nn.Module,
    layer_slice: Any,
    logger: logging.Logger | None = None,
) -> torch.nn.Module:
    """Optionally replace decoder layers with a configured slice for local debug runs."""
    start = _select(layer_slice, "start")
    stop = _select(layer_slice, "stop")
    step = _select(layer_slice, "step")
    reindex_layer_idx = bool(_select(layer_slice, "reindex_layer_idx", False))

    if start is None and stop is None and step is None:
        return model

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("debug.layer_slice requires a model with model.layers.")

    layers = model.model.layers
    original_num_layers = len(layers)
    selected_indices = list(range(original_num_layers))[slice(start, stop, step)]
    if not selected_indices:
        raise ValueError(
            "debug.layer_slice selected no layers "
            f"(start={start}, stop={stop}, step={step}, original_num_layers={original_num_layers})."
        )

    model.model.layers = torch.nn.ModuleList([layers[idx] for idx in selected_indices])
    model.config.num_hidden_layers = len(selected_indices)
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = len(selected_indices)

    if reindex_layer_idx:
        for new_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "layer_idx"):
                layer.layer_idx = new_idx
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = new_idx

    model.debug_layer_slice = {
        "start": start,
        "stop": stop,
        "step": step,
        "original_num_layers": original_num_layers,
        "selected_indices": selected_indices,
        "num_hidden_layers": len(selected_indices),
        "reindex_layer_idx": reindex_layer_idx,
    }

    if logger is not None:
        logger.warning(
            "Applied debug layer slice: original_num_layers=%d selected_indices=%s reindex_layer_idx=%s",
            original_num_layers,
            selected_indices,
            reindex_layer_idx,
        )
    return model


def summarize_parametrization_index_mismatches(
    missing_keys: list[str] | tuple[str, ...],
    unexpected_keys: list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Group load-state mismatches that only differ by parametrization slot index.

    This is mainly useful for transform-subset checkpoint diagnostics: PyTorch stores
    parametrizations under numeric slots (``parametrizations.weight.0``, ``.1``, ...).
    If a subset changes transform attachment order, checkpoint keys for the same logical
    transform can move between slots and surface as paired missing/unexpected keys.
    """

    def _group(keys: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for key in keys:
            match = _PARAMETRIZATION_INDEX_KEY_RE.match(key)
            if match is None:
                continue
            logical_key = f"{match.group('prefix')}{{index}}{match.group('suffix')}"
            entry = grouped.setdefault(logical_key, {"indices": set(), "keys": []})
            entry["indices"].add(int(match.group("index")))
            entry["keys"].append(key)
        return grouped

    missing_grouped = _group(missing_keys)
    unexpected_grouped = _group(unexpected_keys)
    summaries: list[dict[str, Any]] = []
    for logical_key in sorted(set(missing_grouped) & set(unexpected_grouped)):
        missing_entry = missing_grouped[logical_key]
        unexpected_entry = unexpected_grouped[logical_key]
        summaries.append(
            {
                "logical_key": logical_key,
                "missing_indices": sorted(missing_entry["indices"]),
                "unexpected_indices": sorted(unexpected_entry["indices"]),
                "missing_keys": sorted(missing_entry["keys"]),
                "unexpected_keys": sorted(unexpected_entry["keys"]),
            }
        )
    return summaries


def build_transform_parametrization_key_maps(model: torch.nn.Module) -> tuple[dict[str, str], dict[str, str]]:
    """Build actual<->canonical key maps for transform parametrization state_dict entries.

    Canonical keys replace the PyTorch parametrization slot index with the logical
    transform name, which stays stable across subset compositions.
    """

    actual_to_canonical: dict[str, str] = {}
    canonical_to_actual: dict[str, str] = {}
    for module_path, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            continue
        for tensor_name in parametrizations:
            for index, parametrization in enumerate(parametrizations[tensor_name]):
                if not isinstance(parametrization, ft.TransformOperation):
                    continue
                transform_name = parametrization.transform.name or f"slot{index}"
                transform_state = parametrization.transform.state_dict()
                for subkey in transform_state:
                    actual_key = (
                        f"{module_path}.parametrizations.{tensor_name}.{index}._transform.{subkey}"
                        if module_path
                        else f"parametrizations.{tensor_name}.{index}._transform.{subkey}"
                    )
                    canonical_key = (
                        f"{module_path}.parametrizations.{tensor_name}[{transform_name}]._transform.{subkey}"
                        if module_path
                        else f"parametrizations.{tensor_name}[{transform_name}]._transform.{subkey}"
                    )
                    actual_to_canonical[actual_key] = canonical_key
                    canonical_to_actual[canonical_key] = actual_key
    return actual_to_canonical, canonical_to_actual


def remap_state_dict_transform_parametrization_keys(
    state_dict: Mapping[str, torch.Tensor],
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Remap transform parametrization keys from a source composition onto a target composition."""

    source_actual_to_canonical, _ = build_transform_parametrization_key_maps(source_model)
    _, target_canonical_to_actual = build_transform_parametrization_key_maps(target_model)

    remapped: dict[str, torch.Tensor] = {}
    renamed: list[dict[str, str]] = []
    collisions: list[dict[str, str]] = []
    for key, value in state_dict.items():
        canonical_key = source_actual_to_canonical.get(key)
        new_key = target_canonical_to_actual.get(canonical_key, key) if canonical_key is not None else key
        if new_key in remapped and new_key != key:
            collisions.append({"source_key": key, "target_key": new_key})
            continue
        if new_key != key:
            renamed.append({"source_key": key, "target_key": new_key})
        remapped[new_key] = value

    summary = {
        "renamed_count": len(renamed),
        "collision_count": len(collisions),
        "renamed_sample": renamed[:20],
        "collision_sample": collisions[:20],
    }
    return remapped, summary
