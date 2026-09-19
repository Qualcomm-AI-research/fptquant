# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from fastforward import mpath

from .operation import TransformOperation
from .transforms.common import MatmulTransform, Transform


@dataclass(kw_only=True)
class TransformOperationConfig:
    pattern: str
    inv: bool
    transpose: bool = False
    kwargs: Dict[str, Any] = field(default_factory=dict)
    inference: Optional[bool] = None


@dataclass(kw_only=True)
class HookConfig(TransformOperationConfig):
    """This corresponds to a single transformation operation. Multiple operations may be used for the same transform"""

    dim: int
    pre_hook: bool = True


@dataclass(kw_only=True)
class ParametrizationConfig(TransformOperationConfig):
    dim: int
    tensor_name: str


@dataclass(kw_only=True)
class LinearParametrizationConfig(TransformOperationConfig):
    input: bool


# Note this is not a dataclass for a hydra quirk, please don't make it into one
class TransformConfig:
    """This class contains both the transform module, as well as all the operations"""

    def __init__(
        self,
        transform: Transform,
        operation_configs: Sequence[TransformOperationConfig],
        per_layer: bool,
        priority: int,
    ):
        self.transform = transform
        self.operation_configs = operation_configs
        self.per_layer = per_layer
        self.priority = priority


def _add_key_to_dict(
    keys: Sequence[str],
    value: torch.nn.Module,
    module_dict: torch.nn.ModuleDict,
):
    if len(keys) == 1:
        module_dict[keys[0]] = value
    else:
        fragment = keys[0]
        if fragment not in module_dict:
            module_dict[fragment] = torch.nn.ModuleDict()
        _add_key_to_dict(keys[1:], value, module_dict)


def add_transform_operations(
    config: TransformOperationConfig, transform: Transform, module: torch.nn.Module
) -> List[TransformOperation]:
    kwargs: Dict[str, Any] = {}
    kwargs.update(config.kwargs)
    transform_ops = []

    if isinstance(transform, MatmulTransform):
        kwargs["transpose"] = config.transpose

    # Hook configuration
    if isinstance(config, HookConfig):
        # Add a TransformHook for each matching submodule
        transform_hook = TransformOperation(
            transform=transform,
            inv=config.inv,
            dim=config.dim,
            inference=config.inference,
            **kwargs,
        )
        transform_hook.attach_hook(module, pre_hook=config.pre_hook)
        transform_ops.append(transform_hook)

    # Parametrization Configuration
    elif isinstance(config, ParametrizationConfig):
        transform_parametrization = TransformOperation(
            transform=transform,
            inv=config.inv,
            dim=config.dim,
            inference=config.inference,
            **kwargs,
        )
        transform_parametrization.parametrize(module, tensor_name=config.tensor_name)
        transform_ops.append(transform_parametrization)

    # Parametrized Linear configuration. This essentially gets resolved into a couple of parametrizations
    elif isinstance(config, LinearParametrizationConfig):
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"LinearTransformOperationConfig can be applied only to instance of 'torch.nn.Linear'. {module.__class__.__name__} was found."
            )

        weight_parametrization_config = ParametrizationConfig(
            pattern=config.pattern,
            tensor_name="weight",
            dim=-1 if config.input else -2,
            inv=config.inv,
            transpose=not config.transpose if config.input else config.transpose,
            inference=config.inference,
            kwargs=config.kwargs,
        )
        transform_ops = add_transform_operations(weight_parametrization_config, transform=transform, module=module)
        if module.bias is not None and not config.input:
            bias_parametrization_config = ParametrizationConfig(
                pattern=config.pattern,
                tensor_name="bias",
                dim=-1,
                inv=config.inv,
                transpose=config.transpose,
                inference=config.inference,
                kwargs=config.kwargs,
            )
            transform_ops += add_transform_operations(bias_parametrization_config, transform=transform, module=module)
    else:
        raise ValueError(f"Unrecognized transform configuration type: {config.__class__.__name__}")

    return transform_ops


def transforms_from_config(
    config: TransformConfig,
    model: torch.nn.Module,
    name: str | None = None,
    aliases: Any | None = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    transform: Transform = config.transform
    transform.name = name if name else ""

    transform_ops: torch.nn.Module
    # If per_layer is True, we create a Transform for each layer
    if config.per_layer:
        n_repeat: int | None = None
        sorted_names: List[List[str]] = []
        suffixes: list[list[str]] | None = None

        transforms = torch.nn.ModuleDict()
        transform_ops = torch.nn.ModuleDict()

        # First, we check that all patterns resolve to the same number of entries
        for transform_op_config in config.operation_configs:
            res = mpath.search(transform_op_config.pattern, model, aliases=aliases)

            if len(res) == 0:
                raise ValueError(f"The query {transform_op_config.pattern} does not match any module")

            if n_repeat is None:
                n_repeat = len(res)

            if n_repeat != len(res):
                raise ValueError(
                    f'The number of results for the query "{transform_op_config.pattern}" for transform {name} does not match the other queries. {n_repeat}!={len(res)}'
                )

            names = [r.full_name for r in res]

            # Split each name into parts, converting numeric parts to integers for sorting
            def split_key(name):
                parts = name.split(".")
                return [int(p) if p.isdigit() else p for p in parts]

            # Sort using the numeric-aware key
            names = sorted(names, key=split_key)

            # Extract all numeric parts (suffixes) for each name
            if suffixes is None:
                suffixes = [re.findall(r"\d+", name) for name in names]

            sorted_names.append(names)

        assert n_repeat is not None
        assert suffixes is not None

        for idx in range(n_repeat):
            # Update its name to reflect the order
            transform_copy = copy.deepcopy(transform)
            transform_copy.metadata._index = ".".join(suffixes[idx])

            sub_ops = torch.nn.ModuleList()
            for i, transform_op_config in enumerate(config.operation_configs):
                submodule = model.get_submodule(sorted_names[i][idx])
                for transform_op in add_transform_operations(
                    config=transform_op_config, transform=transform_copy, module=submodule
                ):
                    sub_ops.append(transform_op)

                _add_key_to_dict(suffixes[idx], sub_ops, transform_ops)
                _add_key_to_dict(suffixes[idx], transform_copy, transforms)
        return transforms, transform_ops
    # If per_layer is false it means that all hooks are associated to the same transform
    else:
        transform_ops = torch.nn.ModuleList()

        for transform_op_config in config.operation_configs:
            for res in mpath.search(transform_op_config.pattern, model, aliases=aliases):
                for transform_op in add_transform_operations(
                    config=transform_op_config, transform=transform, module=res.module
                ):
                    transform_ops.append(transform_op)
        return transform, transform_ops


def transforms_from_config_dict(
    config_dict: Dict[str, TransformConfig],
    model: torch.nn.Module,
    aliases: Any | None = None,
) -> Tuple[torch.nn.ModuleDict, torch.nn.ModuleDict]:
    named_transforms = torch.nn.ModuleDict()
    named_transform_ops = torch.nn.ModuleDict()

    # Sort the transforms by the priority (higher priority first)
    ordered_names = sorted(
        config_dict,
        key=lambda name: config_dict[name].priority,
        reverse=True,
    )

    for name in ordered_names:
        config = config_dict[name]
        transforms, transform_ops = transforms_from_config(
            config=config,
            name=name,
            model=model,
            aliases=aliases,
        )
        named_transforms[name] = transforms
        named_transform_ops[name] = transform_ops
    return named_transforms, named_transform_ops
