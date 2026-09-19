# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from . import utils
from .config import add_config_to_store
from .nn.collections import (
    HookConfig,
    LinearParametrizationConfig,
    ParametrizationConfig,
    TransformConfig,
    transforms_from_config,
    transforms_from_config_dict,
)
from .nn.operation import TransformOperation, find_transform_operations, fuse_transforms, get_transform_operations
from .nn.transforms.common import Transform
from .nn.transforms.utils import find_transforms, get_transforms
from .overrides import disable_transforms, enable_transforms
from .parameter_setting import estimate_parameters

__all__ = [
    "TransformConfig",
    "HookConfig",
    "ParametrizationConfig",
    "LinearParametrizationConfig",
    "Transform",
    "TransformOperation",
    "add_config_to_store",
    "get_transforms",
    "find_transforms",
    "get_transform_operations",
    "find_transform_operations",
    "fuse_transforms",
    "transforms_from_config_dict",
    "transforms_from_config",
    "enable_transforms",
    "disable_transforms",
    "estimate_parameters",
    "utils",
]

from typing import Any

from fastforward import mpath
from fastforward._import import QualifiedNameReference
from fastforward.mpath import register_mpath_query_extension
from typing_extensions import override


class AbsoluteClassPathFragment(mpath.fragments.ClassFragment):
    @override
    @classmethod
    def from_raw_string_with_context(cls, raw_str: str, context: dict[str, Any]) -> mpath.Selector:
        obj = QualifiedNameReference(raw_str).import_()
        return mpath.Selector(None, cls(fragment_class=obj))


register_mpath_query_extension("abscls", AbsoluteClassPathFragment, override=False)
