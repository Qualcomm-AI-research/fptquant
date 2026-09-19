# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections import namedtuple
from enum import Enum
from enum import auto as enum_auto
from functools import partial

from fastforward.nn import DynamicLinearQuantizer, LinearQuantizer


class BaseEnumOptions(Enum):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)


MethodMap = partial(namedtuple("MethodMapTuple", ["value", "cls"]), enum_auto())  # type: ignore[arg-type]


class TorchDtype(BaseEnumOptions):
    auto = enum_auto()
    bfloat16 = enum_auto()
    float16 = enum_auto()
    float32 = enum_auto()


class QuantScheme(ClassEnumOptions):
    linear = MethodMap(LinearQuantizer)  # type: ignore[call-arg]
    dynamic_linear = MethodMap(DynamicLinearQuantizer)  # type: ignore[call-arg]
