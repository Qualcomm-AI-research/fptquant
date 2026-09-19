# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward
import torch
from fastforward.mpath import Fragment, Selector

from fastforward_contrib.typing import override

from . import nn, quantization, range_setting
from .quantization import granularity


class RangeFragment(Fragment):
    def __init__(self, lo: int, hi: int) -> None:
        self._lo, self._hi = lo, hi

    @override
    def match(self, fragment_name: str, module: torch.nn.Module) -> bool:
        try:
            int_value = int(fragment_name)
        except ValueError:
            return False
        return self._lo <= int_value <= self._hi

    @classmethod
    def from_raw_string(cls, raw_str: str) -> Selector:
        lo, hi = [int(v) for v in raw_str.split("-")]
        fragment = cls(lo, hi)
        return Selector(None, fragment)

    @override
    def __str__(self) -> str:
        return f"[range:{self._lo}-{self._hi}]"

    @override
    def __hash__(self) -> int:
        return hash(("__mpath_fragment_hash", type(self), self._lo, self._hi))


fastforward.mpath.register_mpath_query_extension("range", RangeFragment)

PerToken = granularity.PerToken

__all__ = ["nn", "quantization", "range_setting"]
