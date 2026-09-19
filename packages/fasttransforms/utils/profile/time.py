# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import time
from functools import partial
from typing import Any, Callable, ContextManager, Iterable, List, Tuple

import pandas as pd
import torch
from fastforward.forward_override import OverrideHandle

from fasttransforms.nn import Transform, get_transforms

from .common import Profiler


class TransformProfilerOverride:
    """Count how many time a transform is called."""

    def __init__(self, register_entry: Callable) -> None:
        self._handles: List[OverrideHandle] = []
        self.register_entry = register_entry

    def reset(self):
        self._forward_times = []
        self._inverse_times = []

    @property
    def forward_count(self) -> int:
        return len(self._forward_times)

    @property
    def inverse_count(self) -> int:
        return len(self._inverse_times)

    def __call__(
        self,
        _context: Any,
        callback: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Override function for transfrom profiling."""

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = callback(*args, **kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        self.register_entry(inv=kwargs["inv"], time=dt)

        return out

    def attach_to(self, transforms: Transform | Iterable[Transform]) -> None:
        """Attach this override to one or more transformations."""
        if isinstance(transforms, Transform):
            self._handles.append(transforms.register_override(self))
        else:
            for transform in transforms:  # type: ignore[union-attr]
                self.attach_to(transform)

    def detach(self) -> None:
        """Detach this override."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


class TransformTimeProfiler(Profiler):
    def __init__(self):
        super().__init__()
        self._stats = []

    def add_entry(self, full_name: str, inv: bool, time: float):
        self._stats.append({"full_name": full_name, "operation": "inverse" if inv else "forward", "time": time})

    def prepare(self, model: torch.nn.Module) -> Tuple[List[ContextManager], List[TransformProfilerOverride]]:
        counters = []
        for transform in get_transforms(model):
            name = transform.name if transform.name else ""
            register_data = partial(self.add_entry, full_name=name)
            counter = TransformProfilerOverride(register_data)
            counter.attach_to(transform)
            counters.append(counter)
        return [], counters

    def cleanup(self, metadata: List[TransformProfilerOverride]):
        for counter in metadata:
            counter.detach()

    @property
    def stats(self) -> pd.DataFrame:
        stats = []
        for entry in self._stats:
            entry_copy = {k: v for k, v in entry.items()}
            entry_copy["name"] = entry_copy["full_name"].split(".")[0]
            stats.append(entry_copy)

        return pd.DataFrame(stats)

    @property
    def stats_summary(self) -> pd.DataFrame:
        return self.stats[["name", "operation", "time"]].groupby(["name", "operation"]).agg(["mean", "std", "count"])

    def reset(self):
        self._stats = []
