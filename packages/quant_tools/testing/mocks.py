# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from quant_tools.range_setting import core as range_core


def make_quantizer_result(full_name: str, module) -> SimpleNamespace:
    """Build the minimal FastForward quantizer search result used in tests."""
    return SimpleNamespace(full_name=full_name, module=module)


def make_range_settable_module() -> SimpleNamespace:
    """A mock quantizer that satisfies the ``RangeSettable`` protocol.

    ``ff.range_setting.RangeSettable`` is a runtime-checkable protocol requiring
    ``quantization_range`` and ``granularity`` attributes. ``estimate_quantizer_ranges``
    selects quantizers via ``isinstance(module, RangeSettable)``, so test doubles standing
    in for settable quantizers (static linear, clip quantizers) must expose both.
    """
    return SimpleNamespace(quantization_range=(None, None), granularity=None)


def patch_range_contexts(monkeypatch) -> dict[str, list]:
    """Patch range-setting context managers with recording no-ops for unit tests."""
    calls: dict[str, list[Any]] = {"estimate_ranges": [], "strict_quantization": []}

    @contextmanager
    def fake_estimate_ranges(quantizers, range_estimator):
        calls["estimate_ranges"].append((list(quantizers), range_estimator))
        yield

    @contextmanager
    def fake_strict_quantization(enabled):
        calls["strict_quantization"].append(enabled)
        yield

    monkeypatch.setattr(range_core.ff.range_setting, "estimate_ranges", fake_estimate_ranges)
    monkeypatch.setattr(range_core.ff, "strict_quantization", fake_strict_quantization)
    monkeypatch.setattr(range_core, "tqdm", lambda iterable, **kwargs: iterable)
    return calls
