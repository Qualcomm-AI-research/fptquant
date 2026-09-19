# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from quant_tools.range_setting import core as range_core
from quant_tools.testing import make_quantizer_result, patch_range_contexts


def test_make_quantizer_result_builds_minimal_namespace():
    module = object()

    result = make_quantizer_result("layer.weight_quantizer", module)

    assert result.full_name == "layer.weight_quantizer"
    assert result.module is module


def test_patch_range_contexts_records_calls(monkeypatch):
    estimator = object()
    quantizer = object()
    calls = patch_range_contexts(monkeypatch)

    with range_core.ff.strict_quantization(False):
        with range_core.ff.range_setting.estimate_ranges([quantizer], estimator):
            pass

    assert calls == {
        "estimate_ranges": [([quantizer], estimator)],
        "strict_quantization": [False],
    }
