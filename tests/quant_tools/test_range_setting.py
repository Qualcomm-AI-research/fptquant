# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Offline tests for quantizer range-setting orchestration."""

import pytest
import torch

from quant_tools.range_setting import core as range_core
from quant_tools.testing import (
    ToyEvalTrackingModule,
    make_quantizer_result,
    make_range_settable_module,
    patch_range_contexts,
    synthetic_calibration_loader,
)


@pytest.mark.unit
def test_range_setting_skips_stubs_and_dynamic_quantizers(monkeypatch, caplog):
    """Only settable quantizers are calibrated; stubs and dynamic quantizers are ignored.

    Selection is by the ``RangeSettable`` protocol (``quantization_range`` + ``granularity``
    attributes). Plain dynamic quantizers and stubs lack those attributes, so a bare object
    stands in for both here.
    """
    model = ToyEvalTrackingModule()
    estimator = object()

    class NonSettableQuantizer:
        """Neither a stub nor range-settable (no quantization_range/granularity)."""

    monkeypatch.setattr(
        range_core.ff,
        "find_quantizers",
        lambda model, pattern: [
            make_quantizer_result("layer.input_quantizer", NonSettableQuantizer()),
            make_quantizer_result("layer.weight_quantizer", NonSettableQuantizer()),
        ],
    )

    calls = patch_range_contexts(monkeypatch)

    with caplog.at_level("CRITICAL"):
        result = range_core.estimate_quantizer_ranges(
            model,
            calibration_loader=synthetic_calibration_loader([[1.0], [2.0]]),
            range_estimator=estimator,
            logger=range_core.logging.getLogger("test"),
        )

    assert result is model
    assert model.eval_called
    assert model.forward_batches == []
    assert calls["estimate_ranges"] == []
    assert "no settable quantizers found" in caplog.text


@pytest.mark.unit
def test_range_setting_skips_clip_quantizer_in_seed_mode(monkeypatch, caplog):
    """Clip quantizers with a non-None ``clip_init`` seed directly and are not range-set.

    Such a quantizer still satisfies the ``RangeSettable`` protocol structurally (the
    ``quantization_range`` / ``granularity`` attributes exist), so it must be excluded by
    the explicit ``clip_init`` check — otherwise range setting would overwrite the intended
    seeded initialization.
    """
    model = ToyEvalTrackingModule()
    estimator = object()

    # Range-settable by protocol, but in seed mode (clip_init set) → must be skipped.
    seeded_clip = make_range_settable_module()
    seeded_clip.clip_init = 0.982

    monkeypatch.setattr(
        range_core.ff,
        "find_quantizers",
        lambda model, pattern: [make_quantizer_result("layer.input_quantizer", seeded_clip)],
    )

    calls = patch_range_contexts(monkeypatch)

    with caplog.at_level("CRITICAL"):
        result = range_core.estimate_quantizer_ranges(
            model,
            calibration_loader=synthetic_calibration_loader([[1.0], [2.0]]),
            range_estimator=estimator,
            logger=range_core.logging.getLogger("test"),
        )

    assert result is model
    assert model.forward_batches == []
    assert calls["estimate_ranges"] == []
    assert "no settable quantizers found" in caplog.text


@pytest.mark.unit
def test_range_setting_includes_clip_quantizer_in_range_mode(monkeypatch):
    """Clip quantizers with ``clip_init=None`` are range-settable and get calibrated."""
    model = ToyEvalTrackingModule()
    estimator = object()

    range_clip = make_range_settable_module()
    range_clip.clip_init = None  # range-settable mode

    monkeypatch.setattr(
        range_core.ff,
        "find_quantizers",
        lambda model, pattern: [make_quantizer_result("layer.input_quantizer", range_clip)],
    )
    calls = patch_range_contexts(monkeypatch)

    result = range_core.estimate_quantizer_ranges(
        model,
        calibration_loader=synthetic_calibration_loader([[1.0], [2.0]]),
        range_estimator=estimator,
        logger=range_core.logging.getLogger("test"),
    )

    assert result is model
    assert calls["estimate_ranges"] == [([range_clip], estimator)]


@pytest.mark.unit
def test_weight_only_range_setting_consumes_one_batch(monkeypatch):
    """Weight-only quantizer calibration consumes a single synthetic batch."""
    model = ToyEvalTrackingModule()
    estimator = object()
    weight_quantizer = make_range_settable_module()
    calls = patch_range_contexts(monkeypatch)

    monkeypatch.setattr(
        range_core.ff,
        "find_quantizers",
        lambda model, pattern: [make_quantizer_result("linear.weight_quantizer", weight_quantizer)],
    )

    result = range_core.estimate_quantizer_ranges(
        model,
        calibration_loader=synthetic_calibration_loader([[1.0], [2.0], [3.0], [4.0]]),
        range_estimator=estimator,
        logger=range_core.logging.getLogger("test"),
        max_num_batches=3,
    )

    assert result is model
    assert model.eval_called
    assert len(model.forward_batches) == 1
    assert model.forward_batches[0].device == model.bias.device
    assert calls["estimate_ranges"] == [([weight_quantizer], estimator)]
    assert calls["strict_quantization"] == [False]


@pytest.mark.unit
def test_activation_range_setting_consumes_max_num_batches_and_moves_to_device(monkeypatch):
    """Activation quantizers use the requested number of batches and send tensors to the model device."""
    model = ToyEvalTrackingModule()
    estimator = object()
    activation_quantizer = make_range_settable_module()
    observed_devices = []

    monkeypatch.setattr(
        range_core.ff,
        "find_quantizers",
        lambda model, pattern: [make_quantizer_result("linear.input_quantizer", activation_quantizer)],
    )
    patch_range_contexts(monkeypatch)

    def forward_call(x):
        observed_devices.append(x.device)
        return model(x)

    result = range_core.estimate_quantizer_ranges(
        model,
        calibration_loader=synthetic_calibration_loader([[1.0], [2.0], [3.0], [4.0]]),
        range_estimator=estimator,
        logger=range_core.logging.getLogger("test"),
        forward_call=forward_call,
        max_num_batches=3,
    )

    assert result is model
    assert len(model.forward_batches) == 3
    assert observed_devices == [model.bias.device] * 3


@pytest.mark.integration
def test_range_setting_default_forward_runs_finitely_on_toy_cpu_module(monkeypatch):
    """Default model.forward calibration executes finite CPU forwards without external resources."""
    model = ToyEvalTrackingModule()
    estimator = object()
    activation_quantizer = make_range_settable_module()

    monkeypatch.setattr(
        range_core.ff,
        "find_quantizers",
        lambda model, pattern: [make_quantizer_result("linear.input_quantizer", activation_quantizer)],
    )
    patch_range_contexts(monkeypatch)

    result = range_core.estimate_quantizer_ranges(
        model,
        calibration_loader=synthetic_calibration_loader([[1.0], [2.0]]),
        range_estimator=estimator,
        logger=range_core.logging.getLogger("test"),
    )

    assert result is model
    assert len(model.forward_batches) == 2
    assert all(torch.isfinite(batch).all() for batch in model.forward_batches)
