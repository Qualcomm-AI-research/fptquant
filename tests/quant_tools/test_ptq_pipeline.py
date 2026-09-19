# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Offline tests for PTQPipeline orchestration."""

import logging

import pytest
import torch

from quant_tools import pipeline as pipeline_module
from quant_tools.pipeline import PTQPipeline
from quant_tools.testing import SyntheticTensorDataset


@pytest.fixture
def logger():
    """Provide a test logger."""
    return logging.getLogger("test")


@pytest.mark.unit
def test_ptq_pipeline_without_range_estimator_does_not_touch_calibration_set(monkeypatch, logger):
    """Without a range estimator, the pipeline returns before touching calibration data."""
    original_model = torch.nn.Linear(1, 1)
    quant_model = torch.nn.Linear(1, 1)
    initialized_model = torch.nn.Linear(1, 1)
    rules = {"weights": object()}
    calls = []

    def fake_get_quant_model(model, *, logger, default_fallback):
        calls.append(("get_quant_model", model, logger, default_fallback))
        return quant_model

    def fake_initialize_quantizers(model, *, rules, logger):
        calls.append(("initialize_quantizers", model, rules, logger))
        return initialized_model

    def fail_calibration_set(*args, **kwargs):
        raise AssertionError("calibration_set must not be accessed without a range estimator")

    def fail_estimate_quantizer_ranges(*args, **kwargs):
        raise AssertionError("range setting must not run without a range estimator")

    monkeypatch.setattr(pipeline_module, "get_quant_model", fake_get_quant_model)
    monkeypatch.setattr(pipeline_module, "initialize_quantizers", fake_initialize_quantizers)
    monkeypatch.setattr(pipeline_module, "estimate_quantizer_ranges", fail_estimate_quantizer_ranges)

    result = PTQPipeline(
        rules=rules,
        range_estimator=None,
        calibration_set=fail_calibration_set,
    )(
        original_model,
        tokenizer=None,
        logger=logger,
    )

    assert result is initialized_model
    assert calls == [
        ("get_quant_model", original_model, logger, False),
        ("initialize_quantizers", quant_model, rules, logger),
    ]


@pytest.mark.unit
def test_ptq_pipeline_builds_loader_and_forwards_custom_hooks(monkeypatch, logger):
    """Calibration builds a synthetic DataLoader and forwards custom hooks to range setting."""
    original_model = torch.nn.Linear(1, 1)
    initialized_model = torch.nn.Linear(1, 1)
    rules = {"weights": object()}
    range_estimator = object()
    forward_call = object()
    collator_calls = []

    def custom_collator(samples):
        collator_calls.append(samples)
        return {"x": torch.stack([sample["x"] for sample in samples]) + 100.0}

    def fake_estimate_quantizer_ranges(
        model,
        *,
        calibration_loader,
        range_estimator,
        forward_call,
        logger,
    ):
        batches = list(calibration_loader)
        assert model is initialized_model
        assert range_estimator is expected_range_estimator
        assert forward_call is expected_forward_call
        assert logger is expected_logger
        assert [batch["x"].tolist() for batch in batches] == [
            [100.0, 101.0],
            [102.0],
        ]
        return "range-set-model"

    expected_range_estimator = range_estimator
    expected_forward_call = forward_call
    expected_logger = logger

    monkeypatch.setattr(
        pipeline_module,
        "get_quant_model",
        lambda model, *, logger, default_fallback: model,
    )
    monkeypatch.setattr(
        pipeline_module,
        "initialize_quantizers",
        lambda model, *, rules, logger: initialized_model,
    )
    monkeypatch.setattr(pipeline_module, "estimate_quantizer_ranges", fake_estimate_quantizer_ranges)

    result = PTQPipeline(
        rules=rules,
        range_estimator=range_estimator,
        calibration_set=lambda tokenizer: SyntheticTensorDataset([0.0, 1.0, 2.0]),
        forward_call=forward_call,
        data_collator=custom_collator,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )(
        original_model,
        tokenizer=None,
        logger=logger,
    )

    assert result == "range-set-model"
    assert len(collator_calls) == 2
