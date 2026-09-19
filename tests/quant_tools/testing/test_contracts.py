# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from copy import deepcopy

import fastforward as ff
import fastforward.nn as ffnn
import pytest
import torch

from quant_tools import (
    QueryQuantizationRule,
    estimate_quantizer_ranges,
    initialize_quantizers,
    quantize_with_registry_fallback,
)
from quant_tools.testing import (
    ToyLinearStack,
    assert_ptq_contract,
    assert_quantizers_initialized,
    count_quantizers,
    synthetic_calibration_loader,
)


@pytest.fixture
def logger():
    return logging.getLogger("test")


def _make_quantized_stack(logger):
    fp_model = ToyLinearStack(in_features=4, hidden=4, n_layers=2)
    qmodel = quantize_with_registry_fallback(deepcopy(fp_model), logger=logger)
    return fp_model, qmodel


def _freeze_base_parameters(model):
    for name, parameter in model.named_parameters():
        if "quantizer" not in name:
            parameter.requires_grad = False


def test_count_quantizers_and_initialized_assertions(logger):
    _, qmodel = _make_quantized_stack(logger)

    summary = count_quantizers(qmodel)

    assert summary
    assert all(count == 1 for count in summary.values())
    assert_quantizers_initialized(qmodel, pattern="**/weight_quantizer")


def test_assert_quantizers_initialized_rejects_wrong_type(logger):
    _, qmodel = _make_quantized_stack(logger)
    initialize_quantizers(
        qmodel,
        {"weights": QueryQuantizationRule("**/weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=8))},
        logger=logger,
    )

    with pytest.raises(AssertionError, match="Expected all quantizers"):
        assert_quantizers_initialized(
            qmodel,
            pattern="**/weight_quantizer",
            expected_type=ffnn.QuantizerStub,
        )


def test_assert_ptq_contract_passes_for_quantized_stack(logger):
    fp_model, qmodel = _make_quantized_stack(logger)
    initialize_quantizers(
        qmodel,
        {"weights": QueryQuantizationRule("**/weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=2))},
        logger=logger,
    )
    estimate_quantizer_ranges(
        qmodel,
        calibration_loader=synthetic_calibration_loader(
            [torch.randn(4), torch.randn(4), torch.randn(4)],
            batch_size=1,
        ),
        range_estimator=ff.range_setting.running_minmax,
        logger=logger,
        max_num_batches=3,
    )
    _freeze_base_parameters(qmodel)

    assert_ptq_contract(qmodel, {"x": torch.randn(2, 4)}, fp_model=fp_model)


def test_assert_ptq_contract_fails_when_quantization_is_a_noop(logger):
    fp_model, qmodel = _make_quantized_stack(logger)
    _freeze_base_parameters(qmodel)

    with pytest.raises(AssertionError, match="Expected PTQ output to differ"):
        assert_ptq_contract(qmodel, {"x": torch.randn(2, 4)}, fp_model=fp_model)


def test_assert_ptq_contract_fails_for_trainable_base_parameters(logger):
    _, qmodel = _make_quantized_stack(logger)

    with pytest.raises(AssertionError, match="Trainable parameters"):
        assert_ptq_contract(qmodel, {"x": torch.randn(2, 4)})
