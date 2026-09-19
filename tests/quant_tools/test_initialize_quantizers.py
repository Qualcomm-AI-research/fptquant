# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for initialize_quantizers core functionality."""

import logging

import fastforward as ff
import fastforward.nn as ffnn
import pytest

from quant_tools.quantizer_initialization import initialize_quantizers
from quant_tools.quantizer_initialization._rule import QuantizationRule, QueryQuantizationRule

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def quant_model():
    """Provide a QuantizedLinear(4, 4) with four QuantizerStub slots."""
    return ffnn.QuantizedLinear(4, 4)


@pytest.fixture
def logger():
    """Provide a test logger."""
    return logging.getLogger("test")


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================


class TestInitializeQuantizers:
    """Tests for initialize_quantizers function."""

    def test_quantizers_initialized_with_ff_quantizer_class(self, quant_model, logger):
        """Initialize weight_quantizer with a FF Quantizer class; other slots stay as QuantizerStub."""
        rule = QueryQuantizationRule("weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=8))
        initialize_quantizers(quant_model, {"weights": rule}, logger=logger)

        assert isinstance(quant_model.weight_quantizer, ffnn.LinearQuantizer)
        assert isinstance(quant_model.input_quantizer, ffnn.QuantizerStub)
        assert isinstance(quant_model.bias_quantizer, ffnn.QuantizerStub)
        assert isinstance(quant_model.output_quantizer, ffnn.QuantizerStub)

    def test_quantizers_initialized_with_custom_factory(self, quant_model, logger):
        """Initialize weight_quantizer with a custom callable factory; other slots stay as QuantizerStub."""

        def my_factory(name: str, module) -> ffnn.LinearQuantizer:
            return ffnn.LinearQuantizer(num_bits=4)

        rule = QueryQuantizationRule("**/weight_quantizer", my_factory)
        initialize_quantizers(quant_model, {"weights": rule}, logger=logger)

        assert isinstance(quant_model.weight_quantizer, ffnn.LinearQuantizer)
        assert isinstance(quant_model.input_quantizer, ffnn.QuantizerStub)
        assert isinstance(quant_model.bias_quantizer, ffnn.QuantizerStub)
        assert isinstance(quant_model.output_quantizer, ffnn.QuantizerStub)

    def test_multiple_rules_initialize_different_quantizers(self, quant_model, logger):
        """Two rules targeting different slots each initialize only their own slot."""
        rules = {
            "weights": QueryQuantizationRule("**/weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=8)),
            "inputs": QueryQuantizationRule("**/input_quantizer", ffnn.LinearQuantizer.factory(num_bits=8)),
        }
        initialize_quantizers(quant_model, rules, logger=logger)

        assert isinstance(quant_model.weight_quantizer, ffnn.LinearQuantizer)
        assert isinstance(quant_model.input_quantizer, ffnn.LinearQuantizer)
        assert isinstance(quant_model.bias_quantizer, ffnn.QuantizerStub)
        assert isinstance(quant_model.output_quantizer, ffnn.QuantizerStub)

    def test_all_rules_are_applied(self, quant_model, logger):
        """All rules in the mapping are called in insertion order."""
        applied = []

        class TrackingRule(QuantizationRule):
            def __init__(self, tag: str):
                self.tag = tag

            def initialize(self, model):
                applied.append(self.tag)
                return []

        initialize_quantizers(
            quant_model,
            {"a": TrackingRule("a"), "b": TrackingRule("b")},
            logger=logger,
        )
        assert applied == ["a", "b"]

    def test_returns_same_model_and_quantizers_on_device(self, quant_model, logger):
        """Return the same model object and move all quantizer parameters to the model device."""
        rule = QueryQuantizationRule("**/weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=8))
        result = initialize_quantizers(quant_model, {"w": rule}, logger=logger)

        assert result is quant_model

        model_device = next(quant_model.parameters()).device
        for filter_result in ff.find_quantizers(quant_model, "*"):
            for p in filter_result.module.parameters():
                assert p.device == model_device
