# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Offline coverage for quant-tools Hydra config and rule composition."""

import logging

import fastforward.exceptions as ff_exceptions
import fastforward.nn as ffnn
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate, store
from omegaconf import OmegaConf

import quant_tools
from quant_tools.pipeline import PTQPipeline
from quant_tools.quantizer_initialization import QueryQuantizationRule, initialize_quantizers


@pytest.fixture(autouse=True)
def hydra_state():
    """Keep Hydra composition isolated between tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(scope="session", autouse=True)
def quant_tools_configs():
    """Register quant-tools configs once for this process."""
    quant_tools.add_config_to_store()
    store.add_to_hydra_store()


def _compose(overrides: list[str]):
    with initialize(version_base="1.3"):
        return compose(config_name=None, overrides=overrides)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rule_name", "rule_key", "query", "quantizer_type", "num_bits"),
    [
        ("w_example", "linear_weights", "&linear_weights_decoder", ffnn.LinearQuantizer, 4),
        (
            "a_example",
            "linear_activations",
            "{&na, &nm, {&o_proj, &down_proj}/input_quantizer}",
            ffnn.DynamicLinearQuantizer,
            4,
        ),
        ("kv_example", "kv_cache", "{&ke, &ve}", ffnn.DynamicLinearQuantizer, 4),
    ],
)
def test_builtin_rule_configs_compose_to_query_rules(rule_name, rule_key, query, quantizer_type, num_bits):
    """Built-in rule configs instantiate concrete QueryQuantizationRule objects."""
    cfg = _compose([f"+quant/rules={rule_name}"])
    rules = instantiate(cfg.quant.rules)

    assert list(rules) == [rule_key]
    rule = rules[rule_key]
    assert isinstance(rule, QueryQuantizationRule)
    assert rule._query == query

    quantizer = rule._quantizer_factory("test_quantizer", ffnn.QuantizedLinear(2, 2))
    assert isinstance(quantizer, quantizer_type)
    assert quantizer.num_bits == num_bits


@pytest.mark.unit
@pytest.mark.parametrize("range_estimator", ["smoothed_minmax", "running_minmax", "lp_norm"])
def test_builtin_range_estimator_configs_compose(range_estimator):
    """Public range-estimator configs resolve and instantiate from Hydra."""
    cfg = _compose([f"+quant/range_estimator={range_estimator}"])
    estimator = instantiate(cfg.quant.range_estimator)

    assert estimator is not None


@pytest.mark.unit
def test_mock_quant_config_composes_with_rules_and_range_estimator(mock_quant_config: str):
    """The public PTQ pipeline config wires rules and range estimation together."""
    cfg = _compose([f"+quant={mock_quant_config}", "+quant/rules=w_example", "quant/range_estimator=running_minmax"])
    OmegaConf.resolve(cfg)

    pipeline = instantiate(cfg.quant, calibration_set=lambda tokenizer: [])

    assert isinstance(pipeline, PTQPipeline)
    assert set(pipeline._rules) == {"linear_weights"}
    assert isinstance(pipeline._rules["linear_weights"], QueryQuantizationRule)
    assert pipeline._range_estimator is not None


@pytest.mark.unit
def test_exact_and_glob_queries_initialize_expected_quantizers():
    """Exact and glob query rules select the intended quantizer slots."""
    model = ffnn.QuantizedLinear(4, 4)
    rules = {
        "exact_weight": QueryQuantizationRule(
            "weight_quantizer",
            ffnn.LinearQuantizer.factory(num_bits=4),
        ),
        "glob_input": QueryQuantizationRule(
            "**/input_quantizer",
            ffnn.LinearQuantizer.factory(num_bits=8),
        ),
    }

    initialize_quantizers(model, rules, logger=logging.getLogger("test"))

    assert isinstance(model.weight_quantizer, ffnn.LinearQuantizer)
    assert model.weight_quantizer.num_bits == 4
    assert isinstance(model.input_quantizer, ffnn.LinearQuantizer)
    assert model.input_quantizer.num_bits == 8
    assert isinstance(model.output_quantizer, ffnn.QuantizerStub)


@pytest.mark.unit
def test_no_match_rule_leaves_model_unchanged(caplog):
    """A rule with no matches is allowed and logs a warning."""
    model = ffnn.QuantizedLinear(4, 4)

    with caplog.at_level(logging.WARNING):
        initialize_quantizers(
            model,
            {"missing": QueryQuantizationRule("**/missing_quantizer", ffnn.LinearQuantizer.factory(num_bits=4))},
            logger=logging.getLogger("test"),
        )

    assert isinstance(model.weight_quantizer, ffnn.QuantizerStub)
    assert "applied to 0 quantizers" in caplog.text


@pytest.mark.unit
def test_overlapping_rules_raise_when_reinitialization_would_be_required():
    """Overlapping rules currently follow fastforward's default reinit policy and fail."""
    model = ffnn.QuantizedLinear(4, 4)

    with pytest.raises(ff_exceptions.QuantizationError, match="already initialized"):
        initialize_quantizers(
            model,
            {
                "first": QueryQuantizationRule("**/weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=4)),
                "second": QueryQuantizationRule("**/weight_quantizer", ffnn.LinearQuantizer.factory(num_bits=8)),
            },
            logger=logging.getLogger("test"),
        )
