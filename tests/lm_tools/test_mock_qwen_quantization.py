# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Offline integration-lite coverage for mock Qwen + quant-tools wiring."""

import logging
import math

import fastforward as ff
import fastforward.nn as ffnn
import pytest
import torch
from hydra_zen import instantiate
from pandas import DataFrame
from torch.utils.data import DataLoader
from transformers import default_data_collator

from quant_tools import (
    QueryQuantizationRule,
    estimate_quantizer_ranges,
    get_quant_model,
    initialize_quantizers,
)
from quant_tools.testing import assert_ptq_contract, assert_quantizers_initialized


def _finite_logits(model, batch):
    with torch.no_grad(), ff.strict_quantization(False):
        outputs = model(**batch)

    assert torch.isfinite(outputs.logits).all()
    return outputs.logits


@pytest.mark.integration
def test_mock_qwen35_fallback_quantization_range_setting_and_perplexity_are_offline(compose_cfg):
    """Exercise mock Qwen3.5, quant-tools, one-batch range setting, and one-batch PPL.

    The mock Qwen3.5 config currently has model_type ``qwen3_5_text``. lm-tools
    does not yet register a specialized quantization-ready provider for that
    model type, so this test intentionally validates the CI-safe fallback path
    while still checking quantizer insertion/initialization and calibrated
    one-batch execution on local mock data.
    """
    pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
    torch.manual_seed(0)
    logger = logging.getLogger("test")

    cfg = compose_cfg(
        [
            "+llm=mock_qwen35",
            "+evaluation=perplexity_debug",
            "+paths=default",
        ]
    )

    model = instantiate(cfg.llm.model)
    tokenizer = instantiate(cfg.llm.tokenizer)
    tokenized_dataset = instantiate(cfg.evaluation.tokenized_dataset)(tokenizer=tokenizer)
    evaluation = instantiate(cfg.evaluation)

    assert model.config.model_type == "qwen3_5_text"
    assert len(tokenized_dataset) == 2
    assert set(tokenized_dataset.column_names) == {"input_ids", "attention_mask", "labels"}

    first_batch = default_data_collator([tokenized_dataset[0]])
    original_logits = _finite_logits(model, first_batch)
    assert original_logits.shape[:2] == first_batch["input_ids"].shape

    with pytest.raises(NotImplementedError, match="No quantization provider registered"):
        get_quant_model(model, logger=logger)

    quant_model = get_quant_model(model, logger=logger, default_fallback=True)
    assert isinstance(quant_model, ff.nn.QuantizedModule)
    assert_quantizers_initialized(quant_model, pattern="**/weight_quantizer")

    initialize_quantizers(
        quant_model,
        {
            "weights": QueryQuantizationRule(
                "**/weight_quantizer",
                ffnn.LinearQuantizer.factory(num_bits=8),
            )
        },
        logger=logger,
    )
    assert_quantizers_initialized(
        quant_model,
        pattern="**/weight_quantizer",
        expected_type=ffnn.LinearQuantizer,
    )

    calibration_loader = DataLoader(
        tokenized_dataset.select(range(2)),
        batch_size=1,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    estimate_quantizer_ranges(
        quant_model,
        calibration_loader=calibration_loader,
        range_estimator=ff.range_setting.running_minmax,
        logger=logger,
        max_num_batches=2,
    )

    for name, parameter in quant_model.named_parameters():
        if "quantizer" not in name:
            parameter.requires_grad = False

    quant_logits = _finite_logits(quant_model, first_batch)
    assert quant_logits.shape == original_logits.shape
    assert_ptq_contract(quant_model, first_batch)

    with ff.strict_quantization(False):
        results = evaluation(model=quant_model, tokenizer=tokenizer)

    assert isinstance(results, DataFrame)
    assert results["quantity"].tolist() == ["eval_loss", "perplexity"]
    assert results["sequence_length"].tolist() == [16, 16]
    assert all(math.isfinite(value) for value in results["value"])
