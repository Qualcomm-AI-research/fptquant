# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import math

import pytest
from hydra_zen import instantiate
from pandas import DataFrame

from lm_tools.evaluation.perplexity import PerplexityEvaluation


@pytest.mark.integration
def test_mock_text_tokenization_is_offline_and_fixed_length(compose_cfg):
    cfg = compose_cfg(
        [
            "+llm=mock_llama",
            "+tokenized_dataset=mock_text",
            "+paths=default",
        ]
    )

    tokenizer = instantiate(cfg.llm.tokenizer)
    tokenized_dataset = instantiate(cfg.tokenized_dataset)(tokenizer=tokenizer)

    assert len(tokenized_dataset) == 2
    assert tokenized_dataset.column_names == ["input_ids", "attention_mask", "labels"]
    for row in tokenized_dataset:
        assert len(row["input_ids"]) == 16
        assert len(row["attention_mask"]) == 16
        assert row["labels"] == row["input_ids"]


@pytest.mark.integration
def test_perplexity_pipeline_uses_mock_data_with_stable_schema(compose_cfg):
    cfg = compose_cfg(
        [
            "+llm=mock_llama",
            "+evaluation=perplexity_debug",
            "+paths=default",
        ]
    )

    model = instantiate(cfg.llm.model)
    tokenizer = instantiate(cfg.llm.tokenizer)
    evaluation = instantiate(cfg.evaluation)

    results = evaluation(model=model, tokenizer=tokenizer)

    assert isinstance(results, DataFrame)
    assert set(results.columns) == {"quantity", "value", "sequence_length", "stat"}
    assert results["quantity"].tolist() == ["eval_loss", "perplexity"]
    assert results["sequence_length"].tolist() == [16, 16]
    assert results["stat"].tolist() == ["mean", "mean"]
    assert all(math.isfinite(value) for value in results["value"])


@pytest.mark.integration
def test_hydra_composes_mock_llms_and_perplexity(compose_cfg, mock_llm: str):
    cfg = compose_cfg([f"+llm={mock_llm}", "+evaluation=perplexity_debug", "+paths=default"])

    assert instantiate(cfg.llm.model).config.vocab_size > 0
    assert instantiate(cfg.llm.tokenizer).vocab_size > 0
    assert isinstance(instantiate(cfg.evaluation), PerplexityEvaluation)
