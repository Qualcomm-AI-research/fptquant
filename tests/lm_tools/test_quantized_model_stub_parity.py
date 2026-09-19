# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Stub-only parity checks for custom quantized LM providers."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import fastforward as ff
import pytest
import torch
from fastforward.nn import QuantizerStub
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MinistralConfig,
    MinistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

import lm_tools  # noqa: F401  # Import registers lm-tools quantization providers.
from lm_tools.quantized_models.llama import QuantizedLlamaForCausalLM
from lm_tools.quantized_models.ministral import QuantizedMinistralForCausalLM
from lm_tools.quantized_models.qwen2 import QuantizedQwen2ForCausalLM
from lm_tools.quantized_models.qwen3 import QuantizedQwen3ForCausalLM
from quant_tools import get_quant_model


@dataclass(frozen=True)
class QuantizedModelCase:
    name: str
    config_cls: type[Any]
    model_cls: type[Any]
    expected_quantized_cls: type[torch.nn.Module]
    extra_config: dict[str, Any] = field(default_factory=dict)


MODEL_CASES = [
    QuantizedModelCase(
        name="llama",
        config_cls=LlamaConfig,
        model_cls=LlamaForCausalLM,
        expected_quantized_cls=QuantizedLlamaForCausalLM,
    ),
    QuantizedModelCase(
        name="qwen2",
        config_cls=Qwen2Config,
        model_cls=Qwen2ForCausalLM,
        expected_quantized_cls=QuantizedQwen2ForCausalLM,
    ),
    QuantizedModelCase(
        name="qwen3",
        config_cls=Qwen3Config,
        model_cls=Qwen3ForCausalLM,
        expected_quantized_cls=QuantizedQwen3ForCausalLM,
        extra_config={"use_sliding_window": False},
    ),
    QuantizedModelCase(
        name="ministral",
        config_cls=MinistralConfig,
        model_cls=MinistralForCausalLM,
        expected_quantized_cls=QuantizedMinistralForCausalLM,
        extra_config={"head_dim": 8, "sliding_window": 32},
    ),
]


def _tiny_causal_lm_config(case: QuantizedModelCase) -> Any:
    config = case.config_cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **case.extra_config,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return config


def _logits(model: torch.nn.Module, sequence_length: int) -> torch.Tensor:
    input_ids = torch.arange(1, sequence_length + 1, dtype=torch.long).unsqueeze(0)
    with torch.no_grad(), ff.strict_quantization(False):
        return model(input_ids=input_ids, use_cache=False).logits


@pytest.mark.integration
@pytest.mark.parametrize("case", MODEL_CASES, ids=[case.name for case in MODEL_CASES])
def test_quantized_custom_model_with_stubs_matches_base_model(case: QuantizedModelCase) -> None:
    torch.manual_seed(1234)
    base_model = case.model_cls(_tiny_causal_lm_config(case)).eval()
    quantized_model = get_quant_model(
        copy.deepcopy(base_model),
        logger=logging.getLogger(__name__),
    ).eval()

    assert isinstance(quantized_model, case.expected_quantized_cls)

    quantizers = list(ff.find_quantizers(quantized_model, "**"))
    assert quantizers
    non_stub_quantizers = [
        f"{result.full_name} ({type(result.module).__name__})"
        for result in quantizers
        if not isinstance(result.module, QuantizerStub)
    ]
    assert not non_stub_quantizers

    single_token_base_logits = _logits(base_model, sequence_length=1)
    single_token_quantized_logits = _logits(quantized_model, sequence_length=1)
    assert torch.equal(single_token_quantized_logits, single_token_base_logits)

    multi_token_base_logits = _logits(base_model, sequence_length=4)
    multi_token_quantized_logits = _logits(quantized_model, sequence_length=4)
    torch.testing.assert_close(
        multi_token_quantized_logits,
        multi_token_base_logits,
        rtol=0.0,
        atol=1e-6,
    )
