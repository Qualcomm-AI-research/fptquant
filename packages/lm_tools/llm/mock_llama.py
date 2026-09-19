# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch


def build_mock_llama_config():
    from transformers.models.llama.configuration_llama import LlamaConfig

    model_config = LlamaConfig()
    # update to align with LLama_v32_1B model
    for k, v in {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1280,
        "eos_token_id": [1281, 1288, 1289],
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "max_position_embeddings": 131072,
        "mlp_bias": False,
        "model_type": "llama",
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "rope_theta": 500000.0,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
        "transformers_version": "4.45.2",
        "use_cache": True,
        "vocab_size": 1286,
    }.items():
        setattr(model_config, k, v)
    # add some settings that will make the model very lightweight
    model_config.num_attention_heads = 4
    model_config.num_key_value_heads = 2
    model_config.hidden_size = 16
    model_config.intermediate_size = 32
    model_config.num_hidden_layers = 2
    model_config.head_dim = 4

    return model_config


def build_mock_llama(model_dtype: str | None = None):
    from transformers import AutoModelForCausalLM

    from lm_tools.utils.model_aliases import get_aliases

    model_config = build_mock_llama_config()
    model = AutoModelForCausalLM.from_config(model_config)

    if model_dtype is not None:
        model = model.to(dtype=getattr(torch, model_dtype))

    # Add the aliases for mpath selection
    model.aliases = get_aliases(model.config.model_type)

    return model
