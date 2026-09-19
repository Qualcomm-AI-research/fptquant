# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path

from hydra_zen import builds, store
from omegaconf import II
from transformers import AutoTokenizer

import lm_tools
from lm_tools.llm.mock_llama import build_mock_llama
from lm_tools.llm.mock_qwen35 import build_mock_qwen35
from lm_tools.utils.llm import instantiate_llm

llm_store = store(group="llm", provider=__name__)


###################
# Huggingface LMs #
###################

HF_LLMS = [
    "llama_v32_1B_instruct",
    "llama_v32_3B_instruct",
    "llama_v3_8B",
]

# Register all llms mentioned in `paths.resources.llms`
for name in HF_LLMS:
    # Create the configuration
    model_path = II(f"paths.resources.llms.{name}")
    model_config = builds(
        instantiate_llm,
        model_name_or_path=model_path,
        populate_full_signature=True,
    )
    tokenizer_config = builds(
        AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=model_path,
        populate_full_signature=True,
    )
    config = {
        "model": model_config,
        "tokenizer": tokenizer_config,
    }
    llm_store(config, name=name)

######################
# Test Architectures #
######################
mock_model_config = builds(
    build_mock_llama,
    populate_full_signature=True,
)
tokenizer_config = builds(
    AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=Path(lm_tools.__path__[0]) / "data/mock_tokenizer",
)
mock_llama_config = {
    "model": mock_model_config,
    "tokenizer": tokenizer_config,
}
llm_store(mock_llama_config, name="mock_llama")

mock_qwen35_model_config = builds(
    build_mock_qwen35,
)
mock_qwen35_tokenizer_config = builds(
    AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=Path(lm_tools.__path__[0]) / "data/mock_tokenizer",
)
mock_qwen35_config = {
    "model": mock_qwen35_model_config,
    "tokenizer": mock_qwen35_tokenizer_config,
}
llm_store(mock_qwen35_config, name="mock_qwen35")
