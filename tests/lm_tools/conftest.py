# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest

import lm_tools
import quant_tools
from dev_tools.testing import compose, register_markers

MOCK_LLMS = ["mock_llama", "mock_qwen35"]
MOCK_QUANT_CONFIGS = ["simple_ptq"]


def pytest_configure(config):
    register_markers(config)


@pytest.fixture
def lm_tools_configs():
    lm_tools.add_config_to_store()
    quant_tools.add_config_to_store()


@pytest.fixture
def compose_cfg(lm_tools_configs):
    def _compose(overrides: list[str]):
        return compose(overrides)

    return _compose


@pytest.fixture(params=MOCK_LLMS)
def mock_llm(request) -> str:
    return request.param


@pytest.fixture(params=MOCK_QUANT_CONFIGS)
def mock_quant_config(request) -> str:
    return request.param
