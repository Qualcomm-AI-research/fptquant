# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest

from dev_tools.testing import register_markers

MOCK_QUANT_CONFIGS = ["simple_ptq"]


def pytest_configure(config):
    register_markers(config)


@pytest.fixture(params=MOCK_QUANT_CONFIGS)
def mock_quant_config(request) -> str:
    return request.param
