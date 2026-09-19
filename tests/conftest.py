# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from dev_tools.testing import register_markers


def pytest_configure(config):
    register_markers(config)
