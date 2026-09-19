# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import sys

from dev_tools.testing.gpu import device_for_test, gpu_available


def test_gpu_available_handles_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    assert gpu_available() is False
    assert device_for_test() == "cpu"
