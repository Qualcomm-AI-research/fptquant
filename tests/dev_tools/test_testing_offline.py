# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os

import pytest

from dev_tools.testing.offline import OFFLINE_ENV_VARS, assert_offline_env_set, offline_env


def test_offline_env_sets_and_restores_vars():
    for key in OFFLINE_ENV_VARS:
        os.environ.pop(key, None)

    with pytest.MonkeyPatch.context() as monkeypatch:
        offline_env.__wrapped__(monkeypatch)
        assert_offline_env_set()

    for key in OFFLINE_ENV_VARS:
        assert key not in os.environ
