# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Offline test helpers shared across packages."""

from __future__ import annotations

import os

import pytest

OFFLINE_ENV_VARS: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
}


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    """Force default test runs onto local-only Hugging Face and WandB code paths."""
    for key, value in OFFLINE_ENV_VARS.items():
        monkeypatch.setenv(key, value)


def assert_offline_env_set() -> None:
    """Assert that the canonical offline environment variables are active."""
    for key, value in OFFLINE_ENV_VARS.items():
        assert os.environ.get(key) == value, f"{key} is not set to the expected offline value"
