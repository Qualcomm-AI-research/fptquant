# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Reusable pytest fixtures for test ergonomics and determinism."""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_chdir(tmp_path, monkeypatch):
    """Change the working directory to ``tmp_path`` for the duration of a test."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def frozen_torch_seed():
    """Set deterministic torch seeds when torch is available."""
    try:
        import torch
    except ImportError:
        return 0

    seed = 0
    torch.manual_seed(seed)
    if hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    return seed
