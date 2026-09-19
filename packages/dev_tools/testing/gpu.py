# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""GPU-related test helpers that keep torch as an optional dependency."""

from __future__ import annotations

import pytest


def gpu_available() -> bool:
    """Return whether CUDA is available, falling back to ``False`` without torch."""
    try:
        import torch
    except ImportError:
        return False

    return bool(torch.cuda.is_available())


def requires_gpu_or_skip() -> None:
    """Skip the calling test when a CUDA-capable GPU is unavailable."""
    if not gpu_available():
        pytest.skip("no CUDA-capable GPU available")


def device_for_test() -> str:
    """Return the preferred device string for a test."""
    return "cuda" if gpu_available() else "cpu"
