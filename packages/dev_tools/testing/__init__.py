# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared testing utilities for FastResearch packages."""

from .fixtures import frozen_torch_seed, tmp_chdir
from .gpu import device_for_test, gpu_available, requires_gpu_or_skip
from .hydra import compose, compose_cfg, hydra_clean
from .markers import (
    DEFAULT_PR_GATE_SELECTION,
    MARKER_DEFINITIONS,
    MARKER_NAMES,
    register_markers,
)
from .offline import OFFLINE_ENV_VARS, assert_offline_env_set, offline_env

__all__ = [
    "DEFAULT_PR_GATE_SELECTION",
    "MARKER_DEFINITIONS",
    "MARKER_NAMES",
    "OFFLINE_ENV_VARS",
    "assert_offline_env_set",
    "compose",
    "compose_cfg",
    "device_for_test",
    "frozen_torch_seed",
    "gpu_available",
    "hydra_clean",
    "offline_env",
    "register_markers",
    "requires_gpu_or_skip",
    "tmp_chdir",
]
