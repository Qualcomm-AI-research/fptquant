# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared pytest marker definitions for FastResearch packages."""

from __future__ import annotations

MARKER_NAMES = ["unit", "integration", "requires_network"]

MARKER_DEFINITIONS = [
    "unit: fast, isolated tests for a single function/class/module; must be offline and deterministic",
    "integration: tests that exercise multiple in-repository components together; required CI integration tests must still be offline, deterministic, and cache-independent",
    "requires_network: tests that may download datasets, model weights, benchmark tasks, or other remote resources",
]

DEFAULT_PR_GATE_SELECTION = "not requires_network"


def register_markers(config) -> None:
    """Register the canonical FastResearch pytest markers on ``config``."""
    for definition in MARKER_DEFINITIONS:
        config.addinivalue_line("markers", definition)
