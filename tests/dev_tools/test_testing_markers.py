# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from dev_tools.testing.markers import MARKER_DEFINITIONS, register_markers


class _Config:
    def __init__(self):
        self.lines = []

    def addinivalue_line(self, key, value):
        self.lines.append((key, value))


def test_register_markers_adds_canonical_definitions():
    config = _Config()

    register_markers(config)

    assert config.lines == [("markers", definition) for definition in MARKER_DEFINITIONS]
