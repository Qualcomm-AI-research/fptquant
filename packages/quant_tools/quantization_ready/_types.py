# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Base types for quantization."""

from __future__ import annotations

import fastforward.nn.quantized_module as ffqm


class FakeQuantizedModule(ffqm.QuantizedModule):
    """Placeholder QuantizedModule used to mark modules that are intentionally left in FP."""

    pass
