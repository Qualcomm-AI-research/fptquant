# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Shared quantized components used by multiple model families."""

import torch.nn as nn
from fastforward.nn import QuantizedModule


class QuantizedSharedActivation(QuantizedModule, nn.Module):
    """Mock shared activation that can be used by multiple model families."""

    def __init__(self):
        super().__init__()
        self.name = "shared_activation"

    def forward(self, x):
        return x


class QuantizedSharedLinear(QuantizedModule, nn.Module):
    """Mock shared linear layer."""

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = "shared_linear"

    def forward(self, x):
        return x
