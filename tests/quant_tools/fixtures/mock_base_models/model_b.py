# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Base Model B with components."""

import torch.nn as nn

from .common import SharedDecoder


class ModelBMLP(nn.Module):
    """Model B specific MLP component."""

    def __init__(self):
        super().__init__()
        self.name = "base_model_b_mlp"

    def forward(self, x):
        return x


class ModelB(nn.Module):
    """Complete Model B with all components."""

    def __init__(self):
        super().__init__()
        self.mlp = ModelBMLP()
        self.decoder = SharedDecoder()

        # Add config for quantization_ready
        class Config:
            model_type = "model_b"

        self.config = Config()

    def forward(self, x):
        x = self.mlp(x)
        x = self.decoder(x)
        return x
