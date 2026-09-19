# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Base Model A with components."""

import torch.nn as nn

from .common import SharedDecoder


class ModelAAttention(nn.Module):
    """Model A specific attention component."""

    def __init__(self):
        super().__init__()
        self.name = "base_model_a_attention"

    def forward(self, x):
        return x


class ModelA(nn.Module):
    """Complete Model A with all components."""

    def __init__(self):
        super().__init__()
        self.attention = ModelAAttention()
        self.decoder = SharedDecoder()

        # Add config for quantization_ready
        class Config:
            model_type = "model_a"

        self.config = Config()

    def forward(self, x):
        x = self.attention(x)
        x = self.decoder(x)
        return x


class ModelAVariant(ModelA):
    def __init__(self):
        super().__init__()

        # Add config for quantization_ready
        class Config:
            model_type = "model_a_variant"

        self.config = Config()
