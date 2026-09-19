# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Base Model C with components."""

import torch.nn as nn


class ModelCEncoder(nn.Module):
    """Model C specific encoder component."""

    def __init__(self):
        super().__init__()
        self.name = "base_model_c_encoder"

    def forward(self, x):
        return x


class ModelC(nn.Module):
    """Complete Model C with all components."""

    def __init__(self):
        super().__init__()
        self.encoder = ModelCEncoder()

        # Add config for quantization_ready
        class Config:
            model_type = "model_c"

        self.config = Config()

    def forward(self, x):
        x = self.encoder(x)
        return x
