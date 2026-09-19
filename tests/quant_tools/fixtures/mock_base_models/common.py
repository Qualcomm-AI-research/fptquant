# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Common base components shared across models."""

import torch.nn as nn


class SharedDecoder(nn.Module):
    """Shared decoder component used by multiple models."""

    def __init__(self):
        super().__init__()
        self.name = "base_shared_decoder"

    def forward(self, x):
        return x
