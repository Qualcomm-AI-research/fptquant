# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

import torch


class ToyLinear(torch.nn.Module):
    """Tiny Linear wrapper that keeps test fixtures explicit and portable."""

    def __init__(self, in_features: int = 4, out_features: int = 4, *, bias: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ToyLinearStack(torch.nn.Module):
    """Small Linear+ReLU stack for quantization and calibration tests."""

    def __init__(
        self,
        in_features: int = 4,
        hidden: int = 4,
        *,
        out_features: int | None = None,
        n_layers: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        out_features = hidden if out_features is None else out_features
        layers: list[torch.nn.Module] = []
        current_in = in_features
        for index in range(n_layers):
            current_out = out_features if index == n_layers - 1 else hidden
            layers.append(torch.nn.Linear(current_in, current_out, bias=bias))
            if index != n_layers - 1:
                layers.append(torch.nn.ReLU())
            current_in = current_out
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ToyEvalTrackingModule(torch.nn.Module):
    """CPU-only module that records eval calls and forwarded batches."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(1.0))
        self.eval_called = False
        self.forward_batches: list[torch.Tensor] = []

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_batches.append(x.detach().clone())
        return x + self.bias
