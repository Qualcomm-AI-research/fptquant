# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from abc import abstractmethod
from functools import partial
from typing import List

import torch

from fasttransforms.nn.functional.permutation import permute

from .common import OrthogonalTransform
from .composition import KroneckerPowerTransform


class Permutation(OrthogonalTransform):
    required_parameters: List[str] = ["order"]

    @abstractmethod
    def _make_perm(self, parameter_dim: int) -> torch.LongTensor:
        pass

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        self.order = torch.nn.Parameter(
            self._make_perm(parameter_dim=parameter_dim).to(device=device), requires_grad=False
        )

    @property
    def forward_fn(self):
        return partial(permute, order=self.order)


class RandomPermutation(Permutation):
    def _make_perm(self, parameter_dim: int) -> torch.LongTensor:
        return torch.LongTensor(torch.randperm(parameter_dim).to(torch.long))


class ZigZag2DPermutation(Permutation):
    def _make_perm(self, parameter_dim: int) -> torch.LongTensor:
        size = int(parameter_dim**0.5)
        if size**2 != parameter_dim:
            raise RuntimeError(f"The dimension {parameter_dim} must be a square.")
        order = []
        flip = True
        for i in range(size):
            for j in range(i + 1):
                k = i - j
                if flip:
                    order.append(j + k * size)
                else:
                    order.append(k + j * size)
            flip = not flip
        for i in reversed(range(size - 1)):
            for j in range(i + 1):
                k = size - 1 - i + j
                if flip:
                    order.append(k + (size - 1 - j) * size)
                else:
                    order.append((size - 1 - j) + k * size)
            flip = not flip
        return torch.LongTensor(order)


class WalshPermutation(Permutation):
    def _make_perm(self, parameter_dim: int) -> torch.LongTensor:
        from .hadamard import SylvesterHadamardTransform

        diff = (torch.eye(parameter_dim) - torch.roll(torch.eye(parameter_dim), 1, dims=1))[:-1]
        hadamard = SylvesterHadamardTransform()
        D = hadamard(diff, dim=-1).abs()

        return torch.LongTensor(torch.argsort(D.sum(0)).to(torch.long))


class Walsh2DPermutation(Permutation):
    def _make_perm(self, parameter_dim: int) -> torch.LongTensor:
        transform_func = KroneckerPowerTransform(
            WalshPermutation(),
            exponent=2,
        )
        order = transform_func(torch.arange(parameter_dim)).to(dtype=torch.long)
        return order
