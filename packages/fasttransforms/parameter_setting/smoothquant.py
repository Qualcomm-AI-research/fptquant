# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Iterator, Optional, TypeVar

import torch
from fastforward.forward_override import OverrideHandle
from torchmetrics import Metric

import fasttransforms as ft

from .common import ParameterEstimator, SimpleEstimatorStep

T = TypeVar("T")

logger = logging.getLogger(__name__)


class AbsMax(Metric):
    """Compute elementwise max(abs(x)) over all dims except `dim`, aggregated over updates."""

    _max: torch.Tensor

    def __init__(self):
        super().__init__()
        # We don't know the size of the reduced dimension until the first update,
        # so initialize with an empty tensor and replace it on first update.
        self.add_state("_max", default=torch.tensor([]), dist_reduce_fx="max")

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])

        # Per-index abs max along flattened batch dimension
        x_absmax = x.abs().amax(dim=0)

        # Initialize or update running max
        if self._max.numel() == 0:
            self._max: torch.Tensor = x_absmax
        else:
            # Sanity check in case dim size changes between calls
            if self._max.shape != x_absmax.shape:
                raise ValueError(
                    f"AbsMax received incompatible shapes across updates: "
                    f"state {tuple(self._max.shape)} vs new {tuple(x_absmax.shape)}. "
                    f"Did the size of dim={self.dim} change?"
                )
            self._max = torch.maximum(self._max, x_absmax)

    @property
    def has_been_updated(self) -> bool:
        return self._max.numel() > 0

    def compute(self) -> torch.Tensor:
        # If never updated, return empty tensor (or raise, depending on preference)
        return self._max


class MeanPower(Metric):
    """Per-index mean of x**exp over all dims except `dim`, aggregated over updates.

    Output shape: (x.shape[dim],)
    """

    _sum: torch.Tensor
    _count: torch.Tensor

    def __init__(self, exp: float):
        super().__init__()
        self.exp = exp

        # Running sum (vector) and total count (scalar)
        # dist_reduce_fx="sum" makes it DDP-safe (sums across processes)
        self.add_state("_sum", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])

        # Compute power and reduce over N
        x_pow = torch.pow(x, self.exp)
        batch_sum = x_pow.sum(dim=0)  # shape: (D,)
        batch_count = x.shape[0]  # scalar N

        # Initialize state lazily (we don't know D in __init__)
        if self._sum.numel() == 0:
            self._sum: torch.Tensor = batch_sum
        else:
            if self._sum.shape != batch_sum.shape:
                raise ValueError(
                    f"MeanPower received incompatible shapes across updates: "
                    f"state {tuple(self._sum.shape)} vs new {tuple(batch_sum.shape)}. "
                    f"Did the size of dim={self.dim} change?"
                )
            self._sum += batch_sum
        self._count += batch_count

    @property
    def has_been_updated(self) -> bool:
        return self._count.item() > 0

    def compute(self) -> torch.Tensor:
        if self._count.item() == 0:
            # never updated; return empty tensor
            return self._sum

        # Cast count to match dtype/device for safe division
        return self._sum / self._count.to(self._sum.dtype)


class SmoothQuantEstimatorStep(SimpleEstimatorStep[ft.nn.transforms.ScaleTransform], torch.nn.Module):
    """Estimates the smoothquant scales"""

    def __init__(self, alpha: float = 0.5, pow: str | float = "max") -> None:
        """Initialize the SmoothedMinMaxEstimator.

        Args:
            alpha: The smoothing factor.
            disable_transforms: Flag to disable the transforms.
        """
        super().__init__(disable_transforms=True)
        self.alpha = alpha

        if isinstance(pow, str) and pow != "max":
            raise ValueError(f"Invalid value for norm {pow}.")

        self._pow = pow

        self.forward_metric: Optional[Metric] = None
        self.inverse_metric: Optional[Metric] = None

    def initialize_parameters(self, data: torch.Tensor, dim: int) -> None:
        """Initialize the parameters"""
        if self.forward_metric is None:
            if self._pow == "max":
                self.forward_metric = AbsMax()
            elif isinstance(self._pow, float) or isinstance(self._pow, int):
                self.forward_metric = MeanPower(exp=self._pow)
            else:
                raise NotImplementedError()
        if self.inverse_metric is None:
            if self._pow == "max":
                self.inverse_metric = AbsMax()
            elif isinstance(self._pow, float) or isinstance(self._pow, int):
                self.inverse_metric = MeanPower(exp=self._pow)
            else:
                raise NotImplementedError()

    def estimate_step(
        self, transform: ft.nn.transforms.ScaleTransform, data: torch.Tensor, dim: int, inv: bool, **kwargs
    ) -> None:
        """Perform a single estimation step.

        Args:
            transform: The transform to estimate ranges for.
            data: The input data tensor.
        """
        self.initialize_parameters(data, dim=dim)
        assert self.forward_metric is not None
        assert self.inverse_metric is not None

        # Set the relevant dimension as the last
        data = data.transpose(dim, -1)

        if inv:
            self.inverse_metric.update(data)
        else:
            self.forward_metric.update(data)

        if self.forward_metric.has_been_updated and self.inverse_metric.has_been_updated:
            transform.scale = (
                self.inverse_metric.compute() ** (1.0 - self.alpha) / self.forward_metric.compute() ** self.alpha
            )

    def extra_repr(self) -> str:
        """Return a string representation of the estimator.

        Returns:
            str: A string representation of the estimator.
        """
        return f"forward_stat={self.forward_metric}, inverse_stat={self.inverse_metric}"


class SmoothQuantEstimator(ParameterEstimator[OverrideHandle, ft.Transform]):
    """SmoothQuant for ScaleSetting"""

    def __init__(
        self,
        alpha: float = 0.5,
        disable_other_transforms: bool = True,
    ) -> None:
        self.alpha = alpha
        self.disable_other_transforms = disable_other_transforms

    def prepare(self, module: ft.Transform) -> Optional[OverrideHandle]:
        """Prepare the module for range estimation.

        Args:
            module: The transform module.

        Returns:
            OverrideHandle: The override handle for the module.
        """
        if isinstance(module, ft.nn.transforms.ScaleTransform):
            # We add the Estimator handle to all scale settable transforms
            return module.register_override(SmoothQuantEstimatorStep(alpha=self.alpha))
        else:
            if self.disable_other_transforms:
                # Otherwise we disable the transforms
                return module.register_override(ft.overrides.DisableTransformOverride())
            else:
                # Unless otherwise specified
                return None

    def cleanup(self, module: ft.Transform, metadata: Optional[OverrideHandle]) -> None:
        """Clean up the module after range estimation.

        Args:
            module: The transform module.
            metadata: The override handle for the module.
        """
        del module
        if metadata:
            metadata.remove()

    def split_module(self, module: torch.nn.Module) -> Iterator[ft.Transform]:
        """Split the module into individual transforms.

        Args:
            module: The module to split.

        Yields:
            Iterator[transform]: An iterator over the transforms.
        """
        for transform in ft.get_transforms(module, recurse=True):
            yield transform
