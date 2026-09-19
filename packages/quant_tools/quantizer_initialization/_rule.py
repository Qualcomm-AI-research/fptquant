# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Quantization rule abstractions for initializing quantizers on a model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Literal, cast

import fastforward as ff
import torch


class QuantizationRule(ABC):
    """Abstract base class for quantization rules.

    A rule encapsulates the logic for finding and initializing a set of
    quantizers on a model. Subclasses must implement :meth:`initialize`.
    """

    @abstractmethod
    def initialize(self, model: torch.nn.Module) -> List[ff.nn.Quantizer]:
        """Initialize quantizers on ``model`` according to this rule.

        Args:
            model: The (quantized) model to apply the rule to.

        Returns:
            A list of the quantizer instances that were initialized.
        """
        ...


class QueryQuantizationRule(QuantizationRule):
    """A rule that selects quantizers via an mpath query string.

    The query is evaluated against the model using :func:`ff.mpath.find_quantizers`.
    All matching quantizers are initialized with the provided ``factory`` and
    any additional keyword arguments.

    Args:
        query: An mpath query string identifying the target quantizers
            (e.g. ``"**.weight_quantizer"``).
        factory: Either a :class:`ff.Quantizer` subclass or a callable
            ``(name, module) -> ff.Quantizer`` used to construct each quantizer.
        **factory_kwargs: Extra keyword arguments forwarded to ``factory``.
    """

    def __init__(
        self,
        query: str,
        quantizer: Callable[[str, torch.nn.Module], ff.nn.Quantizer],
        overwrite_policy: Literal["overwrite", "skip", "error"] = "error",
    ):
        self._query = query
        self._quantizer_factory = quantizer
        self._overwrite_policy = overwrite_policy

    def initialize(self, model: torch.nn.Module) -> List[ff.nn.Quantizer]:
        """Find quantizers matching :attr:`_query` and initialize them.

        Args:
            model: The model to search for quantizers.

        Returns:
            A list of the initialized quantizer instances.
        """
        aliases = getattr(model, "aliases", None)
        if not isinstance(aliases, dict):
            aliases = None

        quantizers = ff.find_quantizers(
            model,
            self._query,
            aliases=aliases,
        )
        quantizers.initialize(quantizer_factory=self._quantizer_factory, overwrite_policy=self._overwrite_policy)
        return [cast(ff.nn.Quantizer, quantizer.module) for quantizer in quantizers]

    def __repr__(self) -> str:
        parts = [
            f"pattern='{self._query}'",
            f"quantizer_factory={self._quantizer_factory}",
            f"overwrite_policy='{self._overwrite_policy}'",
        ]
        return self.__class__.__name__ + "(" + ", ".join(parts) + ")"
