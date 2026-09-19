# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Parametrizations for learnable clipping parameters.

Adapted from ``fastforward.nn._parametrizations`` so that
:class:`fastforward_contrib.nn.DynamicLinearClipQuantizer` can offer a
configurable clip-factor transform without depending on a newer fastforward
release.

These parametrizations wrap a raw ``torch.nn.Parameter`` via
:func:`torch.nn.utils.parametrize.register_parametrization` so that reading the
attribute returns the *effective* (transformed) value used by the quantizer,
while the optimizer sees the raw parameter.

Each parametrization class implements:

* ``forward(raw) -> effective``: maps the raw parameter to its effective value.
* ``right_inverse(effective) -> raw``: maps a desired effective value back to
  the raw parameter. This is used both by
  :func:`register_parametrization` to initialize the raw parameter from a given
  effective value, and for data-dependent initialization schemes that want to
  set a target effective value without having to invert the transform by hand.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lower bound used by parametrizations that need to avoid division by zero
# or log(0). Small enough to be a no-op in float32 for sensible values, large
# enough to keep numerics well-defined.
_CLAMP_MIN = 1e-6


class _ScaleIdentity(nn.Module):
    """Effective value: ``clamp(raw, min=1e-6)``.

    Unconstrained raw parameter; useful when gamma may exceed 1
    (e.g. with smoothmax-style reducers that under-estimate the true max).
    """

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.clamp(min=_CLAMP_MIN)

    def right_inverse(self, effective: torch.Tensor) -> torch.Tensor:
        return effective


class _ScaleAbs(nn.Module):
    """Effective value: ``clamp(|raw|, min=1e-6)``.

    Sign-free raw parameter.
    """

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.abs().clamp(min=_CLAMP_MIN)

    def right_inverse(self, effective: torch.Tensor) -> torch.Tensor:
        return effective


class _ScaleSoftplus(nn.Module):
    """Effective value: ``softplus(raw)``.

    Smooth positive transform; raw parameter is unconstrained.
    """

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw)

    def right_inverse(self, effective: torch.Tensor) -> torch.Tensor:
        # inverse of softplus: log(exp(x) - 1) = log(expm1(x))
        return torch.log(torch.expm1(effective))


class _ScaleExp(nn.Module):
    """Effective value: ``exp(raw)``.

    Log-scale space; raw parameter is unconstrained.
    """

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.exp()

    def right_inverse(self, effective: torch.Tensor) -> torch.Tensor:
        return effective.clamp(min=_CLAMP_MIN).log()


class _ScaleSigmoid(nn.Module):
    """Effective value: ``sigmoid(raw) in (0, 1)``.

    This is the classic "learnable clipping factor" form: gamma cannot exceed
    one, so the quantizer can only *shrink* the dynamic range. Not compatible
    with reducers that under-estimate the true max (e.g. ``LpReducer``).
    """

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw)

    def right_inverse(self, effective: torch.Tensor) -> torch.Tensor:
        effective = effective.clamp(min=_CLAMP_MIN, max=1.0 - _CLAMP_MIN)
        return torch.log(effective / (1.0 - effective))


CLIP_PARAMETRIZATION_REGISTRY: dict[str, type[nn.Module]] = {
    "identity": _ScaleIdentity,
    "abs": _ScaleAbs,
    "softplus": _ScaleSoftplus,
    "exp": _ScaleExp,
    "sigmoid": _ScaleSigmoid,
}
"""Registry of clip-factor parametrizations available to
:class:`fastforward_contrib.nn.DynamicLinearClipQuantizer`.

Keys: string identifiers usable as the ``clip_parametrization`` constructor
argument. Values: ``nn.Module`` subclasses implementing ``forward`` and
``right_inverse`` suitable for
:func:`torch.nn.utils.parametrize.register_parametrization`.
"""


def parametrization_module(name: str) -> nn.Module:
    """Instantiate a clip parametrization module by registry key.

    Args:
        name: Key into :data:`CLIP_PARAMETRIZATION_REGISTRY`.

    Returns:
        A freshly-constructed parametrization module.

    Raises:
        ValueError: If ``name`` is not a known parametrization.
    """
    if name not in CLIP_PARAMETRIZATION_REGISTRY:
        msg = f"Unknown clip parametrization '{name}'. Choose from: {list(CLIP_PARAMETRIZATION_REGISTRY)}"
        raise ValueError(msg)
    return CLIP_PARAMETRIZATION_REGISTRY[name]()
