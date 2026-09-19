# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

import torch
from fastforward import forward_override as override

_TRANSFORM_COUNT = 0


class TransformOpType(Enum):
    forward = auto()
    inverse = auto()


def generate_new_name() -> str:
    global _TRANSFORM_COUNT
    name = "T_" + str(_TRANSFORM_COUNT)
    _TRANSFORM_COUNT += 1
    return name


class TransformMetadata:
    def __init__(
        self,
        name: Optional[str] = None,
        parent: Optional["TransformMetadata"] = None,
        index: Optional[str | int] = None,
    ):
        self._name = name
        self._parent = parent
        self._index = index

    @property
    def name(self) -> str:
        if self._name is None:
            return ""
        else:
            suffix = f".{self._index}" if self._index is not None else ""
            if self._parent is None:
                return self._name + suffix
            else:
                return self._parent.name + self._name + suffix

    def __eq__(self, other: object):
        if not isinstance(other, TransformMetadata):
            return False
        return self.name == other.name


class Transform(torch.nn.Module, ABC):
    required_parameters: List[str] = []

    def __init__(
        self,
    ):
        super().__init__()
        self._transform_overrides = collections.OrderedDict()
        self._metadata = TransformMetadata()
        self.reset_parameters()

    def register_override(self, override_fn: override.OverrideFn[torch.Tensor]) -> override.OverrideHandle:
        handle = override.OverrideHandle(self)  # noqa
        self._transform_overrides[handle.handle_id] = override_fn
        return handle

    def remove_override(self, override_id: int) -> override.OverrideFn[torch.Tensor] | None:
        return self._transform_overrides.pop(override_id, None)

    @property
    @abstractmethod
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        pass

    @property
    @abstractmethod
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        pass

    @property
    def metadata(self) -> TransformMetadata:
        return self._metadata

    @property
    def name(self) -> str | None:
        return self.metadata.name

    @name.setter
    def name(self, new_value: str):
        self.metadata._name = new_value

    def reset_parameters(self):
        for parameter_name in self.required_parameters:
            setattr(self, parameter_name, None)

    @abstractmethod
    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        pass

    @property
    def overrides(self) -> Iterator[override.OverrideFn[torch.Tensor]]:
        """Yield overrides attached to this transform."""
        yield from self._transform_overrides.values()

    @property
    def initialized(self) -> bool:
        for param_name in self.required_parameters:
            if getattr(self, param_name) is None:
                return False
        return True

    def transform(self, data: torch.Tensor, inv: bool = False, dim: int = -1, **kwargs) -> torch.Tensor:
        if not self.initialized:
            self.__init_parameters__(data.shape[dim], dtype=data.dtype, device=data.device)

        orig_dtype = data.dtype

        if inv:
            data = self.inverse_fn(data=data, dim=dim, **kwargs)
        else:
            data = self.forward_fn(data=data, dim=dim, **kwargs)

        if data.dtype != orig_dtype:
            raise RuntimeError(
                f"The transformation {self.__class__.__name__} has changed the dtype of the input tensor: {orig_dtype} expected, {data.dtype} found."
            )

        return data

    def forward(self, data: torch.Tensor, inv: bool = False, dim: int = -1, **kwargs) -> torch.Tensor:
        wrapped_transform = override.apply_overrides(self, self.transform, self._transform_overrides)
        return wrapped_transform(data, inv=inv, dim=dim, **kwargs)

    def _extra_repr(self) -> str:
        return ""

    def __repr__(self):
        repr = f"{self.__class__.__name__}" + "("
        extra_repr = self._extra_repr()
        if "\n" in extra_repr:
            repr += "\n  " + extra_repr.replace("\n", "\n  ") + "\n"
        else:
            repr += extra_repr
        repr += ")"
        return repr

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        else:
            eq = True
            eq = eq and self._metadata == other._metadata
            eq = eq and self.__class__ == other.__class__
            eq = eq and self.parameters == other.parameters
            return eq

    def __hash__(self):
        return hash(repr(self))


class MatmulTransform(Transform, ABC):
    def transform(
        self, data: torch.Tensor, inv: bool = False, dim: int = -1, transpose: bool = False, **kwargs
    ) -> torch.Tensor:
        assert len(kwargs) == 0
        return super().transform(
            data=data,
            inv=inv,
            dim=dim,
            transpose=transpose,
        )


class OrthogonalTransform(MatmulTransform, ABC):
    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return _transpose_wrapper(self.forward_fn)


class IdentityTransform(OrthogonalTransform):
    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return _identity_function

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device) -> Dict[str, Any]:
        return {}


class TransposedTransform(MatmulTransform):
    def __init__(self, base_transform: MatmulTransform):
        super().__init__()
        self.base_transform: MatmulTransform = base_transform
        self._metadata = base_transform._metadata

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        pass

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return _transpose_wrapper(self.base_transform.forward_fn)

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return _transpose_wrapper(self.base_transform.inverse_fn)


class InvertedTransform(Transform):
    def __init__(self, base_transform: Transform):
        super().__init__()
        self.base_transform: Transform = base_transform
        self._metadata = base_transform._metadata

    def __init_parameters__(self, parameter_dim: int, dtype: torch.dtype, device: torch.device):
        pass

    @property
    def forward_fn(self) -> Callable[..., torch.Tensor]:
        return self.base_transform.inverse_fn

    @property
    def inverse_fn(self) -> Callable[..., torch.Tensor]:
        return self.base_transform.forward_fn


def transpose(transform: MatmulTransform) -> MatmulTransform:
    if isinstance(transform, TransposedTransform):
        return transform.base_transform
    else:
        return TransposedTransform(base_transform=transform)


def invert(transform: Transform) -> Transform:
    if isinstance(transform, InvertedTransform):
        return transform.base_transform
    else:
        return InvertedTransform(base_transform=transform)


def _transpose_wrapper(transform_function):
    def wrapped(data: torch.Tensor, dim: int, transpose: bool = False) -> torch.Tensor:
        return transform_function(data=data, dim=dim, transpose=not transpose)

    return wrapped


def _identity_function(data, *args, **kwargs):
    return data
