# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib
from typing import Any, Callable, Generator, Iterable, List

import torch
from fastforward.forward_override import OverrideHandle

import fasttransforms as ft


@contextlib.contextmanager
def disable_transforms(model: torch.nn.Module, disable: str = "transform_operations") -> Generator[None, None, None]:
    handles: list[OverrideHandle] = []

    # Disable transform operations
    if disable == "transform_operations":
        for transform_op in ft.get_transform_operations(model):
            handles.append(transform_op.register_override(DisableTransformOpOverride()))

    # Disable transforms
    elif disable == "transforms":
        for transform in ft.get_transforms(model):
            handles.append(transform.register_override(DisableTransformOverride()))

    else:
        raise ValueError("disable can be set to 'transform_operations' or 'transforms'.")

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def enable_transforms(model: torch.nn.Module, disable: str = "transform_operations") -> Generator[None, None, None]:
    with contextlib.ExitStack() as exit_stack:
        # Enable transform operations
        if disable == "transform_operations":
            for transform_operation in ft.get_transform_operations(model):
                for transform_op_override in transform_operation.overrides:
                    if isinstance(transform_op_override, DisableTransformOpOverride):
                        exit_stack.enter_context(transform_op_override.enable_transform())
        # Enable transforms
        elif disable == "transforms":
            for transform in ft.get_transforms(model):
                for transform_override in transform.overrides:
                    if isinstance(transform_override, DisableTransformOverride):
                        exit_stack.enter_context(transform_override.enable_transform())
        else:
            raise ValueError("disable can be set to 'transform_operations' or 'transforms'.")
        yield


class DisableTransformOpOverride:
    """Override to disable transform operation."""

    def __init__(self) -> None:
        self._transform_op_enabled = False
        self._handles: List[OverrideHandle] = []

    @property
    def transform_enabled(self) -> bool:
        """True if transform is enabled, False otherwise."""
        return self._transform_op_enabled

    def enable_transform(self, enabled: bool = True) -> "_TransformationOpContext":
        """Enable transformation."""
        context = _TransformationOpContext(self, self._transform_op_enabled)
        self._transform_op_enabled = enabled
        return context

    def disable_transform(self) -> "_TransformationOpContext":
        """Disable transformation."""
        return self.enable_transform(enabled=False)

    def __call__(
        self,
        _context: Any,
        callback: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Override function for transfrom disabling."""

        if self._transform_op_enabled:
            return callback(*args, **kwargs)
        else:
            return _extract_data_from_args(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(transform_op_enabled={self._transform_op_enabled})"

    def attach_to(self, transform_ops: ft.TransformOperation | Iterable[ft.TransformOperation]) -> None:
        """Attach this override to one or more transformations."""
        if isinstance(transform_ops, ft.TransformOperation):
            self._handles.append(transform_ops.register_override(self))
        else:
            for transform_op in transform_ops:  # type: ignore[union-attr]
                self.attach_to(transform_op)

    def detach(self) -> None:
        """Detach this override."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _TransformationOpContext(contextlib.AbstractContextManager[None]):
    def __init__(self, override: DisableTransformOpOverride, transform_enabled_reset_status: bool) -> None:
        self._override = override
        self._reset_transformation_status = transform_enabled_reset_status

    def __enter__(self) -> None: ...

    def __exit__(self, type, value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._override._transform_op_enabled = self._reset_transformation_status


class DisableTransformOverride:
    """Override to disable transform."""

    def __init__(self) -> None:
        self._transform_enabled = False
        self._handles: List[OverrideHandle] = []

    @property
    def transform_enabled(self) -> bool:
        """True if transform is enabled, False otherwise."""
        return self._transform_enabled

    def enable_transform(self, enabled: bool = True) -> "_TransformationContext":
        """Enable transformation."""
        context = _TransformationContext(self, self._transform_enabled)
        self._transform_enabled = enabled
        return context

    def disable_transform(self) -> "_TransformationContext":
        """Disable transformation."""
        return self.enable_transform(enabled=False)

    def __call__(
        self,
        _context: Any,
        callback: Callable[..., torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Override function for transfrom disabling."""

        if self._transform_enabled:
            return callback(*args, **kwargs)
        else:
            return _extract_data_from_args(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(transform_enabled={self._transform_enabled})"

    def attach_to(self, transforms: ft.Transform | Iterable[ft.Transform]) -> None:
        """Attach this override to one or more transformations."""
        if isinstance(transforms, ft.Transform):
            self._handles.append(transforms.register_override(self))
        else:
            for transform in transforms:  # type: ignore[union-attr]
                self.attach_to(transform)

    def detach(self) -> None:
        """Detach this override."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _TransformationContext(contextlib.AbstractContextManager[None]):
    def __init__(self, override: DisableTransformOverride, transform_enabled_reset_status: bool) -> None:
        self._override = override
        self._reset_transformation_status = transform_enabled_reset_status

    def __enter__(self) -> None: ...

    def __exit__(self, type, value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._override._transform_enabled = self._reset_transformation_status


def _extract_data_from_args(data: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
    return data
