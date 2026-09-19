# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import collections
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import torch
import torch.nn.utils.parametrize as P
from fastforward import forward_override as override
from fastforward import mpath
from fastforward.forward_override import OverrideHandle, apply_overrides

from .transforms.common import Transform

logger = logging.getLogger("fasttransforms")

HOOKS_CONTAINER = "hooks"
TRANSFORM_OP_CONTAINER = "_attached_ops"


class TransformOperation(torch.nn.Module):
    def __init__(
        self,
        transform: Transform,
        dim: int,
        inv: bool = False,
        inference: Optional[bool] = None,
        **call_kwargs: Dict[str, Any],
    ):
        """
        Initialize a transform operation.

        Args:
            transform: Transform associated with this operation.
            dim: Dimension along which the transform is applied.
            inv: If ``True``, apply the inverse transform. Otherwise, apply
                the forward transform.
            inference:
                Controls when the operation is active:

                - ``None``: Always active.
                - ``True``: Disabled during parameter-setting and enabled during
                inference.
                - ``False``: Active only during parameter-setting and disabled
                during inference.
            **call_kwargs: Additional keyword arguments forwarded to the
                transform call.
        """

        super().__init__()

        self._transform = transform

        self._dim = dim
        self._inv = inv
        self._call_kwargs = call_kwargs
        self._transform_op_overrides: collections.OrderedDict = collections.OrderedDict()
        self._attached_to: Optional[torch.nn.Module] = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._pre_hook: Optional[bool] = None
        self._parametrized_tensor: Optional[str] = None
        self._enabled = inference is None or inference
        self._inference = inference

    @property
    def transform(self) -> Transform:
        return self._transform

    def register_override(self, override_fn: override.OverrideFn[torch.Tensor]) -> OverrideHandle:
        handle = OverrideHandle(self)  # noqa
        self._transform_op_overrides[handle.handle_id] = override_fn
        return handle

    def remove_override(self, override_id: int) -> override.OverrideFn[torch.Tensor] | None:
        return self._transform_op_overrides.pop(override_id, None)

    @property
    def overrides(self) -> Iterator[override.OverrideFn[torch.Tensor]]:
        """Yield overrides attached to this transform."""
        yield from self._transform_op_overrides.values()

    def forward(self, data: torch.Tensor):
        if not self.is_enabled:
            return data
        wrapped_transform = apply_overrides(self, self.transform, self._transform_op_overrides)
        return wrapped_transform(data, dim=self._dim, inv=self._inv, **self._call_kwargs)

    @property
    def is_attached(self) -> bool:
        return self._attached_to is not None

    @property
    def is_parametrization(self) -> bool:
        return self._parametrized_tensor is not None

    @property
    def is_hook(self) -> bool:
        return self._hook is not None

    @property
    def attached_to(self) -> Optional[Any]:
        return self._attached_to

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def parametrized_tensor(self) -> Optional[str]:
        return self._parametrized_tensor

    @property
    def inv(self) -> bool:
        return self._inv

    def _pre_hook_func(self, module: torch.nn.Module, inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        data = inputs[0]
        transformed_data = self.forward(data)
        if transformed_data is not None:
            data = transformed_data
        return (data,) + inputs[1:]

    def _post_hook_func(self, module: Callable, inputs: Tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        data = output
        transformed_data = self.forward(data)
        if transformed_data is not None:
            data = transformed_data
        return data

    def attach_hook(self, module: torch.nn.Module, pre_hook: bool = True):
        if self.is_attached:
            raise RuntimeError(f"{self.__class__.__name__} is already attached.")

        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"TransformHook can only be applied to torch.nn.Module. {module.__class__.__name__} was found."
            )

        if not hasattr(module, HOOKS_CONTAINER):
            module.add_module(HOOKS_CONTAINER, TransformHooksContainer())

        hook_collection = module.get_submodule(HOOKS_CONTAINER)
        assert isinstance(hook_collection, TransformHooksContainer)
        hook_collection.append(self)

        if pre_hook:
            self._hook = module.register_forward_pre_hook(self._pre_hook_func)
        else:
            self._hook = module.register_forward_hook(self._post_hook_func)

        self._pre_hook = pre_hook
        # Keep a reference without inheriting the parameters
        object.__setattr__(self, "_attached_to", module)

    def parametrize(self, module: torch.nn.Module, tensor_name: str):
        if self.is_attached:
            raise RuntimeError(f"{self.__class__.__name__} is already attached.")

        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                f"TransformHook can only be applied to torch.nn.Module. {module.__class__.__name__} was found."
            )

        P.register_parametrization(module, tensor_name, self)

        self._parametrized_tensor = tensor_name
        # Keep a reference without inheriting the parameters
        object.__setattr__(self, "_attached_to", module)

    def _detach_hook(self):
        if self._hook is None:
            raise ValueError("The TransformOperation is not hooked.")

        hook_collection = self.attached_to.get_submodule(HOOKS_CONTAINER)
        hook_collection.delete(self)

        if len(hook_collection) == 0:
            delattr(self.attached_to, HOOKS_CONTAINER)

        self._hook.remove()
        self._hook = None

    def _remove_parametrization(self):
        if self._parametrized_tensor is None:
            raise ValueError("The TransformOperation is not hooked.")

        remove_parametrization(self.attached_to, self._parametrized_tensor, parametrization=self)
        self._parametrized_tensor = None

    def remove(self):
        """
        Removes the transform operation from the model, let it be a hook or a parametrization
        """
        if not self.is_attached:
            raise RuntimeError(f"{self.__class__.__name__} is not attached.")

        if self.is_hook:
            self._detach_hook()
        if self.is_parametrization:
            self._remove_parametrization()

        self._attached_to = None

    def __del__(self):
        if self.is_attached:
            self.remove()

    def __repr__(self) -> str:
        sep = ", "
        short_name = self._transform.name.split(".")[0] if self._transform.name else ""
        return (
            f"{self.__class__.__name__}"
            + "("
            + (f'transform="{short_name}"' + sep if self._transform.name is not None else "")
            + f"dim={self._dim}"
            + sep
            + f"inv={self._inv}"
            + (
                sep + sep.join([f"{name}={str(value)}" for name, value in self._call_kwargs.items()])
                if len(self._call_kwargs) > 0
                else ""
            )
            + ")"
        )


class TransformHooksContainer(torch.nn.ModuleList):
    def append(self, item: torch.nn.Module):
        if not isinstance(item, TransformOperation):
            raise ValueError(f"{item.__class__.__name__} is not an instance of {TransformOperation.__name__}.")
        super().append(item)

    def delete(self, item):
        index_to_delete = None
        for i, transform_op in enumerate(self):
            if transform_op is item:
                index_to_delete = i

        if index_to_delete is None:
            raise ValueError(f"{TransformHooksContainer.__name__} has no item {item}")

        del self[index_to_delete]

    def detach(self):
        for transform_op in self:
            transform_op.detach()


def remove_parametrization(module: torch.nn.Module, tensor_name: str, parametrization: torch.nn.Module):
    parametrizations = getattr(module, "parametrizations")[tensor_name]
    P.remove_parametrizations(module, tensor_name, leave_parametrized=False)

    if len(parametrizations) == 1 and parametrizations[0] == parametrization:
        return
    else:
        # Add all the other parametrizations back
        for p in parametrizations:
            if p != parametrization:
                P.register_parametrization(module, tensor_name, p)


def get_transform_operations(model: torch.nn.Module) -> Iterator[TransformOperation]:
    transforms_ops = []
    modules = [module for _, module in model.named_modules()]

    for module in modules:
        if isinstance(module, TransformOperation) and module not in transforms_ops:
            transforms_ops.append(module.transform)
            yield module


def find_transform_operations(root_module: torch.nn.Module, query: str, **kwargs):
    transforms_ops = []
    modules = mpath.search(query, root_module, **kwargs)
    for res in modules:
        module = res.module
        if isinstance(module, TransformOperation) and module not in transforms_ops:
            transforms_ops.append(module)

    return torch.nn.ModuleList(transforms_ops)


def fuse_transforms(root_module: torch.nn.Module) -> List[TransformOperation]:
    # Remove all parametrizations by fusing them into the model parameters
    fused_ops = []

    for transform_op in get_transform_operations(root_module):
        if transform_op.is_parametrization:
            module = transform_op.attached_to
            assert isinstance(module, torch.nn.Module)

            tensor_name = transform_op._parametrized_tensor
            assert isinstance(tensor_name, str)

            # Determine the parametrizations attached to the current module
            parametrizations = getattr(module, "parametrizations")[tensor_name]

            # Remove the parametrizations
            P.remove_parametrizations(module, tensor_name, leave_parametrized=True)

            # Detach and collect all the parametrizations that we are removing
            for parametrization in parametrizations:
                if isinstance(parametrization, TransformOperation):
                    parametrization._parametrized_tensor = None
                    parametrization._attached_to = None
                    fused_ops.append(parametrization)

    return fused_ops
