# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Collection of types, protocols and base implementations for range setting methods."""

import abc
import contextlib
import logging
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
)

import torch
from typing_extensions import override

from fasttransforms.nn.transforms.common import Transform

_T = TypeVar("_T")
_Module = TypeVar("_Module", bound=torch.nn.Module)


class ParameterEstimator(abc.ABC, Generic[_T, _Module]):
    """Abstract base class for parameter estimator methods.

    Subclasses of this base class implement preparation and cleanup methods for
    parameter estimation that setup a module for range estimation before passing
    data through the model and clean up any range estimation related settings
    afterwards.

    Since many parameter estimation methods operate 'locally' on leaf modules for
    which separate prepare and cleanup steps are appropriate, `split_module`
    may take a particular module and yield child modules for which `prepare`
    and `cleanup` is called separately.
    """

    @abc.abstractmethod
    def prepare(self, module: _Module) -> _T:
        """Prepare module for a particular parameter estimation method.

        Any metadata that is returned from this method is fed back to cleanup
        after parameter estimation.

        Args:
            module: Module to prepare for parameter estimation

        Returns:
            None or any Metadata that may be used during cleanup.
        """
        ...

    @abc.abstractmethod
    def cleanup(self, module: _Module, metadata: _T) -> None:
        """Clean up any parameter estimation specific settings from module.

        After the conclusion of this method, it is assumed that module is in
        the same state as before `prepare` was called, except for some
        transform specific parameters that where estimated during the parameter
        estimation step.

        Args:
            module: The module to clean up
            metadata: Any metadata that was returned from the prepare method for `module`
        """
        ...

    @abc.abstractmethod
    def split_module(self, module: torch.nn.Module) -> Iterator[_Module]:
        """Split the module into one or more submodules.

        For each module yielded from this function, prepare and cleanup are
        called separately. If only a single prepare and cleanup should be
        performed from module, only yield module once.

        Yields:
            Submodules for module (or module itself) for which a separate prepare/cleanup
                step is performed.
        """
        ...


class WeightParameterEstimator(abc.ABC):
    """Abstract base class for transform parameter estimators based only on model weights."""

    @abc.abstractmethod
    def estimate(
        self,
        model: torch.nn.Module,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        """Estimate transform parameters without running data through the model."""
        ...


_TransformType = TypeVar("_TransformType", bound=Transform)


class SimpleEstimatorStep(abc.ABC, Generic[_TransformType]):
    """Base class for simple parameter estimator step.

    SimpleEstimatorStep provides a forward pass in which estimate_step is called. During
    the first execution of the forward pass, setup_estimator is called.

    Args:
        disable_transform: If True, during parameter estimation, the parameters are estimated
            based on a non-transformed forward pass, i.e., the transforms will not be applied.
            If False, the transforms will be applied normally.
    """

    def __init__(self, *args: Any, disable_transforms: bool = False, **kwargs: Any) -> None:
        self._initialized = False
        self._disable_transforms = disable_transforms
        super().__init__(*args, **kwargs)

    def setup_estimator(self, data: torch.Tensor) -> None:
        """Perform setup for the estimator.

        The first data batch is passed. this method is only called once during
        the first estimator step.

        Args:
            data: The first batch passed to the estimator.
        """

    @abc.abstractmethod
    def estimate_step(self, transform: _TransformType, data: torch.Tensor, dim: int, inv: bool, **kwargs) -> None:
        """Given transform and data, update the transform parameters of `transform` based on data.

        Args:
            transform: `Transform` module for which parameters should be updated
            data: Data that update should be based on
            dim: The dimension that the transform is applied on
            inv: if the transform is called with inverse or not
        """
        ...

    @override
    def forward(  # type: ignore[misc]
        self,
        transform: _TransformType,
        callback: Callable[[torch.Tensor], torch.Tensor],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        def _data_dim_inv(
            data: torch.Tensor, dim: int, inv: bool, *_args: Any, **_kwargs: Any
        ) -> Tuple[torch.Tensor, int, bool]:
            return data, dim, inv

        data, dim, inv = _data_dim_inv(*args, **kwargs)

        if not transform.initialized:
            transform.__init_parameters__(data.shape[dim], dtype=data.dtype, device=data.device)

        self.estimate_step(transform, data, dim=dim, inv=inv)

        if self._disable_transforms:
            return data
        return callback(data)


@contextlib.contextmanager
def estimate_parameters(
    model_or_layers: torch.nn.Module | Sequence[torch.nn.Module],
    estimator: ParameterEstimator[_T, _Module] | type[ParameterEstimator[_T, _Module]],
    *args: Any,
    **kwargs: Any,
) -> Generator[None, None, None]:
    """Context manager to setup `model_or_layers` for parameter estimation.

    Within the context, any data that is passed to `model_or_layers` will
    trigger a parameter estimation step. Each module will be setup by `estimator`
    and cleaned up at the conclusion of the context.

    If `estimator` is a `ParameterEstimator` type (in contrast to an instance), it is initialized
    with any extra arguments passed to this function. If it is an instance, it is used as is.

    Example:
        This function may be used as follows:

        ```python
        with estimate_parameters([first_module, second_module], SomeEstimatorClass):
            for batch in data_loader:
                first_module(data)
                second_module(data)

        # Here all transforms that executed in the previous context will have their
        # parameters initialized and are ready for further use.
        ```
    """
    if isinstance(model_or_layers, torch.nn.Module):
        model_or_layers = [model_or_layers]

    if isinstance(estimator, type):
        estimator = estimator(*args, **kwargs)
    elif args or kwargs:
        raise ValueError("`estimator` is already initialized so no `args` or `kwargs` can be given.")

    prepared_modules: list[tuple[_Module, _T]] = []

    for module in model_or_layers:
        for module_part in estimator.split_module(module):
            metadata = estimator.prepare(module_part)
            prepared_modules.append((module_part, metadata))
    try:
        yield
    finally:
        for module, metadata in prepared_modules:
            estimator.cleanup(module, metadata)
