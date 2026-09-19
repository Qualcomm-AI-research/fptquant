# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from typing import TYPE_CHECKING, Callable, Literal

import fastforward.quantization.granularity as granularities
import torch
import torch.nn.utils.parametrize as parametrize_utils
from fastforward import exceptions
from fastforward.common import ensure_tensor
from fastforward.nn.dynamic_linear_quantizer import (
    DynamicLinearQuantizer as _FFDynamicLinearQuantizer,
)
from fastforward.quantization import tiled_tensor
from fastforward.quantization.affine import (
    DynamicAffineQuantParams,
    DynamicParamInferenceFn,
    parameters_for_range,
)
from fastforward.quantization.affine import (
    quantization_context as static_quantization_context,
)
from fastforward.quantization.ste import round_ste
from typing_extensions import override

from .clip_parametrization import _CLAMP_MIN, CLIP_PARAMETRIZATION_REGISTRY, parametrization_module

if TYPE_CHECKING:
    from fastforward import QuantizedTensor


def dynamic_scale_offset_with_grad(params: DynamicAffineQuantParams, data: torch.Tensor):
    tile_size = params.granularity.tile_size(data.shape)
    row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
    try:
        min_range = torch.amin(row_representation, dim=1)
        max_range = torch.amax(row_representation, dim=1)
    except IndexError as e:
        msg = f"Cannot dynamically quantize an empty tensor of shape {data.shape}"
        raise exceptions.QuantizationError(msg) from e

    scale, offset = parameters_for_range(
        min_range,
        max_range,
        params.num_bits,
        symmetric=params.symmetric,
        allow_one_sided=params.allow_one_sided,
    )

    if offset is None:
        offset = torch.zeros_like(scale)
    else:
        offset = round_ste(offset)

    return scale, offset


class DynamicLinearQuantizer(_FFDynamicLinearQuantizer):
    """Dynamic Linear quantizer with optional grad-enabled min/max range.

    Thin extension of :class:`fastforward.nn.DynamicLinearQuantizer` that adds
    the ``enable_grad_minmax`` flag. Everything else (granularity handling,
    ``symmetric`` / ``allow_one_sided`` support, ``quantization_parameters``,
    ``quantization_function``) is inherited unchanged from the FastForward base.

    Args:
        num_bits: Bitwidth of quantization output granularity.
        granularity: Granularity object that specifies the quantization
            granularity.
        quantized_dtype: datatype in which the quantized data is stored.
        parameter_inference_fn: function that determines scale and offset
            dynamically. If omitted, a min/max inference method is used.
        allow_one_sided: If symmetric and allow_one_sided, the quantizer may
            fall back to a one-sided (unsigned) quantizer when all lower
            thresholds are above zero.
        symmetric: Use symmetric quantization if True, asymmetric otherwise.
        enable_grad_minmax: if True, let gradients flow through the dynamic
            min/max range estimation to the input data. When True, a custom
            parameter inference function (:func:`dynamic_scale_offset_with_grad`)
            is installed and ``parameter_inference_fn`` must be omitted.
    """

    def __init__(
        self,
        num_bits: int,
        *,
        granularity: granularities.Granularity | None = None,
        quantized_dtype: torch.dtype | None = None,
        parameter_inference_fn: DynamicParamInferenceFn | None = None,
        allow_one_sided: bool = True,
        symmetric: bool = False,
        enable_grad_minmax: bool = False,
    ):
        if enable_grad_minmax:
            if parameter_inference_fn is not None:
                raise ValueError(
                    "enable_grad_minmax=True installs its own parameter_inference_fn; "
                    "pass parameter_inference_fn only when enable_grad_minmax=False."
                )
            parameter_inference_fn = dynamic_scale_offset_with_grad
        super().__init__(
            num_bits=num_bits,
            granularity=granularity,
            quantized_dtype=quantized_dtype,
            parameter_inference_fn=parameter_inference_fn,
            allow_one_sided=allow_one_sided,
            symmetric=symmetric,
        )
        self.enable_grad_minmax = enable_grad_minmax


class DynamicLinearClipQuantizer(DynamicLinearQuantizer):
    """Dynamic Linear quantizer with learnable clipping.

    For simulating FlatQuant learnable activation quantizers,
    https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/quant_utils.py.

    The per-tile data range is shrunk (or, for unbounded parametrizations,
    rescaled) by learnable clip factors before quantization::

        min_eff = gamma_min * min(data)
        max_eff = gamma_max * max(data)

    where ``gamma_min`` / ``gamma_max`` are the *effective* clip factors. The
    transform from the raw learnable parameter to the effective factor is
    configurable via ``clip_parametrization`` (see
    :data:`fastforward_contrib.nn.clip_parametrization.CLIP_PARAMETRIZATION_REGISTRY`).
    The classic FlatQuant behaviour corresponds to
    ``clip_parametrization='sigmoid'`` (effective gamma in ``(0, 1)``).

    Two initialization modes are available, selected by ``clip_init``:

    1. ``clip_init=<float>`` (default): the clip factors are materialized on
       the first forward pass and seeded to ``clip_init`` in the *effective*
       (post-parametrization) space. No range setting is needed.
    2. ``clip_init=None``: range-settable mode. The clip factors are not seeded
       automatically; instead the standard ``LinearQuantizer`` range-setting
       flow (e.g. ``ff.estimate_ranges`` with a min/max estimator) determines a
       target ``(min, max)`` grid, and the clip factors are solved so that
       ``gamma * row_minmax(data) ≈ target`` on the next forward. This class
       implements :class:`fastforward.range_setting.SupportsRangeBasedOperator`,
       so MSE-style range setting also works.

    Args:
        num_bits: Bitwidth of quantization output granularity.
        granularity: Granularity object that specifies the quantization
            granularity.
        quantized_dtype: datatype in which the quantized data is stored.
        parameter_inference_fn: function that determines scale and offset
            dynamically. If omitted, a min/max inference method is used.
        allow_one_sided: If symmetric and allow_one_sided, the quantizer may
            fall back to a one-sided (unsigned) quantizer when all lower
            thresholds are above zero.
        symmetric: if True, use symmetric quantization.
        enable_grad_minmax: if True, let gradients flow through the dynamic
            min/max range estimation to the input data.
        param_dtype: dtype of the (raw) clip-factor parameters.
        clip_granularity: ``"tensor"`` materializes a single scalar clip factor
            shared across all tiles; ``"follow_quantizer"`` materializes one clip
            factor per tile.
        clip_parametrization: key into
            :data:`CLIP_PARAMETRIZATION_REGISTRY` selecting the transform from
            the raw parameter to the effective clip factor. ``'sigmoid'``
            (default) bounds gamma to ``(0, 1)`` (classic learnable clip);
            ``'identity'`` / ``'abs'`` / ``'softplus'`` / ``'exp'`` allow
            gamma > 1.
        clip_init: initial *effective* clip factor, or ``None`` to put the
            quantizer in range-settable mode. When a float, the default
            ``0.982`` equals ``sigmoid(4.0)``, matching FlatQuant's raw-4.0
            seed under the sigmoid parametrization.
        device: device for the clip parameters.
    """

    # Target quantization range stored by the `quantization_range` setter.
    # ``None`` means "no range has been set". The canonical source of the
    # effective range is the clip factors plus the live data min/max; these
    # buffers only (a) stash the target until the next forward solves gamma
    # and (b) let the getter report what was last set.
    _target_min_range: torch.Tensor | None
    _target_max_range: torch.Tensor | None

    def __init__(
        self,
        num_bits: int,
        *,
        granularity: granularities.Granularity | None = None,
        quantized_dtype: torch.dtype | None = None,
        parameter_inference_fn: DynamicParamInferenceFn | None = None,
        allow_one_sided: bool = False,
        symmetric: bool = False,
        enable_grad_minmax: bool = False,
        param_dtype: torch.dtype | None = None,
        clip_granularity: Literal["tensor", "follow_quantizer"] = "tensor",
        clip_parametrization: str = "sigmoid",
        clip_init: float | None = 0.982,
        device: torch.device | str = "cpu",
    ):
        super().__init__(
            num_bits,
            granularity=granularity,
            quantized_dtype=quantized_dtype,
            parameter_inference_fn=parameter_inference_fn,
            allow_one_sided=allow_one_sided,
            symmetric=symmetric,
            enable_grad_minmax=enable_grad_minmax,
        )
        if clip_parametrization not in CLIP_PARAMETRIZATION_REGISTRY:
            msg = (
                f"Unknown clip_parametrization '{clip_parametrization}'. "
                f"Choose from: {list(CLIP_PARAMETRIZATION_REGISTRY)}"
            )
            raise ValueError(msg)
        if clip_granularity not in ("tensor", "follow_quantizer"):
            msg = f"Unknown clip_granularity '{clip_granularity}'. Choose from: ['tensor', 'follow_quantizer']."
            raise ValueError(msg)
        self.clip_factor_min = torch.nn.UninitializedParameter(dtype=param_dtype, device=device, requires_grad=True)
        self.clip_factor_max = torch.nn.UninitializedParameter(dtype=param_dtype, device=device, requires_grad=True)
        self.clip_granularity = clip_granularity
        self.clip_parametrization = clip_parametrization
        self.clip_init = clip_init
        self._param_device: torch.device | str = device
        self._clip_parametrization_registered = False
        # Pending range-setting target; consumed on the next forward.
        self.register_buffer("_target_min_range", None)
        self.register_buffer("_target_max_range", None)
        self._needs_range_reinit: bool = False

    @override
    def reset_parameters(self) -> None:
        with torch.no_grad():
            for name in ("clip_factor_min", "clip_factor_max"):
                if parametrize_utils.is_parametrized(self, name):
                    parametrize_utils.remove_parametrizations(self, name, leave_parametrized=False)
            self.clip_factor_min = torch.nn.UninitializedParameter(
                device=self.clip_factor_min.device,  # type: ignore[call-arg]
                dtype=self.clip_factor_min.dtype,
            )
            self.clip_factor_max = torch.nn.UninitializedParameter(
                device=self.clip_factor_max.device,  # type: ignore[call-arg]
                dtype=self.clip_factor_max.dtype,
            )
        self._clip_parametrization_registered = False
        self._target_min_range = None
        self._target_max_range = None
        self._needs_range_reinit = False

    # ------------------------------------------------------------------
    # Parameter materialization
    # ------------------------------------------------------------------

    def _clip_factor_shape_for_data(self, data: torch.Tensor) -> tuple[int, ...]:
        """Shape of the clip-factor parameters for a given input tensor."""
        if self.clip_granularity == "tensor":
            return (1,)
        tile_size = self.granularity.tile_size(data.shape)
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
        return (len(row_representation),)

    def _clip_factor_shape_from_range(self, min_t: torch.Tensor) -> tuple[int, ...]:
        """Shape of the clip-factor parameters when initialized via the range setter."""
        if self.clip_granularity == "tensor":
            return (1,)
        # "follow_quantizer": one clip factor per tile. The estimator already
        # sized min/max according to the quantizer's granularity.
        return tuple(min_t.shape) if min_t.ndim > 0 else (1,)

    def _materialize_clip_factors(self, shape: tuple[int, ...], *, seed: float) -> None:
        """Materialize the clip-factor parameters and register parametrizations.

        Fills with ``seed`` in the *effective* (post-parametrization) space;
        ``register_parametrization`` then applies ``right_inverse(seed)`` to get
        the raw value so that the parametrization's forward yields ``seed`` again.
        """
        with torch.no_grad():
            self.clip_factor_max.materialize(shape)
            self.clip_factor_min.materialize(shape)
            self.clip_factor_max.fill_(seed)
            self.clip_factor_min.fill_(seed)

        if not self._clip_parametrization_registered:
            parametrize_utils.register_parametrization(
                self, "clip_factor_max", parametrization_module(self.clip_parametrization)
            )
            parametrize_utils.register_parametrization(
                self, "clip_factor_min", parametrization_module(self.clip_parametrization)
            )
            self._clip_parametrization_registered = True

    def _initialize_parameters(self, data: torch.Tensor) -> None:  # pylint: disable=arguments-differ
        """Materialize the uninitialized clip-factor parameters of the quantizer.

        Sources, highest to lowest priority:

        1. A pending range-setting target (``_needs_range_reinit``), in which
           case the clip factors are sized from the target and gamma is solved
           so that ``gamma * row_minmax(data) ≈ target``.
        2. ``clip_init`` (float): seed the effective clip factor to ``clip_init``.
        3. ``clip_init=None`` with no pending range: raise, mirroring
           ``LinearQuantizer``'s uninitialized-forward behaviour.
        """
        if not self.has_uninitialized_params:  # type: ignore[unused-ignore]
            return
        if self._needs_range_reinit:
            # Range-setter target present: size from it (handled in _lazy_init).
            shape = self._clip_factor_shape_from_range(self._target_min_range)  # type: ignore[arg-type]
            self._materialize_clip_factors(shape, seed=1.0)
        elif self.clip_init is not None:
            shape = self._clip_factor_shape_for_data(data)
            self._materialize_clip_factors(shape, seed=float(self.clip_init))
        else:
            msg = (
                "Tried to quantize a tensor using an uninitialized "
                f"{type(self).__name__}. This quantizer was constructed with "
                "clip_init=None; its clip factors are initialized through a "
                "range-setting method. Either set `quantization_range` directly "
                "or use `ff.estimate_ranges(...)` before calling forward."
            )
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # `RangeSettable` / `SupportsRangeBasedOperator` implementation
    # ------------------------------------------------------------------

    @property
    def quantization_range(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Getter: the ``(min, max)`` last passed to the setter, or ``(None, None)``.

        The dynamic clip quantizer does not keep a single canonical
        scale/offset pair; the effective range is recomputed per-forward from
        ``gamma * row_minmax(data)``. This getter mirrors
        :attr:`LinearQuantizer.quantization_range` in spirit.

        Part of :class:`fastforward.range_setting.RangeSettable`.
        """
        return self._target_min_range, self._target_max_range

    @quantization_range.setter
    def quantization_range(self, quant_range: tuple[torch.Tensor | float, torch.Tensor | float]) -> None:
        """Setter: stash the target ``(min, max)`` and defer gamma solving.

        The actual ``target → gamma`` conversion happens on the next forward,
        which is when the data batch (and thus the per-tile min/max) is
        available. In the standard :func:`fastforward.range_setting.estimate_ranges`
        flow the setter is called from inside the estimator's forward override
        and ``quantize(data)`` runs immediately afterwards on the same batch.

        Part of :class:`fastforward.range_setting.RangeSettable`.
        """
        try:
            min_t, max_t = (ensure_tensor(t, device=self._param_device) for t in quant_range)
        except (TypeError, ValueError) as e:
            msg = "Tried to set quantization range with a non 2-tuple. A 2-tuple is expected."
            raise ValueError(msg) from e

        if self.has_uninitialized_params:
            shape = self._clip_factor_shape_from_range(min_t)
            self._materialize_clip_factors(shape, seed=1.0)

        self._target_min_range = min_t.detach().clone()
        self._target_max_range = max_t.detach().clone()
        self._needs_range_reinit = True

    @property
    def symmetric(self) -> bool:  # type: ignore[override]
        """True if symmetric quantization, False otherwise.

        Part of :class:`fastforward.range_setting.SupportsRangeBasedOperator`.
        """
        return self._symmetric

    @symmetric.setter
    def symmetric(self, value: bool) -> None:
        # `DynamicLinearQuantizer.__init__` assigns ``self.symmetric``; back it
        # with a private attribute so we can expose a read property required by
        # the SupportsRangeBasedOperator protocol.
        self._symmetric = value

    def operator_for_range(
        self,
        min_range: torch.Tensor,
        max_range: torch.Tensor,
        data_shape: torch.Size,
    ) -> Callable[[torch.Tensor], "QuantizedTensor"]:
        """Return a *static* affine quantization operator for the given range.

        Identical in spirit to :meth:`LinearQuantizer.operator_for_range`: it
        builds a data-independent ``(scale, offset)`` from
        ``(min_range, max_range)``. Used by MSE-style range estimators that ask
        "what would the error be if we statically used this range?".

        Part of :class:`fastforward.range_setting.SupportsRangeBasedOperator`.
        """
        del data_shape
        scale, offset = parameters_for_range(
            min_range, max_range, self.num_bits, symmetric=self.symmetric, allow_one_sided=True
        )
        quant_context = static_quantization_context(
            scale=scale,
            offset=offset,
            num_bits=self.num_bits,
            granularity=self.granularity,
            output_dtype=self.quantized_dtype,
        )

        def _quant_operator(data: torch.Tensor) -> "QuantizedTensor":
            return quant_context.quantization_fn.quantize(data, quant_context.quantization_params)

        return _quant_operator

    # ------------------------------------------------------------------
    # Gamma computation from target ranges
    # ------------------------------------------------------------------

    def _raw_value_for_effective(self, effective: torch.Tensor) -> torch.Tensor:
        """Raw parameter value whose parametrized forward equals ``effective``."""
        module = parametrization_module(self.clip_parametrization)
        raw: torch.Tensor = module.right_inverse(effective)  # type: ignore[operator]
        return raw

    def _write_gamma_from_target(self, data: torch.Tensor) -> None:
        """Solve gamma so that ``gamma * row_minmax(data) ≈ target``; write raw params.

        ``gamma_max`` matches the positive side (``target_max`` vs ``max(rows)``)
        and ``gamma_min`` the negative side (``|target_min|`` vs ``|min(rows)|``).
        """
        assert self._target_min_range is not None and self._target_max_range is not None
        tile_size = self.granularity.tile_size(data.shape)
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        with torch.no_grad():
            rows = tiled_tensor.tiles_to_rows(data, tile_size)
            row_max = rows.max(1).values
            row_min_abs = (-rows.min(1).values).clamp(min=0)

            target_max = self._target_max_range.clamp(min=0)
            target_min_abs = (-self._target_min_range).clamp(min=0)

            gamma_p = target_max / row_max.clamp(min=_CLAMP_MIN)
            gamma_n = target_min_abs / row_min_abs.clamp(min=_CLAMP_MIN)

            if self.clip_granularity == "tensor":
                gamma_p = gamma_p.mean().reshape((1,))
                gamma_n = gamma_n.mean().reshape((1,))

            self.parametrizations.clip_factor_max.original.copy_(  # type: ignore[union-attr]
                self._raw_value_for_effective(gamma_p)
            )
            self.parametrizations.clip_factor_min.original.copy_(  # type: ignore[union-attr]
                self._raw_value_for_effective(gamma_n)
            )

    def _lazy_init(self, data: torch.Tensor) -> None:
        """Run one-shot materialization and any pending range-setter re-seeding."""
        if self.has_uninitialized_params:
            self._initialize_parameters(data)
        if self._needs_range_reinit:
            self._write_gamma_from_target(data)
            self._needs_range_reinit = False

    def clip(self, data: torch.Tensor) -> "QuantizedTensor":
        orig_dtype = data.dtype
        self._lazy_init(data)
        params = self.quantization_parameters()
        tile_size = params.granularity.tile_size(data.shape)
        tile_size = data.shape if tile_size == "data_shape" else tile_size
        row_representation = tiled_tensor.tiles_to_rows(data, tile_size)
        min_ = row_representation.min(1).values
        max_ = row_representation.max(1).values

        if not self.enable_grad_minmax:
            min_ = min_.detach()
            max_ = max_.detach()

        # Reading the (parametrized) clip factors returns the effective gamma.
        min_ = min_ * self.clip_factor_min
        max_ = max_ * self.clip_factor_max
        row_representation = torch.clamp(
            row_representation,
            min=min_[:, None].to(dtype=orig_dtype),
            max=max_[:, None].to(dtype=orig_dtype),
        )
        assert row_representation.dtype == orig_dtype
        return tiled_tensor.rows_to_tiles(row_representation, data.shape, tile_size)

    def quantize(self, data: torch.Tensor) -> "QuantizedTensor":
        data = self.clip(data)
        return super().quantize(data)
