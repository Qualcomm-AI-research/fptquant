# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import fastforward as ff
import torch


def count_quantizers(model: torch.nn.Module, pattern: str = "**") -> dict[str, int]:
    """Return a simple count summary for quantizers matched by ``pattern``."""
    return dict(Counter(result.full_name for result in ff.find_quantizers(model, pattern)))


def assert_quantizers_initialized(
    model: torch.nn.Module,
    *,
    pattern: str = "**/weight_quantizer",
    expected_type: type[torch.nn.Module] | None = None,
) -> None:
    """Assert that at least one quantizer matches and all matches have the expected type."""
    matches = list(ff.find_quantizers(model, pattern))
    summary = count_quantizers(model, pattern="**")
    assert matches, f"No quantizers matched pattern {pattern!r}. Quantizer summary: {summary}"

    if expected_type is None:
        return

    wrong_type = [
        f"{result.full_name} ({type(result.module).__name__})"
        for result in matches
        if not isinstance(result.module, expected_type)
    ]
    assert not wrong_type, (
        f"Expected all quantizers matching {pattern!r} to be {expected_type.__name__}; got: {wrong_type}"
    )


def assert_ptq_contract(
    qmodel: torch.nn.Module,
    calib_data: Any,
    *,
    fp_model: torch.nn.Module | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> None:
    """Assert the basic PTQ wiring contract for an inference-ready quantized model."""
    wrap_candidates = []
    missing_quantizers = []
    for name, module in qmodel.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            wrap_candidates.append(name or "<root>")
            if not hasattr(module, "weight_quantizer"):
                missing_quantizers.append(name or "<root>")

    assert not missing_quantizers, (
        "Expected every Linear/Conv2d in the quantized model to expose a weight quantizer. "
        f"Missing: {missing_quantizers}"
    )
    assert wrap_candidates, "Expected the quantized model to contain at least one Linear or Conv2d module."

    trainable_base_parameters = [
        name for name, parameter in qmodel.named_parameters() if parameter.requires_grad and "quantizer" not in name
    ]
    assert not trainable_base_parameters, (
        f"Expected PTQ base parameters to be frozen for inference. Trainable parameters: {trainable_base_parameters}"
    )

    if fp_model is None:
        return

    qmodel.eval()
    fp_model.eval()

    with torch.no_grad(), ff.strict_quantization(False):
        quant_output = _extract_model_tensor_output(qmodel, calib_data)
        fp_output = _extract_model_tensor_output(fp_model, calib_data)

    assert quant_output.shape == fp_output.shape, (
        "Expected quantized and floating-point outputs to have matching shapes. "
        f"Got {tuple(quant_output.shape)} vs {tuple(fp_output.shape)}"
    )

    diff = torch.linalg.vector_norm((quant_output - fp_output).float()).item()
    fp_norm = torch.linalg.vector_norm(fp_output.float()).item()
    threshold = rtol * fp_norm + atol
    assert diff > threshold, (
        "Expected PTQ output to differ from floating-point output by more than the tolerance. "
        f"Observed diff={diff:.6g}, threshold={threshold:.6g}"
    )


def _extract_model_tensor_output(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    if isinstance(batch, Mapping):
        output = model(**batch)
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes, bytearray)):
        output = model(*batch)
    else:
        output = model(batch)

    tensor = _first_tensor(output)
    if tensor is None:
        raise AssertionError(f"Could not find a tensor output in {type(output).__name__}")
    return tensor.detach()


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "logits") and isinstance(value.logits, torch.Tensor):
        return value.logits
    if isinstance(value, Mapping):
        if "logits" in value and isinstance(value["logits"], torch.Tensor):
            return value["logits"]
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None
