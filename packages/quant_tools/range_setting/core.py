# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import itertools
import logging
from typing import Callable, Optional

import fastforward as ff
import torch
from accelerate.utils import send_to_device  # type: ignore[import-untyped]
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]


def estimate_quantizer_ranges(
    model: ff.nn.QuantizedModule,
    calibration_loader: DataLoader,
    range_estimator: ff.range_setting.common.RangeEstimator,
    logger: logging.LoggerAdapter | logging.Logger,
    forward_call: Optional[Callable] = None,
    max_num_batches: Optional[int] = None,
) -> ff.nn.QuantizedModule:

    # When forward is not specified, use the model forward as default
    if forward_call is None:
        forward_call = model.forward

    model.eval()

    device = next(model.parameters()).device

    # A quantizer is range-settable iff it implements the ``RangeSettable`` protocol
    # (exposes ``quantization_range`` / ``granularity``). This includes clip quantizers
    # like ``DynamicLinearClipQuantizer(clip_init=None)`` whose clip factors are seeded
    # from an estimated range, while still excluding plain dynamic quantizers and stubs.
    # NB: do *not* blacklist ``DynamicLinearQuantizer`` here — its range-settable
    # subclasses (clip quantizers) must pass through.
    #
    # Exception: a clip quantizer with a non-``None`` ``clip_init`` seeds its clip factors
    # directly on the first forward and is *not* range-set (see ``DynamicLinearClipQuantizer``).
    # It still satisfies the ``RangeSettable`` protocol structurally (the attributes exist),
    # so it must be filtered out explicitly — otherwise range setting would run and overwrite
    # the intended seeded initialization.
    def _is_range_settable(module: object) -> bool:
        if not isinstance(module, ff.range_setting.RangeSettable):
            return False
        if isinstance(module, ff.nn.quantizer.QuantizerStub):
            return False
        # Clip quantizers in seed mode (clip_init is not None) initialize directly, not via
        # range setting. ``getattr`` keeps this robust for quantizers without ``clip_init``.
        if getattr(module, "clip_init", None) is not None:
            return False
        return True

    range_settable_quantizers = [res for res in ff.find_quantizers(model, "**") if _is_range_settable(res.module)]

    if len(range_settable_quantizers) == 0:
        logger.critical(
            "Skipping range setting: no settable quantizers found. "
            "This is expected only when all quantizers are dynamic."
        )
        return model

    weights_only = all(q.full_name.endswith("weight_quantizer") for q in range_settable_quantizers)
    if not max_num_batches:
        max_num_batches = len(calibration_loader)
    effective_batches = 1 if weights_only else max_num_batches

    logger.info(
        f"Range estimation: {len(range_settable_quantizers)} quantizers, "
        f"estimator={range_estimator}, "
        f"batches={effective_batches}" + (" (weights only — single batch sufficient)" if weights_only else "")
    )

    with (
        torch.no_grad(),
        ff.strict_quantization(False),
        ff.range_setting.estimate_ranges(
            [quantizer.module for quantizer in range_settable_quantizers], range_estimator
        ),
    ):
        for batch in tqdm(
            itertools.islice(calibration_loader, effective_batches),
            total=effective_batches,
            desc="Range estimation",
            leave=False,
        ):
            batch = send_to_device(batch, device)
            forward_call(**batch)

    logger.info("Range estimation complete.")
    return model
