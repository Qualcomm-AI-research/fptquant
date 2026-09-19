# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Any, Callable, Mapping, Optional

import fastforward as ff
import torch
from torch.utils.data import DataLoader, Dataset

from .quantization_ready import get_quant_model
from .quantizer_initialization import QuantizationRule, initialize_quantizers
from .range_setting import estimate_quantizer_ranges


class PTQPipeline:
    def __init__(
        self,
        rules: Mapping[str, QuantizationRule],
        range_estimator: Optional[ff.range_setting.common.RangeEstimator],
        calibration_set: Callable[..., "Dataset"],
        max_num_batches: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
        data_collator: Optional[Callable] = None,
        forward_call: Optional[Callable] = None,
        lazy_ff: bool = False,  # Quantize only using default ff classes (nn.Linear, ...)
    ):
        self._rules = rules
        self._range_estimator = range_estimator
        self._calibration_set = calibration_set
        self._max_num_batches = max_num_batches
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle
        self._lazy_ff = lazy_ff

        if data_collator is None:
            from transformers import default_data_collator

            data_collator = default_data_collator

        self._data_collator = data_collator
        self._forward_call = forward_call

    def __call__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        logger: logging.Logger | logging.LoggerAdapter,
    ) -> ff.nn.QuantizedModule:

        # 1) Switch the model classes to their Quantization-ready counterpart
        model = get_quant_model(model, logger=logger, default_fallback=self._lazy_ff)

        # 2) Apply the specified Quantization rules
        model = initialize_quantizers(model, rules=self._rules, logger=logger)

        if self._range_estimator is None:
            return model

        # 3.1) Build a dataloader
        calibration_set = self._calibration_set(tokenizer=tokenizer)
        calibration_loader = DataLoader(
            calibration_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=self._shuffle,
            collate_fn=self._data_collator,
        )

        # 3.2) Calibrate the quantizers
        model = estimate_quantizer_ranges(
            model,
            calibration_loader=calibration_loader,
            range_estimator=self._range_estimator,
            forward_call=self._forward_call,
            logger=logger,
        )

        return model
