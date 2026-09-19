# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Any, Callable, Dict, Mapping, Optional

import fastforward as ff
import torch
from torch.utils.data import DataLoader, Dataset

import fasttransforms as ft
from fasttransforms.parameter_setting import WeightParameterEstimator
from quant_tools.quantization_ready import get_quant_model
from quant_tools.quantizer_initialization import QuantizationRule, initialize_quantizers
from quant_tools.range_setting import estimate_quantizer_ranges


class TransformPTQPipeline:
    def __init__(
        self,
        rules: Mapping[str, QuantizationRule],
        calibration_set: Callable[..., "Dataset"],
        transforms: Dict[str, ft.TransformConfig],
        range_estimator: ff.range_setting.common.RangeEstimator,
        parameter_estimator: Optional[ft.parameter_setting.common.ParameterEstimator] = None,
        max_num_batches: int = 1,
        batch_size: int = 1,
        num_workers: int = 1,
        shuffle: bool = True,
        fuse_transforms: bool = False,
        data_collator: Optional[Callable] = None,
        forward_call: Optional[Callable] = None,
    ):
        if range_estimator is None:
            raise ValueError(
                "TransformPTQPipeline requires a `range_estimator`; got None. "
                "Configure a quant/range_estimator (e.g. 'lp_norm' or 'smoothed_minmax')."
            )
        self._rules = rules
        self._transforms = transforms
        self._range_estimator = range_estimator
        self._parameter_estimator = parameter_estimator
        self._calibration_set = calibration_set
        self._max_num_batches = max_num_batches
        self._fuse_transforms = fuse_transforms
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._shuffle = shuffle

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
    ) -> ff.nn.QuantizedModule | torch.nn.Module:
        # 1) Switch the model classes to their Quantization-ready counterpart
        model = get_quant_model(model, logger=logger)

        # 2) Apply the specified Quantization rules
        model = initialize_quantizers(model, rules=self._rules, logger=logger)

        # 3.1) Convert the LayerNorm into rmsnorm if R1 is present
        # Both the "R1" and "R1_train" config options register under
        # package="quant.transforms.R1", so they always land at the key "R1". Assert "R1_train" name never survives into the composed dict.
        if "R1_train" in self._transforms:
            raise ValueError("Unexpected 'R1_train' key in transforms")
        if "R1" in self._transforms:
            logger.info("Converting LayerNorm into RMSNorm")
            ft.utils.merge_rmsnorm(model, merge_head=True)

        # 3.2) Add the transforms
        logger.info("Adding transforms to the model...")
        transforms, transform_ops = ft.transforms_from_config_dict(
            self._transforms, model=model, aliases=model.aliases
        )

        logger.info(f"Instantiated {len(transforms)} transformations")
        logger.info(transforms)

        # Weight-only estimators operate directly on the attached transform parametrizations.
        if isinstance(self._parameter_estimator, WeightParameterEstimator):
            self._parameter_estimator.estimate(
                model=model,
                logger=logger,
            )

        needs_forward_parameter_estimation = self._parameter_estimator is not None and not isinstance(
            self._parameter_estimator, WeightParameterEstimator
        )

        # 4) Instantiate the calibration dataset used for parameter/range estimation
        calibration_set = self._calibration_set(tokenizer=tokenizer)
        calibration_loader = DataLoader(
            calibration_set,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=self._shuffle,
            collate_fn=self._data_collator,
        )

        # 5) If a parameter estimation is specified, use it to estimate the transform parameters
        if needs_forward_parameter_estimation:
            assert self._parameter_estimator is not None  # implied by needs_forward_parameter_estimation
            ft.parameter_setting.estimate_transform_parameters(
                model=model,
                calibration_loader=calibration_loader,
                parameter_estimator=self._parameter_estimator,
                logger=logger,
            )

        # 6) Fuse the transform parametrizations if specified
        if self._fuse_transforms:
            fused_ops = ft.fuse_transforms(model)
            logger.info(f"Fused {len(fused_ops)} transformation operations.")

        # 7) Calibrate the ranges of the weight quantizers
        estimate_quantizer_ranges(
            model,
            calibration_loader=calibration_loader,
            range_estimator=self._range_estimator,
            forward_call=self._forward_call,
            logger=logger,
        )

        return model
