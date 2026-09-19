# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from fasttransforms.parameter_setting.smoothquant import SmoothQuantEstimator

from ...parameter_setting.weight_optimization import WeightLpEstimator

parameter_setting_store = store(group="quant/parameter_estimator", provider=__name__)
parameter_setting_store(
    builds(SmoothQuantEstimator, populate_full_signature=True),
    name="smoothquant",
)
parameter_setting_store(
    builds(WeightLpEstimator, populate_full_signature=True),
    name="weight_lp",
)
