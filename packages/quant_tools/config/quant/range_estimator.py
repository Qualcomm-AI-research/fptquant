# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
from hydra_zen import builds, store

import fastforward_contrib as ffc

store = store(provider=__name__)

range_estimator_store = store(group="quant/range_estimator")
for name, _RangeEstimator in [
    ("smoothed_minmax", ff.range_setting.smoothed_minmax),
    ("running_minmax", ff.range_setting.running_minmax),
    ("lp_norm", ffc.range_setting.lp_norm),
]:
    range_estimator_store(
        builds(
            _RangeEstimator,
            populate_full_signature=True,
        ),
        name=name,
    )
