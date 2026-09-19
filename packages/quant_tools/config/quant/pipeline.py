# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from quant_tools.pipeline import PTQPipeline

quant_store = store(group="quant", provider=__name__)

quant_store(
    builds(
        PTQPipeline,
        range_estimator=None,
        rules={},
        hydra_defaults=[
            "_self_",
            {"range_estimator": "smoothed_minmax"},
        ],
        populate_full_signature=True,
    ),
    name="simple_ptq",
)
