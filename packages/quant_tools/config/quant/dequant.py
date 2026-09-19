# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from quant_tools.convert import dequantize

dequant_store = store(group="dequant")
dequant_store(
    builds(
        dequantize,
        populate_full_signature=True,
    ),
    name="weight_only",
)
