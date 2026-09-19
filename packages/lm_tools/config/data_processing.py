# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from lm_tools.data_processing.simple import SimpleDataProcessing

data_processing_store = store(group="data_processing", provider=__name__)

# hydra_zen registration
config = builds(
    SimpleDataProcessing,
    populate_full_signature=True,
)
data_processing_store(config, name="simple")
