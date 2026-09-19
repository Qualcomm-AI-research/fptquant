# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from lm_tools.data_processing.simple import PreprocessingType, SimpleDataProcessing
from lm_tools.tokenized_dataset.base import process_dataset

tokenized_dataset_store = store(group="tokenized_dataset", provider=__name__)

# Basic tokenized dataset, nothing specified
config = builds(
    process_dataset,
    hydra_defaults=[{"/raw_dataset": "???"}, {"/data_processing": "simple"}],
    populate_full_signature=True,
    zen_partial=True,
)
tokenized_dataset_store(config, name="base")

# mock_text
mock_data_processing_config = builds(
    SimpleDataProcessing,
    preprocessing_type=PreprocessingType.join_nn,
    sequence_length=16,
    take=2,
)
config = builds(
    process_dataset,
    data_processing=mock_data_processing_config,
    populate_full_signature=True,
    hydra_defaults=[{"/raw_dataset": "mock_text"}],
    zen_partial=True,
)
tokenized_dataset_store(config, name="mock_text")

# Wikitext
data_processing_config_2048 = builds(
    SimpleDataProcessing,
    preprocessing_type=PreprocessingType.join_nn,
    sequence_length=2048,
    populate_full_signature=True,
)
data_processing_config_4096 = builds(
    SimpleDataProcessing,
    preprocessing_type=PreprocessingType.join_nn,
    sequence_length=4096,
    populate_full_signature=True,
)
data_processing_config_8192 = builds(
    SimpleDataProcessing,
    preprocessing_type=PreprocessingType.join_nn,
    sequence_length=8192,
    populate_full_signature=True,
)

config = builds(
    process_dataset,
    data_processing=data_processing_config_2048,
    populate_full_signature=True,
    hydra_defaults=[{"/raw_dataset": "wikitext2"}],
    zen_partial=True,
)
tokenized_dataset_store(config, name="wikitext2_2048")

config = builds(
    process_dataset,
    data_processing=data_processing_config_2048,
    hydra_defaults=[
        {"/raw_dataset": "wikitext2_test"},
    ],
    populate_full_signature=True,
    zen_partial=True,
)
tokenized_dataset_store(config, name="wikitext2_2048_test")

config = builds(
    process_dataset,
    data_processing=data_processing_config_4096,
    hydra_defaults=[
        {"/raw_dataset": "wikitext2_test"},
    ],
    populate_full_signature=True,
    zen_partial=True,
)
tokenized_dataset_store(config, name="wikitext2_4096_test")

config = builds(
    process_dataset,
    data_processing=data_processing_config_8192,
    hydra_defaults=[
        {"/raw_dataset": "wikitext2_test"},
    ],
    populate_full_signature=True,
    zen_partial=True,
)
tokenized_dataset_store(config, name="wikitext2_8192_test")

config = builds(
    process_dataset,
    data_processing=data_processing_config_2048,
    hydra_defaults=[
        {"/raw_dataset": "wikitext2_train"},
    ],
    populate_full_signature=True,
    zen_partial=True,
)
tokenized_dataset_store(config, name="wikitext2_2048_train")
