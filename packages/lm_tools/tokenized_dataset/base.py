# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

from lm_tools.data_processing.simple import DataProcessing


def process_dataset(
    raw_dataset: "Dataset",
    data_processing: DataProcessing,
    tokenizer: "PreTrainedTokenizerBase",
) -> "Dataset":
    processed_dataset = data_processing(
        dataset=raw_dataset,
        tokenizer=tokenizer,
    )

    return processed_dataset
