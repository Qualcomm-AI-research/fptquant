# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from abc import abstractmethod
from enum import Enum
from enum import auto as enum_auto
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from datasets import Dataset


class PreprocessingType(Enum):
    line_by_line = enum_auto()
    join = enum_auto()
    join_nn = enum_auto()


class DataProcessing:
    @abstractmethod
    def __call__(self, dataset: "Dataset", tokenizer: Any) -> "Dataset":
        pass


class SimpleDataProcessing(DataProcessing):
    def __init__(
        self,
        sequence_length: int,
        preprocessing_type: Union[PreprocessingType, str],
        text_column_name: str = "text",
        batch_size: Optional[int] = 0,  # provide the full dataset as a single batch to `function`
        num_workers: int = 0,
        shuffle: bool = False,
        take: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ):
        self.sequence_length = sequence_length
        self.preprocessing_type = (
            preprocessing_type
            if isinstance(preprocessing_type, PreprocessingType)
            else PreprocessingType[preprocessing_type]
        )
        self.text_column_name = text_column_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        if return_tensors not in {None, "pt"}:
            raise ValueError(f"Value {return_tensors=} is not supported")
        self.shuffle = shuffle
        self.take = take
        self.return_tensors = return_tensors

    def __call__(self, dataset: "Dataset", tokenizer: Any) -> "Dataset":
        from accelerate import Accelerator

        from lm_tools.utils.data_processing import extract_sequences, tokenize_dataset

        accelerator = Accelerator()
        logger = logging.getLogger(__name__)

        # Tokenize it
        tokenized_dataset = tokenize_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            accelerator=accelerator,
            preprocessing_type=self.preprocessing_type,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            text_column_name=self.text_column_name,
            logger=logger,
        )

        # Extract sequences of the specified length
        sequence_dataset = extract_sequences(
            dataset=tokenized_dataset,
            accelerator=accelerator,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            logger=logger,
        )

        if self.return_tensors == "pt":
            sequence_dataset.set_format(type="torch", columns=sequence_dataset.column_names)

        if self.shuffle:
            sequence_dataset = sequence_dataset.shuffle()

        if self.take:
            sequence_dataset = sequence_dataset.select(range(self.take))

        return sequence_dataset
