# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from functools import partial
from itertools import chain
from typing import Any, Optional

from accelerate import Accelerator
from datasets import Dataset, IterableDataset

from lm_tools.data_processing.simple import PreprocessingType


def extract_sequences(
    dataset: Dataset,
    accelerator: Accelerator,
    sequence_length: int,
    batch_size: Optional[int],
    num_workers: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Dataset:
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Extracting sequences")

    # Main data processing function that will concatenate all texts from our dataset and generate
    # chunks of max_seq_length.

    def group_texts(examples, sequence_length):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of
        # this drop, you can customize this part to your needs.
        if total_length >= sequence_length:
            total_length = (total_length // sequence_length) * sequence_length
        else:
            total_length = 0
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts
    # throws away a remainder for each of those groups of 1,000 texts. You can adjust that
    # batch_size here but a higher value might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method
    # for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        dataset_kwargs = {
            "num_proc": num_workers if num_workers > 0 else 1,
            "desc": f"Grouping sequences in chunks of {sequence_length}",
        }
        if isinstance(dataset, IterableDataset):
            del dataset_kwargs["num_proc"]
            logger.info(dataset_kwargs["desc"])
            del dataset_kwargs["desc"]

        sequence_dataset = dataset.map(
            partial(group_texts, sequence_length=sequence_length),
            batched=True,
            batch_size=batch_size,
            **dataset_kwargs,
        )

    return sequence_dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: Any,
    accelerator: Accelerator,
    preprocessing_type: PreprocessingType,
    logger: Optional[logging.Logger] = None,
    text_column_name: str = "text",
    num_workers: int = 0,
    batch_size: Optional[int] = None,
) -> Dataset:
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Tokenizing the dataset")
    # Tokenize all the texts.
    if dataset.column_names[0] != text_column_name:
        dataset = dataset.rename_column(dataset.column_names[0], text_column_name)
    column_names = dataset.column_names

    # Define tokenization function
    def tokenize_function_line_by_line(examples, tokenizer, text_column_name):
        return tokenizer(examples[text_column_name])

    def tokenize_function_join(examples, tokenizer, text_column_name):
        return tokenizer(["".join(examples[text_column_name])])

    def tokenize_function_join_nn(examples, tokenizer, text_column_name):
        return tokenizer(["\n\n".join(examples[text_column_name])])

    if preprocessing_type == PreprocessingType.line_by_line:
        tokenize_function = tokenize_function_line_by_line
    elif preprocessing_type == PreprocessingType.join:
        tokenize_function = tokenize_function_join
    elif preprocessing_type == PreprocessingType.join_nn:
        tokenize_function = tokenize_function_join_nn
    else:
        raise ValueError(f"Unknown preprocessing type {preprocessing_type}")

    # make the default batch size for text pre-processing explicit
    with accelerator.main_process_first():
        dataset_kwargs = {
            "writer_batch_size": batch_size,
            "num_proc": num_workers if num_workers > 0 else 1,
            "desc": "Running tokenizer on datasets",
        }
        if isinstance(dataset, IterableDataset):
            del dataset_kwargs["writer_batch_size"]
            del dataset_kwargs["num_proc"]
            logger.info(dataset_kwargs["desc"])
            del dataset_kwargs["desc"]

        tokenized_dataset = dataset.map(
            function=partial(
                tokenize_function,
                tokenizer=tokenizer,
                text_column_name=text_column_name,
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=column_names,
            **dataset_kwargs,
        )

    return tokenized_dataset
