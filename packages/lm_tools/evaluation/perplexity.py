# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import TYPE_CHECKING, Any, Callable, Optional

import pandas as pd
from torch.utils.data import DataLoader

from lm_tools.evaluation._base import Evaluation

if TYPE_CHECKING:
    from datasets import Dataset


class PerplexityEvaluation(Evaluation):
    def __init__(
        self,
        tokenized_dataset: Callable[..., "Dataset"],
        max_num_batches: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        super().__init__()
        self.max_num_batches = max_num_batches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenized_dataset = tokenized_dataset

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
    ) -> pd.DataFrame:
        from datasets import IterableDataset
        from transformers import default_data_collator

        from lm_tools.utils.perplexity import evaluate_perplexity

        # The tokenized_dataset is a partial function, we instantiate it passing the tokenizer
        self.logger.info("Processing the dataset")
        tokenized_dataset = self.tokenized_dataset(tokenizer=tokenizer)

        # If the evaluation dataset is iterable, check that the number of workers is compatible
        # with the number of shards in the dataset
        if isinstance(tokenized_dataset, IterableDataset):
            self.logger.warning(
                "Using num_workers>1 for IterableDataset may drop some of the data. "
                "If you want to ensure reproducibility, set evaluation.num_workers=0"
            )
            # For IterableDatasets, the number of workers shouldn't
            # exceed the number of shards
            if self.num_workers > tokenized_dataset.num_shards:
                self.logger.warning(
                    "The number of workers exceeds the number of shards. "
                    f"Only {tokenized_dataset.num_shards} workers will be used."
                )
                self.num_workers = tokenized_dataset.num_shards

        # Determine the sequence length
        sequence_length = len(tokenized_dataset[0]["input_ids"])

        # Create an instance of the dataloader for the tokenized dataset
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            num_workers=self.num_workers,
            shuffle=False,
        )

        # Run the perplexity evaluation
        results = evaluate_perplexity(
            model=model,
            dataloader=dataloader,
            max_num_batches=self.max_num_batches,
        )

        # Add a few columns of the dataset to the result table
        results["sequence_length"] = sequence_length
        results["stat"] = "mean"

        # Log the perplexity to stdout
        self.logger.info(f"perplexity: {results[results['quantity'] == 'perplexity'].value.values[0]:.4f}")

        return results
