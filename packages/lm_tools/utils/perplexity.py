# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import math
from typing import Any, Dict, List

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import tqdm
from torch.utils.data import DataLoader

from .batch import determine_batch_size, determine_sequence_length, determine_shape

logger = logging.getLogger(__name__)


def check_sequence_length(
    model_config: Dict,
    batch: Any,
):
    batch_shape = determine_shape(batch=batch)

    if len(batch_shape) != 2:
        raise ValueError(f"The dataloader is expected to return 2D tensor, got shape {batch_shape}")

    sequence_length = determine_sequence_length(batch)

    # Check if the model supports the sequence length
    if not hasattr(model_config, "max_position_embeddings"):
        raise ValueError("max_position_embeddings is missing in the model config")
    if sequence_length > model_config.max_position_embeddings:
        # Since max_position_embeddings can be set incorrectly (https://github.com/meta-llama/llama/issues/359),
        # we trust the user and show a warning only
        logger.warning(
            f"  The specified sequence_length ({sequence_length}) is larger than the maximum "
            f"length for the model ({model_config.max_position_embeddings}). "
        )


def extract_model_config(model: Any) -> Any:
    model_config = model.config  # after wrapping with accelerate, the config may become inaccessible
    # For VLM, the config is stored in model.config.text_config instead
    if hasattr(model_config, "text_config"):
        model_config = model_config.text_config
    return model_config


def evaluate_perplexity(
    model: Any,
    dataloader: DataLoader,
    max_num_batches=None,
) -> pd.DataFrame:
    model_config = extract_model_config(model)

    accelerator = Accelerator()
    dummy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # required by deepspeed zero2
    model, dataloader, dummy_optimizer = accelerator.prepare(model, dataloader, dummy_optimizer)

    losses: List[torch.Tensor] = []
    first_batch = True
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if first_batch:
                check_sequence_length(model_config=model_config, batch=batch)
                first_batch = False

            if max_num_batches is not None and batch_idx >= max_num_batches:
                break

            outputs = model(**batch)

            try:
                loss = outputs.loss
            except AttributeError:
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                    assert len(loss.shape) == 0  # we are looking for a scalar

            batch_size = determine_batch_size(batch)
            loss_ = accelerator.gather_for_metrics(loss.repeat(batch_size)).cpu()
            losses.append(loss_)

    try:
        eval_loss = torch.mean(torch.cat(losses)).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
        eval_loss = float("inf")

    results = pd.DataFrame(
        [
            {"quantity": "eval_loss", "value": eval_loss},
            {"quantity": "perplexity", "value": perplexity},
        ]
    )

    return results
