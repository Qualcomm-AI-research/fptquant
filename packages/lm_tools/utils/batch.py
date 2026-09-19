# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import TypeAlias

import torch

Batch: TypeAlias = torch.Tensor | dict[str, torch.Tensor] | list[torch.Tensor]


def determine_batch_size(batch: Batch) -> int:
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 1:
            return 1
        elif batch.ndim == 2:
            return batch.shape[0]
        else:
            raise ValueError(f"Can't determine the batch size for a batch with shape {batch.shape}")
    elif isinstance(batch, dict):
        return determine_batch_size(next(iter(batch.values())))
    elif isinstance(batch, list):
        return determine_batch_size(batch[0])
    else:
        raise TypeError(f"Unsupported type {type(batch)}")


def determine_sequence_length(batch: Batch) -> int:
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 1:
            return batch.shape[0]
        elif batch.ndim == 2:
            return batch.shape[1]
        else:
            raise ValueError(f"Can't determine the sequence length for a batch with shape {batch.shape}")
    elif isinstance(batch, dict):
        return determine_sequence_length(next(iter(batch.values())))
    elif isinstance(batch, list):
        return determine_sequence_length(batch[0])
    else:
        raise TypeError(f"Unsupported type {type(batch)}")


def determine_shape(batch: Batch) -> torch.Size:
    if isinstance(batch, torch.Tensor):
        return batch.shape
    elif isinstance(batch, dict):
        return determine_shape(next(iter(batch.values())))
    elif isinstance(batch, list):
        return determine_shape(batch[0])
    else:
        raise TypeError(f"Unsupported type {type(batch)}")
