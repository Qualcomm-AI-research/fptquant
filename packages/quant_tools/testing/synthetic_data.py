# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticTensorDataset(Dataset):
    """Small dict-backed tensor dataset for calibration and PTQ tests."""

    def __init__(self, values: Iterable[Any]):
        self._values = [torch.as_tensor(value, dtype=torch.float32) for value in values]

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._values[idx]}


def synthetic_calibration_loader(
    values: Iterable[Any],
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    collate_fn: Any = None,
) -> DataLoader:
    """Build a deterministic calibration loader from in-memory values."""
    return DataLoader(
        SyntheticTensorDataset(values),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
