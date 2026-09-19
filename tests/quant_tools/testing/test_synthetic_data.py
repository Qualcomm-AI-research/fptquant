# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch

from quant_tools.testing import SyntheticTensorDataset, synthetic_calibration_loader


def test_synthetic_tensor_dataset_wraps_values_as_x_tensors():
    dataset = SyntheticTensorDataset([0.0, [1.0, 2.0]])

    assert len(dataset) == 2
    assert torch.equal(dataset[0]["x"], torch.tensor(0.0))
    assert torch.equal(dataset[1]["x"], torch.tensor([1.0, 2.0]))


def test_synthetic_calibration_loader_batches_values():
    loader = synthetic_calibration_loader([[0.0], [1.0], [2.0]], batch_size=2, shuffle=False)
    batches = list(loader)

    assert [batch["x"].tolist() for batch in batches] == [[[0.0], [1.0]], [[2.0]]]
