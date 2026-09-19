# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pytest
import torch

from quant_tools.testing import ToyEvalTrackingModule, ToyLinear, ToyLinearStack


def test_toy_linear_runs_forward_with_expected_shape():
    model = ToyLinear(in_features=3, out_features=2)

    output = model(torch.randn(4, 3))

    assert output.shape == (4, 2)


def test_toy_linear_stack_builds_requested_number_of_linear_layers():
    model = ToyLinearStack(in_features=3, hidden=5, out_features=2, n_layers=3)

    output = model(torch.randn(4, 3))
    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]

    assert output.shape == (4, 2)
    assert len(linear_layers) == 3


def test_toy_linear_stack_rejects_empty_stacks():
    with pytest.raises(ValueError, match="n_layers"):
        ToyLinearStack(n_layers=0)


def test_toy_eval_tracking_module_records_eval_and_batches():
    model = ToyEvalTrackingModule()
    batch = torch.tensor([[1.0], [2.0]])

    model.eval()
    output = model(batch)

    assert model.eval_called
    assert len(model.forward_batches) == 1
    assert torch.equal(model.forward_batches[0], batch)
    assert torch.equal(output, batch + 1.0)
