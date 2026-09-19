# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from types import SimpleNamespace

import torch
from torch import nn
from transformers import TrainingArguments

from lm_tools.train.kd import LossType
from lm_tools.utils.train.kd_trainer import KDTrainer


class _LabelsSensitiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.saw_labels = False

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if labels is not None:
            self.saw_labels = True
            return SimpleNamespace(logits=None)
        logits = torch.nn.functional.one_hot(input_ids, num_classes=8).float() * self.scale
        return SimpleNamespace(logits=logits)


def test_kd_compute_loss_does_not_pass_labels_to_model(tmp_path) -> None:
    model = _LabelsSensitiveModel()
    trainer = KDTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True),
        loss_type=LossType.jsd.value,
    )

    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, -100, 3]]),
        },
    )

    assert not model.saw_labels
    assert torch.isfinite(loss)
