# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import TYPE_CHECKING, Optional

from ._base import BaseTraining, filter_training_args_for_init

if TYPE_CHECKING:
    from datasets import Dataset
    from torch import nn
    from transformers import Trainer


class CETraining(BaseTraining):
    def make_trainer(
        self, model: "nn.Module", train_dataset: "Dataset", eval_dataset: Optional["Dataset"]
    ) -> "Trainer":
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(**filter_training_args_for_init(self.training_args, TrainingArguments))

        # Default trainer does end-2-end training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=self.callbacks,
        )

        return trainer
