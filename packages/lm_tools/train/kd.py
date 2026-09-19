# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from omegaconf import DictConfig, DictKeyType

from ._base import BaseTraining, filter_training_args_for_init

if TYPE_CHECKING:
    from datasets import Dataset
    from torch import nn
    from transformers import Trainer, TrainerCallback


class LossType(Enum):
    jsd = "jsd"
    kl = "kl"
    reverse_kl = "reverse_kl"


class KDTraining(BaseTraining):
    """
    Knowledge Distillation Training Config with student-teacher training
    """

    def __init__(
        self,
        training_args: Dict[DictKeyType, Any] | DictConfig,
        train_dataset: Callable[[Any], "Dataset"],
        loss_type: LossType,
        eval_dataset: Optional[Callable[[Any], "Dataset"]] = None,
        skip_params: Optional[List[str]] = None,
        resume_from_checkpoint: Union[str, bool] = True,
        checkpoint_parameter_substrings: Optional[List[str]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        eval_callback: Optional["TrainerCallback"] = None,
    ):
        super().__init__(
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            skip_params=skip_params,
            resume_from_checkpoint=resume_from_checkpoint,
            checkpoint_parameter_substrings=checkpoint_parameter_substrings,
            callbacks=callbacks,
            eval_callback=eval_callback,
        )
        self.loss_type = loss_type

    def make_trainer(
        self, model: "nn.Module", train_dataset: "Dataset", eval_dataset: Optional["Dataset"]
    ) -> "Trainer":
        from transformers import TrainingArguments

        from lm_tools.utils.train.kd_trainer import KDTrainer

        training_args = TrainingArguments(**filter_training_args_for_init(self.training_args, TrainingArguments))

        # Default trainer does end-2-end training
        trainer = KDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss_type=self.loss_type,
            callbacks=self.callbacks,
        )

        return trainer
