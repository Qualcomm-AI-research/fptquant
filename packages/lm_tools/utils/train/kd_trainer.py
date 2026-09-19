# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator, Optional, Tuple, Union

import fastforward as ff
import torch
from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
from torch import Tensor, nn
from torch.distributions import Categorical, kl_divergence
from transformers import PreTrainedModel, Trainer
from transformers.utils import ModelOutput

from lm_tools.train.kd import LossType


@contextmanager
def control_producing_logits(
    model: nn.Module,
    no_logits: bool,
) -> Iterator[None]:
    """Return hidden states instead of logits while using Liger's fused linear JSD loss."""
    if not no_logits:
        yield
        return

    assert isinstance(model.lm_head, nn.Module), "model.lm_head must be an instance of nn.Module"
    lm_head_forward = model.lm_head.forward

    try:

        def _identity(self, x):
            return x

        model.lm_head.forward = MethodType(_identity, model.lm_head)
        yield
    finally:
        model.lm_head.forward = lm_head_forward


class KDTrainer(Trainer):
    def __init__(self, loss_type: LossType | str, **kwargs):
        """
        Trainer for Knowledge Distillation with student-teacher training
        :param loss_type: Divergence used to match the student and teacher logits (options: kl, reverse_kl, jsd).
        :param kwargs:
        """
        super().__init__(**kwargs)
        loss_type_value = loss_type.value if isinstance(loss_type, LossType) else loss_type
        assert loss_type_value in [member.value for member in LossType], f"Unknown {loss_type=}"
        self.loss_type = loss_type_value

    def _compute_loss_func(self, student_outputs: ModelOutput, teacher_outputs: ModelOutput) -> Tensor:
        """
        Computes the loss function for Knowledge Distillation given the student and teacher outputs
        :param student_outputs: Model outputs of the student.
        :param teacher_outputs: Model outputs of the teacher.
        :return:
        """
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        if student_logits is None or teacher_logits is None:
            raise ValueError("KD training requires student and teacher logits, but the model did not return logits.")

        student_dist = Categorical(logits=student_logits)
        teacher_dist = Categorical(logits=teacher_logits.detach())

        if self.loss_type == LossType.jsd.value:
            return (kl_divergence(student_dist, teacher_dist) + kl_divergence(teacher_dist, student_dist)) / 2.0
        elif self.loss_type == LossType.kl.value:
            return kl_divergence(teacher_dist, student_dist)
        elif self.loss_type == LossType.reverse_kl.value:
            return kl_divergence(student_dist, teacher_dist)
        else:
            raise RuntimeError(f"Unknown {self.loss_type=}")

    def _compute_liger_jsd_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Any,
        return_outputs: bool,
    ) -> Union[Tuple[Tensor, Any], Tensor]:
        if self.loss_type != LossType.jsd.value:
            raise RuntimeError(f"Liger fused KD loss only supports {LossType.jsd.value}, got {self.loss_type}.")

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        with control_producing_logits(model=model, no_logits=True):
            student_outputs = model(**model_inputs)

        with torch.no_grad(), ff.disable_quantization(model), control_producing_logits(model=model, no_logits=True):
            teacher_outputs = model(**model_inputs)

        assert isinstance(model.lm_head, nn.Module), "model.lm_head must be an instance of nn.Module"
        student_hidden_states = student_outputs.logits.to(model.lm_head.weight.dtype)
        teacher_hidden_states = teacher_outputs.logits.to(model.lm_head.weight.dtype)
        loss = LigerFusedLinearJSDFunction.apply(
            student_hidden_states.flatten(0, 1),
            model.lm_head.weight,
            teacher_hidden_states.flatten(0, 1),
            model.lm_head.weight,
            inputs["labels"].flatten(0, 1),
            0.5,
        )
        return (loss, student_outputs) if return_outputs else loss

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[Tuple[Tensor, Any], Tensor]:
        """
        Computes the loss for Knowledge Distillation on a given input batch.
        :param model: The model used for the forward pass
        :param inputs: The batch of inputs
        :param return_outputs: Flag indicating whether to return the output of the model
        :param num_items_in_batch: number of items in the batch
        :return: Loss for the given inputs
        """
        if self.args.use_liger_kernel and self.loss_type == LossType.jsd.value:
            return self._compute_liger_jsd_loss(model=model, inputs=inputs, return_outputs=return_outputs)

        # Liger's fused linear CE path skips materializing logits when labels are
        # passed in training mode. KD computes its own distribution loss, so the
        # model forwards must not receive labels.
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        student_outputs = model(**model_inputs)
        with torch.no_grad(), ff.disable_quantization(model):
            teacher_outputs = model(**model_inputs)

        loss = self._compute_loss_func(student_outputs=student_outputs, teacher_outputs=teacher_outputs)

        # this follows the behaviour of other loss functions
        # E.g. see transformers.loss.loss_utils.ForCausalLMLoss
        # remove padded input from grad computation
        # Create a mask for the ignored indices
        ignore_index = -100
        padding_mask = inputs["labels"].eq(ignore_index)
        loss.masked_fill_(padding_mask, 0.0)

        # Reduction
        if num_items_in_batch is None:
            num_active_elements = padding_mask.numel() - padding_mask.long().sum()
            # Compute the number of active elements

            # Sum the losses and divide by the number of active elements
            loss = loss.sum() / num_active_elements
        else:
            loss = loss.sum() / num_items_in_batch
        return (loss, student_outputs) if return_outputs else loss
