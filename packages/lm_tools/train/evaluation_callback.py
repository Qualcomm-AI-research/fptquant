# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from lm_tools.evaluation._base import Evaluation

logger = logging.getLogger(__name__)


class EvaluationCallback(TrainerCallback):
    def __init__(
        self,
        evaluation: Evaluation,
        output_dir: str = "eval_callback",
        metric_prefix: str = "eval_callback",
    ):
        self.evaluation = evaluation
        self.output_dir = output_dir
        self.metric_prefix = metric_prefix
        self.tokenizer = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        import fastforward as ff
        import torch.nn.utils.parametrize as P
        from accelerate import Accelerator

        accelerator = Accelerator()
        if not accelerator.is_main_process:
            return control

        model = kwargs.get("model")
        tokenizer = self.tokenizer or kwargs.get("processing_class")
        if model is None or tokenizer is None:
            logger.warning("Skipping checkpoint evaluation at step %s: missing model or tokenizer.", state.global_step)
            return control

        was_training = model.training
        step = int(state.global_step)
        logger.info("Running checkpoint evaluation at step %d", step)
        model.eval()
        with torch.no_grad(), ff.strict_quantization(False), P.cached():
            results = self.evaluation(model=model, tokenizer=tokenizer)
        if was_training:
            model.train()

        results = results.copy()
        results["step"] = step
        callback_output_dir = Path(args.output_dir) / self.output_dir
        callback_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = callback_output_dir / f"step-{step}.csv"
        results.to_csv(output_path, index=False)
        logger.info("Checkpoint evaluation results saved to %s", output_path)

        log_entry: dict[str, Any] = {"step": step}
        for row in results.to_dict(orient="records"):
            quantity = row.get("quantity")
            value = row.get("value")
            if quantity is None or value is None:
                continue
            evaluation = row.get("evaluation")
            metric_name = f"{self.metric_prefix}/{quantity}"
            if evaluation is not None:
                metric_name = f"{self.metric_prefix}/{evaluation}/{quantity}"
            log_entry[metric_name] = value
        if len(log_entry) > 1:
            state.log_history.append(log_entry)

        return control
