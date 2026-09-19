# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Any, Dict

import pandas as pd
from hydra_zen import instantiate

from lm_tools.evaluation._base import Evaluation


class GroupEvaluation(Evaluation):
    def __init__(self, steps: Dict[str, Any]):
        super().__init__()
        self.steps: Dict[str, Evaluation] = {k: instantiate(v) for k, v in steps.items()}

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
    ) -> pd.DataFrame:
        """
        Evaluate a model with multiple evaluations sequentially
        :param model: The model to evaluate
        :param tokenizer: The tokenizer associated to the model
        :return: A DataFrame containing the concatenated evaluation results
        """
        logger = logging.getLogger(__name__)
        results = []

        # For each evaluation step
        for name, evaluation in self.steps.items():
            # Perform the corresponding evaluation
            logger.info(f"Evaluating {name}")
            result = evaluation(model=model, tokenizer=tokenizer)
            result["evaluation"] = name
            results.append(result)

        # Return the concatenated results
        return pd.concat(results)
