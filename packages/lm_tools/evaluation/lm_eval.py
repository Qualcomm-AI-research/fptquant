# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import os
from typing import Any, List, Union

import pandas as pd
import yaml

from ._base import Evaluation


class LMEval(Evaluation):
    def __init__(
        self,
        tasks: List[str],
        batch_size: Union[int, str, None] = 1,
        save_all: bool = False,
        **eval_kwargs,  # see lm_eval.evaluator.simple_evaluate for possible arguments
    ):
        super().__init__()
        self.tasks = tasks
        self.batch_size = batch_size
        self.save_all = save_all
        self.eval_kwargs = eval_kwargs

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
    ) -> pd.DataFrame:
        """
        Wrapper for the LM-Evaluation Harness package
        :return: The results of LM-Evaluation Harness on the specified tasks
        """

        from lm_tools.utils.lm_eval import make_dataframe, run_lm_eval

        # Run LM-Evaluation-Harness
        lm_eval_output = run_lm_eval(
            tasks=self.tasks,
            batch_size=self.batch_size,
            model=model,
            tokenizer=tokenizer,
            **self.eval_kwargs,
        )

        # Store all the evaluation results
        if self.save_all:
            filepath = os.path.abspath("./all_lmeval.yaml")
            self.logger.info(f"Saving all lm_eval results to {filepath}")
            with open(filepath, "w") as f:
                yaml.dump(lm_eval_output, f, indent=4)

        # Convert the results to a dataframe to store only the relevant part
        pd_results = make_dataframe(lm_eval_output)
        return pd_results
