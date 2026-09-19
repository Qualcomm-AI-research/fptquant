# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from abc import abstractmethod
from typing import Any

import pandas as pd


class Evaluation:
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
    ) -> pd.DataFrame:
        pass

    def __call__(
        self,
        model: Any,
        tokenizer: Any,
    ) -> pd.DataFrame:
        """
        Abstract method for evaluating a model. Should be overridden in derived classes.
        :param model: The LLM to evaluate
        :param tokenizer: The tokenizer associated with the model
        :param accelerator: An instance of accelerator that can be used for evaluation
        :param logger: A logger instance
        :return: A DataFrame containing evaluation results
        """
        results = self.evaluate(model.eval(), tokenizer)
        self.logger.info(f"\n{results.to_markdown()}")

        return results
