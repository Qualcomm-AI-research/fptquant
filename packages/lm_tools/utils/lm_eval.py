# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any, Dict, List, Union

import pandas as pd
from lm_eval import evaluator as lmeval_evaluator
from lm_eval.models.huggingface import HFLM


def run_lm_eval(
    tasks: List[str],
    batch_size: Union[int, str, None],
    model: Any,
    tokenizer: Any,
    **eval_kwargs,
):
    """Run lm-eval given a model, a tokenizer, and parameters"""

    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    results = lmeval_evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=list(tasks),
        batch_size=batch_size,
        **eval_kwargs,
    )

    # convert some attributes to make them serializable
    if not isinstance(results["config"]["model"], str):
        results["config"]["model"] = str(model.__class__)
    if not isinstance(results["config"].get("model_dtype", ""), str):
        results["config"]["model_dtype"] = str(results["config"]["model_dtype"])

    return results


def make_dataframe(out: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a dataframe from a dictionary of outputs
    :param out: the output of a lm_eval.simple_evaluate
    :return: a Dataframe containing each quantity, version, stats, filters, n_shots and values
    """
    pd_results = []
    for metric_name, stat_values in out["results"].items():
        version = out["versions"].get(metric_name, "N/A")
        n = str(out["n-shot"].get(metric_name, "N/A"))

        if "alias" in stat_values:
            metric_name = stat_values.pop("alias")

        for stat, value in stat_values.items():
            stat, _, f = stat.partition(",")

            if stat == "acc":
                stat = "mean"
            if stat == "acc_stderr":
                stat = "std"
            if stat == "acc_norm":
                stat = "norm"
            if stat == "acc_norm_stderr":
                stat = "norm_std"

            pd_results.append(
                {
                    "task": metric_name,
                    "stat": stat,
                    "value": value,
                    "version": version,
                    "filter": f,
                    "n_shots": n,
                }
            )
    return pd.DataFrame(pd_results)
