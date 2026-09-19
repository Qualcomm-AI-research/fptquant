# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from hydra_zen import builds, store

from lm_tools.train.ce import CETraining
from lm_tools.train.evaluation_callback import EvaluationCallback
from lm_tools.train.kd import KDTraining

train_store = store(group="train", provider=__name__)
train_eval_callback_store = store(group="train/eval_callback", provider=__name__)

# CE
config = builds(
    CETraining,
    training_args={
        "push_to_hub": False,
        "output_dir": ".",
    },
    populate_full_signature=True,
)
train_store(config, name="ce")

eval_callback_config: Any = builds(
    EvaluationCallback,
    hydra_defaults=[
        {"/evaluation": "wikitext2_2048_ppl"},
    ],
    populate_full_signature=True,
)
train_eval_callback_store(eval_callback_config, name="evaluation")

# KD
kd_config = builds(
    KDTraining,
    training_args={
        "push_to_hub": False,
        "output_dir": ".",
    },
    populate_full_signature=True,
)
train_store(kd_config, name="kd")
