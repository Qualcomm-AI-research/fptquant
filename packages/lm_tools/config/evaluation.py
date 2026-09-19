# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra_zen import builds, store

from lm_tools.evaluation.group_evaluation import GroupEvaluation
from lm_tools.evaluation.lm_eval import LMEval
from lm_tools.evaluation.perplexity import PerplexityEvaluation

evaluation_store = store(group="evaluation", provider=__name__)

####################
# Group Evaluation #
####################

config = builds(
    GroupEvaluation,
    populate_full_signature=True,
    hydra_recursive=False,
)
evaluation_store(config, name="group_evaluation")


#################
# LM Eval-based #
#################

config = builds(LMEval, num_fewshot=5, batch_size=1, limit=None, populate_full_signature=True)
evaluation_store(config, name="lm_eval")

config = builds(
    LMEval,
    tasks=["arc_easy"],
    num_fewshot=5,
    batch_size=1,
    limit=None,
)
evaluation_store(config, name="arc_easy")

config = builds(
    LMEval,
    tasks=[
        "boolq",
        "piqa",
        "winogrande",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
    ],
    num_fewshot=0,
    batch_size="auto",
    limit=None,
)
evaluation_store(config, name="common_reasoning")

config = builds(
    LMEval,
    tasks=[
        "lambada_openai",
        "piqa",
        "winogrande",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
    ],
    num_fewshot=0,
    batch_size="auto",
    limit=None,
)
evaluation_store(config, name="common_reasoning_flatquant")

config = builds(
    LMEval,
    tasks=["mmlu"],
    num_fewshot=5,
    batch_size="auto",
    limit=None,
)
evaluation_store(config, name="mmlu")

###############
# Perplexity  #
###############

config = builds(
    PerplexityEvaluation,
    populate_full_signature=True,
)
evaluation_store(config, name="perplexity")

config = builds(
    PerplexityEvaluation,
    max_num_batches=None,
    batch_size=1,
    num_workers=0,
    hydra_defaults=[
        {"/tokenized_dataset": "wikitext2_2048_test"},
    ],
)
evaluation_store(config, name="wikitext2_2048_ppl")

config = builds(
    PerplexityEvaluation,
    max_num_batches=None,
    batch_size=1,
    num_workers=0,
    hydra_defaults=[
        {"/tokenized_dataset": "wikitext2_2048"},
    ],
)
evaluation_store(config, name="wikitext2_2048_val")

# Debug variant: uses mock_text (seq_len=16), 1 batch — for fast tests
config = builds(
    PerplexityEvaluation,
    max_num_batches=1,
    batch_size=1,
    num_workers=0,
    hydra_defaults=[
        {"/tokenized_dataset": "mock_text"},
    ],
)
evaluation_store(config, name="perplexity_debug")


config = builds(
    PerplexityEvaluation,
    max_num_batches=None,
    batch_size=1,
    num_workers=0,
    hydra_defaults=[
        {"/tokenized_dataset": "wikitext2_4096_test"},
    ],
)
evaluation_store(config, name="wikitext2_4096_ppl")

config = builds(
    PerplexityEvaluation,
    max_num_batches=None,
    batch_size=1,
    num_workers=0,
    hydra_defaults=[
        {"/tokenized_dataset": "wikitext2_8192_test"},
    ],
)
evaluation_store(config, name="wikitext2_8192_ppl")
