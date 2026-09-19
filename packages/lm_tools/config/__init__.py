# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

_CONFIG_ADDED = False


def add_config_to_store():
    global _CONFIG_ADDED
    if _CONFIG_ADDED:
        return

    from .data_processing import data_processing_store
    from .evaluation import evaluation_store
    from .llm import llm_store
    from .quant.transforms import transforms_store
    from .raw_dataset import raw_dataset_store
    from .tokenized_dataset import tokenized_dataset_store
    from .train import train_eval_callback_store, train_store

    data_processing_store.add_to_hydra_store()
    evaluation_store.add_to_hydra_store()
    llm_store.add_to_hydra_store()
    raw_dataset_store.add_to_hydra_store()
    tokenized_dataset_store.add_to_hydra_store()
    train_store.add_to_hydra_store()
    train_eval_callback_store.add_to_hydra_store()
    transforms_store.add_to_hydra_store()
    _CONFIG_ADDED = True
