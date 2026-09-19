# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from datasets import load_dataset
from hydra_zen import builds, store
from omegaconf import II

from lm_tools.utils.mock_text_generation import create_mock_text_dataset

raw_dataset_store = store(group="raw_dataset", provider=__name__)

# Bookcorpus
config = builds(
    load_dataset,
    path="bookcorpus/bookcorpus",
    name=None,
    cache_dir=II("paths.cache.local"),
    split="validation",
    trust_remote_code=True,
)
raw_dataset_store(config, name="bookcorpus")

# PTB
config = builds(
    load_dataset,
    path="ptb-text-only/ptb_text_only",
    name=None,
    cache_dir=II("paths.cache.local"),
    split="validation",
    trust_remote_code=True,
)
raw_dataset_store(config, name="ptb")

# Wikitext
config = builds(
    load_dataset,
    path="Salesforce/wikitext",
    name="wikitext-2-raw-v1",
    cache_dir=II("paths.cache.local"),
    split="validation",
)
raw_dataset_store(config, name="wikitext2")

config = builds(
    load_dataset,
    path="Salesforce/wikitext",
    name="wikitext-2-raw-v1",
    cache_dir=II("paths.cache.local"),
    split="test",
)
raw_dataset_store(config, name="wikitext2_test")

config = builds(
    load_dataset,
    path="Salesforce/wikitext",
    name="wikitext-2-raw-v1",
    cache_dir=II("paths.cache.local"),
    split="train",
)
raw_dataset_store(config, name="wikitext2_train")

config = builds(
    load_dataset,
    path="Salesforce/wikitext",
    name="wikitext-103-raw-v1",
    cache_dir=II("paths.cache.local"),
    split="validation",
)
raw_dataset_store(config, name="wikitext103")

# Mock dataset for testing
config = builds(
    create_mock_text_dataset,
    n_sentences=32,
    max_seq_length=128,
)
raw_dataset_store(config, name="mock_text")
