# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import random
import string

from datasets import Dataset


def generate_random_characters(n: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=n - 1)) + "\n"


def create_mock_text_dataset(n_sentences: int, max_seq_length: int) -> Dataset:
    my_dict = {"text": [generate_random_characters(random.randint(0, max_seq_length)) for _ in range(n_sentences)]}
    return Dataset.from_dict(my_dict)
