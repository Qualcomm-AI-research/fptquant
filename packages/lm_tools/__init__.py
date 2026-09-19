# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from . import (
    data_processing,
    evaluation,
    llm,
    raw_dataset,
    tokenized_dataset,
    train,
)
from .config import add_config_to_store

__all__ = [
    "data_processing",
    "evaluation",
    "llm",
    "raw_dataset",
    "tokenized_dataset",
    "train",
    "add_config_to_store",
]

# Registration of Quantization-ready models
import fastforward

import fastforward_contrib
from lm_tools.quantized_models import common as common_pkg
from quant_tools import register_quantization_providers_from_package

from . import quantized_models as quant_pkg

# Each package in quant_pkg.<package_name> is imported when quantizing models
# with model_type <package_name>.
register_quantization_providers_from_package(
    package=quant_pkg,
    # Always add the quantized Models defined in fastforward and fastforward_contrib
    generic_modules=[
        fastforward,
        fastforward_contrib,
        common_pkg,
    ],
    # We exclude "common" since that is not a model and __pycache__
    skip=("common", "__pycache__"),
    default_priority=100,
)
