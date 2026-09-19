# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Model B quantized components."""

from fastforward.nn import QuantizedModule

from tests.quant_tools.fixtures.mock_base_models.common import SharedDecoder
from tests.quant_tools.fixtures.mock_base_models.model_b import ModelB, ModelBMLP


class QuantizedModelB(QuantizedModule, ModelB):
    """Quantized Model B container."""

    pass


class QuantizedModelBMLP(QuantizedModule, ModelBMLP):
    """Quantized MLP for Model B."""

    def __init_quantization__(self):
        self.name = "quantized_model_b_mlp"


class QuantizedSharedDecoder(QuantizedModule, SharedDecoder):
    """Quantized shared decoder for Model B (different implementation than Model A)."""

    def __init_quantization__(self):
        self.name = "quantized_shared_decoder_model_b"
