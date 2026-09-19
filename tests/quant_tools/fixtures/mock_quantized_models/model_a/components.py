# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Model A quantized components."""

from fastforward.nn import QuantizedModule

from tests.quant_tools.fixtures.mock_base_models.common import SharedDecoder
from tests.quant_tools.fixtures.mock_base_models.model_a import ModelA, ModelAAttention, ModelAVariant


class QuantizedModelA(QuantizedModule, ModelA):
    """Quantized Model A container."""

    pass


class QuantizedModelAVariant(QuantizedModule, ModelAVariant):
    """Quantized Model A container."""

    pass


class QuantizedModelAAttention(QuantizedModule, ModelAAttention):
    """Quantized attention for Model A."""

    def __init_quantization__(self):
        self.name = "quantized_model_a_attention"


class QuantizedSharedDecoder(QuantizedModule, SharedDecoder):
    """Quantized shared decoder for Model A."""

    def __init_quantization__(self):
        self.name = "quantized_shared_decoder_model_a"
