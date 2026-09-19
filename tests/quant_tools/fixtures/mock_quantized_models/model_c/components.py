# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Model C quantized components."""

from fastforward.nn import QuantizedModule

from tests.quant_tools.fixtures.mock_base_models.model_c import ModelC, ModelCEncoder


class QuantizedModelC(QuantizedModule, ModelC):
    """Quantized Model C container."""

    pass


class QuantizedModelCEncoder(QuantizedModule, ModelCEncoder):
    """Quantized encoder for Model C."""

    def __init_quantization__(self):
        self.name = "quantized_model_c_encoder"
