# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

from fastforward.nn import QuantizedModule
from fastforward.quantization import strict_quantization
from transformers import LlamaForCausalLM, LlamaModel

logger = logging.getLogger(__name__)


class QuantizedLlamaForCausalLM(LlamaForCausalLM, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        # The following parts are usually not quantized:
        strict_quantization.strict_quantization_for_module(False, self.model.embed_tokens)
        strict_quantization.strict_quantization_for_module(False, self.lm_head)

    # Required for LM eval
    def tie_weights(self):
        pass


class QuantizedLlamaModel(LlamaModel, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
