# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

from fastforward.nn import QuantizedModule
from fastforward.quantization import strict_quantization
from transformers import Qwen3ForCausalLM, Qwen3Model

logger = logging.getLogger(__name__)


class QuantizedQwen3ForCausalLM(Qwen3ForCausalLM, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        # The following parts are not quantized:
        strict_quantization.strict_quantization_for_module(False, self.model.embed_tokens)
        strict_quantization.strict_quantization_for_module(False, self.lm_head)

    # Required for LM eval
    def tie_weights(self):
        pass


class QuantizedQwen3Model(Qwen3Model, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
