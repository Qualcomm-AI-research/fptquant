# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward.nn.functional as FFF
import torch
from fastforward.nn import QuantizedModule, QuantizerStub
from fastforward.quantization.strict_quantization import strict_quantization
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class QuantizedLlamaRMSNorm(LlamaRMSNorm, QuantizedModule):
    """
    LlamaRMSNorm is equivalent to T5LayerNorm
    """

    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.output_quantizer = QuantizerStub(output_quantizer=True)
        self.weight_quantizer = QuantizerStub(weight_quantizer=True)

    def forward(self, hidden_states):
        # NOTE: this module is not strict-quantized!
        with strict_quantization(False):
            hidden_states = self.input_quantizer(hidden_states)
            input_dtype = hidden_states.dtype
            if hidden_states.dtype != torch.float64:
                hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = hidden_states.to(input_dtype)

            weight = self.weight_quantizer(self.weight)
            out = FFF.mul(weight, hidden_states, output_quantizer=self.output_quantizer)
        return out
