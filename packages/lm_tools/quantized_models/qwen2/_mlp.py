# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

import fastforward.nn.functional as FFF
from fastforward.nn import QuantizedModule, QuantizerStub
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

logger = logging.getLogger(__name__)


class QuantizedQwen2MLP(Qwen2MLP, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.gated_up_proj_output_quantizer = QuantizerStub(output_quantizer=True)

    def forward(self, x):
        x = self.input_quantizer(x)
        gate = self.gate_proj(x)
        gate_act = self.act_fn(gate)
        gated_up_proj = FFF.mul(gate_act, self.up_proj(x), output_quantizer=self.gated_up_proj_output_quantizer)
        down_proj = self.down_proj(gated_up_proj)
        return down_proj
