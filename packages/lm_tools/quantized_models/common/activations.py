# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as FF
from fastforward.nn import QuantizedModule, QuantizerStub
from transformers.activations import SiLUActivation


class QuantizedActivation(QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.output_quantizer = QuantizerStub(output_quantizer=True)


class QuantizedSiLUActivation(SiLUActivation, QuantizedActivation):
    def forward(self, x):
        x = self.input_quantizer(x)
        x = FF.nn.functional.silu(x, output_quantizer=self.output_quantizer)
        return x
