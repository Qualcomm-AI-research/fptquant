# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Optional, Tuple

import fastforward.nn.functional as FFF
import torch
from fastforward import set_strict_quantization
from fastforward.nn import QuantizedModule, QuantizerStub
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

logger = logging.getLogger(__name__)


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer, QuantizedModule):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        # activation quantizers
        self.input_quantizer = QuantizerStub(input_quantizer=True)
        self.attn_res_act_quantizer = QuantizerStub(output_quantizer=True)
        self.mlp_res_act_quantizer = QuantizerStub(output_quantizer=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor`, *optional*):
                Kept for backward compatibility; not forwarded to attention.
            past_key_values (`Cache`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`.
                Passed through to the attention module.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that inject code into the model.
        """
        # Transform parametrizations are reused multiple times within a layer.
        # Cache them for old transformations-pipeline parity and to avoid
        # recomputing parametrized weights during a single forward pass.
        with torch.nn.utils.parametrize.cached():
            # quantize input
            hidden_states = self.input_quantizer(hidden_states)

            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

            # NOTE: embeddings are not quantized, thus residual is not quantized here:
            with set_strict_quantization(False):
                hidden_states = FFF.add(residual, hidden_states, output_quantizer=self.attn_res_act_quantizer)

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            hidden_states = FFF.add(residual, hidden_states, output_quantizer=self.mlp_res_act_quantizer)
        return hidden_states
