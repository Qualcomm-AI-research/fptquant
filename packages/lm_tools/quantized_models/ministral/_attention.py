# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Optional, Tuple

import torch
from fastforward.nn import QuantizedModule, QuantizerStub
from transformers import Cache
from transformers.models.ministral.modeling_ministral import MinistralAttention

from fastforward_contrib.nn.functional import scaled_dot_product_attention

from ..utils import repeat_kv
from ._rotary_embedding import apply_rotary_pos_emb

logger = logging.getLogger(__name__)


class QuantizedMinistralAttention(MinistralAttention, QuantizedModule):
    """Ministral attention with quantizers around the SDPA computation."""

    def __init_quantization__(self) -> None:
        super().__init_quantization__()

        # assumptions
        if self.config._attn_implementation != "sdpa":
            raise NotImplementedError(
                f"Quantized ministral only supported for SDPA, but this model is using '{self.config._attn_implementation}'"
            )

        if self.config.sliding_window:
            logger.warning("Not supporting sliding window.")

        # add missing attributes
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads

        # quantizers
        self.input_quantizer = QuantizerStub(input_quantizer=True)

        self.rope_q_quantizer = QuantizerStub("act_head_dim", output_quantizer=True)  # post-RoPE
        self.rope_k_quantizer = QuantizerStub("act_head_dim", "kv_cache", output_quantizer=True)  # post-RoPE
        self.v_quantizer = QuantizerStub("act_head_dim", "kv_cache", output_quantizer=True)

        self.sdpa_matmul_quantizer = QuantizerStub("act_head_dim", output_quantizer=True)
        self.sdpa_softmax_quantizer = QuantizerStub("act_head_dim", output_quantizer=True)
        self.sdpa_dropout_quantizer = QuantizerStub("act_head_dim", output_quantizer=True)
        self.sdpa_output_quantizer = QuantizerStub("act_head_dim", output_quantizer=True)

    # Adapted from MinistralAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.input_quantizer(hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # position_embeddings is now mandatory
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            q_output_quantizer=self.rope_q_quantizer,
            k_output_quantizer=self.rope_k_quantizer,
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        value_states = self.v_quantizer(value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            matmul_k_quantizer=self.sdpa_matmul_quantizer,
            softmax_quantizer=self.sdpa_softmax_quantizer,
            dropout_quantizer=self.sdpa_dropout_quantizer,
            output_quantizer=self.sdpa_output_quantizer,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_values
