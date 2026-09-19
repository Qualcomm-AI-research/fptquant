# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import copy
import logging
from typing import Any

from fastforward import mpath
from fastforward.nn import QuantizedLinear  # noqa: F401

logger = logging.getLogger(__name__)


# ALL: q.k.v.o.ke.qe.aw.ap.ad.ao.g.gs.d.u.mm.na.nm.ra.rm
# attn: q.k.v.o.ke.qe.aw.ap.ad.ao
# mlp: g.gs.d.u.mm
# layernorm: na.nm
# residuals: ra.rm
_LLAMA_ALIASES = dict(
    ### 1) Aliases for parts of the network
    # groups (decoder layers)
    decoder="model",
    decoder_layers="model/layers/*",
    # embeddings
    embed_tokens="model/embed_tokens",
    # head
    lm_head="lm_head",
    # attention
    attn_norm="&decoder_layers/input_layernorm",  # norm before attention
    attn="&decoder_layers/self_attn",
    q_proj="&attn/q_proj",
    k_proj="&attn/k_proj",
    v_proj="&attn/v_proj",
    o_proj="&attn/o_proj",
    # MLP
    mlp_norm="&decoder_layers/post_attention_layernorm",  # norm before mlp
    mlp="&decoder_layers/mlp",
    up_proj="&mlp/up_proj",
    gate_proj="&mlp/gate_proj",
    down_proj="&mlp/down_proj",
    act_fn="&mlp/act_fn",
    # (final) output norm
    norm="model/norm",
    ### 2) Individual activation quantizers
    # Embeddings (activations)
    emb="&embed_tokens/output_quantizer",
    # Self-attn: QKVO (output quantizers)
    q="&q_proj/output_quantizer",
    k="&k_proj/output_quantizer",
    v="&v_proj/output_quantizer",
    o="&o_proj/output_quantizer",
    # Self-attn: QKVO (input quantizers)
    qi="&q_proj/input_quantizer",
    ki="&k_proj/input_quantizer",
    vi="&v_proj/input_quantizer",
    oi="&o_proj/input_quantizer",
    # Self-attn: QKVO (extra)
    ke="&attn/rope_k_quantizer",  # key embeddings (key+embeddings)
    qe="&attn/rope_q_quantizer",  # query embeddings (query+embeddings)
    ve="&attn/v_quantizer",  # different shape to `v` due to #heads (which affects per head dynamic quant)
    # Self-attn: attn scores/probabilities
    aw="&attn/sdpa_matmul_quantizer",  # attention weights (pre softmax)
    ap="&attn/sdpa_softmax_quantizer",  # attention probabilities (post softmax)
    ao="&attn/sdpa_output_quantizer",  # attention output (after att_weight*values)
    # MLP (output quantizers)
    g="&gate_proj/output_quantizer",
    gs="&act_fn/output_quantizer",  # gate-silu quantizer (act=silu)
    d="&down_proj/output_quantizer",
    u="&up_proj/output_quantizer",
    mm="&mlp/gated_up_proj_output_quantizer",  # MLP elementwise multiplication
    # MLP (input quantizers)
    gi="&gate_proj/input_quantizer",
    ui="&up_proj/input_quantizer",
    # MLP (extra)
    silu_sigmoid="&mlp/act_fn/sigmoid_quantizer",  # quantizer on sigmoid inside SiLU
    # norm
    na="&attn_norm/output_quantizer",  # layer norm before attention
    nm="&mlp_norm/output_quantizer",  # layer norm before mlp
    # residuals
    ra="&decoder_layers/attn_res_act_quantizer",  # residual add after attention
    rm="&decoder_layers/mlp_res_act_quantizer",  # residual add after mlp
    # post (final) norm
    post_norm="&norm/output_quantizer",
    # LM head output
    head="&lm_head/output_quantizer",
    ### 3) Groups of quantizers
    all_acts_decoder="&decoder_layers/**/[quantizer:activation/output]",
    all_weights_decoder="&decoder_layers/**/[quantizer:parameter/weight]",
    linear_weights_decoder="&decoder/**/[cls:QuantizedLinear]/[quantizer:parameter/weight]",
)


## Qwen3 has extra pre-QK RMSNorms
# layernorm: na.nm.nq.nk
# ALL: q.k.v.o.ke.qe.aw.ap.ad.ao.g.gs.d.u.mm.na.nm.nq.nk.ra.rm
_QWEN3_ALIASES = _LLAMA_ALIASES.copy()
_QWEN3_ALIASES.update(
    dict(
        q_norm="&attn/q_norm",
        k_norm="&attn/k_norm",
        nq="&q_norm/output_quantizer",
        nk="&k_norm/output_quantizer",
    )
)


## Qwen3.5 has gated attention (gate_sig) in addition to Qwen3's pre-QK RMSNorms,
## and GatedDeltaNet (linear attention) layers.
# ALL: q.k.v.o.ke.qe.aw.ap.ad.ao.g.gs.d.u.mm.na.nm.nq.nk.ra.rm.gate_sig
# GatedDeltaNet activations: gdn_in.gdn_q.gdn_k.gdn_v.gdn_z.gdn_beta.gdn_sp_in.gdn_sp_out.gdn_g.gdn_conv.gdn_dr.gdn_state
# GatedDeltaNet weights:     gdn_qkv_w.gdn_z_w.gdn_b_w.gdn_a_w.gdn_out_w.gdn_conv_w.gdn_norm_w
_QWEN35_ALIASES = _QWEN3_ALIASES.copy()
_QWEN35_ALIASES.update(
    dict(
        gate_sig="&attn/gate_sig_quantizer",  # gated attention: sigmoid(gate) output
        # GatedDeltaNet (linear attention) layer aliases
        # GatedDeltaNet lives at ``linear_attn`` on the decoder layer (not ``self_attn``)
        linear_attn="&decoder_layers/linear_attn",
        # --- projection module paths (used to build weight quantizer aliases below) ---
        gdn_qkv_proj="&linear_attn/in_proj_qkv",
        gdn_z_proj="&linear_attn/in_proj_z",
        gdn_b_proj="&linear_attn/in_proj_b",
        gdn_a_proj="&linear_attn/in_proj_a",
        gdn_out_proj="&linear_attn/out_proj",
        gdn_conv1d="&linear_attn/conv1d",
        gdn_norm="&linear_attn/norm",  # RMSNormGated
        # --- activation quantizers ---
        gdn_in="&linear_attn/input_quantizer",  # GatedDeltaNet module input
        gdn_q="&linear_attn/q_quantizer",  # query (after conv1d + split)
        gdn_k="&linear_attn/k_quantizer",  # key (after conv1d + split)
        gdn_v="&linear_attn/v_quantizer",  # value (after conv1d + split)
        gdn_z="&linear_attn/z_quantizer",  # z projection (input to RMSNormGated)
        gdn_beta="&linear_attn/beta_quantizer",  # beta = sigmoid(in_proj_b)
        gdn_sp_in="&linear_attn/softplus_input_quantizer",  # softplus input (a + dt_bias)
        gdn_sp_out="&linear_attn/softplus_output_quantizer",  # softplus output
        gdn_g="&linear_attn/g_quantizer",  # gating signal g = A_log_exp * softplus(...)
        gdn_conv="&linear_attn/conv1d_out_quantizer",  # conv1d output (after activation)
        gdn_dr="&linear_attn/delta_rule_quantizer",  # delta rule kernel output
        gdn_state="&linear_attn/attention_state_quantizer",  # recurrent state (GDN cache)
        gdn_dr_chunk="&linear_attn/chunk_delta_rule_quantizers/*",
        gdn_dr_recur="&linear_attn/recurrent_delta_rule_quantizers/*",
        # --- weight quantizers ---
        gdn_qkv_w="&gdn_qkv_proj/weight_quantizer",  # in_proj_qkv weight
        gdn_z_w="&gdn_z_proj/weight_quantizer",  # in_proj_z weight
        gdn_b_w="&gdn_b_proj/weight_quantizer",  # in_proj_b weight
        gdn_a_w="&gdn_a_proj/weight_quantizer",  # in_proj_a weight
        gdn_out_w="&gdn_out_proj/weight_quantizer",  # out_proj weight
        gdn_conv_w="&gdn_conv1d/weight_quantizer",  # conv1d weight
        gdn_norm_w="&gdn_norm/weight_quantizer",  # RMSNormGated weight
    )
)


_OPT_ALIASES = dict(
    # decoder layer
    decoder="model/decoder",
    decoder_layers="&decoder/layers/*",
    all_weights_decoder="&decoder_layers/**/[quantizer:parameter/weight]",
    linear_weights_decoder="&decoder_layers/**/[cls:QuantizedLinear]/[quantizer:parameter/weight]",
)


class Aliases(dict):
    """
    Utility class for aliases.

    Allows extending a list of aliases with more aliases. This is not possible with `mpath.aliases`, as that function
    does not have aliases as an argument. For example, if we have mpath.aliases(a="a"), we cannot add dict(b="&a/b")),
    since the latter will not know what `&a` means.
    """

    def __init__(self, **kwargs):
        super().__init__(**mpath.aliases(**kwargs))

    def extend(self, dictionary: dict[str, str], context: dict[str, Any] | None = None, inplace=False):
        if inplace:
            aliases = self
        else:
            aliases = copy.deepcopy(self)

        if isinstance(dictionary, Aliases):
            aliases.update(dictionary)
        else:
            context = context or mpath._parser.get_caller_context()
            for name, value in dictionary.items():
                aliases[name] = mpath._parser.parse(value, context=context, aliases=aliases)

        return aliases


class NotExistingAliases(Aliases):
    def __init__(self, model_type: str):
        self.model_type = model_type

    def raise_error(self):
        raise NotImplementedError(
            f"Aliases not implemented for model type {self.model_type}. Add model-specific aliases to file {__file__}"
        )

    def __getitem__(self, _):
        self.raise_error()

    def __repr__(self):
        self.raise_error()

    def __iter__(self):
        self.raise_error()


def get_aliases(model_type: str) -> Aliases:
    match model_type:
        case "llama":
            aliases = _LLAMA_ALIASES
        case "llama3":
            aliases = _LLAMA_ALIASES
        case "ministral":
            aliases = _LLAMA_ALIASES
        case "qwen2":
            aliases = _LLAMA_ALIASES
        case "qwen3":
            aliases = _QWEN3_ALIASES
        case "qwen3_5" | "qwen3_5_text":
            aliases = _QWEN35_ALIASES
        case "opt":
            aliases = _OPT_ALIASES
        case _:
            logger.warning(f"Model type {model_type}'s aliases are unknown.")
            return NotExistingAliases(model_type)
    return Aliases(**aliases)
