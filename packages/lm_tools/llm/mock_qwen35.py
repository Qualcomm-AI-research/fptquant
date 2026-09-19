# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


def build_mock_qwen35():
    """Build a tiny randomly-initialised Qwen3.5 text model for testing.

    The model is scaled down to be very lightweight while still exercising
    **both** layer types present in the real Qwen3.5 architecture:
      - ``linear_attention``  (GatedDeltaNet-style recurrent layers)
      - ``full_attention``    (standard multi-head attention with RoPE)

    The layer pattern mirrors the real 4B model: 3 linear + 1 full, repeated
    once (4 layers total).

    Dimensions are chosen so that the mRoPE section constraint is satisfied:
        sum(mrope_section) == head_dim * partial_rotary_factor / 2
        sum([1, 1, 2])     == 32 * 0.25 / 2  ==  4  ✓
    """
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from lm_tools.utils.model_aliases import get_aliases

    model_config = Qwen3_5TextConfig(
        # ── Transformer core ──────────────────────────────────────────────
        hidden_size=128,  # = num_attention_heads * head_dim = 4 * 32
        intermediate_size=256,
        num_hidden_layers=2,
        # ── Full-attention heads ──────────────────────────────────────────
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        # ── Layer schedule: 1 linear + 1 full ───────
        layer_types=[
            "linear_attention",
            "full_attention",
        ],
        full_attention_interval=4,
        # ── Linear-attention (GatedDeltaNet) heads ────────────────────────
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_num_value_heads=2,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        # ── RoPE (mRoPE) ─────────────────────────────────────────────────
        # rotary_dim = head_dim * partial_rotary_factor = 32 * 0.25 = 8
        # sum(mrope_section) must equal rotary_dim / 2 = 4
        rope_parameters={
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 2],  # sums to 4 ✓
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        },
        # ── Misc ──────────────────────────────────────────────────────────
        vocab_size=248320,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        dtype="bfloat16",
        use_cache=True,
        tie_word_embeddings=True,
    )

    model = AutoModelForCausalLM.from_config(model_config, attn_implementation="sdpa")

    # Attach mpath aliases so quantization path-selection works out of the box.
    model.mpath_aliases = get_aliases(model_config.model_type)

    return model
