# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from hydra_zen import builds, store
from omegaconf import DictConfig, OmegaConf

from fasttransforms.nn.collections import HookConfig, LinearParametrizationConfig, TransformConfig

# Resolve the LLM config keys
if OmegaConf._get_resolver("llm_config") is None:
    from transformers import AutoConfig, Qwen2Config

    def resolve_llm_config(config_key: str, _root_: DictConfig):
        llm_cfg = OmegaConf.select(_root_, "llm")

        if "mock_llama" in llm_cfg.model["_target_"]:
            from lm_tools.llm.mock_llama import build_mock_llama_config

            config = build_mock_llama_config()
        else:
            config = AutoConfig.from_pretrained(llm_cfg.model.model_name_or_path)

        # Qwen2 does not keep track of the head_dim directly, so we add it manually
        if isinstance(config, Qwen2Config):
            config.head_dim = config.hidden_size // config.num_attention_heads

        for k in config_key.split("."):
            config = getattr(config, k)
        return config

    OmegaConf.register_new_resolver("llm_config", resolve_llm_config)


transforms_store = store(group="quant/transforms")

# No transform
transforms_store(
    {},
    name="no_transforms",
)


# R1
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "???"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&embed_tokens/output_quantizer",
                inv=False,
                dim=-1,
            ),
            LinearParametrizationConfig(
                pattern="{&lm_head,&q_proj,&k_proj,&v_proj,&up_proj,&gate_proj}",
                inv=True,
                input=True,
            ),
            LinearParametrizationConfig(
                pattern="{&o_proj,&down_proj}",
                inv=False,
                input=False,
            ),
        ],
        per_layer=False,
        priority=1,
    ),
    name="R1",
    package="quant.transforms.R1",
)

# R1_train keeps the standard RMSNorm merge behavior, but avoids placing the
# inverse R1 on lm_head during QAT. The lm_head path is expensive because tied
# embeddings must be duplicated; for training, undo R1 before the final RMSNorm.
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "rotation"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&embed_tokens/output_quantizer",
                inv=False,
                dim=-1,
            ),
            HookConfig(
                pattern="&norm",
                inv=True,
                dim=-1,
            ),
            LinearParametrizationConfig(
                pattern="{&q_proj,&k_proj,&v_proj,&up_proj,&gate_proj}",
                inv=True,
                input=True,
            ),
            LinearParametrizationConfig(
                pattern="{&o_proj,&down_proj}",
                inv=False,
                input=False,
            ),
        ],
        per_layer=False,
        priority=1,
    ),
    name="R1_train",
    package="quant.transforms.R1",
)

# MM (Down Projection input)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "???"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&mm",
                inv=False,
                dim=-1,
            ),
            LinearParametrizationConfig(
                pattern="&down_proj",
                inv=True,
                input=True,
            ),
        ],
        per_layer=True,
        priority=0,
    ),
    name="MM",
    package="quant.transforms.MM",
)


# Na (Attention input)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "???"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&na",
                inv=False,
                dim=-1,
            ),
            LinearParametrizationConfig(
                pattern="&q_proj",
                inv=True,
                input=True,
            ),
            LinearParametrizationConfig(
                pattern="&k_proj",
                inv=True,
                input=True,
            ),
            LinearParametrizationConfig(
                pattern="&v_proj",
                inv=True,
                input=True,
            ),
        ],
        per_layer=True,
        priority=0,
    ),
    name="Na",
    package="quant.transforms.Na",
)

# Nm (MLP input)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "???"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&nm",
                inv=False,
                dim=-1,
            ),
            LinearParametrizationConfig(
                pattern="&gate_proj",
                input=True,
                inv=True,
            ),
            LinearParametrizationConfig(
                pattern="&up_proj",
                input=True,
                inv=True,
            ),
        ],
        per_layer=True,
        priority=0,
    ),
    name="Nm",
    package="quant.transforms.Nm",
)

# Ph (Post RoPE)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "???"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&qe",
                inv=False,
                dim=-1,
            ),
            HookConfig(
                pattern="&ke",
                inv=True,
                transpose=True,
                dim=-1,
            ),
        ],
        per_layer=True,
        priority=0,
    ),
    name="Ph",
    package="quant.transforms.Ph",
)


# Pv (per-head on to_v and to_o)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[
            {"/transform": "values_batched"},
        ],
        hydra_convert="all",
        operation_configs=[
            LinearParametrizationConfig(
                pattern="&v_proj",
                input=False,
                inv=False,
            ),
            LinearParametrizationConfig(
                pattern="&o_proj",
                input=True,
                inv=True,
            ),
        ],
        per_layer=True,
        priority=1,
    ),
    name="Pv",
    package="quant.transforms.Pv",
)


# Po (across-head on post-attention)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[
            {"/transform": "cross_head"},
        ],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&oi",
                inv=False,
                dim=-1,
            ),
            LinearParametrizationConfig(
                pattern="&o_proj",
                input=True,
                inv=True,
            ),
        ],
        per_layer=True,
        priority=0,
    ),
    name="Po",
    package="quant.transforms.Po",
)

transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[
            {"/transform": "head_mixing"},
        ],
        hydra_convert="all",
        operation_configs=[
            HookConfig(
                pattern="&ao",
                inv=False,
                dim=1,
            ),
            LinearParametrizationConfig(
                pattern="&o_proj",
                input=True,
                inv=True,
            ),
        ],
        per_layer=True,
        priority=0,
    ),
    name="Po_paper",
    package="quant.transforms.Po",
)


transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "scale"}],
        hydra_convert="all",
        operation_configs=[
            LinearParametrizationConfig(pattern="&up_proj", inv=False, input=False),
            LinearParametrizationConfig(pattern="&down_proj", inv=True, input=True),
        ],
        per_layer=True,
        priority=1,
    ),
    name="Pu",
    package="quant.transforms.Pu",
)

transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "dynamic_scale"}],
        hydra_convert="all",
        operation_configs=[
            HookConfig(pattern="&decoder_layers/input_quantizer", inv=False, dim=-1, kwargs={"first": True}),
            HookConfig(
                pattern="&ap",
                inv=False,
                dim=-1,
                kwargs={"rescale": True, "expand_dim": 1},
            ),
            HookConfig(pattern="&ra", inv=False, dim=-1),
            HookConfig(pattern="&mm", inv=False, dim=-1, kwargs={"rescale": True}),
            HookConfig(pattern="&rm", inv=True, dim=-1, pre_hook=False),
        ],
        per_layer=True,
        priority=0,
    ),
    name="PDR",
    package="quant.transforms.PDR",
)

transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[{"/transform": "prerope"}],
        hydra_convert="all",
        operation_configs=[
            LinearParametrizationConfig(pattern="&k_proj", inv=False, input=False),
            LinearParametrizationConfig(pattern="&q_proj", inv=True, input=False),
        ],
        per_layer=True,
        priority=1,
    ),
    name="PreRoPE",
    package="quant.transforms.PreRoPE",
)


# f_Pv (per-head in-between v_proj and o_proj. This is the same as Pv, but it is fused afterwards)
transforms_store(
    builds(
        TransformConfig,
        hydra_defaults=[
            {"/transform": "per_head"},
        ],
        hydra_convert="all",
        operation_configs=[
            # Initially, hook to the input of out_proj
            HookConfig(
                pattern="&oi",
                inv=False,
                dim=-1,
                inference=False,
            ),
            # Fuse to the output of v_proj during inference
            LinearParametrizationConfig(
                pattern="&v_proj",
                input=False,
                inv=False,
                inference=True,
            ),
            LinearParametrizationConfig(
                pattern="&o_proj",
                input=True,
                inv=True,
            ),
        ],
        per_layer=True,
        priority=1,
    ),
    name="f_Pv",
    package="quant.transforms.f_Pv",
)
