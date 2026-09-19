# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Any

from omegaconf import OmegaConf


def _hook_config(pattern: str, inv: bool = False, dim: int = -1, **kwargs: Any) -> dict[str, Any]:
    """Build a transform hook configuration for Hydra composition."""
    config: dict[str, Any] = {
        "_target_": "fasttransforms.nn.collections.HookConfig",
        "pattern": pattern,
        "inv": inv,
        "dim": dim,
    }
    if kwargs:
        config["kwargs"] = kwargs
    return config


def pdr_chain_operation_configs(
    attention_pattern: str, mlp_pattern: str, attention_expand_dim: int | None = None
) -> str:
    """Return YAML for the PDR transform-operation chain resolver."""
    attention_kwargs: dict[str, Any] = {"rescale": True}
    if attention_expand_dim is not None:
        attention_kwargs["expand_dim"] = attention_expand_dim
    configs = [
        _hook_config("&decoder_layers/input_quantizer", first=True),
        _hook_config(attention_pattern, **attention_kwargs),
        _hook_config("&ra"),
        _hook_config(mlp_pattern, rescale=True),
        {
            **_hook_config("&rm", inv=True),
            "pre_hook": False,
        },
    ]
    return OmegaConf.to_yaml(configs)


def register_config_resolvers() -> None:
    """Register the FPTQuant OmegaConf resolvers once per process."""
    if OmegaConf._get_resolver("fptquant_pdr_chain") is None:
        OmegaConf.register_new_resolver("fptquant_pdr_chain", pdr_chain_operation_configs)
