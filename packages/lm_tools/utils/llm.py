# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, AutoModelForCausalLM, PreTrainedModel

from lm_tools.utils.enums import TorchDtype
from lm_tools.utils.model_aliases import get_aliases

if TYPE_CHECKING:
    from transformers import MODEL_MAPPING, PreTrainedModel

    # define the available model types
    MODEL_CONFIG_CLASSES = list(set(MODEL_MAPPING.keys()))
else:
    MODEL_CONFIG_CLASSES = []
ModelTypes = Enum("ModelTypes", [conf.model_type for conf in MODEL_CONFIG_CLASSES])  # type: ignore


class AttnImplementation(str, Enum):
    AUTO = "auto"
    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    EAGER = "eager"
    FLEX_ATTENTION = "flex_attention"


def get_attn_implementation(value: str) -> AttnImplementation:
    try:
        return AttnImplementation(value)
    except ValueError as e:
        raise ValueError(f"Invalid attention implementation: {value}") from e


def instantiate_llm(
    model_type: Optional[ModelTypes] = None,  # type: ignore
    model_name_or_path: Optional[Union[Path, str]] = None,
    config_name: Optional[str] = None,
    revision: str = "main",
    load_weights: bool = True,
    cache_dir: Optional[Path] = None,
    model_dtype: TorchDtype = TorchDtype.auto,
    config_overrides: Optional[Dict[str, Any]] = None,
    use_cache: bool = False,
    low_cpu_mem_usage: bool = False,
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = False,
    attn_implementation: str = "auto",
    logger: Optional[logging.Logger] = None,
) -> "PreTrainedModel":
    if logger is None:
        logger = logging.getLogger(__name__)

    if isinstance(model_name_or_path, Path):
        model_name_or_path = str(model_name_or_path)

    if not any([config_name, model_type, model_name_or_path]):
        raise ValueError('At least one of the "config_name", "model_name_or_path", "model_type" has to be specified')

    if load_weights and model_name_or_path is None:
        raise ValueError('If "load_weight" is True, then "model_name_or_path" must be specified')

    # In distributed training, the .from_pretrained methods guarantee that only one local process
    # can concurrently download model & vocab.

    # Get HuggingFace model config
    hf_config_kwargs = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    if config_name is not None and model_name_or_path is None:
        logger.warning(f'Using same config as "{config_name}" but for a model with randomly initialized weights')
        hf_config = CONFIG_MAPPING[config_name](**hf_config_kwargs)

    elif config_name is not None:
        logger.warning(f'Using a config from pretrained model with config_name = "{config_name}"')
        hf_config = AutoConfig.from_pretrained(config_name, **hf_config_kwargs)

    elif model_name_or_path is not None:
        logger.warning(f'Using a config from model_name_or_path = "{model_name_or_path}"')
        hf_config = AutoConfig.from_pretrained(model_name_or_path, **hf_config_kwargs)

    elif model_type is not None:
        logger.warning(
            f"You are instantiating a new HuggingFace config instance from scratch for model_type = {model_type}"
        )
        hf_config = CONFIG_MAPPING[model_type]()

    else:
        raise ValueError('At least one of the "config_name", "model_name_or_path", "model_type" has to be specified')

    # override config
    if config_overrides is not None:
        logger.info(f"Overriding HF config: {config_overrides}")
        hf_config.update(config_overrides)
        logger.info(f"New HF config: {hf_config}")

    logger.info(f"Setting use_cache to {use_cache}")
    hf_config.use_cache = use_cache

    logger.info(f"HuggingFace config after user changes: {hf_config}")

    attn_implementation = get_attn_implementation(attn_implementation)
    if attn_implementation == AttnImplementation.AUTO:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available() and "V100" not in torch.cuda.get_device_name():
            attn_implementation = AttnImplementation.FLASH_ATTENTION_2
        else:
            attn_implementation = AttnImplementation.SDPA

    logger.info(f"Using attention implementation: {attn_implementation.value}")

    # Load and prepare model
    # Note: gradient_checkpointing is no longer set on the config.
    # It is enabled on the model instance after creation (see below).
    if model_name_or_path and load_weights:
        # determine dtype
        if model_dtype.name is None:
            raise ValueError(f"model_dtype {model_dtype!r} has no name attribute")
        torch_dtype = model_dtype.name if model_dtype == TorchDtype.auto else getattr(torch, model_dtype.name)

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=hf_config,
            cache_dir=cache_dir,
            revision=revision,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation.value,
        )
    else:
        logger.info("Model weights are not loaded")
        model = AutoModelForCausalLM.from_config(
            hf_config,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation.value,
        )

    # Enable gradient checkpointing on the model instance (replaces the deprecated config attribute)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Add the aliases for mpath selection
    model.aliases = get_aliases(model.config.model_type)

    return model
