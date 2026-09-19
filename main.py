#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from pathlib import Path
from typing import Any, Sequence

import fastforward as ff
import hydra
import torch
import torch.nn.utils.parametrize as P
from accelerate import Accelerator
from hydra_zen import instantiate
from omegaconf import DictConfig, ListConfig

import fasttransforms as ft
import lm_tools
import quant_tools
from config_resolvers import register_config_resolvers
from debug_utils import apply_debug_layer_slice
from lm_tools.utils import set_seed

lm_tools.add_config_to_store()
quant_tools.add_config_to_store()
ft.add_config_to_store()
register_config_resolvers()


def _set_trainable_parameters(
    model: torch.nn.Module, trainable_parameter_substrings: Sequence[str] | ListConfig | None
) -> None:
    """Freeze model parameters except those matching configured substrings."""
    if trainable_parameter_substrings is None:
        return
    if isinstance(trainable_parameter_substrings, ListConfig):
        trainable_parameter_substrings = list(trainable_parameter_substrings)

    trainable_count = 0
    frozen_count = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(substring in name for substring in trainable_parameter_substrings)
        if parameter.requires_grad:
            trainable_count += parameter.numel()
        else:
            frozen_count += parameter.numel()

    logging.getLogger(__name__).info(
        "Configured trainable parameters: %d trainable, %d frozen, patterns=%s",
        trainable_count,
        frozen_count,
        trainable_parameter_substrings,
    )


def _verify_trainable_parameter_contract(
    model: torch.nn.Module, trainable_parameter_substrings: Sequence[str]
) -> None:
    """Raise when parameter trainability differs from the configured contract."""
    incorrectly_trainable = []
    incorrectly_frozen = []
    for name, parameter in model.named_parameters():
        should_train = any(s in name for s in trainable_parameter_substrings)
        if parameter.requires_grad and not should_train:
            incorrectly_trainable.append(name)
        elif should_train and not parameter.requires_grad:
            incorrectly_frozen.append(name)

    if incorrectly_trainable or incorrectly_frozen:
        preview_trainable = incorrectly_trainable[:10]
        preview_frozen = incorrectly_frozen[:10]
        raise AssertionError(
            "Training must update only quantizer and transform parameters. "
            f"Unexpected trainable parameters: {preview_trainable}; "
            f"unexpected frozen quantizer/transform parameters: {preview_frozen}."
        )


def _cast_quantizer_parameters_to_float32(model: torch.nn.Module) -> int:
    """Cast initialized quantizer parameters to float32 and return their count."""
    cast_count = 0
    for name, parameter in model.named_parameters():
        if "quantizer" not in name:
            continue
        if isinstance(parameter, torch.nn.UninitializedParameter):
            continue
        if parameter.data.dtype != torch.float32:
            with torch.no_grad():
                parameter.copy_(parameter.to(torch.float32))
            cast_count += parameter.numel()
    return cast_count


def _verify_quantizer_parameters_float32(model: torch.nn.Module) -> None:
    """Raise when an initialized quantizer parameter is not float32."""
    non_float32_quantizer_parameters = []
    for name, parameter in model.named_parameters():
        if "quantizer" not in name:
            continue
        if isinstance(parameter, torch.nn.UninitializedParameter):
            continue
        if parameter.dtype != torch.float32:
            non_float32_quantizer_parameters.append((name, str(parameter.dtype)))

    if non_float32_quantizer_parameters:
        preview = non_float32_quantizer_parameters[:10]
        raise AssertionError(f"Quantizer parameters must be float32. Non-float32 parameters: {preview}")


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config: DictConfig) -> Any:
    """Run the configured FPTQuant quantization, training, and evaluation workflow."""
    set_seed(config.seed)
    logger = logging.getLogger()
    accelerator = Accelerator()

    model = instantiate(config.llm.model)
    model = apply_debug_layer_slice(model, config.get("debug", {}).get("layer_slice"), logger=logger)
    tokenizer = instantiate(config.llm.tokenizer)

    model = accelerator.prepare(model)

    if "quant" in config:
        quant_pipeline = instantiate(config.quant)
        model = quant_pipeline(model, tokenizer, logger)
        cast_count = _cast_quantizer_parameters_to_float32(model)
        _verify_quantizer_parameters_float32(model)
        logger.info("Casted %d quantizer parameters to float32.", cast_count)

    if config.get("skip_training", False):
        logger.info("Skipping training because skip_training=true.")
    elif "train" in config:
        _set_trainable_parameters(model, config.get("trainable_parameter_substrings"))
        if config.get("trainable_parameter_substrings") is not None:
            _verify_trainable_parameter_contract(model, config.trainable_parameter_substrings)
        training_procedure = instantiate(config.train)
        with ff.strict_quantization(False):
            training_procedure(model=model, tokenizer=tokenizer)

    if config.get("skip_final_evaluation", False):
        logger.info("Skipping final evaluation because skip_final_evaluation=true.")
        return None

    evaluation = instantiate(config.evaluation)
    with ff.strict_quantization(False), P.cached():
        results = evaluation(model=model, tokenizer=tokenizer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        result_file = Path("./results.csv").resolve()
        results.to_csv(result_file)
        logger.info(f"Results saved to {result_file}")

    return results


if __name__ == "__main__":
    main()
