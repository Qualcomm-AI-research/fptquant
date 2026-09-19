# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
import logging
import os
from abc import abstractmethod
from inspect import Parameter, signature
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, DictKeyType, ListConfig, OmegaConf
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_NAME

from lm_tools.utils import match_patterns

if TYPE_CHECKING:
    from datasets import Dataset
    from torch import nn
    from transformers import Trainer, TrainerCallback


logger = logging.getLogger(__name__)


_SAFE_GLOBALS: list[Any] = [
    np.core.multiarray._reconstruct,  # type: ignore
    np.ndarray,
    np.dtype,
    type(np.dtype("uint32")),
]

torch.serialization.add_safe_globals(_SAFE_GLOBALS)


def skip_some_trainable_parameters(model: "nn.Module", skip_params: Optional[list[str]] = None):
    if skip_params:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                logger.debug(f"Parameter {name} does not require gradients")
            elif match_patterns(name, skip_params):
                logger.info(f"Parameter {name} will be excluded from fine-tuning")
                param.requires_grad = False
            else:
                param.requires_grad = True


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.name.rsplit("-", maxsplit=1)[-1]), path.name)
    except ValueError:
        return (-1, path.name)


def _get_last_checkpoint_path(output_dir: Path) -> Optional[Path]:
    checkpoints = [
        path for path in output_dir.glob("checkpoint-*") if path.is_dir() and (path / "trainer_state.json").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def _resolve_resume_checkpoint(
    resume_from_checkpoint: Union[str, bool], output_dir: Path
) -> tuple[Optional[Union[str, bool]], Optional[Path]]:
    if isinstance(resume_from_checkpoint, str):
        resume_path = Path(resume_from_checkpoint)
        if resume_path.is_dir() and (resume_path / "trainer_state.json").exists():
            return resume_from_checkpoint, resume_path
        return resume_from_checkpoint, None

    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resolved_resume_checkpoint = _get_last_checkpoint_path(output_dir)
        if resolved_resume_checkpoint is not None:
            return str(resolved_resume_checkpoint), resolved_resume_checkpoint

    return None, None


def _get_checkpoint_global_step(checkpoint_path: Path) -> Optional[int]:
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return None

    try:
        with open(trainer_state_path, encoding="utf-8") as f:
            return int(json.load(f).get("global_step", -1))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        logger.exception("Failed to read %s; proceeding with trainer.train() as usual.", trainer_state_path)
        return None


def _checkpoint_satisfies_max_steps(checkpoint_path: Path, max_steps: int) -> tuple[bool, Optional[int]]:
    if max_steps <= 0:
        return False, None

    checkpoint_global_step = _get_checkpoint_global_step(checkpoint_path)
    if checkpoint_global_step is None:
        return False, None

    return 0 <= max_steps <= checkpoint_global_step, checkpoint_global_step


def _load_checkpoint_weights(model: "nn.Module", checkpoint_path: Path) -> None:
    weights_path = checkpoint_path / WEIGHTS_NAME
    safe_weights_path = checkpoint_path / SAFE_WEIGHTS_NAME

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    elif safe_weights_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safe_weights_path), device="cpu")
    else:
        raise FileNotFoundError(
            f"Could not find checkpoint weights in {checkpoint_path}; expected {WEIGHTS_NAME} or {SAFE_WEIGHTS_NAME}."
        )

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if incompatible_keys.unexpected_keys:
        logger.warning(
            "Loaded checkpoint %s with %d unexpected key(s), e.g. %s",
            checkpoint_path,
            len(incompatible_keys.unexpected_keys),
            incompatible_keys.unexpected_keys[:5],
        )


def filter_training_args_for_init(training_args: dict[DictKeyType, Any], args_cls: type) -> dict[DictKeyType, Any]:
    init_signature = signature(args_cls)
    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in init_signature.parameters.values()):
        return dict(training_args)

    accepted_args = set(init_signature.parameters)
    filtered_args = {key: value for key, value in training_args.items() if key in accepted_args}
    dropped_args = sorted(set(training_args) - set(filtered_args), key=str)
    if dropped_args:
        logger.info("Ignoring unsupported %s argument(s): %s", args_cls.__name__, dropped_args)

    return filtered_args


def _patch_filtered_checkpoints(trainer: "Trainer", parameter_substrings: list[str]) -> None:
    """Restrict checkpoint weights to parameters whose name contains one of ``parameter_substrings``.

    This intentionally reimplements (rather than delegates to) ``transformers.Trainer._save``. In
    transformers 5.x, ``save_pretrained`` always serialises to safetensors, which cannot represent the
    shared/tied tensors produced by the transform parametrizations (e.g. ``...R.original`` aliasing
    ``...R.0.base``, or transforms tied across q/k and up/down projections). We therefore write the
    filtered weights with ``torch.save`` to :data:`WEIGHTS_NAME` (``pytorch_model.bin``); HF's resume
    path checks safetensors first and falls back to this file, loading it with ``strict=False`` onto
    the full model. Config/generation_config/training_args are saved alongside for parity with the
    stock checkpoint.

    The configured substrings are validated **here**, at training start (before any compute is spent),
    so a misconfiguration fails fast rather than at the first checkpoint: matching zero tensors, or
    leaving any currently-trainable parameter unmatched (it could not be restored on resume), raises.
    Buffers (e.g. quantizer scales) are not parameters and so are not required to match here; they are
    still saved whenever they match a substring.
    """

    def _matches(name: str) -> bool:
        return any(substring in name for substring in parameter_substrings)

    state_names = list(trainer.model.state_dict().keys())
    if not any(_matches(name) for name in state_names):
        raise ValueError(
            f"checkpoint_parameter_substrings {parameter_substrings} matched 0 of {len(state_names)} "
            f"tensors; nothing would be saved. Check the configured substrings."
        )

    missed_trainable = sorted(
        name for name, param in trainer.model.named_parameters() if param.requires_grad and not _matches(name)
    )
    if missed_trainable:
        raise ValueError(
            f"{len(missed_trainable)} trainable parameter(s) are not matched by "
            f"checkpoint_parameter_substrings {parameter_substrings} and would not be saved; they could "
            f"not be restored on resume (e.g. {missed_trainable[:5]}). Adjust the substrings or skip_params."
        )

    def _save(
        self: "Trainer", output_dir: str | None = None, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if state_dict is None:
            state_dict = model.state_dict()

        filtered_state_dict = {name: tensor.detach().cpu() for name, tensor in state_dict.items() if _matches(name)}
        logger.info(
            "Saving filtered checkpoint to %s with %d/%d tensors matching %s",
            output_dir,
            len(filtered_state_dict),
            len(state_dict),
            parameter_substrings,
        )
        torch.save(filtered_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # Save the same sidecar files HF would (minus the tokenizer, which is managed separately).
        if getattr(model, "config", None) is not None:
            model.config.save_pretrained(output_dir)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    trainer._save = MethodType(_save, trainer)


def _patch_incompatible_optimizer_resume(trainer: "Trainer") -> None:
    if not hasattr(trainer, "_load_optimizer_and_scheduler"):
        return

    original_load_optimizer_and_scheduler = trainer._load_optimizer_and_scheduler

    def _load_optimizer_and_scheduler(checkpoint: str | None) -> None:
        try:
            original_load_optimizer_and_scheduler(checkpoint)
        except ValueError as exc:
            if checkpoint is None or "parameter group" not in str(exc):
                raise
            logger.warning(
                "Skipping optimizer/scheduler state from %s because it is incompatible with the current "
                "trainable parameter groups; continuing from checkpoint model weights and trainer_state.json.",
                checkpoint,
            )

    trainer._load_optimizer_and_scheduler = _load_optimizer_and_scheduler


class BaseTraining:
    def __init__(
        self,
        training_args: dict[DictKeyType, Any] | DictConfig,
        train_dataset: Callable[..., "Dataset"],
        eval_dataset: Optional[Callable[..., "Dataset"]] = None,
        skip_params: Optional[list[str]] = None,
        resume_from_checkpoint: Union[str, bool] = True,
        checkpoint_parameter_substrings: Optional[list[str]] = None,
        callbacks: Optional[list["TrainerCallback"]] = None,
        eval_callback: Optional["TrainerCallback"] = None,
    ):
        if isinstance(training_args, DictConfig):
            training_args_ = OmegaConf.to_container(training_args)
            if not isinstance(training_args_, dict):
                raise RuntimeError(f"Unexpected training arguments type {type(training_args_)}")
            training_args = training_args_

        # Make sure we are not pushing to Huggingface by mistake
        assert not training_args.get("push_to_hub", None)

        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.skip_params = skip_params
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_parameter_substrings = checkpoint_parameter_substrings

        if isinstance(callbacks, ListConfig):
            callbacks_ = OmegaConf.to_container(callbacks)
            if not isinstance(callbacks_, list):
                raise RuntimeError(f"Unexpected callbacks type {type(callbacks_)}")
            callbacks = callbacks_
        self.callbacks = callbacks
        self.eval_callback = eval_callback

    @abstractmethod
    def make_trainer(
        self, model: "nn.Module", train_dataset: "Dataset", eval_dataset: Optional["Dataset"]
    ) -> "Trainer":
        pass

    def __call__(
        self,
        model: "nn.Module",
        tokenizer: Any,
    ) -> None:
        from accelerate import Accelerator

        accelerator = Accelerator()
        max_steps = int(self.training_args.get("max_steps", -1) or -1)
        output_dir = Path(str(self.training_args.get("output_dir", ".")))
        resume_from_checkpoint, resolved_resume_checkpoint = _resolve_resume_checkpoint(
            self.resume_from_checkpoint, output_dir
        )

        if resolved_resume_checkpoint is not None:
            checkpoint_complete, checkpoint_global_step = _checkpoint_satisfies_max_steps(
                resolved_resume_checkpoint, max_steps
            )
            if checkpoint_complete:
                logger.info(
                    "Resolved checkpoint %s has global_step=%d >= max_steps=%d; "
                    "loading checkpoint weights and skipping trainer construction.",
                    resolved_resume_checkpoint,
                    checkpoint_global_step,
                    max_steps,
                )
                _load_checkpoint_weights(accelerator.unwrap_model(model), resolved_resume_checkpoint)
                accelerator.wait_for_everyone()
                return

        # Prepare model and datasets
        logger.info("Preparing the training Dataset")
        train_dataset = self.train_dataset(tokenizer=tokenizer)

        if self.eval_dataset is not None:
            logger.info("Preparing the evaluation Dataset")
            eval_dataset = self.eval_dataset(tokenizer=tokenizer)
        else:
            eval_dataset = None

        # Skip parameters from training
        skip_some_trainable_parameters(model, self.skip_params)

        callbacks = list(self.callbacks or [])
        if self.eval_callback is not None:
            if hasattr(self.eval_callback, "set_tokenizer"):
                self.eval_callback.set_tokenizer(tokenizer)
            callbacks.append(self.eval_callback)
        self.callbacks = callbacks or None

        # Instantiate the trainer
        trainer = self.make_trainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
        _patch_incompatible_optimizer_resume(trainer)
        if self.checkpoint_parameter_substrings is not None:
            _patch_filtered_checkpoints(trainer, list(self.checkpoint_parameter_substrings))

        # Move the model and datasets to the correct device
        logger.info("Preparing the model and dataset for training")
        trainer = accelerator.prepare(trainer)

        if resume_from_checkpoint:
            resume_display = (
                resolved_resume_checkpoint if resolved_resume_checkpoint is not None else resume_from_checkpoint
            )
            logger.info("Found existing QAT checkpoints, will resume training from %s", resume_display)

        # Train the model
        logger.info("Starting the training procedure")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.accelerator.wait_for_everyone()

        # Unwrap the model for further use
        model = accelerator.unwrap_model(model)
