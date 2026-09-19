# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import fastforward as ff
import pytest
import torch
from hydra_zen import instantiate
from transformers.trainer_callback import TrainerState

import lm_tools

PARAMETER_CHANGE_EPS = 1e-8
CHECKPOINT_TRAINERS = [
    ("ce", []),
    ("kd", ["+train.loss_type=kl"]),
]


def _train_once(cfg, tmp_path):
    model = instantiate(cfg.llm.model)
    tokenizer = instantiate(cfg.llm.tokenizer)
    train = instantiate(cfg.train)
    train_dataset = train.train_dataset(tokenizer=tokenizer)

    trainer = train.make_trainer(model=model, train_dataset=train_dataset, eval_dataset=None)
    result = trainer.train()

    assert trainer.args.output_dir == str(tmp_path)
    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    assert any(path.name.startswith("checkpoint-") for path in tmp_path.iterdir())

    return model, tokenizer


def _compose_quantized_training_cfg(compose_cfg, tmp_path, train_name: str, extra_overrides: list[str]):
    return compose_cfg(
        [
            "+llm=mock_llama",
            "+quant=simple_ptq",
            "+quant/rules=w_example",
            "+tokenized_dataset@quant.calibration_set=mock_text",
            f"+train={train_name}",
            "+tokenized_dataset@train.train_dataset=mock_text",
            "+train.training_args.per_device_train_batch_size=1",
            "+train.training_args.max_steps=1",
            "+train.training_args.save_steps=1",
            "+train.training_args.report_to=[]",
            "+train.training_args.use_cpu=true",
            "+train.training_args.remove_unused_columns=false",
            "train.resume_from_checkpoint=false",
            f"train.training_args.output_dir={tmp_path}",
            "+paths=default",
            *extra_overrides,
        ]
    )


def _compose_checkpoint_training_cfg(compose_cfg, tmp_path, train_name: str, extra_overrides: list[str]):
    return compose_cfg(
        [
            "+llm=mock_llama",
            f"+train={train_name}",
            "+tokenized_dataset@train.train_dataset=mock_text",
            "+train.training_args.per_device_train_batch_size=1",
            "+train.training_args.max_steps=1",
            "+train.training_args.save_strategy=steps",
            "+train.training_args.save_steps=1",
            "+train.training_args.report_to=[]",
            "+train.training_args.use_cpu=true",
            "+train.training_args.remove_unused_columns=false",
            "train.resume_from_checkpoint=false",
            f"train.training_args.output_dir={tmp_path}",
            "+paths=default",
            *extra_overrides,
        ]
    )


def _mock_tokenizer_path() -> str:
    return str(Path(lm_tools.__path__[0]) / "data" / "mock_tokenizer")


def _set_mock_model_name_or_path(model) -> None:
    model.config._name_or_path = _mock_tokenizer_path()


def _make_checkpoint_trainer(cfg):
    model = instantiate(cfg.llm.model)
    tokenizer = instantiate(cfg.llm.tokenizer)
    _set_mock_model_name_or_path(model)
    train = instantiate(cfg.train)
    train_dataset = train.train_dataset(tokenizer=tokenizer)
    trainer = train.make_trainer(model=model, train_dataset=train_dataset, eval_dataset=None)
    return model, tokenizer, trainer


def _quantize_model_for_training(cfg, logger):
    model = instantiate(cfg.llm.model)
    _set_mock_model_name_or_path(model)
    tokenizer = instantiate(cfg.llm.tokenizer)
    quant_pipeline = instantiate(cfg.quant)

    quant_model = quant_pipeline(
        model,
        tokenizer=tokenizer,
        logger=logger,
    )
    quantizer_snapshots = _snapshot_quantizers_with_parameters(quant_model)
    assert quantizer_snapshots, "expected weight-only quantization to create parameterized quantizers"
    _set_mock_model_name_or_path(quant_model)

    return quant_model, tokenizer


def _snapshot_named_parameters(model, *, include_quantizers: bool) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ("quantizer" in name) is include_quantizers
    }


def _snapshot_quantizers_with_parameters(model) -> dict[str, dict[str, torch.Tensor]]:
    snapshots = {}
    for result in ff.find_quantizers(model, "**"):
        parameters = {
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in result.module.named_parameters()
        }
        if parameters:
            snapshots[result.full_name] = parameters
    return snapshots


def _set_only_quantizer_parameters_trainable(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "quantizer" in name


def _assert_only_quantizer_parameters_trainable(model) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]

    assert trainable, "expected at least one trainable quantizer parameter"
    assert frozen, "expected non-quantizer parameters to be frozen"
    assert all("quantizer" in name for name in trainable), (
        "expected only quantizer parameters to remain trainable, "
        f"but found non-quantizer trainables: {[name for name in trainable if 'quantizer' not in name]}"
    )
    assert all("quantizer" not in name for name in frozen), (
        "expected all non-quantizer parameters to be frozen, "
        f"but found frozen quantizers: {[name for name in frozen if 'quantizer' in name]}"
    )


def _parameter_changed(before: torch.Tensor, after: torch.Tensor) -> bool:
    return bool(torch.max(torch.abs(after.detach() - before)).item() > PARAMETER_CHANGE_EPS)


def _assert_all_named_parameters_unchanged(before: dict[str, torch.Tensor], model) -> None:
    after = dict(model.named_parameters())
    changed = [name for name, before_value in before.items() if _parameter_changed(before_value, after[name])]

    assert not changed, f"expected non-quantizer parameters to stay fixed, but changed: {changed}"


def _assert_each_quantizer_changed(before: dict[str, dict[str, torch.Tensor]], model) -> None:
    after = {
        result.full_name: dict(result.module.named_parameters())
        for result in ff.find_quantizers(model, "**")
        if any(True for _ in result.module.named_parameters())
    }
    unchanged = [
        quantizer_name
        for quantizer_name, parameter_snapshots in before.items()
        if not any(
            _parameter_changed(before_value, after[quantizer_name][parameter_name])
            for parameter_name, before_value in parameter_snapshots.items()
        )
    ]

    assert not unchanged, f"expected each parameterized quantizer to change, but did not change: {unchanged}"


def _snapshot_logits(model, tokenizer) -> torch.Tensor:
    model.to("cpu")
    model.eval()
    inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt")
    with torch.no_grad():
        return model(**inputs).logits.detach().clone()


def _clone_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return deepcopy(value)


def _assert_states_equal(expected: Any, actual: Any, path: str = "state") -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected tensor, got {type(actual).__name__}"
        assert torch.equal(expected.cpu(), actual.detach().cpu()), f"{path}: tensors differ"
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert expected.keys() == actual.keys(), f"{path}: keys differ"
        for key in expected:
            _assert_states_equal(expected[key], actual[key], f"{path}.{key}")
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(expected) == len(actual), f"{path}: lengths differ"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_states_equal(expected_item, actual_item, f"{path}[{index}]")
        return

    if isinstance(expected, tuple):
        assert isinstance(actual, tuple), f"{path}: expected tuple, got {type(actual).__name__}"
        assert len(expected) == len(actual), f"{path}: lengths differ"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _assert_states_equal(expected_item, actual_item, f"{path}[{index}]")
        return

    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def _load_checkpoint_without_training(trainer, checkpoint_dir: Path) -> None:
    trainer.create_optimizer_and_scheduler(num_training_steps=trainer.args.max_steps)
    trainer._load_from_checkpoint(str(checkpoint_dir), model=trainer.model)
    trainer._load_optimizer_and_scheduler(str(checkpoint_dir))
    trainer.state = TrainerState.load_from_json(str(checkpoint_dir / "trainer_state.json"))


@pytest.mark.integration
@pytest.mark.parametrize(
    "train_name, extra_overrides",
    [
        ("ce", []),
        ("kd", ["+train.loss_type=kl"]),
    ],
)
def test_training_pipeline_runs_one_step_with_mock_data(
    tmp_path, monkeypatch, compose_cfg, mock_llm: str, train_name, extra_overrides
):
    monkeypatch.chdir(tmp_path)
    cfg = compose_cfg(
        [
            f"+llm={mock_llm}",
            f"+train={train_name}",
            "+tokenized_dataset@train.train_dataset=mock_text",
            "+train.training_args.per_device_train_batch_size=1",
            "+train.training_args.max_steps=1",
            "+train.training_args.save_steps=1",
            "+train.training_args.report_to=[]",
            "+train.training_args.use_cpu=true",
            f"train.training_args.output_dir={tmp_path}",
            "+paths=default",
            *extra_overrides,
        ]
    )

    _train_once(cfg, tmp_path)


@pytest.mark.integration
@pytest.mark.parametrize("train_name, extra_overrides", CHECKPOINT_TRAINERS)
def test_trainer_checkpoint_save_load_restores_model_and_training_state(
    tmp_path, monkeypatch, compose_cfg, train_name, extra_overrides
):
    monkeypatch.chdir(tmp_path)
    cfg = _compose_checkpoint_training_cfg(compose_cfg, tmp_path, train_name, extra_overrides)
    model, tokenizer, trainer = _make_checkpoint_trainer(cfg)

    result = trainer.train()
    checkpoint_dir = tmp_path / "checkpoint-1"

    assert trainer.state.global_step == 1
    assert math.isfinite(result.training_loss)
    assert checkpoint_dir.is_dir()

    expected_parameters = _snapshot_named_parameters(model, include_quantizers=False)
    expected_logits = _snapshot_logits(model, tokenizer)
    expected_optimizer_state = _clone_state(trainer.optimizer.state_dict())
    expected_scheduler_state = _clone_state(trainer.lr_scheduler.state_dict())
    expected_global_step = trainer.state.global_step

    loaded_model, loaded_tokenizer, loaded_trainer = _make_checkpoint_trainer(cfg)
    _load_checkpoint_without_training(loaded_trainer, checkpoint_dir)

    _assert_states_equal(expected_parameters, _snapshot_named_parameters(loaded_model, include_quantizers=False))
    assert torch.equal(expected_logits, _snapshot_logits(loaded_model, loaded_tokenizer))
    _assert_states_equal(expected_optimizer_state, _clone_state(loaded_trainer.optimizer.state_dict()))
    _assert_states_equal(expected_scheduler_state, _clone_state(loaded_trainer.lr_scheduler.state_dict()))
    assert loaded_trainer.state.global_step == expected_global_step


@pytest.mark.integration
@pytest.mark.parametrize(
    "train_name, extra_overrides",
    [
        ("ce", []),
        ("kd", ["+train.loss_type=kl"]),
    ],
)
def test_single_model_quantized_trainers_update_quantizers_without_changing_base_weights(
    tmp_path, monkeypatch, compose_cfg, train_name, extra_overrides
):
    monkeypatch.chdir(tmp_path)
    cfg = _compose_quantized_training_cfg(compose_cfg, tmp_path, train_name, extra_overrides)
    model, tokenizer = _quantize_model_for_training(cfg, logging.getLogger("test"))
    _set_only_quantizer_parameters_trainable(model)
    _assert_only_quantizer_parameters_trainable(model)

    base_parameter_snapshots = _snapshot_named_parameters(model, include_quantizers=False)
    quantizer_snapshots = _snapshot_quantizers_with_parameters(model)

    train = instantiate(cfg.train)
    with ff.strict_quantization(False):
        train(model=model, tokenizer=tokenizer)

    _assert_each_quantizer_changed(quantizer_snapshots, model)
    _assert_all_named_parameters_unchanged(base_parameter_snapshots, model)
