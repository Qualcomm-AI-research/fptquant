# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
import pytest
import torch

from lm_tools.evaluation._base import Evaluation
from lm_tools.train._base import BaseTraining, _patch_incompatible_optimizer_resume, filter_training_args_for_init
from lm_tools.train.evaluation_callback import EvaluationCallback


class _DummyAccelerator:
    is_main_process = True

    def prepare(self, obj):
        return obj

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


class _DummyTrainer:
    def __init__(self, model: torch.nn.Module, output_dir: Path):
        self.model = model
        self.args = SimpleNamespace(output_dir=str(output_dir))
        self.accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
        self.resume_from_checkpoint = None

    def train(self, resume_from_checkpoint=None):
        self.resume_from_checkpoint = resume_from_checkpoint


class _DummyTraining(BaseTraining):
    def __init__(self, *, output_dir: Path, train_dataset, max_steps: int = 2, eval_callback=None):
        super().__init__(
            training_args={"output_dir": str(output_dir), "max_steps": max_steps, "push_to_hub": False},
            train_dataset=train_dataset,
            eval_dataset=None,
            resume_from_checkpoint=True,
            eval_callback=eval_callback,
        )
        self.make_trainer_calls = 0
        self.last_trainer: Optional[_DummyTrainer] = None
        self.callbacks_seen = None

    def make_trainer(self, model, train_dataset, eval_dataset):
        self.make_trainer_calls += 1
        self.callbacks_seen = self.callbacks
        self.last_trainer = _DummyTrainer(model, Path(self.training_args["output_dir"]))
        return self.last_trainer


def _write_checkpoint(checkpoint_dir: Path, *, global_step: int, model: torch.nn.Module) -> None:
    checkpoint_dir.mkdir(parents=True)
    with open(checkpoint_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step}, f)
    torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")


@pytest.fixture(autouse=True)
def _dummy_accelerator(monkeypatch):
    monkeypatch.setattr("accelerate.Accelerator", _DummyAccelerator)


def test_completed_checkpoint_skips_dataset_and_trainer_construction(tmp_path: Path) -> None:
    """A checkpoint already at max_steps should be loaded before dataset or trainer allocation."""
    output_dir = tmp_path / "run"
    checkpoint_model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        checkpoint_model.weight.fill_(7.0)
        checkpoint_model.bias.fill_(3.0)
    _write_checkpoint(output_dir / "checkpoint-2", global_step=2, model=checkpoint_model)

    def train_dataset(tokenizer):
        raise AssertionError("train dataset should not be materialized for a completed checkpoint")

    model = torch.nn.Linear(1, 1)
    training = _DummyTraining(output_dir=output_dir, train_dataset=train_dataset, max_steps=2)
    training(model=model, tokenizer=None)

    assert training.make_trainer_calls == 0
    torch.testing.assert_close(model.weight, checkpoint_model.weight)
    torch.testing.assert_close(model.bias, checkpoint_model.bias)


def test_incomplete_checkpoint_still_constructs_trainer_and_resumes(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    checkpoint_model = torch.nn.Linear(1, 1)
    _write_checkpoint(output_dir / "checkpoint-1", global_step=1, model=checkpoint_model)

    dataset_calls = 0

    def train_dataset(tokenizer):
        nonlocal dataset_calls
        dataset_calls += 1
        return object()

    model = torch.nn.Linear(1, 1)
    training = _DummyTraining(output_dir=output_dir, train_dataset=train_dataset, max_steps=2)
    training(model=model, tokenizer=None)

    assert dataset_calls == 1
    assert training.make_trainer_calls == 1
    assert training.last_trainer is not None
    assert training.last_trainer.resume_from_checkpoint == str(output_dir / "checkpoint-1")


def test_filter_training_args_drops_transformers_unsupported_dataset_keys() -> None:
    from transformers import TrainingArguments

    filtered_args = filter_training_args_for_init(
        {
            "output_dir": "out",
            "max_steps": 1,
            "max_length": 2048,
            "dataset_kwargs": {"skip_prepare_dataset": True},
        },
        TrainingArguments,
    )

    assert filtered_args == {"output_dir": "out", "max_steps": 1}
    TrainingArguments(**filtered_args)


def test_incompatible_optimizer_state_is_skipped() -> None:
    class TrainerWithBadOptimizer:
        def __init__(self) -> None:
            self.calls = 0

        def _load_optimizer_and_scheduler(self, checkpoint):
            self.calls += 1
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
            )

    trainer = TrainerWithBadOptimizer()
    _patch_incompatible_optimizer_resume(trainer)

    trainer._load_optimizer_and_scheduler("checkpoint-896")

    assert trainer.calls == 1


def test_unrelated_optimizer_load_value_error_is_not_swallowed() -> None:
    class TrainerWithOtherError:
        def _load_optimizer_and_scheduler(self, checkpoint):
            raise ValueError("some other optimizer failure")

    trainer = TrainerWithOtherError()
    _patch_incompatible_optimizer_resume(trainer)

    with pytest.raises(ValueError, match="some other optimizer failure"):
        trainer._load_optimizer_and_scheduler("checkpoint-896")


def test_eval_callback_is_attached_and_receives_tokenizer(tmp_path: Path) -> None:
    class DummyCallback:
        def __init__(self) -> None:
            self.tokenizer = None

        def set_tokenizer(self, tokenizer):
            self.tokenizer = tokenizer

    callback = DummyCallback()
    tokenizer = object()
    training = _DummyTraining(
        output_dir=tmp_path / "run", train_dataset=lambda tokenizer: object(), eval_callback=callback
    )

    training(model=torch.nn.Linear(1, 1), tokenizer=tokenizer)

    assert callback.tokenizer is tokenizer
    assert training.callbacks_seen == [callback]


def test_evaluation_callback_runs_on_save_and_records_results(tmp_path: Path) -> None:
    class DummyEvaluation(Evaluation):
        def evaluate(self, model: Any, tokenizer: Any) -> pd.DataFrame:
            assert not model.training
            assert tokenizer == "tokenizer"
            return pd.DataFrame([{"quantity": "perplexity", "value": 12.25, "stat": "mean"}])

    model = torch.nn.Linear(1, 1)
    model.train()
    callback = EvaluationCallback(evaluation=DummyEvaluation())
    callback.set_tokenizer("tokenizer")
    args = SimpleNamespace(output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=128, log_history=[])
    control = SimpleNamespace()

    assert callback.on_save(args, state, control, model=model) is control

    result_path = tmp_path / "eval_callback" / "step-128.csv"
    assert result_path.exists()
    assert pd.read_csv(result_path).iloc[0]["value"] == 12.25
    assert state.log_history == [{"step": 128, "eval_callback/perplexity": 12.25}]
    assert model.training
