# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from __future__ import annotations

from collections.abc import Iterable

import pytest
from omegaconf import DictConfig

AUTO_COMPOSE_GROUPS = (
    "data_processing",
    "evaluation",
    "train",
)

EXTRA_OVERRIDES = {
    ("train", "kd"): ["+train.loss_type=kl"],
}


def _registered_configs(groups: Iterable[str]) -> list[tuple[str, str]]:
    """Return registered hydra-zen store entries for a narrow test-only subset."""
    from hydra_zen import store

    repo = getattr(store, "_internal_repo", None)
    if not isinstance(repo, dict):
        pytest.fail(
            "hydra-zen store discovery needs updating: expected store._internal_repo "
            f"to be a dict, got {type(repo).__name__}."
        )

    selected_groups = set(groups)
    names_by_group: dict[str, list[str]] = {group: [] for group in selected_groups}
    for key in repo:
        if not isinstance(key, tuple) or len(key) != 2 or not isinstance(key[0], str) or not isinstance(key[1], str):
            pytest.fail(
                "hydra-zen store discovery needs updating: expected store keys to be "
                f"(group, name) string tuples, got {key!r}."
            )

        group, name = key
        if group in selected_groups:
            names_by_group[group].append(name)

    missing_groups = sorted(group for group, names in names_by_group.items() if not names)
    if missing_groups:
        pytest.fail(f"No registered configs found for expected groups: {missing_groups}.")

    return [(group, name) for group in AUTO_COMPOSE_GROUPS for name in sorted(names_by_group[group])]


def _compose_overrides(group: str, name: str) -> list[str]:
    overrides = [f"+{group}={name}"]
    if group != "paths":
        overrides.append("+paths=default")
    overrides.extend(EXTRA_OVERRIDES.get((group, name), []))
    return overrides


def _assert_group_selected(cfg: DictConfig, group: str) -> None:
    assert group in cfg


def test_registered_predictable_group_configs_compose(lm_tools_configs, compose_cfg):
    for group, name in _registered_configs(AUTO_COMPOSE_GROUPS):
        overrides = _compose_overrides(group, name)
        try:
            cfg = compose_cfg(overrides)
        except Exception as exc:
            pytest.fail(f"Failed to compose {group}={name} with overrides {overrides}: {exc}")

        _assert_group_selected(cfg, group)


@pytest.mark.parametrize(
    ("override", "group"),
    [
        ("+llm=mock_llama", "llm"),
        ("+llm=mock_qwen35", "llm"),
        ("+raw_dataset=mock_text", "raw_dataset"),
        ("+tokenized_dataset=mock_text", "tokenized_dataset"),
    ],
)
def test_mock_resource_configs_compose(compose_cfg, override: str, group: str):
    cfg = compose_cfg([override, "+paths=default"])

    _assert_group_selected(cfg, group)
