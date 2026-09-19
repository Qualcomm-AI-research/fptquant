# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Hydra helpers shared across tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest


@pytest.fixture(autouse=True)
def hydra_clean():
    """Clear Hydra global state before and after each test."""
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def compose(
    overrides: list[str],
    *,
    version_base: str = "1.3",
    config_name: str | None = None,
):
    """Compose a Hydra config after flushing hydra-zen's dynamic store."""
    from hydra import compose as hydra_compose
    from hydra import initialize
    from hydra_zen import store

    store.add_to_hydra_store()
    with initialize(version_base=version_base):
        return hydra_compose(config_name=config_name, overrides=overrides)


@pytest.fixture
def compose_cfg() -> Callable[[list[str]], object]:
    """Return a convenience callable for repeated in-test Hydra composition."""

    def _compose(overrides: list[str]):
        return compose(overrides)

    return _compose
