# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import store

from dev_tools.testing.hydra import compose, hydra_clean


def test_compose_adds_store_entries_and_returns_cfg():
    demo_store = store(group="testing_demo")
    demo_store({"answer": 42}, name="cfg")

    cfg = compose(["+testing_demo=cfg"])

    assert cfg.testing_demo.answer == 42


def test_hydra_clean_clears_before_and_after():
    with initialize(version_base="1.3"):
        assert GlobalHydra.instance().is_initialized()

    cleanup = hydra_clean.__wrapped__()
    next(cleanup)
    assert not GlobalHydra.instance().is_initialized()

    with initialize(version_base="1.3"):
        assert GlobalHydra.instance().is_initialized()

    try:
        next(cleanup)
    except StopIteration:
        pass

    assert not GlobalHydra.instance().is_initialized()
