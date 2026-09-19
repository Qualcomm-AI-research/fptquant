# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

_CONFIG_ADDED = False


def add_config_to_store():
    global _CONFIG_ADDED
    if _CONFIG_ADDED:
        return

    from .quant import dequant_store, quant_store, range_estimator_store, rule_store

    rule_store.add_to_hydra_store()
    range_estimator_store.add_to_hydra_store()
    quant_store.add_to_hydra_store()
    dequant_store.add_to_hydra_store()
    _CONFIG_ADDED = True
