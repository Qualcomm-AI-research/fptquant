# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


def add_config_to_store():
    from fasttransforms.config.pipeline import quant_store
    from fasttransforms.config.quant.parameter_setting import parameter_setting_store
    from fasttransforms.config.transform import transform_func_store

    transform_func_store.add_to_hydra_store()
    parameter_setting_store.add_to_hydra_store()
    quant_store.add_to_hydra_store()


__all__ = ["add_config_to_store"]
