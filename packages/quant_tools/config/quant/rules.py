# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import fastforward as ff
from hydra_zen import builds, store

import fastforward_contrib as ffc
from quant_tools.quantizer_initialization import QueryQuantizationRule

rule_store = store(group="quant/rules", provider=__name__)

rule_store(
    {
        "linear_weights": builds(
            QueryQuantizationRule,
            query="&linear_weights_decoder",
            quantizer=builds(
                ff.nn.LinearQuantizer.factory,
                num_bits=4,
                granularity=builds(ff.PerChannel, channel_dim=0),
                symmetric=True,
            ),
            overwrite_policy="error",
        ),
    },
    name="w_example",
)

rule_store(
    {
        "linear_activations": builds(
            QueryQuantizationRule,
            query="{&na, &nm, {&o_proj, &down_proj}/input_quantizer}",
            quantizer=builds(
                ff.nn.DynamicLinearQuantizer.factory,
                num_bits=4,
                granularity=builds(ffc.PerToken),
            ),
            overwrite_policy="error",
        ),
    },
    name="a_example",
)

rule_store(
    {
        "kv_cache": builds(
            QueryQuantizationRule,
            query="{&ke, &ve}",
            quantizer=builds(
                ff.nn.DynamicLinearQuantizer.factory,
                num_bits=4,
                granularity=builds(ffc.PerToken),
            ),
            overwrite_policy="error",
        ),
    },
    name="kv_example",
)
