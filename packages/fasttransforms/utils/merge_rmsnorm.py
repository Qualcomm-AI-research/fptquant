# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
from fastforward import mpath

RMSNORM_MERGE_OPERATIONS = {
    "&attn_norm": ("&q_proj", "&k_proj", "&v_proj"),
    "&mlp_norm": ("&up_proj", "&gate_proj"),
}


def remove_base_from_pat(pat, pat_to_remove, aliases):
    pat = mpath._parser.parse(pat, aliases=aliases)
    # Ensure we remove the layer part from the root, as we want to explicitly select just a single layer
    return str(pat).replace(str(mpath.query(pat_to_remove, aliases=aliases)) + "/", "")


def get_pat_for_layer(pat, i, aliases):
    # This is a patch
    subpath = mpath.query("/".join(remove_base_from_pat(pat, "&decoder", aliases).split("/")[2:]))
    return mpath.query(f"model/layers/{i}").extends(subpath)


def merge_rmsnorm(model, merge_head=False):
    aliases = model.aliases
    for i in range(len(model.model.layers)):
        for rms_pat, linear_target_pat in RMSNORM_MERGE_OPERATIONS.items():
            rms_pat = get_pat_for_layer(rms_pat, i, aliases)
            rms_op = mpath.search(rms_pat, model, aliases=aliases)[0].module
            rms_weight = rms_op.weight.data
            for target in linear_target_pat:
                target = get_pat_for_layer(target, i, aliases)
                lm_head = mpath.search(target, model, aliases=aliases)[0].module
                # merge rms weight into input_channels of linear
                lm_head.weight.data = lm_head.weight.data * rms_weight

            # remove rms weight from rms op by setting to scaler 1
            rms_op.weight.data.fill_(1.0)

    # merge last RMSnorm too. This is a separate option only for backwards compatibility
    if merge_head:
        rms_pat = "&norm"
        target = "&lm_head"
        rms_op = mpath.search(rms_pat, model, aliases=aliases)[0].module
        rms_weight = rms_op.weight.data
        lm_head = mpath.search(target, model, aliases=aliases)[0].module
        # merge rms weight into input_channels of linear
        if model.config.tie_word_embeddings:
            model.config.tie_word_embeddings = False
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data * rms_weight, requires_grad=False)
        else:
            lm_head.weight.data = lm_head.weight.data * rms_weight
        # remove rms weight from rms op by setting to scaler 1
        rms_op.weight.data.fill_(1.0)
