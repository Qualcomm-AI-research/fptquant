# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from fastforward import mpath


def remove_base_from_pat(pat, pat_to_remove, aliases):
    pat = mpath._parser.parse(pat, aliases=aliases)
    # Ensure we remove the layer part from the root, as we want to explicitly select just a single layer
    return mpath.query(str(pat).replace(str(mpath.query(pat_to_remove, aliases=aliases)) + "/", ""))


def get_pat_for_layer(pat, i, aliases):
    return mpath.query(f"model/layers/{i}").extends(remove_base_from_pat(pat, "&decoder", aliases))
