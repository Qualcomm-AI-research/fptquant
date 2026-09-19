# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from typing import Callable

import fastforward as ff
import torch


class LPNormRangeEstimator(ff.range_setting.min_error.MinErrorGridRangeEstimator):
    def __init__(
        self,
        p: float = 2.0,
        num_candidates: int = 100,
        search_grid_generator: ff.range_setting.min_error._SearchGridGenerator = ff.range_setting.min_error._UniformSearchGrid(),
        update_range_policy: Callable[["ff.range_setting.min_error._MinAvgErrorGridEstimator", int], bool]
        | None = None,
        skip_unsupported_quantizers: bool = False,
    ):
        super().__init__(
            error_fn=lambda x, y: torch.sum((x - y).abs() ** p, dim=1),
            num_candidates=num_candidates,
            search_grid_generator=search_grid_generator,
            update_range_policy=update_range_policy,
            skip_unsupported_quantizers=skip_unsupported_quantizers,
        )


lp_norm = LPNormRangeEstimator
