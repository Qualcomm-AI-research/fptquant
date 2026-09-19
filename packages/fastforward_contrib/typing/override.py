# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import sys

if sys.version_info >= (3, 12):
    from typing import override as override
else:
    from overrides import override as override
