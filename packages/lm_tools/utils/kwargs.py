# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import inspect
from typing import Any, Callable, Dict


def extract_required_kwargs(func: Callable, **kwargs) -> Dict[str, Any]:
    parameters = inspect.signature(func).parameters

    # If the function/class has **kwargs in the signature, pass everything
    for parameter in parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    # Otherwise pass only the ones in the signature
    required_kwargs = {}
    for name, value in kwargs.items():
        if name in parameters:
            required_kwargs[name] = value
    return required_kwargs
