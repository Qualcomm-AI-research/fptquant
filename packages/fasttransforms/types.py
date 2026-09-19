# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from enum import Enum


class StrEnum(str, Enum):
    def __new__(cls, value):
        # Ensure the member is a real `str` object
        if not isinstance(value, str):
            value = str(value)
        obj = str.__new__(cls, value)  # build the `str` instance
        obj._value_ = value  # attach the enum value
        return obj

    def __str__(self):
        # Return the underlying string value
        return self._value_

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        # Make `auto()` use the member name as the value
        return name
