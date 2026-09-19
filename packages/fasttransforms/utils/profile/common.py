# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import contextlib
from typing import Any, ContextManager, List, Tuple

import pandas as pd
import torch


class Profiler:
    def prepare(self, model: torch.nn.Module) -> Tuple[List[ContextManager], Any]:
        raise NotImplementedError()

    def cleanup(self, metadata: Any):
        raise NotImplementedError()

    @property
    def stats(self) -> pd.DataFrame:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    @contextlib.contextmanager
    def __call__(self, model):
        contexts, metadata = self.prepare(model)
        try:
            with contextlib.ExitStack() as stack:
                # Enter each context manager and optionally capture its __enter__ result
                for cm in contexts:
                    stack.enter_context(cm)
                yield
        finally:
            self.cleanup(metadata)
