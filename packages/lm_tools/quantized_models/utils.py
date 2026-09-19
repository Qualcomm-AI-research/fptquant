# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from abc import ABCMeta, abstractmethod

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    return torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep)


class FastActCachingLLMMixin(metaclass=ABCMeta):
    """
    A mixin class that provides unified access to different parts of LLM network
    (used for faster data caching).
    """

    @abstractmethod
    def get_embeddings(self):
        """
        Returns
        -------
        embeddings: iterable
            Embeddings and other parameters before the attention blocks.
        """
        pass

    @abstractmethod
    def get_attention_blocks(self):
        """
        Returns
        -------
        blocks: iterable
            Attention blocks
        """
        pass

    @abstractmethod
    def get_lm_head(self):
        """
        Returns
        -------
        lm_head: nn.Module
            LM head
        """
        pass

    @abstractmethod
    def apply_block(self, block, hidden_states, **kw):
        """
        Apply the attention `block` to the `hidden_states`.

        We have this function in case we need to pass some extra arguments (`kw`) to the
        `block(hidden_states)` or perform other model-specific manipulations before/after.
        """
        pass
