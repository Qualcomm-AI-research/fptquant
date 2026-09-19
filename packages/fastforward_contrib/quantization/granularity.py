# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from typing import Any, Iterable

import torch
from fastforward import PerChannel
from fastforward.quantization.granularity import Granularity

logger = logging.getLogger(__name__)


class PerBlock(Granularity):
    def __init__(
        self,
        block_dim: int | list[int] | None,
        block_size: int | list[int] | None,
        per_channel_dims: int | list[int] | None = None,
    ):
        self.block_dim = block_dim
        self.block_size = block_size
        self.per_channel_dims = []
        if per_channel_dims is not None:
            self.per_channel_dims = per_channel_dims if isinstance(per_channel_dims, Iterable) else [per_channel_dims]

    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        tile_size = list(data_shape)
        if self.per_channel_dims is not None:
            for dim in self.per_channel_dims:
                tile_size[dim] = 1

        if self.block_dim is not None and self.block_size is not None:
            block_dims = [self.block_dim] if isinstance(self.block_dim, int) else self.block_dim
            block_sizes = [self.block_size] if isinstance(self.block_size, int) else self.block_size
            for block_dim, block_size in zip(block_dims, block_sizes):
                if block_size > data_shape[block_dim]:
                    logging.warning(
                        f"Can't apply per block quantization using block-size={block_size}"
                        f" over dimension {block_dim} for a tensor with shape {data_shape}. "
                        f"block-dim={data_shape[block_dim]} will be used instead"
                    )
                else:
                    tile_size[block_dim] = block_size

        tile_size = torch.Size(tile_size)
        return tile_size

    def repr_args(self) -> dict[str, Any]:
        return {"block_dim": self.block_dim, "block_size": self.block_size, "per_channel_dims": self.per_channel_dims}


class PerChannelSelfGuidance(PerChannel):
    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        # self guidance use the first BS elements for noise_pred_unconditioned
        # and the next BS elements for noise_pred_text, so the actual BS is 2 times BS.
        # Here we want to create two groups with different quantization parameters:
        #   - A first one of size BS for noise_pred_unconditioned elements.
        #   - A second one of size BS for noise_pred_text elements.
        new_data_shape = [s for s in data_shape]
        new_data_shape[0] //= 2
        data_shape = torch.Size(new_data_shape)
        return super().tile_size(data_shape)


class PerToken(Granularity):
    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        """Return the tile size for per-token quantization

        Args:
            data_shape: The shape of the input data.

        Returns:
            The tile size for per-token quantization (scale shared on last dimension).
        """
        tile_size = [1] * len(data_shape)
        tile_size[-1] = data_shape[-1]

        return torch.Size(tile_size)

    def repr_args(self) -> dict[str, Any]:
        return {}


class BlockPerToken(Granularity):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def tile_size(self, data_shape: torch.Size) -> torch.Size:
        """Return the tile size for per-token quantization

        Args:
            data_shape: The shape of the input data.

        Returns:
            The tile size for per-token quantization (one scale for each dimension with exception of the last).
        """
        tile_size = [1] * len(data_shape)
        tile_size[-1] = self.block_size

        return torch.Size(tile_size)

    def repr_args(self) -> dict[str, Any]:
        return {"block_size": self.block_size}
