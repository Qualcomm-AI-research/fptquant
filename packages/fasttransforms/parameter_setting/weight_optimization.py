# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
from collections import defaultdict
from typing import Iterator

import torch
from tqdm.auto import tqdm

from fasttransforms.nn.operation import TransformOperation
from fasttransforms.nn.transforms.common import Transform

from .common import WeightParameterEstimator


def _iter_transform_operations(module: torch.nn.Module) -> Iterator[TransformOperation]:
    seen: set[int] = set()
    for child in module.modules():
        if isinstance(child, TransformOperation) and id(child) not in seen:
            seen.add(id(child))
            yield child


def _iter_transforms(module: torch.nn.Module) -> Iterator[Transform]:
    seen: set[int] = set()
    for child in module.modules():
        if isinstance(child, Transform) and id(child) not in seen:
            seen.add(id(child))
            yield child


def _weight_from_parametrization(operation: TransformOperation) -> torch.Tensor:
    module = operation.attached_to
    tensor_name = operation.parametrized_tensor
    if module is None or tensor_name is None:
        raise ValueError("Weight optimization can only use attached parametrization operations.")

    # Plain attribute access triggers PyTorch's parametrization property, which
    # replays the registered transform stack. We rely on the built-in logic here
    # rather than reconstructing it. This backpropagates into the underlying
    # weight (`.original`) leaf, so callers must freeze those leaves for the
    # duration of the optimization -- see `_parametrization_leaves`. Otherwise
    # every leaf allocates a full-size grad buffer that persists for the whole
    # run (a second copy of most of the model's weights on large models).
    return getattr(module, tensor_name)


def _parametrization_leaves(operations: list[TransformOperation]) -> list[torch.nn.Parameter]:
    """Return the underlying `.original` weight leaves behind `operations`.

    These are ordinary model parameters that the Lp loss backpropagates through.
    They are not being optimized here, so they are frozen while the transform
    parameters are optimized to avoid retaining their gradient buffers.
    """
    leaves: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for operation in operations:
        module = operation.attached_to
        tensor_name = operation.parametrized_tensor
        if module is None or tensor_name is None:
            continue
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None or tensor_name not in parametrizations:
            continue
        leaf = parametrizations[tensor_name].original
        if id(leaf) not in seen:
            seen.add(id(leaf))
            leaves.append(leaf)
    return leaves


def _lp_norm(tensor: torch.Tensor, p: float, chunk_size: int = 1_048_576, eps: float = 1e-12) -> torch.Tensor:
    flat_tensor = tensor.reshape(-1)
    if flat_tensor.numel() == 0:
        return flat_tensor.float().sum()

    total: torch.Tensor | None = None
    for chunk in flat_tensor.split(chunk_size):
        chunk_sum = chunk.float().abs().pow(p).sum()
        total = chunk_sum if total is None else total + chunk_sum

    assert total is not None
    # `eps` keeps the outer root away from zero: `pow(1/p)` has an infinite derivative at 0, so
    # an all-zero tensor would otherwise produce a NaN gradient that silently poisons the
    # transform parameters and propagates into range setting.
    return (total / flat_tensor.numel() + eps).pow(1.0 / p)


class WeightLpEstimator(WeightParameterEstimator):
    """Optimize transform parameters from parametrized weights only.

    This mirrors the old transformations branch local weight initializer: for each
    trainable transform, minimize the L_p norm of the weights after the currently
    attached transform parametrization stack. No calibration batches or model
    forwards are used.
    """

    def __init__(
        self,
        p: float = 4.0,
        lr: float = 1e-2,
        num_steps: int = 200,
        max_ops_per_backward: int | None = 16,
    ) -> None:
        self.p = p
        self.lr = lr
        self.num_steps = num_steps
        self.max_ops_per_backward = max_ops_per_backward

    def estimate(
        self,
        model: torch.nn.Module,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        if logger is None:
            logger = logging.getLogger(__name__)

        # Both transform operations (registered as parametrizations or hooks) and
        # their child `Transform` modules are reachable by walking the model, so we
        # derive them here rather than threading them through the call site.
        weight_ops = [
            operation for operation in _iter_transform_operations(model) if operation.parametrized_tensor == "weight"
        ]
        logger.info(
            "Starting Weight Lp transform optimization: weight_ops=%d p=%s lr=%s num_steps=%d",
            len(weight_ops),
            self.p,
            self.lr,
            self.num_steps,
        )

        # Initialize lazy transform parameters before constructing optimizers.
        for operation in tqdm(
            weight_ops,
            desc="WeightLp init",
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            with torch.no_grad():
                _weight_from_parametrization(operation)

        ops_by_transform: dict[int, list[TransformOperation]] = defaultdict(list)
        for operation in weight_ops:
            ops_by_transform[id(operation.transform)].append(operation)

        optimized = 0
        skipped = 0
        transforms = list(_iter_transforms(model))
        logger.info("Weight Lp transform optimization will inspect %d transforms", len(transforms))
        for transform in tqdm(
            transforms,
            desc="WeightLp transforms",
            disable=not logger.isEnabledFor(logging.INFO),
        ):
            params = [parameter for parameter in transform.parameters() if parameter.requires_grad]
            if not params:
                skipped += 1
                continue

            operations = ops_by_transform.get(id(transform), [])
            if not operations:
                logger.info("Skipping weight Lp optimization for %s: no weight parametrization operations", transform)
                skipped += 1
                continue

            # The Lp objective only counterbalances when a transform has both a forward and an
            # inverse leg on weights: shrinking one grows the other. When every weight leg is an
            # inverse, the paired forward leg lives on an activation hook that contributes nothing
            # to this loss, so the objective is unbounded -- it drives the weight to zero by
            # inflating the transform, relocating the magnitude onto the activations (where a
            # static activation quantizer then range-sets on an inflated tensor).
            if all(operation.inv for operation in operations):
                logger.warning(
                    "Skipping weight Lp optimization for %s: all %d weight operations are "
                    "inverse-only, so the Lp objective is unbounded (it shrinks weights by "
                    "inflating the transform). The counterbalancing leg is an activation hook, "
                    "which this estimator does not observe.",
                    transform.__class__.__name__,
                    len(operations),
                )
                skipped += 1
                continue

            logger.info(
                "Optimizing %s with %d parameter tensors and %d weight operations",
                transform.__class__.__name__,
                len(params),
                len(operations),
            )
            optimizer = torch.optim.AdamW(params, lr=self.lr)
            # The Lp loss backpropagates through the `.original` weight leaves, which are not
            # being optimized here. Freeze them for the duration so they do not each retain a
            # full-size gradient buffer for every step of the optimization.
            frozen_leaves = [(leaf, leaf.requires_grad) for leaf in _parametrization_leaves(operations)]
            for leaf, _ in frozen_leaves:
                leaf.requires_grad_(False)
            try:
                # Split the weight operations into groups that are each backpropagated
                # separately. Because the loss is a plain sum over operations, accumulating
                # gradients from per-group `backward()` calls is mathematically identical to
                # one `backward()` over the whole sum -- but only one group's transformed
                # weights are retained at a time.
                #
                # This matters for shared transforms. R1 (`per_layer=False`) spans all seven
                # projections of every layer, so a single graph over all of its operations
                # materializes one transformed copy of nearly the whole model: ~26 GiB in
                # float32 for Llama-3-8B (224 operations), which does not fit alongside the
                # resident model on one GPU. Per-layer transforms have only a handful of
                # operations and are unaffected either way.
                groups: list[list[TransformOperation]]
                if self.max_ops_per_backward is None or self.max_ops_per_backward >= len(operations):
                    groups = [operations]
                else:
                    step = self.max_ops_per_backward
                    groups = [operations[i : i + step] for i in range(0, len(operations), step)]

                for _ in tqdm(
                    range(self.num_steps),
                    desc=f"WeightLp {transform.__class__.__name__}",
                    leave=False,
                    disable=not logger.isEnabledFor(logging.INFO),
                ):
                    optimizer.zero_grad()
                    stepped = False
                    non_finite = False
                    for group in groups:
                        # `cached()` computes the applied transform once and reuses it across
                        # this group's weight operations, instead of replaying the whole stack
                        # per operation. The cache is autograd-aware, so gradients are
                        # equivalent (verified: bitwise identical at the default group size;
                        # only float32 reassociation differs, ~2e-9, at group size 1).
                        loss = None
                        with torch.nn.utils.parametrize.cached():
                            for operation in group:
                                operation_loss = _lp_norm(_weight_from_parametrization(operation), self.p)
                                loss = operation_loss if loss is None else loss + operation_loss

                        if loss is None:
                            continue
                        if not torch.isfinite(loss):
                            non_finite = True
                            break
                        # Accumulate into the transform parameters' `.grad` and release this
                        # group's graph before building the next one.
                        loss.backward()
                        stepped = True

                    if non_finite:
                        logger.warning(
                            "Stopping weight Lp optimization for %s: non-finite loss encountered; "
                            "leaving the transform parameters at their last finite values.",
                            transform.__class__.__name__,
                        )
                        optimizer.zero_grad()
                        break
                    if not stepped:
                        break
                    optimizer.step()
            finally:
                for leaf, was_requiring_grad in frozen_leaves:
                    leaf.requires_grad_(was_requiring_grad)

            optimized += 1

        # Clear residual grads on the transform parameters: the last transform's loop leaves
        # bystander grads behind, and nothing should leak into a downstream optimizer. The
        # `.original` weight leaves are frozen during optimization (see above), so they no
        # longer accumulate grads that need clearing here.
        for parameter in model.parameters():
            parameter.grad = None

        logger.info(
            "Weight Lp transform optimization complete: optimized=%d skipped=%d p=%s lr=%s num_steps=%d",
            optimized,
            skipped,
            self.p,
            self.lr,
            self.num_steps,
        )
