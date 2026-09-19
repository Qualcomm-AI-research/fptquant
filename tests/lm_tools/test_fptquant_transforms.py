# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging

import fastforward as ff
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate, store

import fasttransforms as ft
import fasttransforms.nn.transforms as T
import lm_tools
from fasttransforms.nn.collections import LinearParametrizationConfig, add_transform_operations
from fasttransforms.nn.transforms.flatquant import InvertibleKroneckerMatrixTransformWithDiagonal
from fasttransforms.nn.transforms.matrices import InvertibleMatrixInitType
from fasttransforms.parameter_setting import WeightLpEstimator
from quant_tools import get_quant_model

TRANSFORM_CASES = [
    ("R1", ["transform@quant.transforms.R1.transform=rotation"]),
    ("MM", ["transform@quant.transforms.MM.transform=hadamard"]),
    ("Pu", []),
    ("Pv", []),
    ("PDR", []),
    ("PreRoPE", []),
    ("Po_paper", []),
]

NO_QUANT_TRANSFORM_TOLERANCE = dict(atol=6e-3, rtol=6e-3)


@pytest.mark.unit
def test_invertible_matrix_inverse_eps_matches_old_direct_identity_damping():
    transform = T.InvertibleMatrixTransform(init_type=InvertibleMatrixInitType.identity, inverse_eps=1e-5)
    data = torch.eye(4)

    actual = transform(data, inv=True)
    expected = torch.eye(4) / (1.0 + 1e-5)

    torch.testing.assert_close(actual, expected)


@pytest.mark.unit
def test_invertible_matrix_can_use_old_direct_float32_inverse():
    transform = T.InvertibleMatrixTransform(
        init_type=InvertibleMatrixInitType.identity,
        inverse_eps=1e-5,
        inverse_compute_dtype="float32",
    )
    data = torch.eye(4)
    _ = transform(data)
    matrix = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.1, 1.0, 0.3, 0.0],
            [0.0, -0.2, 1.0, 0.4],
            [0.0, 0.0, 0.1, 1.0],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        transform.M.copy_(matrix)

    expected = torch.linalg.inv(matrix + 1e-5 * torch.eye(4, dtype=torch.float32))

    torch.testing.assert_close(transform(data, inv=True), expected)


@pytest.mark.unit
def test_flatquant_kronecker_diagonal_inverse_eps_matches_old_direct_diagonal():
    transform = InvertibleKroneckerMatrixTransformWithDiagonal(
        n_blocks=2,
        kron_init_type=InvertibleMatrixInitType.identity,
        diagonal_inverse_eps=1e-6,
    )
    data = torch.eye(4)
    _ = transform(data)

    expected = data / (1.0 + 1e-6)

    torch.testing.assert_close(transform(data, inv=True), expected)


def _register_configs() -> None:
    lm_tools.add_config_to_store()
    ft.add_config_to_store()
    store.add_to_hydra_store()


def _compose_transform(transform_name: str, overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=None):
        return compose(
            config_name=None, overrides=["+llm=mock_llama", f"+quant/transforms={transform_name}", *overrides]
        )


@pytest.mark.integration
@pytest.mark.parametrize(("transform_name", "overrides"), TRANSFORM_CASES)
def test_fptquant_transform_config_loads_and_attaches(transform_name: str, overrides: list[str]):
    _register_configs()
    cfg = _compose_transform(transform_name, overrides)
    model = instantiate(cfg.llm.model)
    model.eval()
    model = get_quant_model(model, logger=logging.getLogger("test-fptquant-transforms"))
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    with ff.strict_quantization(False), ff.disable_quantization(model), torch.no_grad():
        untransformed_logits = model(input_ids=input_ids).logits.detach()

    transform_config = instantiate(cfg.quant.transforms)
    transforms, transform_ops = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    transform_key = "Po" if transform_name == "Po_paper" else transform_name
    assert transform_key in transforms
    assert transform_key in transform_ops
    assert len(list(transform_ops[transform_key].modules())) > 1

    with ff.strict_quantization(False), ff.disable_quantization(model), torch.no_grad():
        output = model(input_ids=input_ids)

    assert output.logits.shape == (1, 8, model.config.vocab_size)
    torch.testing.assert_close(output.logits, untransformed_logits, **NO_QUANT_TRANSFORM_TOLERANCE)


@pytest.mark.unit
def test_prerope_transform_supports_key_and_query_shapes():
    transform = T.PreRopeTransform(num_heads=4, num_key_value_heads=2)
    key = torch.randn(2, 3, 8)
    query = torch.randn(2, 3, 16)

    transformed_key = transform(key)
    transformed_query = transform(query, inv=True)

    assert transformed_key.shape == key.shape
    assert transformed_query.shape == query.shape
    assert transformed_key.dtype == key.dtype
    assert transformed_query.dtype == query.dtype


@pytest.mark.unit
def test_prerope_weight_paths_match_old_merge_formulas():
    torch.manual_seed(1061)
    transform = T.PreRopeTransform(num_heads=4, num_key_value_heads=2)
    k_weight = torch.randn(8, 8)
    q_weight = torch.randn(16, 8)
    _ = transform(k_weight, dim=-2)

    with torch.no_grad():
        transform.angles.copy_(torch.randn_like(transform.angles))
        transform.scales.copy_(torch.rand_like(transform.scales) + 0.5)

    new_k = transform(k_weight, dim=-2, inv=False)
    old_k = transform(k_weight.T, dim=-1, inv=False).T
    new_q = transform(q_weight, dim=-2, inv=True)
    old_q = transform(q_weight.T, dim=-1, inv=True).T

    torch.testing.assert_close(new_k, old_k)
    torch.testing.assert_close(new_q, old_q)


@pytest.mark.unit
def test_value_batched_transform_matches_old_values_transform_batched_formula():
    torch.manual_seed(1061)
    transform = T.ValueBatchedTransform(num_heads=4, num_key_value_heads=2, head_dim=3)
    value = torch.randn(2, 6)
    output = torch.randn(2, 12)
    _ = transform(value)

    matrices = torch.stack(
        [
            torch.eye(3) + 0.1 * torch.randn(3, 3),
            torch.eye(3) + 0.1 * torch.randn(3, 3),
        ]
    )
    with torch.no_grad():
        transform.matrix.copy_(matrices)

    expected_value = torch.einsum("... h j, h j k -> ... h k", value.view(2, 2, 3), matrices).reshape_as(value)
    inv_matrices = torch.linalg.inv(matrices + 1e-6 * torch.eye(3).unsqueeze(0))
    expected_output = torch.einsum(
        "... h g j, h j k -> ... h g k",
        output.view(2, 2, 2, 3),
        inv_matrices,
    ).reshape_as(output)

    torch.testing.assert_close(transform(value), expected_value)
    torch.testing.assert_close(transform(output, inv=True), expected_output)


@pytest.mark.unit
def test_value_batched_transform_initializes_to_identity_per_key_value_head():
    transform = T.ValueBatchedTransform(num_heads=4, num_key_value_heads=2, head_dim=3)
    data = torch.randn(2, 6)

    torch.testing.assert_close(transform(data), data)
    assert transform.matrix.shape == (2, 3, 3)


@pytest.mark.unit
def test_value_batched_transform_can_share_one_matrix_across_heads():
    torch.manual_seed(1061)
    transform = T.ValueBatchedTransform(num_heads=6, num_key_value_heads=2, head_dim=3, shared_across_heads=True)
    value = torch.randn(2, 6)
    output = torch.randn(2, 18)
    _ = transform(value)

    matrix = torch.eye(3) + 0.1 * torch.randn(3, 3)
    with torch.no_grad():
        transform.matrix.copy_(matrix)

    expected_value = torch.einsum("... h j, j k -> ... h k", value.view(2, 2, 3), matrix).reshape_as(value)
    inv_matrix = torch.linalg.inv(matrix + 1e-6 * torch.eye(3))
    expected_output = torch.einsum(
        "... h g j, j k -> ... h g k",
        output.view(2, 2, 3, 3),
        inv_matrix,
    ).reshape_as(output)

    assert transform.matrix.shape == (3, 3)
    torch.testing.assert_close(transform(value), expected_value)
    torch.testing.assert_close(transform(output, inv=True), expected_output)


@pytest.mark.unit
def test_head_mixing_transform_matches_head_axis_and_flattened_forms():
    torch.manual_seed(1061)
    transform = T.HeadMixingTransform(num_heads=4, head_dim=3, init_type="identity")
    explicit_heads = torch.randn(2, 4, 5, 3)
    flattened = explicit_heads.transpose(1, 2).reshape(2, 5, 12)
    _ = transform(explicit_heads, dim=1)

    matrix = torch.eye(4) + 0.1 * torch.randn(4, 4)
    with torch.no_grad():
        transform.matrix.copy_(matrix)

    expected_explicit = torch.einsum("b h s d, h k -> b k s d", explicit_heads, matrix)
    expected_flattened = torch.einsum(
        "b s h d, h k -> b s k d",
        flattened.reshape(2, 5, 4, 3),
        matrix,
    ).reshape_as(flattened)
    inverse = torch.linalg.inv(matrix + 1e-6 * torch.eye(4))
    expected_flattened_inverse = torch.einsum(
        "b s h d, h k -> b s k d",
        flattened.reshape(2, 5, 4, 3),
        inverse,
    ).reshape_as(flattened)

    torch.testing.assert_close(transform(explicit_heads, dim=1), expected_explicit)
    torch.testing.assert_close(transform(flattened, dim=-1), expected_flattened)
    torch.testing.assert_close(transform(flattened, dim=-1, inv=True), expected_flattened_inverse)


@pytest.mark.unit
def test_head_mixing_o_proj_parametrization_cancels_explicit_head_transform():
    torch.manual_seed(1061)
    transform = T.HeadMixingTransform(num_heads=4, head_dim=3, init_type="identity")
    explicit_heads = torch.randn(2, 4, 5, 3)
    flattened = explicit_heads.transpose(1, 2).reshape(2, 5, 12)
    linear = torch.nn.Linear(12, 7, bias=False)
    baseline = linear(flattened).detach()
    _ = transform(explicit_heads, dim=1)

    matrix = torch.eye(4) + 0.1 * torch.randn(4, 4)
    with torch.no_grad():
        transform.matrix.copy_(matrix)

    add_transform_operations(
        LinearParametrizationConfig(pattern="&o_proj", input=True, inv=True),
        transform=transform,
        module=linear,
    )
    transformed_heads = transform(explicit_heads, dim=1)
    transformed_flattened = transformed_heads.transpose(1, 2).reshape_as(flattened)

    torch.testing.assert_close(linear(transformed_flattened), baseline)


@pytest.mark.unit
def test_pv_config_uses_old_compatible_batched_values_transform():
    _register_configs()
    cfg = _compose_transform("Pv", [])

    transform_config = instantiate(cfg.quant.transforms)

    assert isinstance(transform_config["Pv"].transform, T.ValueBatchedTransform)
    assert transform_config["Pv"].transform.shared_across_heads is False


@pytest.mark.unit
def test_po_paper_config_uses_head_mixing_transform():
    _register_configs()
    cfg = _compose_transform("Po_paper", [])

    transform_config = instantiate(cfg.quant.transforms)

    assert isinstance(transform_config["Po"].transform, T.HeadMixingTransform)
    assert [op.pattern for op in transform_config["Po"].operation_configs] == ["&ao", "&o_proj"]
    assert transform_config["Po"].operation_configs[0].dim == 1


@pytest.mark.unit
def test_dynamic_scaling_transform_round_trip():
    transform = T.DynamicScalingTransform()
    data = torch.randn(2, 3, 4)

    scaled = transform(data)
    recovered = transform(scaled, inv=True)

    torch.testing.assert_close(recovered, data)


@pytest.mark.unit
def test_weight_lp_estimator_config_instantiates():
    _register_configs()
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=None):
        cfg = compose(config_name=None, overrides=["+quant/parameter_estimator=weight_lp"])

    estimator = instantiate(cfg.quant.parameter_estimator)
    assert isinstance(estimator, WeightLpEstimator)


@pytest.mark.integration
def test_weight_lp_chunked_backward_matches_single_graph_gradients():
    """`max_ops_per_backward` must not change the gradients it produces.

    The Lp loss is a plain sum over weight operations, so accumulating gradients from
    several smaller `backward()` calls is mathematically identical to one `backward()`
    over the whole sum -- while retaining only one chunk's transformed weights at a
    time. That bound is what makes the estimator fit larger models, so pin the
    equivalence here.
    """
    _register_configs()

    def captured_gradients(max_ops_per_backward: int | None) -> list[list[torch.Tensor]]:
        torch.manual_seed(1061)
        GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path=None):
            cfg = compose(
                config_name=None,
                overrides=[
                    "+llm=mock_llama",
                    # R1 is the shared (`per_layer=False`) rotation whose Lp step spans every
                    # projection of every layer -- the case chunking exists for.
                    "+quant/transforms=R1",
                    "transform@quant.transforms.R1.transform=rotation",
                ],
            )
        model = instantiate(cfg.llm.model)
        model.eval()
        model = get_quant_model(model, logger=logging.getLogger("test-weight-lp-chunking"))
        ft.utils.merge_rmsnorm(model, merge_head=True)
        ft.transforms_from_config_dict(instantiate(cfg.quant.transforms), model=model, aliases=model.aliases)

        snapshots: list[list[torch.Tensor]] = []
        original_step = torch.optim.AdamW.step

        def spy(self, *args, **kwargs):
            snapshots.append(
                [
                    parameter.grad.detach().clone()
                    for group in self.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
            )
            return original_step(self, *args, **kwargs)

        torch.optim.AdamW.step = spy
        try:
            # lr=0 keeps the parameters fixed, so every step sees identical inputs
            # across the two variants and the gradients are directly comparable.
            WeightLpEstimator(p=4.0, lr=0.0, num_steps=2, max_ops_per_backward=max_ops_per_backward).estimate(
                model, logger=logging.getLogger("test-weight-lp-chunking")
            )
        finally:
            torch.optim.AdamW.step = original_step
        return snapshots

    whole_graph = captured_gradients(None)
    chunked = captured_gradients(1)

    assert whole_graph, "expected at least one optimizer step with gradients"
    assert sum(len(step) for step in whole_graph) > 0
    assert len(chunked) == len(whole_graph)
    for chunked_step, whole_step in zip(chunked, whole_graph):
        assert len(chunked_step) == len(whole_step)
        for chunked_grad, whole_grad in zip(chunked_step, whole_step):
            # Only float32 reassociation differs, so this is far tighter than the
            # transform tolerances used elsewhere in this module.
            torch.testing.assert_close(chunked_grad, whole_grad, atol=1e-7, rtol=1e-7)
