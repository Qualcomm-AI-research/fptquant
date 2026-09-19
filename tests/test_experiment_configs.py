# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import logging
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import fastforward as ff
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import instantiate, store
from omegaconf import OmegaConf

import fasttransforms as ft
import fasttransforms.nn.transforms as T
import lm_tools
import quant_tools
from quant_tools import get_quant_model, initialize_quantizers

CONFIG_DIR = Path(__file__).parents[1] / "config"
PROJECT_ROOT = Path(__file__).parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_resolvers import register_config_resolvers  # noqa: E402
from main import (  # noqa: E402
    _cast_quantizer_parameters_to_float32,
    _verify_quantizer_parameters_float32,
    _verify_trainable_parameter_contract,
)

EXPECTED_TRANSFORMS = {
    "MM",
    "PDR",
    "PreRoPE",
    "Pu",
    "Pv",
    "R1",
}
EXPECTED_TRANSFORMS_NO_PDR = EXPECTED_TRANSFORMS - {"PDR"}
EXPECTED_TRANSFORM_PRIORITIES = {
    "MM": 0,
    "PDR": 0,
    "PreRoPE": 1,
    "Pu": 1,
    "Pv": 1,
    "R1": 1,
}
# PDR placement variants: (attention-side quantizer, MLP-side quantizer) that the PDR chain
# normalizes at. `ao`/`o` select the attention output site, and the suffix selects the MLP site.
PDR_PLACEMENT_VARIANTS = {
    "fptquant_pdr_ao_ui": (
        "model.layers.{layer_idx}.self_attn.sdpa_output_quantizer",
        "model.layers.{layer_idx}.mlp.up_proj.input_quantizer",
    ),
    "fptquant_pdr_ao_mm": (
        "model.layers.{layer_idx}.self_attn.sdpa_output_quantizer",
        "model.layers.{layer_idx}.mlp.gated_up_proj_output_quantizer",
    ),
    "fptquant_pdr_ao_d": (
        "model.layers.{layer_idx}.self_attn.sdpa_output_quantizer",
        "model.layers.{layer_idx}.mlp.down_proj.output_quantizer",
    ),
    "fptquant_pdr_o_mm": (
        "model.layers.{layer_idx}.self_attn.o_proj.output_quantizer",
        "model.layers.{layer_idx}.mlp.gated_up_proj_output_quantizer",
    ),
    "fptquant_pdr_o_d": (
        "model.layers.{layer_idx}.self_attn.o_proj.output_quantizer",
        "model.layers.{layer_idx}.mlp.down_proj.output_quantizer",
    ),
    "fptquant_pdr_ao_u": (
        "model.layers.{layer_idx}.self_attn.sdpa_output_quantizer",
        "model.layers.{layer_idx}.mlp.up_proj.output_quantizer",
    ),
    "fptquant_pdr_o_u": (
        "model.layers.{layer_idx}.self_attn.o_proj.output_quantizer",
        "model.layers.{layer_idx}.mlp.up_proj.output_quantizer",
    ),
}
# The `_u` variants rescale at the up-projection output, which is not identity-preserving
# against the no-PDR reference, so they are excluded from the identity sweep.
PDR_SWEEP_VARIANTS = tuple(
    name
    for name in PDR_PLACEMENT_VARIANTS
    if name not in {"fptquant_pdr_ao_u", "fptquant_pdr_o_u"}
)

NO_QUANT_TRANSFORM_TOLERANCE = dict(atol=1e-2, rtol=1e-2)
OpSummary = tuple[str, str, int, bool, tuple[tuple[str, object], ...]]


def _register_configs() -> None:
    lm_tools.add_config_to_store()
    quant_tools.add_config_to_store()
    ft.add_config_to_store()
    register_config_resolvers()
    store.add_to_hydra_store()


def _compose_fptquant(overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _resolve_without_hydra_subdir(cfg) -> None:
    # `choice_of` depends on HydraConfig, which is not populated by bare compose().
    OmegaConf.update(cfg, "subdir", "test", force_add=False)
    OmegaConf.update(cfg, "hydra.sweep.subdir", "test", force_add=True)
    OmegaConf.resolve(cfg)


def _orthogonal_like(tensor: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn_like(tensor.float()))
    return q.to(dtype=tensor.dtype, device=tensor.device)


def _randomize_transform_parameters(transforms: torch.nn.Module) -> None:
    with torch.no_grad():
        for transform in transforms.modules():
            if isinstance(transform, T.RotationTransform) and transform.R is not None:
                transform.R.copy_(_orthogonal_like(transform.R))
            elif isinstance(transform, T.InvertibleMatrixTransform) and transform.M is not None:
                transform.M.copy_(_orthogonal_like(transform.M))
            elif isinstance(transform, T.ScaleTransform) and transform.scale is not None:
                transform.scale = torch.exp(torch.randn_like(transform.scale.float()) * 0.2).to(
                    dtype=transform.scale.dtype,
                    device=transform.scale.device,
                )
            elif isinstance(transform, T.PreRopeTransform) and transform.angles is not None:
                transform.angles.copy_(torch.empty_like(transform.angles).uniform_(-math.pi, math.pi))
                transform.scales.copy_(
                    torch.exp(torch.randn_like(transform.scales.float()) * 0.2).to(transform.scales)
                )
            elif isinstance(transform, T.ValueBatchedTransform) and transform.matrix is not None:
                if transform.matrix.ndim == 2:
                    transform.matrix.copy_(_orthogonal_like(transform.matrix))
                else:
                    for head_idx in range(transform.matrix.shape[0]):
                        transform.matrix[head_idx].copy_(_orthogonal_like(transform.matrix[head_idx]))


def _pipeline_prepare_model(cfg):
    model = instantiate(cfg.llm.model)
    model.eval()
    model = get_quant_model(model, logger=logging.getLogger("test-fptquant-main-exp"))
    ft.utils.merge_rmsnorm(model, merge_head=True)
    return model


def _apply_trainable_parameter_filter(model: torch.nn.Module, substrings) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(substring in name for substring in substrings)


class _QuantizerParamModule(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.output_quantizer = torch.nn.Module()
        self.output_quantizer.scale = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        self.output_quantizer.offset = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
        self.output_quantizer.temperature = torch.nn.Parameter(torch.full((2,), 0.5, dtype=dtype))
        self.plain_module = torch.nn.Module()
        self.plain_module.scale = torch.nn.Parameter(torch.ones(2, dtype=dtype))


def _normalize_kwargs(kwargs: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((key, value) for key, value in kwargs.items() if not (key == "transpose" and value is False)))


def _attachment_summary(model: torch.nn.Module, transform_ops: torch.nn.Module) -> dict[str, list[OpSummary]]:
    module_names = {id(module): name for name, module in model.named_modules()}
    summary: dict[str, list[OpSummary]] = {}
    for transform_name, ops in transform_ops.items():
        transform_summary = []
        for op in ops.modules():
            if not isinstance(op, ft.TransformOperation):
                continue

            if op.parametrized_tensor is not None:
                kind = op.parametrized_tensor
            else:
                kind = "hook-pre" if op._pre_hook else "hook-post"

            transform_summary.append(
                (
                    module_names[id(op.attached_to)],
                    kind,
                    op._dim,
                    op._inv,
                    _normalize_kwargs(op._call_kwargs),
                )
            )
        summary[transform_name] = sorted(transform_summary)
    return summary


def _expected_fptquant_attachment_summary(
    num_layers: int,
    include_pdr: bool = True,
    pdr_internal_patterns: tuple[str, str] | None = None,
) -> dict[str, list[OpSummary]]:
    def op(
        module: str,
        kind: str,
        dim: int,
        inv: bool,
        kwargs: tuple[tuple[str, object], ...] = (),
    ) -> OpSummary:
        return (module, kind, dim, inv, kwargs)

    r1 = [
        op("model.embed_tokens.output_quantizer", "hook-pre", -1, False),
        op("model.norm", "hook-pre", -1, True),
    ]
    pv = []
    prerope = []
    mm = []
    pu = []
    pdr = []
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        r1.extend(
            [
                op(f"{prefix}.self_attn.q_proj", "weight", -1, True, kwargs=(("transpose", True),)),
                op(f"{prefix}.self_attn.k_proj", "weight", -1, True, kwargs=(("transpose", True),)),
                op(f"{prefix}.self_attn.v_proj", "weight", -1, True, kwargs=(("transpose", True),)),
                op(f"{prefix}.mlp.up_proj", "weight", -1, True, kwargs=(("transpose", True),)),
                op(f"{prefix}.mlp.gate_proj", "weight", -1, True, kwargs=(("transpose", True),)),
                op(f"{prefix}.self_attn.o_proj", "weight", -2, False),
                op(f"{prefix}.mlp.down_proj", "weight", -2, False),
            ]
        )
        pv.extend(
            [
                op(f"{prefix}.self_attn.v_proj", "weight", -2, False),
                op(f"{prefix}.self_attn.o_proj", "weight", -1, True, kwargs=(("transpose", True),)),
            ]
        )
        prerope.extend(
            [
                op(f"{prefix}.self_attn.k_proj", "weight", -2, False),
                op(f"{prefix}.self_attn.q_proj", "weight", -2, True),
            ]
        )
        mm.extend(
            [
                op(f"{prefix}.mlp.gated_up_proj_output_quantizer", "hook-pre", -1, False),
                op(f"{prefix}.mlp.down_proj", "weight", -1, True, kwargs=(("transpose", True),)),
            ]
        )
        pu.extend(
            [
                op(f"{prefix}.mlp.up_proj", "weight", -2, False),
                op(f"{prefix}.mlp.down_proj", "weight", -1, True, kwargs=(("transpose", True),)),
            ]
        )
        attention_pdr = f"{prefix}.self_attn.sdpa_softmax_quantizer"
        mlp_pdr = f"{prefix}.mlp.gated_up_proj_output_quantizer"
        attention_pdr_kwargs: tuple[tuple[str, object], ...] = (("expand_dim", 1), ("rescale", True))
        if pdr_internal_patterns is not None:
            attention_pdr = pdr_internal_patterns[0].format(layer_idx=layer_idx)
            mlp_pdr = pdr_internal_patterns[1].format(layer_idx=layer_idx)
            attention_pdr_kwargs = (("rescale", True),)
            if attention_pdr.endswith(".sdpa_output_quantizer"):
                attention_pdr_kwargs = (("expand_dim", 1), ("rescale", True))
        pdr.extend(
            [
                op(f"{prefix}.input_quantizer", "hook-pre", -1, False, kwargs=(("first", True),)),
                op(attention_pdr, "hook-pre", -1, False, attention_pdr_kwargs),
                op(f"{prefix}.attn_res_act_quantizer", "hook-pre", -1, False),
                op(mlp_pdr, "hook-pre", -1, False, (("rescale", True),)),
                op(f"{prefix}.mlp_res_act_quantizer", "hook-post", -1, True),
            ]
        )
    expected = {
        "R1": sorted(r1),
        "Pv": sorted(pv),
        "PreRoPE": sorted(prerope),
        "MM": sorted(mm),
        "Pu": sorted(pu),
        "PDR": sorted(pdr),
    }
    if not include_pdr:
        del expected["PDR"]
    return expected


@pytest.mark.integration
def test_main_exp_composes_for_llama_v32_1b_instruct():
    _register_configs()

    cfg = _compose_fptquant(["paths=default"])
    # NOTE: do not call OmegaConf.resolve(cfg) here. Resolving the whole config eagerly
    # builds the tokenizer from `paths.resources.llms.*` and can download from Hugging Face.
    # Attribute access below resolves lazily, per key.

    assert cfg.llm.model.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct"
    assert set(cfg.quant.transforms) == EXPECTED_TRANSFORMS
    assert {
        name: transform.priority for name, transform in cfg.quant.transforms.items()
    } == EXPECTED_TRANSFORM_PRIORITIES
    assert set(cfg.quant.rules) == {"linear_weights", "linear_activations_inputs", "kv_cache"}
    assert cfg.quant.rules.linear_activations_inputs.query == "{&ao, &mm, &na, &nm}"
    assert cfg.quant.rules.linear_activations_inputs.quantizer.granularity._target_.endswith(".PerTensor")
    assert cfg.quant.rules.linear_activations_inputs.quantizer.symmetric is False
    assert cfg.quant.rules.kv_cache.quantizer.granularity._target_.endswith(".PerTensor")
    assert cfg.quant.rules.kv_cache.quantizer.symmetric is False
    assert cfg.quant.rules.linear_weights.quantizer.granularity._target_.endswith(".PerChannel")
    assert cfg.quant.rules.linear_weights.quantizer.granularity.channel_dim == 0
    assert cfg.quant.rules.linear_weights.quantizer.symmetric is True
    assert cfg.quant.range_estimator._target_ == "fastforward_contrib.range_setting.norm.LPNormRangeEstimator"
    assert cfg.quant.range_estimator.p == 3.0
    assert (
        cfg.quant.parameter_estimator._target_
        == "fasttransforms.parameter_setting.weight_optimization.WeightLpEstimator"
    )
    assert cfg.quant.parameter_estimator.p == 4.0
    assert cfg.quant.parameter_estimator.lr == 1e-2
    assert cfg.quant.parameter_estimator.num_steps == 200
    assert list(cfg.trainable_parameter_substrings) == ["quantizer", "._transform"]
    assert cfg.quant.max_num_batches == 4
    assert cfg.quant.batch_size == 16
    assert cfg.train.training_args.warmup_ratio == 0.1
    assert cfg.train._target_ == "lm_tools.train.kd.KDTraining"
    assert cfg.train.loss_type == "jsd"
    assert cfg.train.train_dataset._target_ == "lm_tools.tokenized_dataset.base.process_dataset"
    assert cfg.train.train_dataset.data_processing.sequence_length == 2048
    assert cfg.train.train_dataset.data_processing.num_workers == 0
    assert cfg.train.training_args.max_length == 2048
    assert cfg.train.training_args.dataset_kwargs.skip_prepare_dataset is True
    assert list(cfg.train.checkpoint_parameter_substrings) == ["quantizer", "._transform"]

    # NOTE: `instantiate(cfg.train)` is deliberately not exercised here -- building the
    # KDTraining object downloads/loads the real wikitext dataset. `cfg.train._target_`
    # above already pins the training class. The KDTraining object itself is covered by
    # tests/lm_tools/test_training_pipeline.py against mock models.


@pytest.mark.integration
def test_main_exp_static_release_preset_composes():
    """The static W4A4KV4 release entrypoint: `experiment=main_exp_static`."""
    _register_configs()

    cfg = _compose_fptquant(["paths=default", "experiment=main_exp_static"])
    # Assert on the raw run-directory template; the timestamp resolves at runtime.
    subdir_template = OmegaConf.to_container(cfg, resolve=False)["subdir"]

    assert cfg.llm.model.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct"
    assert set(cfg.quant.transforms) == EXPECTED_TRANSFORMS

    # Static quantization: plain LinearQuantizer on weights, activations and kv-cache.
    assert set(cfg.quant.rules) == {"linear_weights", "linear_activations_inputs", "kv_cache"}
    assert (
        cfg.quant.rules.linear_weights.quantizer._target_ == "fastforward.nn.linear_quantizer.LinearQuantizer.factory"
    )
    assert cfg.quant.rules.linear_weights.quantizer.granularity._target_.endswith(".PerChannel")
    assert cfg.quant.rules.linear_weights.quantizer.symmetric is True
    assert cfg.quant.rules.linear_activations_inputs.quantizer._target_ == (
        "fastforward.nn.linear_quantizer.LinearQuantizer.factory"
    )
    assert cfg.quant.rules.kv_cache.quantizer._target_ == "fastforward.nn.linear_quantizer.LinearQuantizer.factory"

    # The static release preset inherits the shared base training configuration.
    assert cfg.train.training_args.learning_rate == 1e-3
    assert cfg.train.training_args.max_steps == 1024
    assert cfg.train.training_args.gradient_accumulation_steps == 4
    assert cfg.train.training_args.per_device_train_batch_size == 4
    assert cfg.train.training_args.bf16 is True
    assert cfg.train.training_args.use_liger_kernel is True
    assert set(cfg.evaluation.steps) == {"wikitext2_2048_ppl"}
    assert subdir_template == "main_exp/${now:%Y-%m-%d}/${now:%H-%M-%S}_seed${seed}"


@pytest.mark.integration
def test_main_exp_dynamic_release_preset_composes():
    """The dynamic W4A4KV4D release entrypoint: `experiment=main_exp_dynamic`."""
    _register_configs()

    cfg = _compose_fptquant(["paths=default", "experiment=main_exp_dynamic"])
    # See note in the static preset test about the runtime directory template.
    subdir_template = OmegaConf.to_container(cfg, resolve=False)["subdir"]

    assert cfg.llm.model.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct"
    assert set(cfg.quant.transforms) == EXPECTED_TRANSFORMS

    # Dynamic quantization: DynamicLinearClipQuantizer with grad-based min/max everywhere.
    assert set(cfg.quant.rules) == {"linear_weights", "linear_activations", "kv_cache"}
    for rule_name in ("linear_weights", "linear_activations", "kv_cache"):
        quantizer = cfg.quant.rules[rule_name].quantizer
        assert quantizer._target_.endswith("DynamicLinearClipQuantizer.factory")
        assert quantizer.num_bits == 4
        assert quantizer.enable_grad_minmax is True
        assert quantizer.clip_parametrization == "identity"
    assert cfg.quant.rules.linear_weights.quantizer.granularity._target_.endswith(".PerChannel")
    assert cfg.quant.rules.linear_weights.quantizer.clip_granularity == "follow_quantizer"
    assert cfg.quant.rules.linear_activations.quantizer.granularity._target_.endswith(".PerToken")
    assert cfg.quant.rules.kv_cache.quantizer.granularity._target_.endswith(".PerToken")

    assert cfg.train.training_args.learning_rate == 1e-3
    assert cfg.train.training_args.max_steps == 1024
    assert cfg.train.training_args.bf16 is True
    assert cfg.train.training_args.use_liger_kernel is True
    assert set(cfg.evaluation.steps) == {"wikitext2_2048_ppl"}
    assert subdir_template == "main_exp/${now:%Y-%m-%d}/${now:%H-%M-%S}_seed${seed}"


@pytest.mark.integration
def test_main_exp_no_pdr_transform_preset_composes_for_llama_v32_1b_instruct():
    _register_configs()

    cfg = _compose_fptquant(["paths=default", "quant/transforms=fptquant_no_pdr"])
    # See note above: resolving eagerly would build the tokenizer from a placeholder path.

    assert set(cfg.quant.transforms) == EXPECTED_TRANSFORMS_NO_PDR
    assert {name: transform.priority for name, transform in cfg.quant.transforms.items()} == {
        name: priority for name, priority in EXPECTED_TRANSFORM_PRIORITIES.items() if name != "PDR"
    }
    assert "PDR" not in cfg.quant.transforms


@pytest.mark.integration
def test_fptquant_pv_uses_nonshared_transform_per_key_value_head():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama", "quant/transforms=fptquant_pdr_o_d"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    transform_config = instantiate(cfg.quant.transforms)
    transforms, _ = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    for layer_transform in transforms["Pv"].values():
        assert isinstance(layer_transform, T.ValueBatchedTransform)
        assert layer_transform.shared_across_heads is False
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        with torch.no_grad():
            layer_transform(torch.zeros(1, model.config.num_key_value_heads * head_dim), dim=-1)
        assert layer_transform.matrix.shape == (model.config.num_key_value_heads, head_dim, head_dim)


@pytest.mark.integration
def test_fptquant_recipe_trains_only_transform_and_quantizer_parameters():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    initialize_quantizers(model, rules=instantiate(cfg.quant.rules), logger=logging.getLogger("test-trainable"))
    transform_config = instantiate(cfg.quant.transforms)
    ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    _apply_trainable_parameter_filter(model, cfg.trainable_parameter_substrings)

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]

    assert trainable_names
    assert frozen_names
    assert any("._transform" in name for name in trainable_names)
    assert any("quantizer" in name for name in trainable_names)
    assert all("._transform" in name or "quantizer" in name for name in trainable_names)
    assert all("._transform" not in name and "quantizer" not in name for name in frozen_names)
    _verify_trainable_parameter_contract(model, cfg.trainable_parameter_substrings)


def test_trainable_contract_rejects_base_parameters_and_frozen_quantizer_params():
    module = _QuantizerParamModule(torch.float32)

    module.plain_module.scale.requires_grad = True
    with pytest.raises(AssertionError, match="Unexpected trainable parameters"):
        _verify_trainable_parameter_contract(module, ["quantizer", "._transform"])

    module.plain_module.scale.requires_grad = False
    module.output_quantizer.scale.requires_grad = False
    with pytest.raises(AssertionError, match="unexpected frozen quantizer/transform parameters"):
        _verify_trainable_parameter_contract(module, ["quantizer", "._transform"])


def test_cast_quantizer_parameters_to_float32_casts_all_quantizer_parameters():
    module = _QuantizerParamModule(torch.bfloat16)

    cast_count = _cast_quantizer_parameters_to_float32(module)

    assert cast_count == 6
    assert module.output_quantizer.scale.dtype == torch.float32
    assert module.output_quantizer.offset.dtype == torch.float32
    assert module.output_quantizer.temperature.dtype == torch.float32
    assert module.plain_module.scale.dtype == torch.bfloat16
    _verify_quantizer_parameters_float32(module)


def test_quantizer_float32_contract_rejects_non_float32_quantizer_parameters():
    module = _QuantizerParamModule(torch.bfloat16)

    with pytest.raises(AssertionError, match="Quantizer parameters must be float32"):
        _verify_quantizer_parameters_float32(module)


@pytest.mark.integration
def test_main_exp_fptquant_transforms_attach_to_mock_llama():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    with ff.strict_quantization(False), ff.disable_quantization(model), torch.no_grad():
        untransformed_logits = model(input_ids=input_ids).logits.detach()

    transform_config = instantiate(cfg.quant.transforms)
    transforms, transform_ops = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    assert set(transforms) == EXPECTED_TRANSFORMS
    assert set(transform_ops) == EXPECTED_TRANSFORMS

    with ff.strict_quantization(False), ff.disable_quantization(model), torch.no_grad():
        output = model(input_ids=input_ids)

    assert output.logits.shape == (1, 8, model.config.vocab_size)
    torch.testing.assert_close(output.logits, untransformed_logits, **NO_QUANT_TRANSFORM_TOLERANCE)


@pytest.mark.integration
def test_main_exp_fptquant_transform_placements_match_expected_surface():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    transform_config = instantiate(cfg.quant.transforms)
    _, transform_ops = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    assert _attachment_summary(model, transform_ops) == _expected_fptquant_attachment_summary(
        model.config.num_hidden_layers,
        pdr_internal_patterns=PDR_PLACEMENT_VARIANTS["fptquant_pdr_o_d"],
    )


@pytest.mark.integration
def test_main_exp_down_projection_parametrization_order_matches_training_default():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    transform_config = instantiate(cfg.quant.transforms)
    ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    parametrizations = model.model.layers[0].mlp.down_proj.parametrizations.weight
    ordered_transform_names = [
        parametrization.transform.name.split(".")[0]
        for parametrization in parametrizations
        if isinstance(parametrization, ft.TransformOperation)
    ]

    assert ordered_transform_names == ["R1", "Pu", "MM"]


@pytest.mark.integration
def test_main_exp_no_pdr_transform_placements_match_expected_subset():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama", "quant/transforms=fptquant_no_pdr"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    transform_config = instantiate(cfg.quant.transforms)
    _, transform_ops = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    assert _attachment_summary(model, transform_ops) == _expected_fptquant_attachment_summary(
        model.config.num_hidden_layers,
        include_pdr=False,
    )


@pytest.mark.integration
def test_main_exp_fptquant_transforms_are_identity_for_random_initializations_without_quantization():
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    input_ids_cases = [
        torch.randint(0, model.config.vocab_size, (1, 8)),
        torch.randint(0, model.config.vocab_size, (2, 8)),
        torch.randint(0, model.config.vocab_size, (1, 5)),
    ]

    with ff.strict_quantization(False), ff.disable_quantization(model), torch.no_grad():
        untransformed_logits = [model(input_ids=input_ids).logits.detach() for input_ids in input_ids_cases]

    transform_config = instantiate(cfg.quant.transforms)
    transforms, _ = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)
    _randomize_transform_parameters(transforms)

    with ff.strict_quantization(False), ff.disable_quantization(model), torch.no_grad():
        transformed_logits = [model(input_ids=input_ids).logits.detach() for input_ids in input_ids_cases]

    for transformed, untransformed in zip(transformed_logits, untransformed_logits):
        torch.testing.assert_close(transformed, untransformed, **NO_QUANT_TRANSFORM_TOLERANCE)


@pytest.mark.integration
@pytest.mark.parametrize("variant,pdr_internal_patterns", PDR_PLACEMENT_VARIANTS.items())
def test_main_exp_pdr_placement_variants_match_expected_surface(variant, pdr_internal_patterns):
    _register_configs()
    torch.manual_seed(1061)

    cfg = _compose_fptquant(["paths=default", "llm=mock_llama", f"quant/transforms={variant}"])
    _resolve_without_hydra_subdir(cfg)

    model = _pipeline_prepare_model(cfg)
    transform_config = instantiate(cfg.quant.transforms)
    _, transform_ops = ft.transforms_from_config_dict(transform_config, model=model, aliases=model.aliases)

    assert _attachment_summary(model, transform_ops) == _expected_fptquant_attachment_summary(
        model.config.num_hidden_layers,
        pdr_internal_patterns=pdr_internal_patterns,
    )


@pytest.mark.integration
@pytest.mark.parametrize("variant", PDR_SWEEP_VARIANTS)
def test_main_exp_pdr_sweep_variants_are_identity_without_quantization(variant):
    _register_configs()
    torch.manual_seed(1061)

    no_pdr_cfg = _compose_fptquant(["paths=default", "llm=mock_llama", "quant/transforms=fptquant_no_pdr"])
    variant_cfg = _compose_fptquant(["paths=default", "llm=mock_llama", f"quant/transforms={variant}"])
    _resolve_without_hydra_subdir(no_pdr_cfg)
    _resolve_without_hydra_subdir(variant_cfg)

    no_pdr_model = _pipeline_prepare_model(no_pdr_cfg)
    variant_model = _pipeline_prepare_model(variant_cfg)
    variant_model.load_state_dict(no_pdr_model.state_dict(), strict=False)

    input_ids_cases = [
        torch.randint(0, variant_model.config.vocab_size, (1, 8)),
        torch.randint(0, variant_model.config.vocab_size, (2, 8)),
        torch.randint(0, variant_model.config.vocab_size, (1, 5)),
    ]

    no_pdr_transform_config = instantiate(no_pdr_cfg.quant.transforms)
    variant_transform_config = instantiate(variant_cfg.quant.transforms)
    no_pdr_transforms, _ = ft.transforms_from_config_dict(
        no_pdr_transform_config,
        model=no_pdr_model,
        aliases=no_pdr_model.aliases,
    )
    variant_transforms, _ = ft.transforms_from_config_dict(
        variant_transform_config,
        model=variant_model,
        aliases=variant_model.aliases,
    )
    _randomize_transform_parameters(no_pdr_transforms)
    variant_transforms.load_state_dict(no_pdr_transforms.state_dict(), strict=False)

    with ff.strict_quantization(False), ff.disable_quantization(no_pdr_model), torch.no_grad():
        no_pdr_logits = [no_pdr_model(input_ids=input_ids).logits.detach() for input_ids in input_ids_cases]

    with ff.strict_quantization(False), ff.disable_quantization(variant_model), torch.no_grad():
        variant_logits = [variant_model(input_ids=input_ids).logits.detach() for input_ids in input_ids_cases]

    for variant_output, no_pdr_output in zip(variant_logits, no_pdr_logits):
        torch.testing.assert_close(variant_output, no_pdr_output, **NO_QUANT_TRANSFORM_TOLERANCE)

    # Backward: layer-0 transform-param gradients must stay within the same tolerance as forward.
    # PDR changes activations by O(epsilon_mismatch), so backward diffs are small but nonzero.
    labels = torch.randint(0, no_pdr_model.config.vocab_size, (1, 8))

    no_pdr_model.zero_grad()
    with ff.strict_quantization(False), ff.disable_quantization(no_pdr_model):
        no_pdr_loss = torch.nn.functional.cross_entropy(
            no_pdr_model(input_ids=input_ids_cases[0]).logits.view(-1, no_pdr_model.config.vocab_size),
            labels.view(-1),
        )
    no_pdr_loss.backward()
    no_pdr_grads = {
        n: p.grad.detach().clone()
        for n, p in no_pdr_model.named_parameters()
        if "layers.0." in n and p.requires_grad and p.grad is not None
    }
    no_pdr_model.zero_grad()

    variant_model.zero_grad()
    with ff.strict_quantization(False), ff.disable_quantization(variant_model):
        variant_loss = torch.nn.functional.cross_entropy(
            variant_model(input_ids=input_ids_cases[0]).logits.view(-1, variant_model.config.vocab_size),
            labels.view(-1),
        )
    variant_loss.backward()
    variant_grads = {
        n: p.grad.detach().clone()
        for n, p in variant_model.named_parameters()
        if "layers.0." in n and p.requires_grad and p.grad is not None
    }
    variant_model.zero_grad()

    for name in no_pdr_grads:
        if name in variant_grads:
            torch.testing.assert_close(variant_grads[name], no_pdr_grads[name], **NO_QUANT_TRANSFORM_TOLERANCE)
