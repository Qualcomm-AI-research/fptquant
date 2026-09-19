# FPTQuant

FPTQuant is a reference implementation of [FPTQuant: Function-Preserving Transforms for LLM Quantization](https://openreview.net/forum?id=2jpCo5lnSZ), accepted at ICML 2026. It provides quantization pipelines, transform parameterization methods, and Hydra configurations for experiments.

## Requirements

- Python 3.11 or newer.
- A supported PyTorch environment. GPU training requires a CUDA-compatible
  PyTorch installation.
- Network access to Hugging Face on the first run. The default Meta Llama
  repositories are gated, so accept their license and authenticate once with
  `hf auth login`.

## Setup

Install the project and development dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

The default entries in [`config/paths/default.yaml`](config/paths/default.yaml) are Hugging Face model and dataset IDs. To use a local model checkpoint instead, override its resource path (`config/paths`). Models and datasets download on first use into the standard Hugging Face cache; experiment outputs are written to `outputs/`. 


## Run an experiment

The default Hydra configuration is [`config/config.yaml`](config/config.yaml)
and composes the static `main_exp_static` experiment preset. Start it with:

```bash
uv run main.py \
  experiment=main_exp_static \
  llm=llama_v32_1B_instruct \
  quant.batch_size=4 \
  quant.max_num_batches=16 \
  train.training_args.use_liger_kernel=false \
  train.training_args.per_device_train_batch_size=1 \
  train.training_args.gradient_accumulation_steps=16
```

Output:
```bash
...
|    | quantity   |    value |   sequence_length | stat   | evaluation         |
|---:|:-----------|---------:|------------------:|:-------|:-------------------|
|  0 | eval_loss  |  2.81001 |              2048 | mean   | wikitext2_2048_ppl |
|  1 | perplexity | 16.6102  |              2048 | mean   | wikitext2_2048_ppl |
```

Use Hydra overrides to select a different experiment or customize a setting:

```bash
uv run python main.py experiment=main_exp_dynamic train.training_args.max_steps=10
```

### FPTQuant and baselines

The release presets select `quant/parameter_estimator=weight_lp` by default.
That estimator initializes trainable transform parameters before range setting (i.e. the local optimization step based on $L_p$ norm, see paper). _It is separate from the quantizer `lp_norm` range estimator._  

To attach the selected transforms without Weight-Lp initialization, add `quant.parameter_estimator=null` as shown below. The transforms then use their configured initial values and can still be optimized during global distillation training.

```bash
# Full FPTQuant recipe (the static preset default; includes Weight-Lp initialization)
uv run python main.py experiment=main_exp_static quant/transforms=fptquant_pdr_o_d

# FPTQuant without PDR
uv run python main.py experiment=main_exp_static \
  quant/transforms=fptquant_no_pdr

# QuaRot: fixed Hadamard rotations
uv run python main.py experiment=main_exp_static \
  quant/transforms=quarot_opt quant.parameter_estimator=null

# SpinQuant: learnable rotation matrices
uv run python main.py experiment=main_exp_static \
  quant/transforms=spinquant quant.parameter_estimator=null

# FlatQuant: Kronecker-factored invertible matrices
uv run python main.py experiment=main_exp_static \
  quant/transforms=flatquant quant.parameter_estimator=null

# RTN-opt: no transforms; range estimation only
uv run python main.py experiment=main_exp_static \
  quant/transforms=rtn_opt quant.parameter_estimator=null
```

Use `experiment=main_exp_dynamic` instead for the dynamic-activation/KV
quantizer preset. Additional FPTQuant PDR placements are available as
`fptquant_pdr_ao_ui`, `fptquant_pdr_ao_mm`, `fptquant_pdr_ao_d`, `fptquant_pdr_o_mm`, `fptquant_pdr_o_d`, `fptquant_pdr_ao_u`, and `fptquant_pdr_o_u`.

Experiment outputs and Hydra logs are written below `paths.log`, using the
configured run subdirectory. The training configuration can require local
model files and may download public datasets when they are not cached.

## Configuration

- `config/experiment/_exp_base.yaml` contains the shared FPTQuant setup;
  `main_exp_static` and `main_exp_dynamic` select its static and dynamic
  quantization rules, respectively.
- `config/quant/` contains quantization rules, transform placements,
  calibration datasets, and parameter-estimation settings.
- `config/paths/default.yaml` documents every environment-specific path.

The default preset uses 4-bit weights, activations, and KV cache (`W4A4KV4`).


## Development and validation

Run the test suite with:

```bash
uv run pytest
```

Check linting and formatting with:

```bash
uv run ruff check .
uv run ruff format --check .
```

Install the optional pre-commit hooks with `uv run pre-commit install`.

<details>
<summary><strong>Debugging: creating cheap runs</strong></summary>

Run the static and dynamic release presets locally with the tiny `mock_llama`
model and generated text. These commands exercise quantization, transform
attachment, one Weight-Lp initialization step, range setting, and one training
step without requiring a model snapshot or downloaded dataset. Final evaluation
is skipped so this remains a fast, fully local smoke test.

```bash
uv run python main.py experiment=main_exp_static llm=mock_llama \
  'tokenized_dataset@quant.calibration_set=base' \
  'raw_dataset@quant.calibration_set.raw_dataset=mock_text' \
  'tokenized_dataset@train_dataset=base' \
  'raw_dataset@train_dataset.raw_dataset=mock_text' \
  'tokenized_dataset@eval_dataset=base' \
  'raw_dataset@eval_dataset.raw_dataset=mock_text' \
  '+skip_final_evaluation=true' \
  quant.max_num_batches=1 quant.batch_size=1 \
  quant.calibration_set.data_processing.preprocessing_type=join_nn \
  quant.calibration_set.data_processing.sequence_length=16 \
  quant.calibration_set.data_processing.take=1 \
  quant.parameter_estimator.num_steps=1 \
  train_dataset.data_processing.preprocessing_type=join_nn \
  train_dataset.data_processing.sequence_length=16 train_dataset.data_processing.take=4 \
  eval_dataset.data_processing.preprocessing_type=join_nn \
  eval_dataset.data_processing.sequence_length=16 eval_dataset.data_processing.take=4 \
  train.training_args.max_steps=1 train.training_args.per_device_train_batch_size=1 \
  train.training_args.gradient_accumulation_steps=1 train.training_args.save_strategy=no \
  train.training_args.report_to=[] '+train.training_args.use_cpu=true' \
  train.training_args.bf16=false train.training_args.use_liger_kernel=false \
  train.resume_from_checkpoint=false paths.log=/tmp/fptquant-release-smoke subdir=static-mock
```

For the dynamic preset, use the same command with
`experiment=main_exp_dynamic subdir=dynamic-mock`.

The final lines from the local runs were:

```text
# static
{'loss': '0.0001822', 'grad_norm': '0.0009077', 'learning_rate': '0', 'epoch': '0.25'}
{'train_runtime': '0.7496', 'train_samples_per_second': '1.334', 'train_steps_per_second': '1.334', 'train_loss': '0.0001822', 'epoch': '0.25'}
[root][INFO] - Skipping final evaluation because skip_final_evaluation=true.

# dynamic
{'loss': '0.004578', 'grad_norm': '0.02333', 'learning_rate': '0', 'epoch': '0.25'}
{'train_runtime': '1.247', 'train_samples_per_second': '0.802', 'train_steps_per_second': '0.802', 'train_loss': '0.004578', 'epoch': '0.25'}
[root][INFO] - Skipping final evaluation because skip_final_evaluation=true.
```

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards. Project
changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## License

FPTQuant is distributed under the [BSD-3-Clause-Clear License](LICENSE).

## Cite
```
@inproceedings{
breugel2026fptquant,
title={{FPTQ}uant: Function-Preserving Transforms for {LLM} Quantization},
author={Boris van Breugel and Yelysei Bondarenko and Paul N. Whatmough and Markus Nagel},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=2jpCo5lnSZ}
}
```
