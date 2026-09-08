# Flood-depth estimation from SAR and terrain

This is the production code path for flood-depth estimation using Sentinel-1
state, event, change, acquisition-reliability, and terrain inputs. PA-HydroKAN
uses labels and label-derived masks only for supervision and evaluation. The
separately documented comparison models use `valid_depth_mask` directly as the
user-required flood-range oracle; no model predicts flood extent.

The production model is **PA-HydroKAN** (`pa_hydrokan`): a SAR-first,
terrain-aware depth estimator with a TAE-KAN terrain-connectivity prior.
`configs/pa_hydrokan.xml` is its sole model configuration.

The comparison workflow is deliberately separated from the production depth
model. Every comparison method uses `valid_depth_mask` directly as its flood
range; no flood-range prediction model is included.

## Layout

- `configs/pa_hydrokan.xml` — PA-HydroKAN training and inference configuration.
- `configs/base/` — common runtime and dataset configuration fragments.
- `configs/ablation/` — matched PA-HydroKAN ablation configurations.
- `configs/compare/traditional/` — one configuration for each traditional
  comparison model.
- `configs/compare/deep_learning/` — one configuration for each learned
  comparison model.
- `assets/flooddepthnet_s1_terrain/` — audited complete-release contract and
  train-only normalization statistics.
- `models/` — PA-HydroKAN only: SAR encoder, terrain features, TAE-KAN graph,
  decoder, and heads.
- `compare/common/` — shared comparison blocks, terrain primitives, registries,
  and factories.
- `compare/traditional/` — reproducible non-learned comparison methods.
- `compare/deep_learning/` — reproducible learned comparison methods.
- `tools/` — training, evaluation, inference, and comparison runners.
- `runs/` — locally generated training outputs; ignored by Git.

## Train

Install a PyTorch build suitable for the local accelerator, then install the
remaining packages:

```bash
conda activate flood-depth
pip install -r requirements.txt
python train.py configs/pa_hydrokan.xml
```

Training creates the configuration-owned directory
`runs/flooddepthnet_s1_terrain/train/<model>/<run-tag>/`. It includes the
resolved configuration, dataset fingerprint, calibration artifacts, raw and
EMA checkpoints, metrics, and runtime metadata.

## Evaluate

```bash
python evaluate.py configs/pa_hydrokan.xml
```

`train.py` and `evaluate.py` accept every PA-HydroKAN, learned-comparison, and
traditional-comparison XML. The model, device, hyperparameters, run tag,
checkpoint, split, and result directory come from the inherited `<runtime>` and
other configuration sections; no model-specific command-line parameters are
required.

No trained checkpoint is bundled: previous training results were deliberately
removed before this production reset.

The complete FloodDepthNet training protocol, all model commands, full-data
hyperparameters, and result directory layout are in
[docs/FULL_DATA_TRAINING.md](docs/FULL_DATA_TRAINING.md). The active input is
strictly S1/QA/terrain; Sentinel-2 is not read.

## Learned comparisons

The retained learned comparators are DLSIM Attention U-Net, DLSIM LinkNet,
plain U-Net regression, ResNet18 regression, and U-Net++ regression. Each has a
model-named implementation and XML configuration. The root-level unified
operations select them from XML. Their sources and non-selected candidates are recorded in
[docs/COMPARISON_SOURCES.md](docs/COMPARISON_SOURCES.md).

PA-HydroKAN's publication-facing name and module terminology are recorded in
[docs/MODEL_NOMENCLATURE.md](docs/MODEL_NOMENCLATURE.md); the matched ablation
protocol and configuration set are in [docs/ABLATION_PROTOCOL.md](docs/ABLATION_PROTOCOL.md).
