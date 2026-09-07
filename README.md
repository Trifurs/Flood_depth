# Flood-depth estimation from SAR and terrain

This is the production code path for flood-depth estimation using Sentinel-1
state, event, change, acquisition-reliability, and terrain inputs. PA-HydroKAN
uses labels and label-derived masks only for supervision and evaluation. The
separately documented comparison models use `valid_depth_mask` directly as the
user-required flood-range oracle; no model predicts flood extent.

The production model is **PA-HydroKAN** (`pa_hydrokan`): a SAR-first,
terrain-aware depth estimator with an Edge-KAN terrain-connectivity prior.
`configs/pa_hydrokan.xml` is its sole model configuration.

The comparison workflow is deliberately separated from the production depth
model. Every comparison method uses `valid_depth_mask` directly as its flood
range; no flood-range prediction model is included.

## Layout

- `configs/pa_hydrokan.xml` — PA-HydroKAN training and inference configuration.
- `configs/base/` — common runtime and dataset configuration fragments.
- `configs/compare/traditional/` — one configuration for each traditional
  comparison model.
- `configs/compare/deep_learning/` — one configuration for each learned
  comparison model.
- `assets/` — audited dataset contract and train-only normalization statistics.
- `models/` — PA-HydroKAN only: SAR encoder, terrain features, Edge-KAN graph,
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
pip install -r requirements.txt
conda run -n flood-depth python tools/train_pa_hydrokan.py \
  --config configs/pa_hydrokan.xml \
  --device cuda
```

Training creates a timestamped directory under `runs/train/`. It includes the
resolved configuration, dataset fingerprint, calibration artifacts, raw and
EMA checkpoints, metrics, and runtime metadata.

## Evaluate

```bash
conda run -n flood-depth python tools/evaluate_pa_hydrokan.py \
  --config configs/pa_hydrokan.xml \
  --checkpoint runs/train/<run>/best_raw.pth \
  --split val \
  --device cuda
```

## Infer one known sample

```bash
conda run -n flood-depth python tools/infer_pa_hydrokan.py \
  --config configs/pa_hydrokan.xml \
  --checkpoint runs/train/<run>/best_raw.pth \
  --input <sample-id> \
  --device cuda \
  --save-geotiff
```

No trained checkpoint is bundled: previous training results were deliberately
removed before this production reset.

## Learned comparisons

The retained learned comparators are DLSIM Attention U-Net, DLSIM LinkNet,
plain U-Net regression, ResNet18 regression, and U-Net++ regression. Each has a
model-named implementation, XML configuration, training script, and evaluation
script. Their sources and non-selected candidates are recorded in
[docs/COMPARISON_SOURCES.md](docs/COMPARISON_SOURCES.md).
