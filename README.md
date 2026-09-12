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
- `test.py` — directory-driven test entry for all trained and traditional models.
- `validate_k_fold.py` — event-grouped cross-validation entry for all learned models.
- `tools/` — shared training, evaluation, inference, and comparison runners.
- `runs/` — locally generated training outputs; ignored by Git.

## Train

Install a PyTorch build suitable for the local accelerator, then install the
remaining packages:

```bash
conda activate flood-depth
pip install -r requirements.txt
python train.py configs/pa_hydrokan.xml
```

Training creates the configuration-owned, start-time-stamped directory
`runs/flooddepthnet_s1_terrain/train/<model>/<YYYYMMDD-HHMMSS-microseconds>/`.
It includes the resolved configuration, dataset fingerprint, calibration
artifacts, raw checkpoints, metrics, and runtime metadata. The random
seed is recorded as metadata rather than being used as a directory name, so
repeated experiments with one seed cannot overwrite one another.

## Evaluate

```bash
python evaluate.py configs/pa_hydrokan.xml
```

`train.py` accepts every PA-HydroKAN and learned-comparison XML. Traditional
methods have no training stage and are intentionally rejected by this entry.
The model, device, hyperparameters, time-stamped
run directory, checkpoint, split, and result directory come from the inherited `<runtime>` and
other configuration sections; no model-specific command-line parameters are
required.

## Test all models

```bash
python test.py runs/flooddepthnet_s1_terrain/train
```

The command discovers the newest complete checkpoint under each model folder,
tests it on the official test split, and evaluates all traditional methods by
default. Add `--no-traditional` to disable the latter or `--all-runs` to include
every repeated run. Accuracy, model size, synchronized inference efficiency,
peak accelerator memory, and end-to-end timing are written to one report.

## Event-grouped cross-validation

```bash
python validate_k_fold.py       # default k=5
python validate_k_fold.py 10
python validate_k_fold.py --resume  # resume newest unfinished k=5 session
```

All original train, validation, and test samples are pooled and reassigned by
`source_event_id`; an event never crosses train, inner-validation, and outer-test
roles within a run. Each sample belongs to exactly one outer-test fold. In each
outer run, the other four folds form the development pool, from which an
event-disjoint inner validation set is selected for checkpointing and early
stopping. Each fold receives independent train-only statistics. The script trains
the main model, every learned comparator, and the complete seven-configuration
RCP/TCF/TAE-KAN ablation design, then reports each outer fold and mean ± sample
standard deviation across the five outer tests. Every model×fold job runs in its
own process, and completion is committed after every job. Following an
interruption, `--resume` reuses complete jobs and continues an incomplete model
from its last checkpoint.
Formal runs require a healthy CUDA device and will not silently fall back to CPU.
The subprocess environment and NumPy/PyTorch/CUDA stack are preflighted before
model execution; its MKL threading layer is centralized in the base XML.
Resume additionally requires matching resolved-configuration, source-manifest,
runtime-source, and nested-split-protocol fingerprints. Legacy sessions created
before the full-sample outer-test protocol are rejected instead of mixing
incompatible fold results. Because the original test split now participates in
the folds, this protocol intentionally has no separate external holdout set.

## Monitor TensorBoard

After a training or evaluation has started, view all event streams with:

```bash
conda run -n flood-depth tensorboard \
  --logdir runs/flooddepthnet_s1_terrain \
  --port 6006
```

Open <http://localhost:6006>. Training writes `train/*`, `validation/*`, and
`system/*` scalars once per epoch, including epoch duration, elapsed time, ETA,
learning rate, and early-stopping progress. Per-run metadata is in the
TensorBoard **Text** tab. See [docs/RUN_MONITORING.md](docs/RUN_MONITORING.md)
for focused commands and the harmless TensorBoard TensorFlow notice.

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
The lightweight-model validation record is in
[docs/OPTIMIZATION_STUDY.md](docs/OPTIMIZATION_STUDY.md).
