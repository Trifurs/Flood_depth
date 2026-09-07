# Flood-depth estimation from SAR and terrain

This is the production code path for flood-depth estimation using Sentinel-1
state, event, change, acquisition-reliability, and terrain inputs. Labels and
label-derived masks are used only for supervision and evaluation.

The project intentionally contains one model, one dataset-input contract, and
one default configuration. Checkpoints, comparisons, and alternative sensor
paths are not included.

## Layout

- `configs/config.xml` — default training and inference configuration.
- `assets/` — audited dataset contract and train-only normalization statistics.
- `models/` — SAR encoder, terrain features, Edge-KAN graph, decoder, and heads.
- `tools/` — train, evaluate, and single-sample inference commands.
- `runs/` — locally generated training outputs; ignored by Git.

## Train

Install a PyTorch build suitable for the local accelerator, then install the
remaining packages:

```bash
pip install -r requirements.txt
conda run -n flood-depth python tools/train.py \
  --config configs/config.xml \
  --device cuda
```

Training creates a timestamped directory under `runs/train/`. It includes the
resolved configuration, dataset fingerprint, calibration artifacts, raw and
EMA checkpoints, metrics, and runtime metadata.

## Evaluate

```bash
conda run -n flood-depth python tools/evaluate.py \
  --config configs/config.xml \
  --checkpoint runs/train/<run>/best_raw.pth \
  --split val \
  --device cuda
```

## Infer one known sample

```bash
conda run -n flood-depth python tools/infer.py \
  --config configs/config.xml \
  --checkpoint runs/train/<run>/best_raw.pth \
  --input <sample-id> \
  --device cuda \
  --save-geotiff
```

No trained checkpoint is bundled: previous training results were deliberately
removed before this production reset.
