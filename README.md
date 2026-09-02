# PA-HydroKAN flood-depth estimation

This repository implements PA-HydroKAN as the main model and keeps independently
adapted baselines under `compare/` for **event-aggregated continuous flood-depth
reconstruction** from asynchronous Sentinel-1/Sentinel-2 composites and DSM terrain.
It supports dataset auditing, train-only normalization, training,
validation, held-out testing, checkpoint resume, one-sample inference, uncertainty,
and georeferenced GeoTIFF export. The code does not depend on the reference
[DEHCD-Net repository](https://github.com/Trifurs/DEHCD-Net); only its compact
`configs/datasets/models/tools/utils` workflow inspired the organization.

## Scientific scope

S1/T2 and S2/T2 are per-pixel composites over an event interval, not single scenes.
Different pixels in one patch can come from different dates, and S1/T2 and S2/T2 are
generally asynchronous. T1/T2 are therefore not treated as regular time series or
used to estimate temporal derivatives.

The JRC `depth_m` label is an event-interval reconstructed/aggregated reference. Only
pixels with `valid_depth_mask=1` have reliable positive continuous-depth supervision.
`flood_mask` is identical to that positive set in subset150. Unknown,
permanent-water, extreme-high, and label-nodata pixels are **not 0 m depth targets**
and never enter the supervised depth loss. For the separate extent experiment,
`valid_depth_mask` is explicitly adopted as the binary flood label over
`output_valid`; IoU, F1, precision, and recall are reported under that definition.

The elevation source is a DSM, not a bare-earth DTM. PA-HydroKAN is physics-guided:
it uses mask-aware low-frequency terrain proxies and a weak local water-surface
curvature regularizer. It is not a full
PINN, does not solve the shallow-water equations, and does not claim mass
conservation or hydrodynamic simulation.

## Audited input contract

The expected layout is:

```text
subset150/
  README.md
  metadata/training_manifest.csv
  train|val|test/
    label/  DEM/
    S1/T1/ S1/T2/ S1/change/ S1/QA/
    S2/T1/ S2/T2/ S2/change/ S2/QA/
    masks/
```

The manifest's existing train/val/test split is immutable. The loader never creates a
new split and never uses test for normalization, early stopping, model selection,
threshold selection, or depth-bin selection. `subset150` is only the current code
validation dataset; no source code assumes that the dataset contains 150 samples.

The 27 main continuous channels are:

| Group | Channels |
|---|---|
| S1/T1 | `VV_pre_db`, `VH_pre_db`, `angle_pre_deg` |
| S1/T2 | `VV_event_db`, `VH_event_db`, `angle_event_deg` |
| S1/change | `VV_delta_db`, `VH_delta_db`, `anomaly_raw`, `anomaly_selection` |
| S2/T1 | B2/B3/B4/B8/B11/B12 pre-event reflectance |
| S2/T2 | B2/B3/B4/B8/B11/B12 event-composite reflectance |
| S2/change | `NDWI_delta`, `MNDWI_delta`, `water_change_selection` |
| DEM | `elevation_m_DSM`, `slope_deg` |

QA is used only for safe reliability features. Observation counts and selected event
day offsets are allowed. `selected_pre_observation_count`, relative orbit, and orbit
pass are disabled and zeroed. Sensor/DEM validity masks can gate features;
label-derived masks are passed only to loss and evaluation for learned depth models.
The independent extent model uses `valid_depth_mask` as its binary flood label and
minimizes masked Soft-IoU; frozen inference does not receive the label. All geometry
methods reuse that one predicted extent. `persistent_water` is not a learned-depth
or geometry-method input.

The generated runtime contract is
`artifacts/dataset_audit/subset150_contract.json`. Channel lookup is resolved from its
audited band descriptions rather than scattered numeric indices. Source README,
manifest, metadata, contract, and normalization fingerprints are checked before use
and stored in checkpoints.

## Environment

Python 3.10+ is required. Install the PyTorch build that matches the machine's CUDA
first; do not let a generic requirements install replace it.

```bash
conda create -n flood-depth python=3.10 -y
conda activate flood-depth
# Install the matching PyTorch CUDA or CPU build first.
python -m pip install -r requirements.txt
```

On the implementation machine, the validated environment was created by cloning the
existing compatible CUDA environment and adding the two missing tools:

```bash
conda create -n flood-depth --clone landslidenet -y
conda install -n flood-depth pytest tensorboard -y
conda activate flood-depth
```

## Data audit and frozen train statistics

```bash
python tools/inspect_dataset.py \
  --root "/home/whu/桌面/myData/Flood_depth/subset150" \
  --output artifacts/dataset_audit

python tools/build_train_stats.py \
  --config configs/pa_hydrokan/subset150_main.xml

python tools/analyze_physical_consistency.py \
  --config configs/pa_hydrokan/subset150_main.xml
```

The second command scans only valid train pixels, one band at a time. It records
count, mean, standard deviation, minimum, maximum, p0.5, p1, p99, and p99.5; it also
freezes train depth bins and the conservative observed valid-depth fraction used to
initialize/configure nnPU. Val and test pixels are excluded.
The third command audits candidate local physical assumptions on train/val targets;
its split choices deliberately do not include test.

## Model

PA-HydroKAN (Partial-label Asynchronous Hydro-topographic Kolmogorov–Arnold Network)
contains:

1. independent S1 and S2 encoders with shared T1/T2 weights inside each modality;
2. pre/event/difference/absolute-difference/change fusion at four scales;
3. mask-aware online DSM proxies (`z_hyd`, relative height, barrier, gradients, local
   relief) at the retained 9-pixel terrain context, never written back to source data;
4. per-scale masked reliability softmax for asynchronous S1/S2 fusion;
5. a vectorized eight-neighbour Terrain Graph-KAN at 1/8 resolution, whose internal
   `KANLinear` learns fixed-grid cubic B-spline univariate functions;
6. a U-Net/FPN decoder and support, positive-depth, and Laplace-scale heads.

Hydro-v2 separates two estimands. Conditional flood depth is
`d_cond=softplus(raw_depth)` and is the continuous quantity trained/evaluated on the
reliable positive set. `sigmoid(support_logits) * d_cond` is retained as an explicitly
named support-weighted diagnostic, but it no longer attenuates the supervised depth.
The support score is not claimed to be a calibrated complete flood-extent probability.
Graph gates are latent terrain-conditioned connectivity, not real flow, velocity, or
flux. Pre-v2 checkpoints retain their legacy probability-weighted output semantics
when loaded.

Hydro-v5 keeps those output semantics and targets the observed deep-water failure
mode: exact train-only q90/q95/q97.5 boundaries refine the formerly single
0.48--24.82 m tail stratum. A hierarchical reduction first averages refined cells
inside their shallow/mid/deep parent and then gives each non-empty parent equal
weight. This changes neither inputs nor label masks and prevents both shallow-tail
dominance and accidental over-weighting of deep water.

Hydro-v6 keeps the Hydro-v5 depth hierarchy and changes only the weak physical
objective. A train/validation target audit showed that an unconditional
zero-curvature water-surface penalty was not justified: the Hydro-v5 predictions
were already smoother than the reconstructed references. The active loss instead
penalizes predicted depth increasing toward higher `z_hyd` across local positive
neighbour pairs with a 0.02--0.75 m terrain step. It is downweighted for asynchronous
sensor support, starts at epoch 25, and warms up for 15 epochs. Constant depth and
downhill deepening remain unpenalized; this is a local ordering prior, not a
flat-water or flow-equation constraint.

Hydro-v9 retains that objective and adds DSM-only 33/65-pixel signed relative height,
depression-depth proxy, and local relief beside the original 9-pixel context. These
features supply floodplain-scale topographic context without pretending that the DSM
is a riverbed DTM or that a drainage network is available. The frozen test comparison
improves the event--hierarchical composite MAE from 0.230228 to 0.224811, while pixel
RMSE and uncertainty calibration become worse; see the dedicated report for the full
trade-off.

Hydro-v12 is the current pixel-first default. The 33/65-pixel Hydro-v9 contexts and
event-conditioned scale are disabled. Supervised depth and uncertainty losses balance
frozen train-depth strata inside each input raster and then average rasters, without
reading event identity. The best checkpoint is selected by validation
`pixel_micro_mae`; RMSE and error-band accuracies are mandatory secondary checks.
On subset150, all 105 train rasters have distinct source-event IDs, so the older
event-depth reduction was numerically identical to a per-raster reduction. The gain
therefore comes from aligning checkpoint selection with deployment accuracy, not from
learning event-specific behavior.

Hydro-v13 is available as an explicit engineering candidate while the repository's
default configuration remains Hydro-v12. It resolves model inputs by exact contract
band names, reads only the selected continuous raster bands, and records the resolved
`BandSpec` in configuration/checkpoint metadata. The validated compact configuration
uses VV/VH before and during the event, their two deltas plus `anomaly_raw`, incidence
angles as conditioning, B3/B8/B11 at both times, NDWI/MNDWI changes, DSM elevation,
and slope. Its efficient cross-state encoders, content-aware sensor/terrain fusion,
multi-head terrain Graph-KAN, gated FPN decoder, deep supervision, and separate task
heads are configured in `configs/pa_hydrokan/subset150_v13_final.xml`.

To define another input view, copy one of the v13 band configurations and edit only
the exact names under `dataset.model_bands`; missing or duplicate names fail early.
Legacy configurations without `model_bands` retain the full-band behavior.

```bash
conda run --no-capture-output -n flood-depth \
  python tools/train.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --device auto --seed 20260831 \
  --output runs/optimization/hydrov13/final_seed_20260831

conda run --no-capture-output -n flood-depth \
  python tools/evaluate.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --checkpoint runs/optimization/hydrov13/final_seed_20260831/best.pth \
  --weights raw --split val --device auto \
  --output runs/optimization/hydrov13/final_validation_raw

conda run --no-capture-output -n flood-depth \
  python tools/evaluate.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --checkpoint runs/optimization/hydrov13/final_seed_20260831/best.pth \
  --weights ema --split val --device auto \
  --output runs/optimization/hydrov13/final_validation_ema

conda run --no-capture-output -n flood-depth \
  python tools/analyze_band_importance.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --checkpoint runs/train/pa_hydrokan_subset150_hydrov12_raster_depth_balance_20260901_153904/best.pth \
  --split val --device auto \
  --output artifacts/optimization/hydrov13/band_mask_importance

conda run --no-capture-output -n flood-depth \
  python tools/profile_model.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --device auto --batch-size 4 --iterations 50 \
  --output artifacts/optimization/hydrov13/final_model_profile.json

conda run --no-capture-output -n flood-depth \
  python tools/train.py \
  --config configs/pa_hydrokan/subset150_v13_final.xml \
  --device auto \
  --resume runs/optimization/hydrov13/final_seed_20260831/last.pth
```

Resume now checks a semantic training-identity hash in addition to source
fingerprints. Changing the model, bands, reliability schema, loss, optimizer, or
scheduler requires a new run; epochs, logging, workers, device, and output location
remain runtime choices. Full measurements and candidate decisions are recorded in
`docs/HYDRO_V13_ENGINEERING_REPORT.md`.

## Comparison models

The `compare/` directory follows the engineering role of DEHCD-Net's comparison
folder without making the runtime depend on that repository. It contains clean-room,
task-adapted DLSIM LinkNet and Attention U-Net baselines, plus separately evaluated
FwDET-, RICorDE-, and FLEXTH-style terrain methods. The DLSIM models map the seven
safe S1/S2 change bands to one learned change-evidence channel and use normalized
slope as the second channel. This is the closest no-leakage adaptation of DLSIM's
binary-change-plus-slope contract because subset150 has no independent complete
flood-extent raster.

Learned comparison models share the immutable split, train-only statistics, augmentations,
partial-positive objective, weak physical term, optimizer, pixel-first validation
selection, checkpoints, and evaluator with PA-HydroKAN.  They never receive label
masks, event identity, split, or sample origin. The terrain methods instead use the
same frozen extent predicted once by the independent `extent/` workflow from
pre/event Sentinel-1 VV/VH change evidence. Extent code/checkpoints/results remain
under `extent/` and `runs/extent/`; depth estimation remains under `compare/` and its
own run directories. See `extent/README.md`, `compare/README.md`,
`docs/COMPARISON_METHOD_AUDIT.md`, and `docs/COMPARISON_RESULTS.md` for the selection
rationale and frozen held-out results.

## Training and resume

```bash
python tools/train.py \
  --config configs/pa_hydrokan/subset150_main.xml

python tools/train.py \
  --config configs/compare/subset150_dlsim_attention_unet.xml

python tools/train.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --epochs 1 --max-train-batches 2 --max-val-batches 1

python tools/train.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --resume runs/train/<run>/last.pth
```

CLI overrides include `--device`, `--epochs`, `--batch-size`, `--num-workers`,
`--resume`, `--max-train-batches`, `--max-val-batches`, `--no-amp`, and `--seed`.
Single CPU, one GPU, and `torchrun` DDP are supported. Resume restores model,
optimizer, scheduler, GradScaler, epoch, best metric, and random states. A changed
contract/manifest/normalization fingerprint is rejected unless the explicitly
dangerous `--allow-fingerprint-mismatch` flag is supplied. A changed loss
configuration is always rejected for resume and must start a new run.

Each non-DDP training epoch visits every train sample exactly once without replacement.
Hydro-v12 supervised depth and Laplace losses first average errors within each frozen
train-depth stratum of a raster and then average the non-empty raster strata. Event IDs
are ignored by the objective. Best-model selection minimizes validation pixel-micro
MAE, which weights every labelled deployment pixel equally. Event-level metrics remain
available only as diagnostics. The weak WSE-curvature term starts at epoch 5 and warms
up to weight 0.02 over 15 epochs.
`last.pth` and `best.pth`, resolved
configuration, environment, dataset fingerprints, step/epoch CSV, TensorBoard logs,
runtime, and peak GPU memory are stored under `runs/train/`.

## Evaluation, held-out testing, and inference

```bash
python tools/evaluate.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --checkpoint <checkpoint.pth> --split val

python tools/test.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --checkpoint <best.pth> --split test --save-predictions

python extent/train.py \
  --config configs/extent/subset150_ai4g_mobilenet_iou.xml --device cuda \
  --output runs/extent/train/subset150_ai4g_mobilenet_iou_frozen

python extent/predict.py \
  --config configs/extent/subset150_ai4g_mobilenet_iou.xml \
  --checkpoint runs/extent/train/subset150_ai4g_mobilenet_iou_frozen/best.pth \
  --output runs/extent/products/subset150_ai4g_mobilenet_iou_frozen \
  --splits val test --device cuda

python tools/evaluate_geometry.py \
  --config configs/compare/subset150_geometry_predicted_extent.xml \
  --extent-root runs/extent/products/subset150_ai4g_mobilenet_iou_frozen \
  --split val --output runs/geometry_predicted_extent/val_iou_frozen

python tools/evaluate_geometry.py \
  --config configs/compare/subset150_geometry_predicted_extent.xml \
  --extent-root runs/extent/products/subset150_ai4g_mobilenet_iou_frozen \
  --split test --save-predictions \
  --output runs/test/compare_geometry_iou_extent_final

python tools/infer.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --checkpoint <best.pth> --input <sample_id_or_dataset_raster_path> \
  --save-geotiff --save-visualization
```

The primary metrics are pixel-micro MAE and RMSE. Additional diagnostics include
sample macro, event macro, flat event-depth-bin macro, and event-depth-hierarchical macro
MAE/RMSE/median absolute
error/bias/R²/NSE/log1p-MAE/P90 and 0.25/0.50/1.00 m accuracy bands; per-sample,
per-event, and frozen train-depth-bin CSV files are written. Laplace NLL, 50/80/90/95
percent coverage/width, uncertainty-error Spearman correlation, support recall on
known positives, and support area diagnostics are also emitted. Physical diagnostics
include reference-gated local WSE-gradient error, local terrain-order violation
magnitude/fraction, WSE-Laplacian reference error, and high-relief prediction
continuity. These are diagnostics on reliable positive neighbourhoods, not proof of
hydrodynamic validity. Event IDs are not required for inference and do not alter a
prediction; they are used only to group optional diagnostic rows.

Each saved sample contains `predicted_depth_m.tif` (the checkpoint's primary depth
semantics), `conditional_depth_m.tif`, `support_weighted_depth_m.tif`,
`support_probability.tif`, `uncertainty_scale_m.tif`, `prediction_panel.png`, and
metrics. GeoTIFFs retain source
CRS, transform, width, and height. Their output mask is `DEM valid AND (S1 valid OR S2
valid)`; label validity is never used to crop inference, and unsupported pixels are
written as explicit nodata rather than fake 0 m.

The extent workflow separately writes `flood_probability.tif`,
`flood_extent_raw.tif`, and buffered `flood_extent.tif`. The geometry evaluator
writes `predicted_depth_m.tif`, `water_surface_elevation_m.tif`, and a copy named
`input_predicted_flood_extent.tif` for each method and sample. It verifies the
extent-product fingerprint and refuses a manifest that declares label-derived
inference. Target depth and `valid_depth_mask` enter only evaluation after prediction.

## Verification

```bash
python -m pytest -q

python tools/smoke_test.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --device auto --train-batches 2 --val-batches 1
```

The smoke workflow uses real train and val rasters, executes forward/backward and at
least one optimizer update, atomically saves/reloads a full checkpoint, and exports a
real georeferenced prediction. It does not read test during training or validation.

## Current limitations

- The subset was quality-selected for clear S1/S2 change and may overrepresent easier
  scenes.
- Labels are reconstructed references, not necessarily in-situ water-depth surveys.
- Partial positive labels cannot establish complete flood-extent accuracy.
- DSM-based local terrain ordering is not a riverbed or hydraulic solver.
- Extent metrics use the experiment's explicit definition that `valid_depth_mask`
  is the binary flood label; their interpretation is conditional on that definition.
- FwDET/RICorDE/FLEXTH amplify extent and DSM errors; RICorDE additionally uses a
  declared local pseudo-drainage substitute rather than a conditioned DEM/network.
- Asynchronous per-pixel composites limit instantaneous process interpretation.
- Scientific conclusions require the future complete dataset and independent external
  validation.

See `docs/SCIENTIFIC_ASSUMPTIONS.md`, `docs/DATA_CONTRACT.md`,
`docs/MODEL_ARCHITECTURE.md`, `docs/TRAINING_AND_EVALUATION.md`, and
`docs/HYDRO_V5_OPTIMIZATION_REPORT.md` for the first optimization round, and
`docs/HYDRO_V6_LOSS_PHYSICS_REPORT.md` for the frozen loss/physics ablations and
Hydro-v6 validation comparison, and
`docs/HYDRO_V9_TASK_SPECIFIC_OPTIMIZATION_REPORT.md` for the online comparison,
rejected task-specific candidates, and frozen Hydro-v6/Hydro-v9 test benchmark, and
`docs/HYDRO_V12_PIXEL_FIRST_REPORT.md` for the current event-independent objective
and frozen reproduction assets, and `docs/COMPARISON_RESULTS.md` for the frozen
comparison-model test benchmark.
