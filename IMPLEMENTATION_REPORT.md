# PA-HydroKAN implementation report

## Completion status

The requested single-model flood-depth project is implemented and validated on the
real immutable `subset150` dataset. Dataset audit, train-only statistics, structured
loading, PA-HydroKAN forward/backward, masked losses, nnPU, cubic B-spline KAN,
checkpoint save/load, validation metrics, visualization, and georeferenced GeoTIFF
export all passed. No ablation/comparison model was created and no source dataset file
was written, deleted, or rewritten.

Validated environment:

- Conda environment: `flood-depth`
- Python 3.10.20
- PyTorch 2.11.0+cu130
- Host GPU: NVIDIA GeForce RTX 5090, 32 GB
- NVIDIA driver 595.84; host-reported CUDA 13.2
- rasterio 1.4.4; NumPy 2.2.5; pytest 9.0.3; TensorBoard 2.21.0

The default command sandbox hides NVIDIA devices, but an authorized host-side PyTorch
test found one RTX 5090 and completed a CUDA tensor operation. All final model/smoke
GPU checks were therefore run in the authorized host context.

## Created project tree

```text
Flood_depth/
├── README.md
├── IMPLEMENTATION_REPORT.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── README.md
│   ├── base.xml
│   ├── config.xml
│   ├── datasets/flooddepth_subset150.xml
│   └── pa_hydrokan/subset150_main.xml
├── datasets/
│   ├── __init__.py
│   ├── contract.py
│   ├── flooddepth_dataset.py
│   ├── preprocessing.py
│   ├── samplers.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── pa_hydrokan.py
│   ├── encoders.py
│   ├── asynchronous_fusion.py
│   ├── terrain_features.py
│   ├── terrain_graph_kan.py
│   ├── kan_layers.py
│   ├── decoder.py
│   └── heads.py
├── losses/
│   ├── __init__.py
│   ├── depth_losses.py
│   ├── pu_loss.py
│   ├── physics_losses.py
│   └── composite_loss.py
├── metrics/
│   ├── __init__.py
│   ├── depth_metrics.py
│   ├── uncertainty_metrics.py
│   ├── physical_metrics.py
│   └── aggregator.py
├── tools/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   ├── build_train_stats.py
│   ├── train.py
│   ├── evaluate.py
│   ├── test.py
│   ├── infer.py
│   ├── explore_data.py
│   ├── visualize_predictions.py
│   └── smoke_test.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── registry.py
│   ├── seed.py
│   ├── checkpoint.py
│   ├── logging.py
│   ├── raster_io.py
│   ├── visualization.py
│   ├── distributed.py
│   └── misc.py
├── tests/
│   ├── conftest.py
│   ├── test_dataset_contract.py
│   ├── test_dataset_loading.py
│   ├── test_masked_losses.py
│   ├── test_nnpu_loss.py
│   ├── test_kan_layer.py
│   ├── test_model_forward.py
│   ├── test_geotiff_export.py
│   └── test_smoke_training.py
├── docs/
│   ├── DATA_CONTRACT.md
│   ├── MODEL_ARCHITECTURE.md
│   ├── SCIENTIFIC_ASSUMPTIONS.md
│   └── TRAINING_AND_EVALUATION.md
├── artifacts/dataset_audit/
│   ├── subset150_audit.json
│   ├── subset150_contract.json
│   ├── subset150_report.md
│   └── subset150_train_stats.json
└── runs/
    ├── train/smoke_20260901_102131/
    └── infer/smoke_20260901_102131/
```

## Real dataset audit

Audit status: `ready`, with zero fatal errors and zero warnings.

- Manifest: 150 rows; train=105, val=23, test=22.
- Unique source events: 150; event chains: 60.
- Sample origins: 47 audited legacy, 11 expansion non-overlap, 92 strict supplement.
- Every one of 11 raster groups has exactly 105/23/22 files in train/val/test.
- All inspected rows have strict within-sample grid alignment.
- Shape: 256×256; resolution: 20×20 m; CRS: EPSG:27704.
- No dataset write lock was present.
- No valid raster value was NaN/Inf.
- Label: one float32 band `depth_m`, nodata=-9999, already in metres. Across all
  splits, valid label depth ranges from 0.10 to 24.82 m.
- DEM: float32 `elevation_m_DSM`, `slope_deg`, nodata=-9999. Elevation is treated as
  DSM. GeoTIFF/per-band validity is combined with `DEM_valid_mask` and
  `slope_valid_mask`.
- S1/T1: `VV_pre_db`, `VH_pre_db`, `angle_pre_deg`.
- S1/T2: `VV_event_db`, `VH_event_db`, `angle_event_deg`.
- S1/change: `VV_delta_db`, `VH_delta_db`, `anomaly_raw`, `anomaly_selection`.
- S1/QA: `event_observation_count`, `selected_pre_observation_count`,
  `selected_event_day_offset`, `selected_relative_orbit`, `selected_orbit_pass_code`.
- S2/T1 and T2: B2/B3/B4/B8/B11/B12 reflectance (pre and event descriptions).
- S2/change: `NDWI_delta`, `MNDWI_delta`, `water_change_selection`.
- S2/QA: `pre_clear_observation_count`, `event_clear_observation_count`,
  `selected_event_day_offset`.
- Masks: 10 uint8 bands, no nodata metadata. False is 0 and true may be 1 or 255;
  loader semantics are `value > 0`.
- `valid_depth_mask` equals `flood_mask` at every audited pixel. There are 1,256,474
  reliable positive pixels and 8,573,926 unknown pixels across all splits. The label
  raster validity mask equals `valid_depth_mask` exactly.
- S1 event validity is complete in this subset; S2 has 177,645 invalid pixels. DEM/
  slope GeoTIFF validity excludes 120,159 pixels across all splits.

Observed differences and decisions:

1. The expected band names/counts, 27 main channels, shape, CRS, resolution, and label
   unit all match.
2. Binary masks use both 1 and 255 as true encodings. This is scientifically
   equivalent and is explicitly adapted using `value > 0`.
3. The dataset-provided normalization is explicitly train-only, has matching bands,
   and positive standard deviations, but lacks p0.5/p99.5 robust quantiles. It was not
   used directly. Exact train-only statistics were rebuilt in the project.
4. The explicit DEM validity band is all true, while slope/GeoTIFF masks contain edge
   invalidity. Runtime output validity therefore uses their intersection, preventing
   nodata tensor fills from being interpreted as terrain observations.

Train-only statistics scanned one band at a time and excluded val/test. All 27 bands
contain count/mean/std/min/max/p0.5/p1/p99/p99.5. The conservative nnPU prior proxy is
905,471 reliable positives among 6,792,514 eligible train output pixels:
`pi=0.1333042523`. This is logged as an observed-label proxy, not true flood prevalence.

## Key SHA256 fingerprints

| File | SHA256 |
|---|---|
| Source `README.md` | `4feb72ce5417f1b39285a767d08db490d5e323910ad7769cc76f82c12b624d1f` |
| `metadata/training_manifest.csv` | `4a659f6d193302ab0f9bdc1a1a57c3d54c3e9b17f111604c888e7c8cc23703d5` |
| `metadata/algorithm_spec.json` | `abe25629b2ec4755141775f78fe28535b4cbd57688f191cb4df69979e787f66d` |
| `metadata/materialization_policy.json` | `06010cddc4483455d60cedf9fc4732643651766263a8af1bfcb5732961d4f895` |
| `metadata/spatial_policy.json` | `b5cd0a47364693b8c8c714663699941144c6e2bc46c01c2369cc3acfe672d7da` |
| Provided normalization | `ba55ea2589a4830e2425ee16adda0fa895ba51ceb6b0b3b60fc4bc1d427afd90` |
| Generated audit JSON | `63d2a92b22f99a5b5ffd98cf69473a05c8d1bf8e72263d1fb86c78adc72ea45b` |
| Runtime contract | `79e33f3174163f7fcdb137ed2779fe5680b1fff658c30ea1fa71548517744a06` |
| Generated train statistics | `15ef0977ba69137818e76bcd779dd7f0cf1e81195dd326716ba45d3328441319` |

The contract also records hashes for all top-level dataset JSON policies and
`subset_manifest.csv`. Every saved resolved configuration embeds all ten audited
source-file hashes plus contract and generated-normalization hashes. Checkpoints carry
contract/manifest/normalization fingerprints and reject mismatched resume by default.

## PA-HydroKAN implementation and parameter count

| Module | Parameters |
|---|---:|
| S1 temporal/change encoder + incidence FiLM | 5,925,920 |
| S2 temporal/change encoder | 5,920,736 |
| Mask-aware terrain pyramid | 1,958,976 |
| Asynchronous reliability fusion | 1,838,376 |
| 1/8-scale eight-neighbour Terrain Graph-KAN | 131,954 |
| Decoder | 969,024 |
| Three output heads | 99 |
| **Total / trainable** | **16,745,085 / 16,745,085** |

The KAN edge layer contains learnable cubic B-spline coefficients on a fixed open
uniform knot grid plus a SiLU base path. Eight fixed direction shifts are vectorized;
there is no per-pixel Python loop and no torch_geometric dependency. S1/S2 fusion uses
masked per-scale reliability softmax and correctly handles either/both missing
modalities. The model only accepts a label-independent whitelist; label and
label-derived masks are rejected from `model.forward`.

## Commands actually executed

```bash
conda create -n flood-depth --clone landslidenet -y
conda install -n flood-depth pytest tensorboard -y

conda run --no-capture-output -n flood-depth python \
  tools/inspect_dataset.py \
  --root /home/whu/桌面/myData/Flood_depth/subset150 \
  --output artifacts/dataset_audit

conda run --no-capture-output -n flood-depth python \
  tools/build_train_stats.py \
  --config configs/pa_hydrokan/subset150_main.xml

conda run --no-capture-output -n flood-depth python -m pytest -q

conda run --no-capture-output -n flood-depth python \
  tools/smoke_test.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --device auto --train-batches 2 --val-batches 1

conda run --no-capture-output -n flood-depth python \
  tools/infer.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --checkpoint runs/train/smoke_20260901_102131/last.pth \
  --input JRC_2017_244_2017-09-25_OBJ_0056_R000091_C000000 \
  --device auto --output runs/infer/cli_validation_20260901_102131 \
  --save-geotiff --save-visualization

conda run --no-capture-output -n flood-depth python \
  tools/train.py \
  --config configs/pa_hydrokan/subset150_main.xml \
  --device auto --epochs 1 --batch-size 1 --num-workers 0 \
  --max-train-batches 2 --max-val-batches 1
```

Final pytest result: **11 passed in 4.40 s**.

## Final smoke result and products

- Status: passed on `cuda` / NVIDIA GeForce RTX 5090.
- Real train batches: 2; real val batches: 1.
- Successful optimizer updates: 1 (one mixed-precision update was safely skipped by
  GradScaler's overflow detection; the required optimizer update completed).
- Train losses: 0.77443725, 0.83516169.
- Checkpoint save/reload: passed; loaded epoch 0 with full model/optimizer/scheduler/
  GradScaler state and matching fingerprints.
- One-batch validation event-macro MAE: 0.34044370 m. This is an engineering smoke
  metric, not a trained-model scientific result.
- Peak allocated GPU memory: 1,872,594,944 bytes (about 1.74 GiB).
- Checkpoint:
  `runs/train/smoke_20260901_102131/last.pth`
- Prediction:
  `runs/infer/smoke_20260901_102131/JRC_2017_244_2017-09-25_OBJ_0056_R000091_C000000/predicted_depth_m.tif`
- Companion products: `support_probability.tif`, `uncertainty_scale_m.tif`,
  `prediction_panel.png`, `metrics.json`.
- The independent `tools/infer.py` CLI also completed and wrote the same product set
  beneath `runs/infer/cli_validation_20260901_102131/`.

The final prediction is 256×256, EPSG:27704, 20 m, nodata=-9999, and retains affine
`(20,0,6976780,0,-20,1568420)`. It contains 65,025 valid output pixels and 511 explicit
nodata pixels from the DEM/slope validity intersection. The valid predicted range in
this smoke output is 0.0108–0.7484 m.

The unified `tools/train.py` quick-debug entry also completed successfully at
`runs/train/pa_hydrokan_subset150_main_20260901_102403/`. It wrote both `last.pth` and
`best.pth`, step/epoch CSV, TensorBoard events, resolved config with all source hashes,
environment/model/runtime JSON, and logs. Its one-batch validation event-macro MAE was
0.38603 m; this too is only an entry-point validation, not a performance result.

## Issues to address before formal full training

1. The quality-selected subset is biased toward clear S1/S2 change and is too small
   for final scientific claims.
2. Event-reconstructed labels need independent in-situ/external validation; partial
   positives cannot validate complete flood extent.
3. The automatic nnPU prior is only an observed-label fraction proxy. Formal work
   should pre-register a sensitivity analysis without using test to choose it.
4. S1/S2 per-pixel event composites remain asynchronous; predicted depth must not be
   interpreted as an instantaneous hydraulic state.
5. DSM is not riverbed/DTM. Weak WSE curvature and Graph-KAN gates must retain their
   latent/regularization interpretation.
6. Full 120-epoch training was intentionally not started. The smoke metric is not a
   performance benchmark; use validation only for selection and evaluate test once
   after the final protocol is frozen.
