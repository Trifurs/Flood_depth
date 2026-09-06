# Hydro-v14 optimization and sensor ablation report

## Scope and data contract

The implementation targets the audited `subset1000` dataset at
`/home/whu/桌面/myData/Flood_depth/subset1000`. The existing train/validation/test
contract, manifest, normalization statistics, event-chain separation, and labels
were not modified. The contract has 825 train, 89 validation, and 83 test rows.

The selected Hydro-v14 band specification retains the audited `hydro_compact`
19-channel set. Train-only raster-stratified statistics, correlations, and
semantic masking support are complete; new v14 band-combination retraining and
three-seed confirmation remain explicitly marked pending in
`bands/band_candidate_summary.json`.

Hydro-v14 remains a conditional-positive flood-depth estimator. DSM is treated as
a DSM-derived ground-like terrain proxy; the graph is a weak topographic latent
aggregation prior, not a hydraulic solver, PINN, conservation law, or water-flow
simulation.

## Implemented changes

- Added name-resolved, mask-aware terrain features: metric gradients, local mean,
  ground-like DSM proxy, obstacle residual, local relief, and intermediate-pixel
  path barriers for graph edges.
- Added an eight-neighbour Edge-KAN with fixed train-only descriptor calibration,
  symmetric static topographic descriptors, bounded gamma, zero residual
  initialization, and diagnostic/function regularization outputs.
- Added independent sensor/terrain decoder gates; the gates are availability
  weighted but are not a softmax competition.
- Added tolerant terrain-order and WSE-slope objectives, tail underprediction loss,
  frozen train-bin soft balancing support, and auxiliary-loss decay controls.
- Added an explicit `s2_enabled=false` ablation. When disabled, S2 images, S2
  availability, S2 QA, and S1--S2 timing are excluded from the forward path while
  the public batch input contract remains stable.

## Sentinel-2 ablation

The matched exploratory runs use the same seed (`20260831`), CPU device, model
configuration, optimizer, 3 epochs, and 5 train batches per epoch. Training-time
validation used one batch per epoch only to select a checkpoint; both selected
checkpoints were then reevaluated on all 23 validation batches.

| variant | pixel micro MAE (m) | RMSE (m) | P90 abs. error (m) | bias (m) | sample macro MAE (m) | event macro MAE (m) |
|---|---:|---:|---:|---:|---:|---:|
| full inputs | 0.47692 | 0.84267 | 1.07016 | -0.22286 | 0.44929 | 0.44899 |
| no Sentinel-2 | 0.48293 | 0.84176 | 1.05928 | -0.20444 | 0.45845 | 0.45983 |

On the primary pixel-micro MAE, no-S2 is worse by `0.00601 m` (`+1.26%`). It
does improve P90 absolute error by `0.01088 m`, RMSE by `0.00091 m`, and reduces
the negative bias, but the MAE and event/sample macro metrics consistently favor
the full-input variant. Therefore no-S2 is not promoted as the default from this
experiment. The result is exploratory rather than a final multi-seed training
claim because the CPU budget was intentionally short.

## Resumed structural screen

The interrupted six-candidate screen was resumed into new directories without
overwriting the partial pre-interruption directories. Each candidate received
the same one-batch training step and was then reevaluated on all 23 validation
batches. These values are diagnostic only; the delayed physics terms are still
inactive at epoch 0 and the zero-initialized KAN residual has not had enough
training to support a superiority claim.

| candidate | pixel micro MAE (m) | RMSE (m) | P90 abs. error (m) | bias (m) |
|---|---:|---:|---:|---:|
| no graph | 0.52638 | 0.80855 | 0.87227 | 0.03872 |
| matched MLP edge gate | 0.51838 | 0.80881 | 0.95066 | -0.00217 |
| Edge-KAN scale 4 | 0.52638 | 0.80855 | 0.87224 | 0.03872 |
| Edge-KAN scale 8 | 0.52638 | 0.80855 | 0.87224 | 0.03872 |
| terrain-order | 0.52638 | 0.80855 | 0.87224 | 0.03872 |
| WSE-slope | 0.52638 | 0.80855 | 0.87224 | 0.03872 |

The close KAN/no-graph values are consistent with the intended near-identity
initialization, not evidence that KAN is useless. A longer, multi-seed run is
required before selecting KAN or either physics term.

The validation contract itself has high S2 event-composite availability: mean
pixel validity is approximately `97.74%`, median `99.69%`, and no validation
sample has less than 50% valid S2 pixels. This does not disprove cloud-related
failure cases outside this audited subset, but it explains why removing S2 does
not currently provide a clear advantage on this split.

## Reproducibility artifacts

- `artifacts/optimization/hydrov14/sensor_ablation_summary.json`
- `runs/optimization/hydrov14/controlled/full_inputs_3epoch_5batches/`
- `runs/optimization/hydrov14/controlled/no_s2_3epoch_5batches/`
- `artifacts/optimization/hydrov14/controlled/full_inputs_3epoch_5batches_val_full/`
- `artifacts/optimization/hydrov14/controlled/no_s2_3epoch_5batches_val_full/`
- `artifacts/optimization/hydrov14/bands/train_band_report.json`
- `artifacts/optimization/hydrov14/bands/train_depth_statistics.json`
- `artifacts/optimization/hydrov14/bands/graph_edge_stats_scale4.json`
- `artifacts/optimization/hydrov14/bands/graph_edge_stats_scale8.json`
- `artifacts/optimization/hydrov14/kan_diagnostics.json`
- `artifacts/optimization/hydrov14/kan_curves/curve_summary.json`
- `artifacts/optimization/hydrov14/ablation_summary.csv`
- `artifacts/optimization/hydrov14/ablation_summary.json`
- `artifacts/optimization/hydrov14/physics_diagnostics.json`
- `artifacts/optimization/hydrov14/final_decision.json`
- `artifacts/optimization/hydrov14/final_profile.json`
- `artifacts/optimization/hydrov14/bands/selected_band_spec.json`
- `artifacts/optimization/hydrov14/bands/band_candidate_summary.csv`
- `artifacts/optimization/hydrov14/bands/band_candidate_summary.json`
- `artifacts/optimization/hydrov14/controlled/full_inputs_profile.json`
- `artifacts/optimization/hydrov14/controlled/no_s2_profile.json`

The complete test suite after the changes is `116 passed, 2 skipped`.

The minimal CPU profile reports finite gradients for both variants. The full-input
model has 5,228,010 parameters and a measured batch-4 forward mean of about
`15.87 s`; no-S2 has the same parameter count because the public model contract
and inactive S2 module are retained, with a measured batch-4 forward mean of about
`19.87 s` in this run. These are CPU engineering measurements, not GPU throughput
claims.
