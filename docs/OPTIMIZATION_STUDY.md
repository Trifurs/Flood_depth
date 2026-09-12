# PA-HydroKAN lightweight optimization record

## Scope and decision rule

This record uses only the fixed validation split. The official test split was
not read during architecture, calibration, or efficiency selection. Every row
uses one checkpoint for all metrics; metrics are never selected from different
epochs. The formal claim must ultimately be based on the event-grouped
outer-test folds. Per the final experiment protocol, the former official test
rows are included in that full-sample cross-validation pool; they were not used
during the optimization work recorded here.

PA-HydroKAN is mask agnostic: no flood extent, `valid_depth_mask`, or label is a
model input. The learned comparators use `valid_depth_mask` as their prescribed
oracle flood range, so this input asymmetry must accompany every comparison.

## Retained lightweight design

- Joint TCSE replaces redundant pre/event/change encoder pyramids with one
  joint temporal-change pyramid.
- Encoder widths are `[24, 48, 96, 144]`; DGD widths are `[96, 64, 48, 32]`
  with additive skip fusion.
- Full spatial residual blocks are retained where the bottleneck alternative
  lost substantial accuracy and did not improve epoch time.
- CDH uses a 16-channel contextual depth-range calibrator and a 32-channel
  global evidence calibrator. Both consume mask-agnostic SAR, terrain,
  acquisition-reliability, and sensor/DEM-availability evidence.
- The inactive learned uncertainty head is omitted when `lambda_unc=0`, saving
  9,313 parameters without changing conditional-depth predictions.
- Global evidence moments are reduced in four contiguous groups instead of
  materializing one large full-resolution concatenation.
- Exact reverse-path symmetry and one-sided axial pooling remove repeated
  full-resolution TAE-KAN barrier work. Evaluation uses
  `torch.inference_mode()` and omits training-only graph diagnostics.

The selected calibration uses local strength 9.0 above a 1.5 m smooth-tail
threshold, global scale strength 1.8, and global bias strength 2.0. These scalar
strengths introduce no parameters. They remain provisional until cross-fold
confirmation.

## Validation results

All efficiency values below use batch size 12, BF16, 256×256 patches, and an
RTX 5090. Forward timing is CUDA-synchronized and excludes data loading, metric
aggregation, and file output. The rows are diagnostic measurements from the
stored validation runs; publication tables must regenerate every model in one
`python test.py <runs-directory>` session to control timing conditions.

| Model/version | Parameters | Composite ↓ | MAE ↓ | RMSE ↓ | R² ↑ | Throughput (samples/s) ↑ | Mean batch latency (ms) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earlier spatial PA-HydroKAN | 8,011,332 | 0.315486 | 0.188502 | 0.511872 | — | 146.09 | 80.19 |
| Selected lightweight PA-HydroKAN | 2,598,151 | 0.290556 | 0.167431 | 0.440665 | 0.622068 | 286.07 | 40.95 |
| U-Net++ comparator | 1,171,417 | 0.273213 | 0.169940 | 0.376613 | 0.723950 | 741.36 | 15.80 |

Relative to the earlier spatial PA-HydroKAN, the selected version reduces
parameters by 67.6%, lowers composite error by 7.9%, lowers MAE by 11.2%, lowers
RMSE by 13.9%, nearly doubles throughput (+95.8%), and halves mean batch latency
(-48.9%). These are validation measurements, not official-test claims.

Against the current U-Net++ validation checkpoint, PA-HydroKAN is better on six
of eleven primary accuracy indicators: MAE (1.48%), median absolute error
(21.89%), absolute bias (20.09%), P90 absolute error (1.88%), within-0.25 m
accuracy (0.70%), and event-depth-hierarchical MAE (0.05%). U-Net++ remains
clearly better in RMSE/R², composite error, within-0.50/1.00 m accuracy, and
event MAE. Thus “best on all metrics” is not supported; the present result is a
majority-metric validation lead with a material extreme-error weakness.

Final post-refactor audit result:
`runs/flooddepthnet_s1_terrain/optimization/diagnostics/final_lightweight_refactor_audit_val/summary.json`.
It strictly loaded the selected validation-only checkpoint and ran the complete
validation split after dead-code removal and inference-path optimization.

## Rejected alternatives

- A 1.63 M-parameter bottleneck-spatial variant increased validation composite
  error to about 0.382 in its first epoch and was slower per epoch; rejected.
- Graph-scale barrier approximation improved throughput but degraded composite
  error from about 0.292 to 0.296; rejected because accuracy has priority.
- Channels-last memory format reduced throughput on this workload; rejected.
- Independent local scale/bias strength did not improve the Pareto frontier;
  removed.
- A low-learning-rate, RMSE-heavy whole-model refinement produced composites
  0.29231 and 0.29727 in its first two completed epochs; it was stopped and
  rejected rather than extending an adverse trend.

The remaining RMSE gap is concentrated in a small number of deep-water or
event-level outliers. The formal configuration therefore includes direct pixel,
event, depth-bin, and event-depth-hierarchical RMSE terms for the next clean
from-scratch run, while checkpoint selection remains the same balanced
four-metric composite for every learned model.
