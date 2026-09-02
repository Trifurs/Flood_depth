# Flood-depth comparison-method audit

## Leakage-free two-stage protocol

The comparison set has two deployment-oriented routes. PA-HydroKAN and the DLSIM
adapters directly estimate depth without label masks. FwDET-, RICorDE-, and
FLEXTH-style methods first consume one independently predicted flood extent and then
reconstruct depth from that extent and DSM terrain.

The previous `valid_depth_mask`-as-extent experiment has been superseded. Its files
may remain as historical diagnostics, but they are excluded from the formal ranking.
The active evaluator requires an `extent_product.json`, rejects a product declaring
`prediction_uses_valid_depth_mask=true`, verifies the immutable dataset fingerprint
and georeferencing, and passes the same loaded extent array to every geometry method.
Target depth and `valid_depth_mask` are accessed only by metric aggregation after a
prediction exists.

## Selected independent extent extractor

We selected Misra et al., “Mapping global floods with 10 years of satellite radar
data” (*Nature Communications*, 2025) rather than adding an optical or generic
segmentation baseline. Its early-fusion SAR change detector matches subset150's
pre/event Sentinel-1 VV/VH structure, is robust to cloud cover, has an official MIT
licensed implementation, and was externally evaluated on Kuro Siwo. The paper
reports internal IoU 0.67, precision 0.68, recall 0.99, and F1 0.80; its Kuro Siwo
cross-dataset F1 is 0.77. These are literature results, not local scores.

The local clean-room adaptation uses a torchvision MobileNetV2 encoder and U-Net
decoder (6,628,817 parameters). It reproduces the public two binary SAR-change inputs:
event VV below -17.5 dB or VH below -22.5 dB, each with at least a 5 dB pre/event
drop and the published validity floors. The output threshold is 0.5 and the public
80 m buffer becomes four pixels at the local 20 m resolution.

For this comparison, `valid_depth_mask` is explicitly defined as the binary flood
label over `output_valid`. Training minimizes pure masked Soft-IoU, and the best
checkpoint maximizes validation raw pixel-micro IoU at threshold 0.5. Precision,
recall and F1 use the same label definition. The test mask never selects a checkpoint,
threshold, buffer, or geometry parameter.

## Selected geometry methods

**FwDET v2.1 DSM adaptation.** Retains ten 5×5 shoreline-DSM smoothing iterations,
the 0.5% boundary-slope criterion, unit-cost nearest-boundary water-surface
allocation, non-negative depth, and a 3×3 low-pass filter. The ocean filter is
omitted because coastal identity and a compatible vertical datum are unavailable.

**RICorDE local-HAND DSM adaptation.** Declares the lowest 5% of DSM elevations in
each connected predicted component as pseudo-drainage, constructs nearest-drainage
HAND, caps shoreline HAND using q10/q90 with a 0.5 m floor and 7 m cap, and uses
five-neighbour power-2 IDW plus rolling-HAND smoothing. It is not claimed as a full
RICorDE reproduction because hydroconditioned terrain and drainage inputs are absent.

**FLEXTH method A DSM adaptation.** Retains two cross-kernel closing iterations, the
0.05 km² hole threshold, 0.05 slope-ratio threshold, 100-boundary-pixel fallback,
inner q98 elevation, 100-neighbour power-2 IDW, and 0.10 m minimum depth. Outward
expansion is disabled so the single shared extent remains the support supplied to all
three methods.

All terrain methods fill DSM/slope voids from the nearest valid terrain pixel without
using target depth. Their public names omit `oracle`: `fwdet_v21_dsm_extent`,
`ricorde_local_hand_dsm_extent`, and `flexth_method_a_dsm_extent`.

## Candidate disposition

| Candidate family | Decision | Reason |
|---|---|---|
| DLSIM | Keep two learned adapters | continuous output and change-plus-slope roles map without labels |
| FwDET v2.1 | Keep predicted-extent adaptation | direct shoreline water-surface allocation representative |
| RICorDE | Keep local-HAND adaptation | represents HAND/stage reconstruction with explicit missing-input caveat |
| FLEXTH | Keep method-A adaptation | represents extent enhancement and boundary interpolation |
| L/C-band SAR + DEM | Exclude | L-band and external calibration unavailable |
| CSIRO suite | Do not duplicate | overlaps the selected FwDET/HAND families |
| MatFlood, c-HAND, PyFlood | Exclude | require boundary water level/coastal forcing or additional terrain/land-cover inputs |
| NWM-HAND, Google Manifold | Exclude | require stream network, rating curves, discharge or gauge stage |

Primary sources: [extent paper](https://www.nature.com/articles/s41467-025-60973-1),
[extent code](https://github.com/microsoft/ai4g-flood),
[Kuro Siwo](https://proceedings.neurips.cc/paper_files/paper/2024/hash/43612b0662cb6a4986edf859fd6ebafe-Abstract-Datasets_and_Benchmarks_Track.html),
[FwDET](https://doi.org/10.5194/nhess-19-2053-2019),
[RICorDE](https://doi.org/10.5194/nhess-22-1437-2022), and
[FLEXTH](https://doi.org/10.5194/nhess-24-2817-2024).
