# Frozen subset150 comparison results

## Protocol

All models use the immutable 105/23/22 train/validation/test raster split, the same
train-only normalization and depth strata, seed `20260831`, batch size 4, optimizer,
augmentations, partial-positive loss, weak WSE-curvature term, and checkpoint rule.
The best checkpoint minimizes validation pixel-micro MAE.  Event identifiers do not
enter the models, loss, or checkpoint selection.  After both comparison runs were
finished and frozen, each best checkpoint was evaluated on test exactly once.  No
test result was used to alter a model, threshold, loss, or configuration.

The comparison models are DLSIM-style **adaptations**, not exact paper
reproductions.  They use one learned, label-independent S1/S2 change-evidence channel
plus normalized slope and share the project output heads.

A separate two-stage protocol trains one Sentinel-1 flood-extent extractor, freezes
it, and supplies its val/test product to all three terrain methods. The extractor and
depth results use separate code and output roots. The historical label-derived
oracle diagnostic is retained below only to document why it was superseded.

## Frozen model selection and test ranking

| Model | Parameters | Best epoch | Val pixel MAE (m) | Test pixel MAE (m) | Test RMSE (m) | Median AE (m) | Bias (m) | P90 AE (m) | Within 0.50 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PA-HydroKAN v12 | 16,745,085 | 33 | 0.522350 | 0.286643 | 0.411775 | **0.214847** | 0.145427 | 0.585184 | 86.22% |
| DLSIM-LinkNet adapted | 7,850,347 | 16 | 0.492331 | 0.314490 | 0.439464 | 0.264768 | 0.153222 | 0.530570 | 88.98% |
| DLSIM-Attention U-Net adapted | 8,721,951 | 70 | **0.483518** | **0.280380** | **0.409863** | 0.230987 | **0.082435** | **0.469199** | **91.11%** |

Every test row uses the same 169,951 reliable positive-depth pixels.  Under the
declared pixel-first primary metric, Attention U-Net ranks first: its MAE is 0.006263
m (2.18%) below PA-HydroKAN v12 and its RMSE is 0.001912 m (0.46%) lower.  The
difference is descriptive, not a significance claim: all entries are one
deterministic seed.

PA-HydroKAN still has the lowest median error and the best within-0.25 m rate
(57.46% versus 55.26% for Attention U-Net).  Attention U-Net gains mainly by reducing
moderate errors and positive bias: its within-0.50 m rate is 4.89 percentage points
higher and its P90 error is 19.82% lower.

## Depth-regime audit

Pixel MAE by frozen train-depth boundary:

| Test reference depth | Pixels | PA-HydroKAN v12 | LinkNet adapted | Attention U-Net adapted |
|---|---:|---:|---:|---:|
| <0.23 m | 77,016 | **0.290744** | 0.369850 | 0.306494 |
| 0.23–0.48 m | 49,907 | 0.215121 | 0.187596 | **0.135914** |
| 0.48–0.83 m | 31,763 | 0.248270 | **0.180950** | 0.216594 |
| 0.83–1.22 m | 5,923 | **0.356412** | 0.450243 | 0.511002 |
| 1.22–2.14 m | 3,542 | **0.699536** | 1.014086 | 1.045739 |
| ≥2.14 m | 1,800 | **1.729263** | 1.997243 | 2.029114 |

The DLSIM adapters win the numerous 0.23–0.83 m pixels but lose all three bins above
0.83 m.  This is scientifically consistent with their restricted change-plus-slope
input: local change evidence is strong for common shallow/mid-depth patterns, while
PA-HydroKAN's pre/event multimodal context and explicit DSM processing retain more
information for deeper water.  Thus the Attention U-Net is the current primary
pixel-MAE winner, whereas PA-HydroKAN remains the stronger deep-water model.  The
large validation-to-test error shift indicates different split difficulty; it must
not be interpreted as improvement during deployment.

## Superseded oracle-extent diagnostic (not a formal comparison)

The following experiment predates the active predicted-extent workflow. It is kept
for auditability only and must not be cited as a deployable comparison result.

The FwDET, RICorDE local-HAND, and FLEXTH configurations were fixed from published
defaults plus declared DSM/patch adaptations before test. Complete validation was run
first without parameter search. Test then ran once with the mandatory
`--acknowledge-oracle-test-mask` flag. The method interface receives DSM, slope,
terrain validity, resolution, and the binary `valid_depth_mask`; it never receives
the depth values used for scoring.

| Oracle-extent method | Val pixel MAE (m) | Test pixel MAE (m) | Test RMSE (m) | Median AE (m) | Bias (m) | P90 AE (m) | Within 0.50 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| FwDET v2.1 DSM adapted | **0.499080** | 0.236950 | 0.443936 | 0.140368 | -0.119645 | 0.519130 | 89.00% |
| RICorDE local-HAND DSM adapted | 0.524728 | 0.235345 | 0.360583 | 0.156898 | -0.142187 | 0.480781 | 90.69% |
| FLEXTH method A DSM adapted | 0.499155 | **0.171380** | **0.287423** | **0.105958** | **-0.067666** | **0.388573** | **94.00%** |

All rows use the same 169,951 test reference pixels as the learned models. FLEXTH is
the strongest conditional reconstruction method on this test split. Its 0.171380 m
MAE is numerically 38.88% below the best learned-model MAE, but this is **not a fair
deployment gain**: FLEXTH was given the exact label-derived support and uses its
shape/boundary to estimate water level. The proper conclusion is that, once flood
support is externally known, local extent geometry contains substantial additional
depth information.

The oracle methods also show a very large validation-to-test shift (approximately
0.50 m validation MAE versus 0.17–0.24 m test MAE). This mirrors, and exceeds, the
split-difficulty shift seen for learned models. It argues against tuning a geometry
parameter on test or treating the one subset as universal evidence.

Oracle-method test MAE by frozen train-depth boundary:

| Test reference depth | Pixels | FwDET adapted | RICorDE local-HAND | FLEXTH method A |
|---|---:|---:|---:|---:|
| <0.23 m | 77,016 | 0.130129 | 0.130607 | **0.080030** |
| 0.23–0.48 m | 49,907 | 0.208535 | 0.198552 | **0.164314** |
| 0.48–0.83 m | 31,763 | 0.372828 | 0.314093 | **0.276695** |
| 0.83–1.22 m | 5,923 | 0.471739 | 0.603827 | **0.363907** |
| 1.22–2.14 m | 3,542 | 0.873160 | 1.113276 | **0.714389** |
| ≥2.14 m | 1,800 | 1.173112 | 1.407186 | **0.715456** |

The persistent negative biases show that all three still under-reconstruct water
depth, especially in the deep tail. FLEXTH reduces but does not remove that failure.
Its exact 0.10 m minimum-depth rule is particularly effective in the shallow-heavy
test distribution and should remain visible when interpreting the headline MAE.

## Frozen shared-predicted-extent geometry benchmark

The extent checkpoint was selected at epoch 8 by validation raw pixel-micro IoU after
training with pure masked Soft-IoU. It was then used once to export val and test
products. Probability threshold 0.5 and the four-pixel (80 m) buffer were fixed
before test. All geometry configurations were fixed before the single test run.

| Shared-predicted-extent method | Val pixel MAE (m) | Test pixel MAE (m) | Test RMSE (m) | Median AE (m) | Bias (m) | P90 AE (m) | Within 0.50 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| FwDET v2.1 DSM adapted | 4.745565 | 2.103386 | 5.592669 | 0.422817 | 1.957449 | 5.337323 | 53.87% |
| RICorDE local-HAND DSM adapted | **1.484383** | **0.963726** | **1.731423** | **0.324499** | **0.802822** | 2.818880 | **59.48%** |
| FLEXTH method A DSM adapted | 9.230142 | 1.360899 | 4.128163 | 0.400000 | 1.260363 | **2.275596** | 55.98% |

All rows reuse exactly the same frozen binary extent and are evaluated on the same
169,951 positive-depth test pixels. With `valid_depth_mask` defined as the extent
label, test raw IoU/F1/precision/recall are 0.359321/0.528677/0.365170/0.957320.
The buffered product used by geometry has IoU 0.295989 and recall 0.989987.

RICorDE local-HAND is the strongest of these three under pixel MAE, but remains worse
than the learned depth models. The result reverses the oracle diagnostic and
shows why the exact support could not be used in a fair benchmark: predicted extent
contains extra components and imperfect boundaries, while DSM includes buildings and
vegetation. Boundary-derived water-surface interpolation amplifies both errors;
FwDET is especially sensitive and exhibits large positive outliers. This is a
scientifically meaningful negative result, not a reason to retune on test.

## Reproducibility records

- Dataset contract SHA256: `79e33f3174163f7fcdb137ed2779fe5680b1fff658c30ea1fa71548517744a06`
- Train normalization SHA256: `15ef0977ba69137818e76bcd779dd7f0cf1e81195dd326716ba45d3328441319`
- PA-HydroKAN checkpoint: `5e6df04469621b327ce7d2e21b5d88739e051f312d1ff9a8e43a34271e709986`
- LinkNet checkpoint: `dd9af58844915ee70ce507060b29cfbab69338e4e0baef519be58cf4e06f35f3`
- Attention U-Net checkpoint: `ae484ed3981e7a732059b06bdb325fc7eafc3aa688ebb4682125501f68141af6`
- Extent checkpoint: `cefab47004ee772743406254a48afaff624afb385687e8e2bca743b28aa0da28`
- Predicted-extent geometry configuration: `aedf6cab408d6354f9f740f19f44e29949c8578c061236ae2e4ae57d094a6eb4`
- Predicted-extent geometry implementation: `f18a3f75a07c0f3e761ae8d5e976e615883a7cde5222c50dfd9758d0c99642e3`
- Predicted-extent geometry evaluator: `89b6ca32c99ad18692eb23d37291be302dc8e779a9a01b484d33e9a4f1d271bf`

Machine-readable summaries and all 22 georeferenced predictions per model are in
`runs/test/pa_hydrokan_hydrov12_final`,
`runs/test/compare_dlsim_linknet_final`, and
`runs/test/compare_dlsim_attention_unet_final`. Extent products are under
`runs/extent/products/subset150_ai4g_mobilenet_iou_frozen`; the three active
geometry summaries, metric CSVs, and 22 predictions per method are under
`runs/test/compare_geometry_iou_extent_final`.
