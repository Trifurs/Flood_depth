# Comparison models and methods

This directory is the comparison-method counterpart of DEHCD-Net's `compare/`
layout. It contains two deployment-oriented protocol classes:

1. learned DLSIM adapters; and
2. deterministic terrain-geometry methods driven by one frozen, independently
   predicted flood-extent product.

## Implemented baselines

| Registry name | Upstream idea | Local adaptation |
|---|---|---|
| `dlsim_linknet_adapted` | DLSIM LinkNet | seven normalized S1/S2 change bands are projected to one label-free change-evidence channel; normalized slope is the second channel |
| `dlsim_attention_unet_adapted` | DLSIM Attention U-Net | the same two-role input and shared project output heads, with gated skip connections |
| `fwdet_v21_dsm_extent` | FwDET v2.1 | boundary DSM smoothing, nearest-boundary unit-cost allocation, non-negative depth and low-pass filtering |
| `ricorde_local_hand_dsm_extent` | RICorDE | patch-local pseudo-drainage HAND, capped shoreline HAND, IDW rolling stage and conditional smoothing |
| `flexth_method_a_dsm_extent` | FLEXTH method A | extent closing, mild-slope boundary sampling, IDW/inner-q98 water level and the documented 0.10 m minimum depth; outward expansion disabled |

These are adapted baselines, not exact reproductions. The geometry implementations
are clean-room adaptations; no GPL/EUPL source is copied. Their primary references
are [FwDET v2.1](https://github.com/csdms-contrib/fwdet),
[RICorDE](https://github.com/NRCan/RICorDE), and
[FLEXTH](https://github.com/hyunholee26/FLEXTH). Local elevation is a DSM rather
than a DTM. RICorDE is explicitly named `local_hand` because the dataset lacks a
hydrologically conditioned DEM, drainage network, and permanent-water inputs.

## Shared flood-extent product

All three geometry methods consume exactly the same `flood_extent.tif`. It is
generated once by the independent workflow under `extent/`, using an adapted
MobileNetV2--U-Net early-fusion SAR change detector from Misra et al. (Nature
Communications, 2025). `valid_depth_mask` is the binary flood label and pure
masked Soft-IoU is the training objective. The best checkpoint maximizes validation
raw IoU; frozen inference itself uses only the SAR inputs.

The extent extractor, checkpoint, probability rasters, binary extent rasters, and
diagnostics are stored under `runs/extent/`. Geometry depth results are stored under
`runs/geometry_predicted_extent/` and `runs/test/compare_geometry_predicted_extent_final/`.
This directory boundary is intentional.

## Commands

```bash
conda run -n flood-depth python extent/train.py \
  --config configs/extent/subset150_ai4g_mobilenet_iou.xml --device cuda \
  --output runs/extent/train/subset150_ai4g_mobilenet_iou_frozen

conda run -n flood-depth python extent/predict.py \
  --config configs/extent/subset150_ai4g_mobilenet_iou.xml \
  --checkpoint runs/extent/train/subset150_ai4g_mobilenet_iou_frozen/best.pth \
  --output runs/extent/products/subset150_ai4g_mobilenet_iou_frozen \
  --splits val test --device cuda

conda run -n flood-depth python tools/evaluate_geometry.py \
  --config configs/compare/subset150_geometry_predicted_extent.xml \
  --extent-root runs/extent/products/subset150_ai4g_mobilenet_iou_frozen \
  --split val --output runs/geometry_predicted_extent/val_iou_frozen

conda run -n flood-depth python tools/evaluate_geometry.py \
  --config configs/compare/subset150_geometry_predicted_extent.xml \
  --extent-root runs/extent/products/subset150_ai4g_mobilenet_iou_frozen \
  --split test --save-predictions \
  --output runs/test/compare_geometry_iou_extent_final
```

The evaluator verifies dataset fingerprints, georeferencing, the extent-manifest
hash, and `prediction_uses_valid_depth_mask=false`. Target depth and
`valid_depth_mask` enter only metric aggregation after a depth prediction has been
created. Test was run once after the configuration and frozen extent were fixed.
