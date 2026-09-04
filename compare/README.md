# S1-only comparison methods

The comparison package now contains only the extent-conditioned terrain
baselines. They consume a frozen flood-extent product generated from Sentinel-1
change features and local DSM terrain; no Sentinel-2 or optical tensor is read.

The three clean-room geometry adapters are:

- `fwdet_v21_dsm_extent`
- `ricorde_local_hand_dsm_extent`
- `flexth_method_a_dsm_extent`

All methods consume the same `flood_extent.tif`. The extent extractor uses the
independent S1-only configuration under `configs/extent/`, while target depth
and `valid_depth_mask` enter only metric aggregation after prediction.

Example commands:

```bash
conda run -n flood-depth python extent/train.py \
  --config configs/extent/subset1000_s1_ai4g_mobilenet_iou.xml \
  --device cuda --output runs/extent/train/subset1000_s1_ai4g_mobilenet_iou

conda run -n flood-depth python extent/predict.py \
  --config configs/extent/subset1000_s1_ai4g_mobilenet_iou.xml \
  --checkpoint runs/extent/train/subset1000_s1_ai4g_mobilenet_iou/best.pth \
  --output runs/extent/products/subset1000_s1_ai4g_mobilenet_iou \
  --splits val test --device cuda

conda run -n flood-depth python tools/evaluate_geometry.py \
  --config configs/compare/subset1000_s1_geometry_predicted_extent.xml \
  --extent-root runs/extent/products/subset1000_s1_ai4g_mobilenet_iou \
  --split test --output runs/test/compare_geometry_s1_extent
```
