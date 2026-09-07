# Named flood-depth comparison protocol

## Model-specific layout

Every model has a model-named implementation, configuration, and executable
entry point. Shared loading, metric, and training code lives only in private
helpers; it does not select or expose a model itself.
This follows the model/configuration-family separation used by the
[DEHCD-Net layout](https://github.com/Trifurs/DEHCD-Net).

| Identifier | Implementation | Configuration | Executable | Inputs |
|---|---|---|---|---|
| `pa_hydrokan` | `models/pa_hydrokan.py` | `configs/pa_hydrokan.xml` | `tools/train_pa_hydrokan.py` | S1, QA, DSM/slope |
| `fwdet_v2` | `compare/fwdet_v2.py` | `configs/compare/fwdet_v2.xml` | `tools/run_fwdet_v2.py` | `valid_depth_mask` + DSM |
| `tsa` | `compare/tsa.py` | `configs/compare/tsa.xml` | `tools/run_tsa.py` | `valid_depth_mask` + DSM |
| `fldepth` | `compare/fldepth.py` | `configs/compare/fldepth.xml` | `tools/run_fldepth.py` | `valid_depth_mask` + DSM |
| `dlsim_attention_unet` | `models/dlsim_attention_unet.py` | `configs/dlsim_attention_unet.xml` | `tools/train_dlsim_attention_unet.py` / `tools/evaluate_dlsim_attention_unet.py` | S1 change + DSM + `valid_depth_mask` |
| `dlsim_linknet` | `models/dlsim_linknet.py` | `configs/dlsim_linknet.xml` | `tools/train_dlsim_linknet.py` / `tools/evaluate_dlsim_linknet.py` | S1 change + DSM + `valid_depth_mask` |
| `unet_depth_regression` | `models/unet_depth_regression.py` | `configs/unet_depth_regression.xml` | `tools/train_unet_depth_regression.py` / `tools/evaluate_unet_depth_regression.py` | S1 T1/T2/change + DSM/slope + `valid_depth_mask` |
| `resnet18_depth_regression` | `models/resnet18_depth_regression.py` | `configs/resnet18_depth_regression.xml` | `tools/train_resnet18_depth_regression.py` / `tools/evaluate_resnet18_depth_regression.py` | S1 T1/T2/change + DSM/slope + `valid_depth_mask` |
| `unetplusplus_depth_regression` | `models/unetplusplus_depth_regression.py` | `configs/unetplusplus_depth_regression.xml` | `tools/train_unetplusplus_depth_regression.py` / `tools/evaluate_unetplusplus_depth_regression.py` | S1 T1/T2/change + DSM/slope + `valid_depth_mask` |

PA-HydroKAN is a direct conditional-positive-depth regressor: it neither
predicts nor accepts a flood-range input. Every terrain comparison model uses
the same binary flood range, `valid_depth_mask`, directly. No range predictor,
probability raster, threshold, or alternative range-source argument exists.

## Commands

```bash
# Train/evaluate PA-HydroKAN.
conda run -n flood-depth python tools/train_pa_hydrokan.py \
  --config configs/pa_hydrokan.xml --device cuda
conda run -n flood-depth python tools/evaluate_pa_hydrokan.py \
  --config configs/pa_hydrokan.xml --checkpoint runs/train/<pa_run>/best_raw.pth \
  --split val --device cuda

# Evaluate each named terrain model separately.
conda run -n flood-depth python tools/run_fwdet_v2.py \
  --config configs/compare/fwdet_v2.xml --split val --output runs/compare/fwdet_v2/val
conda run -n flood-depth python tools/run_tsa.py \
  --config configs/compare/tsa.xml --split val --output runs/compare/tsa/val
conda run -n flood-depth python tools/run_fldepth.py \
  --config configs/compare/fldepth.xml --split val --output runs/compare/fldepth/val

# Train/evaluate one named learned comparator (repeat with its matching script).
conda run -n flood-depth python tools/train_dlsim_attention_unet.py --device cuda
conda run -n flood-depth python tools/evaluate_dlsim_attention_unet.py \
  --checkpoint runs/train/<dlsim_attention_unet_run>/best_raw.pth --split val --device cuda

# Combine same-domain summaries only.
conda run -n flood-depth python tools/compare_results.py \
  --pa-summary runs/evaluate/<pa_run>/summary.json \
  --baseline-summary runs/compare/fwdet_v2/val/summary.json \
  --baseline-summary runs/compare/tsa/val/summary.json \
  --baseline-summary runs/compare/fldepth/val/summary.json \
  --model-summary runs/evaluate/<dlsim_attention_unet_run>/summary.json \
  --model-summary runs/evaluate/<dlsim_linknet_run>/summary.json \
  --output runs/comparison/val
```

FwDET allocates the nearest valid outer-boundary elevation as water surface.
TSA fits a first-degree water-surface trend per connected component. FlDepth
uses medial-ridge cross-sections and bank-elevation interpolation. The inland
FwDET adaptation retains zero-elevation boundary exclusion; this subset has no
permanent-water layer. See [Cohen et al. 2019](https://nhess.copernicus.org/articles/19/2053/2019/)
and [Chimata et al. 2025](https://nhess.copernicus.org/articles/25/2455/2025/).
The learned-comparison source and adaptation record is in
[COMPARISON_SOURCES.md](COMPARISON_SOURCES.md).
The local one-batch `subset1000` smoke result and parameter table are in
[SUBSET1000_SMOKE.md](SUBSET1000_SMOKE.md).
