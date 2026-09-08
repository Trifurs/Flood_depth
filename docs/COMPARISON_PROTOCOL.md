# Named flood-depth comparison protocol

## Model-specific layout

Every model has a model-named implementation and configuration. Shared loading,
metric, and training code lives only in private helpers; the root-level
`train.py` and `evaluate.py` resolve the selected model from its XML rather than
requiring a model-specific command.
This follows the model/configuration-family separation used by the
[DEHCD-Net layout](https://github.com/Trifurs/DEHCD-Net).
The `compare/common/` directory contains all shared comparison utilities and
registries; `compare/traditional/` and `compare/deep_learning/` contain only
the named model implementations.

| Identifier | Implementation | Configuration | Unified operation | Inputs |
|---|---|---|---|---|
| `pa_hydrokan` | `models/pa_hydrokan.py` | `configs/pa_hydrokan.xml` | `python train.py <config>` / `python evaluate.py <config>` | S1, QA, DSM/slope |
| `fwdet_v2` | `compare/traditional/fwdet_v2.py` | `configs/compare/traditional/fwdet_v2.xml` | `python train.py <config>` (deterministic evaluation) | `valid_depth_mask` + DSM |
| `tsa` | `compare/traditional/tsa.py` | `configs/compare/traditional/tsa.xml` | `python train.py <config>` (deterministic evaluation) | `valid_depth_mask` + DSM |
| `fldepth` | `compare/traditional/fldepth.py` | `configs/compare/traditional/fldepth.xml` | `python train.py <config>` (deterministic evaluation) | `valid_depth_mask` + DSM |
| `dlsim_attention_unet` | `compare/deep_learning/dlsim_attention_unet.py` | `configs/compare/deep_learning/dlsim_attention_unet.xml` | `python train.py <config>` / `python evaluate.py <config>` | S1 change + DSM + `valid_depth_mask` |
| `dlsim_linknet` | `compare/deep_learning/dlsim_linknet.py` | `configs/compare/deep_learning/dlsim_linknet.xml` | `python train.py <config>` / `python evaluate.py <config>` | S1 change + DSM + `valid_depth_mask` |
| `unet_depth_regression` | `compare/deep_learning/unet_depth_regression.py` | `configs/compare/deep_learning/unet_depth_regression.xml` | `python train.py <config>` / `python evaluate.py <config>` | S1 T1/T2/change + DSM/slope + `valid_depth_mask` |
| `resnet18_depth_regression` | `compare/deep_learning/resnet18_depth_regression.py` | `configs/compare/deep_learning/resnet18_depth_regression.xml` | `python train.py <config>` / `python evaluate.py <config>` | S1 T1/T2/change + DSM/slope + `valid_depth_mask` |
| `unetplusplus_depth_regression` | `compare/deep_learning/unetplusplus_depth_regression.py` | `configs/compare/deep_learning/unetplusplus_depth_regression.xml` | `python train.py <config>` / `python evaluate.py <config>` | S1 T1/T2/change + DSM/slope + `valid_depth_mask` |

PA-HydroKAN is a direct conditional-positive-depth regressor: it neither
predicts nor accepts a flood-range input. Every terrain comparison model uses
the same binary flood range, `valid_depth_mask`, directly. No range predictor,
probability raster, threshold, or alternative range-source argument exists.

## Commands

```bash
# Every model reads its own XML; runtime parameters remain in the inherited config.
python train.py configs/pa_hydrokan.xml
python evaluate.py configs/pa_hydrokan.xml
python train.py configs/compare/deep_learning/dlsim_attention_unet.xml
python evaluate.py configs/compare/deep_learning/dlsim_attention_unet.xml
python train.py configs/compare/traditional/fwdet_v2.xml

# Combine same-domain summaries only after all evaluations finish.
conda run -n flood-depth python tools/compare_results.py \
  --pa-summary runs/flooddepthnet_s1_terrain/evaluate/pa_hydrokan/val/<pa-run-id>/summary.json \
  --baseline-summary runs/flooddepthnet_s1_terrain/evaluate/fwdet_v2/val/<evaluation-id>/summary.json \
  --baseline-summary runs/flooddepthnet_s1_terrain/evaluate/tsa/val/<evaluation-id>/summary.json \
  --baseline-summary runs/flooddepthnet_s1_terrain/evaluate/fldepth/val/<evaluation-id>/summary.json \
  --model-summary runs/flooddepthnet_s1_terrain/evaluate/dlsim_attention_unet/val/<dlsim-attention-run-id>/summary.json \
  --model-summary runs/flooddepthnet_s1_terrain/evaluate/dlsim_linknet/val/<dlsim-linknet-run-id>/summary.json \
  --output runs/flooddepthnet_s1_terrain/comparison/val/<comparison-id>
```

FwDET allocates the nearest valid outer-boundary elevation as water surface.
TSA fits a first-degree water-surface trend per connected component. FlDepth
uses medial-ridge cross-sections and bank-elevation interpolation. The inland
FwDET adaptation retains zero-elevation boundary exclusion; this release has no
permanent-water layer. See [Cohen et al. 2019](https://nhess.copernicus.org/articles/19/2053/2019/)
and [Chimata et al. 2025](https://nhess.copernicus.org/articles/25/2455/2025/).
The learned-comparison source and adaptation record is in
[COMPARISON_SOURCES.md](COMPARISON_SOURCES.md).
Formal full-data commands and the directory convention are in
[FULL_DATA_TRAINING.md](FULL_DATA_TRAINING.md).
