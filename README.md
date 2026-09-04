# PA-HydroKAN optical-free flood-depth estimation

This repository contains the active Sentinel-1 SAR + DSM implementation for
event-aggregated flood-depth estimation. Sentinel-2/optical imagery is not read,
passed to, or fused by the active depth model.

## Active models

- `pa_hydrokan_s1_v14`: S1-only baseline.
- `pa_hydrokan_s1_v15`: SAR-first hydrology refactor with masked state/change
  encoding, incidence-angle conditioning, terrain/reliability residual fusion,
  multi-scale context, and conditional positive-depth heads.

The dataset is `/home/whu/桌面/myData/Flood_depth/subset1000`. Its runtime view is
defined by `input_mode=s1_terrain`; invalid S1/DSM pixels are excluded through the
explicit validity contract. The label and masks are never passed to the model.

## Recommended GPU run

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n flood-depth python tools/train.py \
  --config configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml \
  --output runs/optimization/hydrokan_s1_v15/gpu_precision_full \
  --device cuda \
  --init-checkpoint runs/optimization/hydrokan_s1_v15/init_gpu_precision_from_v14.pth
```

Evaluate the best checkpoint with the frozen output-validity definition:

```bash
conda run -n flood-depth python tools/evaluate.py \
  --config configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml \
  --checkpoint runs/optimization/hydrokan_s1_v15/gpu_precision_full/best.pth \
  --split test \
  --output artifacts/optimization/hydrokan_s1_v15/gpu_precision_test_output_valid \
  --device cuda --validity-mask output_valid
```

The matched S1-only v14 baseline uses
`configs/pa_hydrokan/subset1000_s1_v14_gpu.xml`.

## Verification

```bash
conda run -n flood-depth python -m pytest -q
```

The latest verification passed with `126 passed, 2 skipped`. Machine-readable
comparison results are in
`artifacts/optimization/hydrokan_s1_v15/candidate_summary.json`, and the detailed
experiment report is `docs/HYDROKAN_S1_V15_REFACTOR_REPORT.md`.

The old optical/S2 depth-model versions, comparison adapters, configurations,
diagnostics, tests, and their experiment records have been removed from this
working tree. Generic dataset-contract fields retain their audited source schema,
but the active model input contract is S1-only.
