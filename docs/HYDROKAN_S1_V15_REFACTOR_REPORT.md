# PA-HydroKAN-S1-v15 optical-free refactor report

## Scope

This iteration targets `/home/whu/桌面/myData/Flood_depth/subset1000` with
`dataset.input_mode=s1_terrain`. The model does not open, receive, or fuse
Sentinel-2 tensors. S1 pre-event, S1 event, S1 change, S1 QA/reliability, and
terrain are the only model inputs.

The preceding v14 S1-only path remains intact. v15 is a new model identity so
its checkpoint fingerprint cannot be confused with v14.

## Structural changes

- `SARHydrologyEncoderV15` keeps pre/event states, signed and absolute change,
  local SAR detail, QA, branch validity, and reliability as separate paths.
- The local detail path now uses masked S1 branches; invalid pre/event/change
  fill values cannot leak through the high-resolution difference features.
- Angle conditioning is an identity-initialized FiLM residual. It can correct
  incidence-angle effects without forcing the network to relearn the SAR signal.
- `S1HydrologyFusionV15` makes SAR the identity stream and adds bounded,
  hydrology-aware terrain/reliability residuals.
- `HydrologyContextV15` adds multi-dilation context, while the existing
  terrain-aware Edge-KAN and SAR decoder remain available.
- `S1ZeroInflatedDepthHeadsV15` separates conditional positive depth from the
  optional support probability. The support branch is PU-compatible, but it is
  not allowed to suppress positive depth during precision-oriented inference.
- An optional `GlobalEventDepthScale` is available for event-level calibration;
  it is identity-initialized and disabled in the recommended short-run
  candidate because it did not improve the controlled validation result.

## Controlled evidence

All rows below use the same S1-only subset, `output_valid` mask, and
`conditional_positive_v2` output. The short CPU rows are retained as engineering
screening provenance. The two full-budget GPU runs use the full train split,
AMP, batch 8 with gradient accumulation 2, four workers, and early stopping;
the v14 baseline selected epoch 45 and the v15 precision run selected epoch 35.
The v15 model and objective are intentionally different because this comparison
tests the optical-free refactor and its positive-depth precision setting.

| split | candidate | event hierarchical composite MAE | pixel MAE | bias |
|---|---|---:|---:|---:|
| val | v14 control, short screening | 0.36056 | 0.45995 | -0.31980 |
| val | v14 GPU fair baseline, full train | 0.31895 | 0.41140 | -0.27722 |
| val | v15 full transfer | 0.35399 | 0.45639 | -0.28273 |
| val | v15 event-scale control | 0.36260 | 0.46259 | -0.29646 |
| val | v15 GPU precision, full train | 0.30600 | 0.40226 | -0.30551 |
| test | v14 control, short screening | 0.24047 | 0.21541 | -0.03223 |
| test | v14 GPU fair baseline, full train | 0.23015 | 0.20772 | 0.01559 |
| test | v15 full transfer | 0.23633 | 0.21631 | -0.00281 |
| test | v15 GPU precision, full train | 0.21851 | 0.18686 | -0.00042 |

Relative to the full-budget v14 GPU baseline, the v15 GPU precision run
improves the event-level metric by approximately 4.1% on validation and 5.1%
on held-out test. Pixel MAE improves by approximately 2.2% on validation and
10.0% on test; test absolute bias also decreases from 0.01559 m to 0.00042 m.
The support-weighted output was substantially worse in short screening, so use
conditional depth for positive-depth accuracy evaluation. This is a positive
single-seed engineering result, but it should still be confirmed with multiple
seeds before making a publication claim.

Machine-readable results are written to
`artifacts/optimization/hydrokan_s1_v15/candidate_summary.json` by
`tools/summarize_hydrokan_s1_v15.py`.

## Reproduction

Build and train the optical-free v15 GPU precision model with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n flood-depth python tools/train.py \
  --config configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml \
  --output runs/optimization/hydrokan_s1_v15/gpu_precision_full \
  --device cuda \
  --init-checkpoint runs/optimization/hydrokan_s1_v15/init_gpu_precision_from_v14.pth
```

Evaluate the selected best checkpoint:

```bash
conda run -n flood-depth python tools/evaluate.py \
  --config configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml \
  --checkpoint runs/optimization/hydrokan_s1_v15/gpu_precision_full/best.pth \
  --split test --output artifacts/optimization/hydrokan_s1_v15/gpu_precision_test_output_valid \
  --device cuda --validity-mask output_valid
```

The matched full-budget v14 baseline can be reproduced with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n flood-depth python tools/train.py \
  --config configs/pa_hydrokan/subset1000_s1_v14_gpu.xml \
  --output runs/optimization/hydrokan_s1_v15/v14_gpu_fair_full \
  --device cuda \
  --init-checkpoint runs/optimization/hydrokan_s1_v15/v14_control_5x8/best_raw.pth
```

Before publication or deployment, repeat the same GPU protocol with multiple
seeds and select the model using the same frozen validity mask and metric
definition.
