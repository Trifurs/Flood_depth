# Training and evaluation

Use the optical-free v15 GPU configuration for the active experiment:

```bash
conda run -n flood-depth python tools/train.py \
  --config configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml \
  --output runs/optimization/hydrokan_s1_v15/gpu_precision_full \
  --device cuda
```

Checkpoint selection uses validation
`event_hierarchical_composite_mae`. Final evaluation uses the held-out test
split, the raw best checkpoint, and the frozen `output_valid` mask:

```bash
conda run -n flood-depth python tools/evaluate.py \
  --config configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml \
  --checkpoint runs/optimization/hydrokan_s1_v15/gpu_precision_full/best.pth \
  --split test --output artifacts/optimization/hydrokan_s1_v15/gpu_precision_test_output_valid \
  --device cuda --validity-mask output_valid
```

The v14 S1-only configuration provides the matched baseline. Test data is not
used for normalization, early stopping, or model selection. Use multiple seeds
for publication-level uncertainty estimates.
