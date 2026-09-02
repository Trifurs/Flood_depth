# Training and evaluation protocol

- Seed Python, NumPy, CPU/GPU PyTorch, workers, and samplers; deterministic algorithms
  are enabled by default.
- A non-DDP train epoch visits every sample exactly once without replacement. Samples
  are shuffled deterministically by epoch and repeated source events are interleaved.
  DDP uses the same global order and only repeats the minimal padding required to give
  every rank the same number of steps. The legacy replacement sampler remains solely
  for reproducing an old resolved configuration.
- Only train pixels determine clipping/normalization, nnPU prior proxy, and depth-bin
  boundaries. Hydro-v5 refines the original quartile bins with frozen train q90, q95,
  and q97.5 tail boundaries; this prevents the entire 0.48--24.82 m interval from
  collapsing into one optimization cell. Its hierarchical reduction keeps
  shallow/mid/deep regimes equally weighted and averages the refined cells only
  inside their parent regime. Test is never touched by training,
  validation, early stopping, best
  checkpoint selection, or threshold selection.
- AdamW, cosine decay with warmup, AMP, gradient clipping, optional accumulation, and
  GroupNorm are configured in XML. CPU, one GPU, and torchrun DDP are supported.
- Hydro-v12 depth and uncertainty objectives ignore event IDs. Errors are averaged
  within each frozen train-depth stratum of each raster, then across non-empty strata
  and rasters. Best checkpoint selection minimizes validation `pixel_micro_mae`, so
  every labelled deployment pixel has equal evaluation weight. Pixel RMSE, P90 error,
  and 0.25/0.50/1.00 m accuracy bands are mandatory secondary checks; sample/event
  aggregations remain diagnostics only. Checkpoints are
  written atomically and include full optimizer/scheduler/GradScaler/RNG state plus
  resolved config and dataset fingerprints.
- Hydro-v12 retains only the 9-pixel DSM context and uses a weak positive-region WSE
  Laplacian penalty. Its weight is zero before epoch 5 and increases linearly to 0.02
  over 15 epochs. This is a local smoothness prior, not a conservation law or SWE
  residual.
- Formal test is a single application of the already selected `best.pth`; Hydro-v9's
  frozen test evaluation is recorded under `runs/test/test_hydrov9_frozen_final_20260901`.
  That historical test result has already been exposed. Hydro-v12 was not evaluated
  on test, and repeated inspection for tuning would invalidate the held-out protocol;
  its final unbiased assessment therefore requires a new untouched external set.
- Primary conditional-depth metrics are computed only on `valid_depth_mask`. They are
  reported as pixel micro, sample macro, event macro, flat event-depth-bin macro,
  event-depth-hierarchical macro,
  per-event, per-sample, and frozen train-bin views. Complete wet/dry classification
  metrics are intentionally absent.
- Physical diagnostics are reported separately from predictive metrics: local
  terrain-order violation magnitude/fraction, reference-gated WSE-gradient MAE,
  WSE-Laplacian reference error, and high-relief continuity. They use only valid
  positive/output-supported neighbourhoods and must not be interpreted as discharge,
  mass conservation, or shallow-water-equation residuals.
- GeoTIFF inference validity is DEM valid AND (S1 valid OR S2 valid), independent of
  label masks. Unsupported output is nodata.
- Hydro-v2 exports conditional depth, support-weighted depth, support score, and
  conditional-depth uncertainty separately. No hard flood threshold is selected from
  the held-out test set.
