# Unified testing and event-grouped cross-validation

## Official all-model test

Run the newest completed checkpoint for every learned model below a directory,
plus all traditional methods:

```bash
python test.py runs/flooddepthnet_s1_terrain/train
```

Traditional methods are enabled by default. Optional controls are:

```bash
python test.py runs/flooddepthnet_s1_terrain/train --no-traditional
python test.py runs/flooddepthnet_s1_terrain/train --all-runs
```

Shared defaults (traditional inclusion, latest-run selection, prediction export,
batch limit, output override, and CUDA warm-up count) live in
`configs/base/base.xml` under `runtime.test`.

Only directories containing `best_raw.pth`, `resolved_config.json`, and
`training_summary.json` are eligible. The default selection is the latest
completed run per experiment; an incomplete newer run therefore cannot hide a
valid older checkpoint. Each checkpoint is evaluated with the immutable
resolved training configuration saved beside it.

When the supplied directory is named `train`, `test.py` also includes its
`ablation` sibling automatically (when present), while deliberately excluding
the separate `cross_validation` branch. Thus the command above remains the
single entry point after standalone ablation runs are added.

The test session is stored in
`runs/flooddepthnet_s1_terrain/test/<started-at>/`. `metrics_by_model.csv` and
`summary.json` contain all pixel-, sample-, event-, uncertainty-, physical-,
and depth-stratified accuracy metrics available for each model. Learned models
also report:

- total and trainable parameters;
- parameter/buffer storage and checkpoint size;
- synchronized model-forward time, sample and megapixel throughput;
- mean, median, and p95 batch latency after one CUDA warm-up batch;
- peak allocated/reserved CUDA memory, device, and inference precision;
- end-to-end evaluation time.

Pure forward timing excludes file I/O, host-to-device transfer, loss evaluation,
metric aggregation, and prediction export. End-to-end time includes data setup
and metric calculation. Traditional algorithms record the same latency and
throughput fields on CPU with zero learned parameters, so the device field must
be considered when comparing throughput.

## K-fold command

```bash
python validate_k_fold.py       # k=5
python validate_k_fold.py 10
python validate_k_fold.py --resume
```

The protocol is nested, event grouped, and leakage safe:

1. All rows carrying the original `train`, `val`, or `test` labels form one
   full-sample source pool. Those old labels are provenance only during CV.
2. Entire `source_event_id` groups are assigned by descending sample count to
   the currently lightest outer fold; deterministic hash tie-breaking uses the
   base seed.
3. In each run, one outer fold is test-only. The other four outer folds form
   that run's development pool.
4. A deterministic bitset subset-sum allocation selects an inner validation set
   from the four-fold development pool, targeting 12.5% of both its samples and
   events (10% of all samples for five folds). The remaining events train the
   model.
5. Early stopping and best-checkpoint selection use only the inner validation
   set. The outer test fold is not evaluated until training is complete.
6. Normalization, depth strata, positive prior, and PA-HydroKAN depth
   calibration are recomputed from that fold's training rows only.
7. Across the five runs, every source sample is in an outer test set exactly
   once, and no event crosses train/inner-validation/outer-test roles within a
   run.

The default fold count, grouping policy, full source-split list, inner-validation
fraction, fold seed stride, and normalization reservoir capacity are centralized
under `runtime.cross_validation` in `configs/base/base.xml`; the positional `k`
argument overrides only `default_k`.

Because the original test rows are deliberately included in the full-sample
pool, this CV protocol has no additional external holdout. The publication-level
generalization claim is therefore the mean and sample standard deviation across
the five mutually exclusive outer test folds, not repeated evaluation on the
old fixed test split.

For the current release, the pool contains 5,835 samples from 652 events
(original labels: 5,323 train, 258 validation, and 254 test). With `k=5`, every
outer test fold has 1,167 samples; every corresponding inner validation set has
584 samples, leaving 4,084 training samples. Outer event counts are
`[130, 130, 130, 131, 131]`; every inner validation set contains 65 events,
leaving `[457, 457, 457, 456, 456]` training events. The constrained allocator
matches both sample and event targets exactly for this release.

The default run trains 13 unique experiments per fold: PA-HydroKAN once, five
learned comparison models, and seven non-empty RCP/TCF/TAE-KAN removal
combinations. No redundant `full` ablation configuration is present. This is
65 independent training runs when `k=5`.

Outputs are stored under
`runs/flooddepthnet_s1_terrain/cross_validation/k<k>/<started-at>/` and include
the outer-fold and inner-validation event assignments, fold-balance audit,
fold manifests/contracts/train-only
statistics, generated XML overlays, every model checkpoint, inner-validation
and outer-test details, `metrics_by_fold.csv`, `metrics_summary.csv`, and
`summary.json`.
`metrics_summary.csv` reports finite numerical metrics in long form with count,
mean, sample standard deviation (`ddof=1`), minimum, and maximum.

Each model×fold training and its two evaluations run in an isolated subprocess,
so a corrupted CUDA context cannot destroy the controller's record of earlier
jobs. Progress is committed after every completed job. To continue the newest
unfinished session for the selected/default `k`, run:

```bash
python validate_k_fold.py --resume
```

An explicit session is also accepted:

```bash
python validate_k_fold.py --resume \
  runs/flooddepthnet_s1_terrain/cross_validation/k5/<started-at>
```

Complete train/inner-validation/outer-test products are reused. An incomplete training uses
its family-specific last checkpoint (`last.pth` for PA-HydroKAN and
`last_raw.pth` for learned comparators). A partial attempt without a valid last
checkpoint is moved under `failed_attempts/` before a clean retry; it is never
deleted. `runtime.cross_validation.require_cuda=true` performs controller and
worker health checks and blocks accidental CPU fallback. If `nvidia-smi -L`
fails, repair or reboot the NVIDIA driver before using `--resume`.

Every new session records SHA-256 fingerprints for the fully resolved XML of
all 13 experiments, the source manifest, and all runtime Python sources, plus
the full-sample nested split policy and fold balance. The
controller rechecks them before every isolated job, and the worker rechecks its
source model configuration at startup. A changed or legacy protocol is rejected
before old and new fold results can be mixed. Consequently, sessions created
before the current full-sample outer-test protocol cannot safely resume; start
the campaign with `python validate_k_fold.py 5` and use `--resume` only for that
newly created session.

The isolated worker also fixes `MKL_THREADING_LAYER=GNU` through the centralized
`runtime.cross_validation.worker_mkl_threading_layer` setting. This matches the
GNU OpenMP runtime loaded by the installed PyTorch build and prevents the
`mkl-service ... incompatible with libgomp.so.1` startup failure. Before fold
assets or model jobs are touched, a short worker preflight imports NumPy and
PyTorch and initializes CUDA using this exact subprocess environment.

To audit only the fold manifests and train-only assets before committing to all
training jobs:

```bash
python validate_k_fold.py 5 --prepare-only
```
