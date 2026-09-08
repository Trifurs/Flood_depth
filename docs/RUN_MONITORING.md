# Run monitoring and TensorBoard

## Console records

All trainable models use the same concise per-epoch console record. It includes
the current and total epoch, the epoch duration, training loss, validation
metric and best value, learning rate, early-stopping counter, accumulated time,
and ETA. A trailing `*` identifies a new best validation result. Batch-level
progress bars are disabled by default so they do not corrupt console logs; set
`logging.progress_bar` to `true` in the shared base XML only when interactive
batch progress is needed.

Python warnings are routed through the same timestamped logger. They remain
visible by default (`logging.show_python_warnings=true`) instead of being
silently discarded. Set it to `false` only after the warning has been assessed.

## TensorBoard

TensorBoard is enabled in `configs/base/base.xml` and receives a flush at each
epoch boundary, so scalar charts update as soon as an epoch completes. To view
all training, ablation, and evaluation streams:

```bash
conda run -n flood-depth tensorboard \
  --logdir runs/flooddepthnet_s1_terrain \
  --port 6006
```

Then open <http://localhost:6006>. To focus on one PA-HydroKAN run, replace the
log directory with its run folder:

```bash
conda run -n flood-depth tensorboard \
  --logdir runs/flooddepthnet_s1_terrain/train/pa_hydrokan/<started-at>/tensorboard \
  --port 6006
```

Useful scalar groups are:

- `train/*`: objective and train-time components.
- `validation/raw/*` and, for PA-HydroKAN, `validation/ema/*`: validation
  metrics.
- `system/*`: learning rate, best metric, epoch duration, elapsed time, ETA,
  and early-stopping count/remaining patience.
- `evaluation/*`: metrics from standalone evaluation and deterministic
  traditional baselines.

The **Text** tab contains the immutable run metadata (model, run ID, device,
seed, schedule, and source checkpoint). The TensorBoard command-line program
may print `TensorFlow installation not found - running with reduced feature
set.` in this PyTorch-only environment. It is informational: PyTorch
`SummaryWriter` event files are available and scalar/Text dashboards work
without TensorFlow.
