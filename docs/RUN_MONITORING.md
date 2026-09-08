# Run monitoring and TensorBoard

## Console records

All trainable models use the same concise per-epoch console record. It includes
the current and total epoch, the epoch duration, training loss, validation
metric and best value, learning rate, early-stopping counter, accumulated time,
and ETA. A trailing `*` identifies a new best validation result. Interactive
batch-level progress is enabled by default: it shows the current epoch, batch
count, rate, ETA, loss, and learning rate without adding permanent lines to the
log. Set `logging.progress_bar` to `false` in the shared base XML only for a
non-interactive redirected log.

Python warnings are routed through the same timestamped logger. They remain
visible by default (`logging.show_python_warnings=true`) instead of being
silently discarded. Set it to `false` only after the warning has been assessed.
The known PyTorch CUDA adaptive-average-pooling determinism warning is condensed
to one `INFO` reproducibility note: the run remains seed-controlled, but that
PyTorch CUDA kernel cannot promise bitwise-identical backward passes.

## Monitor an already-running job

An already-started Python process cannot apply a later XML change. In a second
terminal, monitor its compact batch heartbeat without interrupting it:

```bash
RUN_DIR=runs/flooddepthnet_s1_terrain/train/pa_hydrokan/<started-at>
tail -n 1 -f "$RUN_DIR/train_steps.csv" | awk -F, '{printf "epoch=%d | batch=%d | loss=%.5f | lr=%.2e | throughput=%.1f samples/s\n", $1 + 1, $2 + 1, $3, $4, $11}'
```

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
