# Operations

The formal full-data workflow, configuration-owned hyperparameters, and
result-directory layout are in [FULL_DATA_TRAINING.md](FULL_DATA_TRAINING.md).
Use `python train.py <model-config.xml>` for trainable models,
`python test.py runs/flooddepthnet_s1_terrain/train` for the complete test suite,
and `python validate_k_fold.py [k]` for event-grouped cross-validation.
The cross-validation command pools all original train/val/test samples, uses
one event-grouped outer fold for testing, and derives early-stopping validation
events only from the other four folds. Every sample is outer-tested exactly
once.
Resume its newest unfinished session with `python validate_k_fold.py --resume`;
completed model×fold jobs and valid last checkpoints are reused.
Resume is accepted only while the recorded resolved configurations, source
manifest, and runtime-source fingerprints still match. Refactor-era legacy
sessions must be replaced by a new run.
`python evaluate.py <model-config.xml>` remains the single-model validation
entry. The active configuration uses complete
FloodDepthNet S1/QA/DEM data only, and Sentinel-2 is disabled.

Each run directory is self-contained and stores the resolved configuration,
dataset fingerprints, calibration state, checkpoints, metrics, and environment
metadata. Select a checkpoint on validation data, then evaluate it once on the
held-out test split using the same canonical output-validity mask for standalone
train/test experiments. In the full-sample cross-validation workflow, the outer
fold is the held-out test set instead.
