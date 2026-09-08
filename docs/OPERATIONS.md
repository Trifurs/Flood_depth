# Operations

The formal full-data workflow, configuration-owned hyperparameters, and
result-directory layout are in [FULL_DATA_TRAINING.md](FULL_DATA_TRAINING.md).
Use `python train.py <model-config.xml>` and
`python evaluate.py <model-config.xml>`; the active configuration uses complete
FloodDepthNet S1/QA/DEM data only, and Sentinel-2 is disabled.

Each run directory is self-contained and stores the resolved configuration,
dataset fingerprints, calibration state, checkpoints, metrics, and environment
metadata. Select a checkpoint on validation data, then evaluate it once on the
held-out test split using the same canonical output-validity mask.
