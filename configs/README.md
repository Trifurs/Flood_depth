# Configuration

`base.xml` contains shared runtime/training defaults, `datasets/flooddepth_subset150.xml`
contains the audited data and preprocessing contract, and
`pa_hydrokan/subset150_main.xml` contains the current main-model experiment.
`config.xml` points to that default experiment. Includes are resolved first and later
values override earlier values. All values carry explicit XML types.

`compare/subset150_dlsim_common.xml` inherits the frozen Hydro-v12 data, objective,
optimizer, and pixel-first selection protocol.  Its LinkNet and Attention U-Net child
configs override only the registered comparison model and run name.  Extra
PA-HydroKAN-only model keys inherited by the common file are ignored by the explicit
comparison builders; the resolved configuration records them for provenance.

`extent/subset150_ai4g_mobilenet_iou.xml` freezes the independent Sentinel-1
MobileNetV2--U-Net flood-extent extractor. `valid_depth_mask` is its binary flood
label, pure masked Soft-IoU is the objective, and validation raw IoU selects the
checkpoint. Probability threshold 0.5 and the 80 m output buffer are fixed before
test.

`compare/subset150_geometry_predicted_extent.xml` freezes the separate FwDET-,
RICorDE-, and FLEXTH-style terrain baselines. Every method consumes the same
previously exported extent product through `--extent-root`; label-derived extent is
rejected by the evaluator. Source defaults are retained where their inputs have a
local analogue, and every DSM/patch substitution is named in the method and report.

The current experiment is Hydro-v12: without-replacement raster epochs, conditional
positive-depth output semantics, frozen train-depth strata balanced independently
inside each raster, a single 9-pixel DSM context, and a weak delayed WSE-curvature
prior. Loss computation and inference ignore source-event identity. Checkpoint
selection uses validation `pixel_micro_mae`; event aggregations are diagnostic only.
Legacy checkpoint output semantics are read from each checkpoint rather than silently
reinterpreted, and resume rejects a changed loss configuration. Historical v6/v9/v10/
v11 and the current v12 configs are frozen beside `subset150_main.xml`.
