# Configurations

The active depth-estimation configurations use the audited
`subset1000_s1_only` contract and `input_mode=s1_terrain`. They select Sentinel-1
pre-event/event/change bands, S1 quality features, and DSM terrain only.

- `config.xml` points to the GPU-optimized optical-free v15 run.
- `pa_hydrokan/subset1000_s1_v14_gpu.xml` is the matched S1-only baseline.
- `pa_hydrokan/subset1000_s1_v15_gpu_precision.xml` is the recommended model.
- `extent/subset1000_s1_ai4g_mobilenet_iou.xml` is the S1-only flood-extent
  extractor used by the optional terrain baselines.
- `compare/subset1000_s1_geometry_predicted_extent.xml` defines those terrain
  baselines.

Includes are resolved first and later values override earlier values. All values
carry explicit XML types. Historical optical/S2 model configurations have been
removed from this working tree.
