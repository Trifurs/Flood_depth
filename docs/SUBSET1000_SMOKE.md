# `subset1000` learned-comparison smoke check

This is a **runnability check, not a final experiment**. On 2026-09-07 each
learned model and PA-HydroKAN was optimized for one epoch with exactly one
training batch (batch size 1, CPU, no AMP). The three terrain baselines are
deterministic and do not require training. Every method below was then scored on
the same first validation loader batch: 12 samples and 73,151
`valid_depth_mask_and_output_valid` pixels.

The metric files and resolved configurations are under
`runs/comparison/subset1000_learned_smoke_val12/`,
`runs/model_inventory/subset1000_learned_smoke/`, and the corresponding
`runs/train/*_subset1000_smoke/` directories. The runs are intentionally ignored
by Git; use them to verify local execution, not to report performance.

| Identifier | Parameters | Pixel MAE (m) | Pixel RMSE (m) | Event MAE (m) | Flood range |
|---|---:|---:|---:|---:|---|
| `pa_hydrokan` | 3,502,810 | 0.2970 | 0.4896 | 0.3783 | not an input |
| `fwdet_v2` | 0 | 0.3626 | 0.5502 | 0.4225 | `valid_depth_mask` |
| `tsa` | 0 | 0.3727 | 0.5434 | 0.4706 | `valid_depth_mask` |
| `fldepth` | 0 | 0.3612 | 0.5858 | 0.4130 | `valid_depth_mask` |
| `dlsim_attention_unet` | 2,409,089 | 0.4145 | 0.5325 | 0.4565 | `valid_depth_mask` |
| `dlsim_linknet` | 1,777,369 | 0.6921 | 0.8050 | 0.8185 | `valid_depth_mask` |
| `unet_depth_regression` | 2,369,977 | 0.4265 | 0.5003 | 0.4809 | `valid_depth_mask` |
| `resnet18_depth_regression` | 13,118,609 | 0.4095 | 0.4765 | 0.4672 | `valid_depth_mask` |
| `unetplusplus_depth_regression` | 1,171,417 | 0.5388 | 0.6291 | 0.5867 | `valid_depth_mask` |

The learned-comparison models are all at random initialization plus a single
optimization update. Their values are therefore expected to be weak and are
included only to demonstrate that the independent model/configuration/script
paths train, checkpoint, reload, and evaluate on the same domain. The parameter
inventory is the authoritative source for full input and configuration details;
the source and adaptation record is [COMPARISON_SOURCES.md](COMPARISON_SOURCES.md).
