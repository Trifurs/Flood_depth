# PA-HydroKAN model

**PA-HydroKAN** expands to **Prior-Aware Hydrologic Kolmogorov-Arnold Network**.
The production identifier is `pa_hydrokan`. It is the reference depth model for
all comparisons; model parameter counts and exact comparison settings are emitted
by `tools/report_model_inventory.py`. The publication-facing nomenclature and
module labels are defined in [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md).

The production configuration is the accuracy-first lightweight variant. It uses
`[24, 48, 96, 144]` encoder widths, additive decoder fusion, and 2,598,151
trainable parameters. The model uses a radar-first information flow:

- TCSE jointly encodes pre-event state, event state, and externally supplied
  change evidence, avoiding three redundant encoder pyramids.
- Branch-validity fractions prevent raster fill values from becoming evidence.
- RCP encodes acquisition reliability once and reuses it at each spatial scale.
- Incidence-angle conditioning is initialized as an identity residual.
- TPP/TCF fuse terrain as a bounded residual rather than replacing the SAR stream.
- TAE-KAN learns topographic affinity over an eight-neighbour graph.
- DGD reconstructs full-resolution features with additive gated skips.
- CDH predicts positive conditional depth and applies a compact local
  depth-range correction plus patch-level, mask-agnostic evidence calibration.
- The default point-estimation configuration omits the learned uncertainty head;
  it returns a fixed schema-compatible scale because uncertainty loss is zero.

PA-HydroKAN never receives `valid_depth_mask`, flood extent, or labels as model
inputs. Sensor/DEM validity maps describe observation availability only. The
comparison models retain their user-required oracle flood-range input.

The graph is a learned spatial prior, not a hydraulic simulator. Physical losses
are optional and disabled in the default training configuration.

The matched ablation configurations and protocol are in
[ABLATION_PROTOCOL.md](ABLATION_PROTOCOL.md).
The validation-only optimization record, including rejected alternatives and
the remaining RMSE limitation, is in [OPTIMIZATION_STUDY.md](OPTIMIZATION_STUDY.md).
