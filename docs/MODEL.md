# PA-HydroKAN model

**PA-HydroKAN** expands to **Prior-Aware Hydrologic Kolmogorov-Arnold Network**.
The production identifier is `pa_hydrokan`. It is the reference depth model for
all comparisons; model parameter counts and exact comparison settings are emitted
by `tools/report_model_inventory.py`. The publication-facing nomenclature and
module labels are defined in [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md).

The model uses a radar-first information flow:

- TCSE separates pre-event state, event state, and externally supplied change branches.
- Branch-validity fractions prevent raster fill values from becoming evidence.
- RCP encodes acquisition reliability once and reuses it at each spatial scale.
- Incidence-angle conditioning is initialized as an identity residual.
- TPP/TCF fuse terrain as a bounded residual rather than replacing the SAR stream.
- TAE-KAN learns topographic affinity over an eight-neighbour graph.
- DGD and CDH predict positive conditional depth and a detached uncertainty scale.

The graph is a learned spatial prior, not a hydraulic simulator. Physical losses
are optional and disabled in the default training configuration.

The matched ablation configurations and protocol are in
[ABLATION_PROTOCOL.md](ABLATION_PROTOCOL.md).
