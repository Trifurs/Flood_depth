# PA-HydroKAN nomenclature

## Canonical model identity

The paper-facing model name is **PA-HydroKAN**, expanded once on first use as
**Prior-Aware Hydrologic Kolmogorov-Arnold Network**. The stable machine
identifier remains `pa_hydrokan`; retaining it preserves configuration,
checkpoint, and result compatibility.

“Prior-Aware” refers to the explicitly represented acquisition-reliability and
topographic-affinity priors. “Hydrologic” describes the flood-depth task and
terrain reasoning only: PA-HydroKAN is not presented as a hydraulic solver or
a physics-informed model when its optional physical loss is disabled.

Recommended first-use wording:

> We propose PA-HydroKAN (Prior-Aware Hydrologic Kolmogorov-Arnold Network),
> a SAR-first conditional flood-depth regressor with reliability conditioning,
> terrain-conditioned fusion, and topographic-affinity message passing.

## Paper module labels

| Label | Paper-facing name | Implementation anchor | Ablation role |
|---|---|---|---|
| RCP | Reliability-Conditioning Pyramid | `SARReliabilityConditioner` | `w/o RCP` |
| TCSE | Temporal-Change SAR Encoder | `JointSARHydrologyEncoder` | retained backbone |
| TPP | Topographic-Prior Pyramid | `TerrainFeaturePyramid` | retained to preserve decoder/graph geometry |
| TCF | Terrain-Conditioned Fusion | `S1HydrologyFusion` | `w/o TCF` |
| TAE-KAN | Topographic-Affinity Edge-KAN | `HydroEdgeKAN` | `w/o TAE-KAN` |
| LCA | Latent Compatibility Adapter | `HydroEdgeKAN.latent_compatibility` | internal to TAE-KAN; not a separate factor |
| DGD | Dual-Gated Decoder | `SARHydroDecoder` | retained decoder |
| CDH | Conditional-Depth Head | `PAHydroKANHeads` | retained prediction head; includes local and patch-level calibration |

The paper labels are intentionally concise while Python class names remain
descriptive implementation symbols. This avoids an unnecessary checkpoint-API
break while keeping figures, tables, and ablation labels consistent.

## Style rules

- Use `PA-HydroKAN` for the full method and `pa_hydrokan` only in code,
  filenames, configuration keys, and tables of identifiers.
- Introduce each acronym once; subsequently use its short label consistently.
- Name ablations as `PA-HydroKAN w/o <module>` rather than inventing a new model
  name for a disabled component.
- Treat RCP, TCF, and TAE-KAN as the three peer ablation factors. LCA is a
  subordinate TAE-KAN mechanism and is disabled together with its parent path.
- Treat contextual depth-range calibration and global evidence calibration as
  internal CDH mechanisms, not additional peer modules or ablation factors.
- Reserve “physics-informed” and “hydraulic” for experiments that actually use
  and validate those mechanisms.
