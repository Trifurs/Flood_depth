# PA-HydroKAN ablation protocol

## Configurations

Every ablation inherits the exact dataset, preprocessing, supervision, loss,
optimizer, scheduler, and seed settings from `configs/pa_hydrokan.xml`. Only
one explicit architecture switch changes per variant.

| Variant | Configuration | Disabled component |
|---|---|---|
| PA-HydroKAN | `configs/ablation/pa_hydrokan_full.xml` | none |
| PA-HydroKAN w/o RCP | `configs/ablation/pa_hydrokan_no_reliability_conditioning.xml` | Reliability-Conditioning Pyramid |
| PA-HydroKAN w/o TCF | `configs/ablation/pa_hydrokan_no_terrain_conditioned_fusion.xml` | Terrain-Conditioned Fusion |
| PA-HydroKAN w/o TAE-KAN | `configs/ablation/pa_hydrokan_no_topographic_affinity_edge_kan.xml` | Topographic-Affinity Edge-KAN |
| PA-HydroKAN w/o LCA | `configs/ablation/pa_hydrokan_no_latent_compatibility.xml` | Latent Compatibility Adapter within TAE-KAN |

Disabled components are bypassed and frozen. Their tensors remain in the state
dictionary so frozen-weight interventions can load the same full-model
checkpoint strictly; a published ablation must nevertheless retrain every
variant from scratch.

The intervention boundaries are deliberately narrow:

- `w/o RCP` supplies zero reliability-conditioning features to TCSE and TCF;
  the S1 observations and their spatial validity masks remain unchanged.
- `w/o TCF` removes terrain/proxy-derived additive paths from TCF, while TPP
  remains available to DGD and TAE-KAN. It isolates fusion rather than claiming
  an invalid “no terrain” experiment.
- `w/o TAE-KAN` bypasses only the topographic-affinity message-passing update;
  TCSE, TPP, TCF, DGD, and CDH remain intact.
- `w/o LCA` makes edge compatibility neutral (one) while retaining all
  topographic affinity and message-passing calculations.

## Final experimental protocol

Train the full model and every variant independently with the same split and
seed set. Report mean and standard deviation across at least three seeds, the
same canonical `valid_depth_mask_and_output_valid` evaluation mask, parameter
counts, and the primary pixel- and event-level depth metrics. Do not claim that
a one-batch probe measures a component's final effect after optimization.

```bash
python train.py configs/ablation/pa_hydrokan_no_topographic_affinity_edge_kan.xml
```

The shared seed, start-time directory format, device, training schedule, and
output policy are in `configs/base/base.xml`; each ablation derives its output
as `runs/flooddepthnet_s1_terrain/ablation/<variant>/<started-at>/`.

## Lightweight functional probe

`tools/run_ablation_probe.py` remains an internal sensitivity diagnostic that
loads a common full checkpoint and measures the immediate output change on the
same data batch. It is not part of the formal training/evaluation workflow and
is not a substitute for retraining every XML configuration through the unified
entry points. Formal experimental parameters and result paths therefore remain
solely configuration-owned.
