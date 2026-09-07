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
conda run -n flood-depth python tools/train_pa_hydrokan.py \
  --config configs/ablation/pa_hydrokan_no_topographic_affinity_edge_kan.xml \
  --device cuda
```

## Lightweight functional probe

`tools/run_ablation_probe.py` checks that every variant loads a common full
checkpoint and measures the immediate output change on the same data batch.
It is a reproducible configuration and sensitivity check, not a substitute for
the retrained study above.

```bash
conda run -n flood-depth python tools/run_ablation_probe.py \
  --checkpoint runs/train/<full_run>/best_raw.pth \
  --split val --device cpu --batch-size 1 --max-batches 1 \
  --output runs/ablation/<probe_name>
```

The probe writes CSV, JSON, and Markdown tables with active trainable parameter
counts and deltas from the full configuration.

## Local one-batch functional check

On 2026-09-07, the local smoke checkpoint
`runs/train/pa_hydrokan_subset1000_named_smoke/best_raw.pth` was evaluated on
the first validation batch (4,638 canonical positive pixels) with the probe.
The reproducible artifact is
`runs/ablation/subset1000_one_batch_frozen_weight_probe/`.

| Variant | Pixel MAE (m) | Change from full (m) | Active trainable parameters |
|---|---:|---:|---:|
| PA-HydroKAN | 0.308934 | 0.000000 | 3,502,810 |
| w/o RCP | 0.310482 | +0.001548 | 3,498,234 |
| w/o TCF | 0.310078 | +0.001143 | 3,439,414 |
| w/o TAE-KAN | 0.308937 | +0.000003 | 3,391,900 |
| w/o LCA | 0.308932 | −0.000003 | 3,502,804 |

The RCP and TCF interventions change the frozen checkpoint's prediction on
this batch. TAE-KAN and its zero-initialized LCA show negligible immediate
effect after only one optimizer update; this is evidence that the smoke check
cannot assess their learned contribution, not evidence that they are
ineffective. The matched retraining protocol above is required for any paper
table or scientific conclusion.

All four removal variants also completed an independent one-update CPU
training/validation smoke run under `runs/ablation_training_smoke/`, confirming
that frozen paths are excluded from optimizer updates without breaking the
training, EMA, loss, or checkpoint workflows.
