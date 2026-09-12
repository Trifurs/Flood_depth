# PA-HydroKAN ablation protocol

## Three-factor design

The ablation factors are the three top-level proposed modules: RCP, TCF, and
TAE-KAN. The design contains every non-empty subset (three single removals,
three pairwise removals, and one three-module removal). All variants inherit
the dataset, preprocessing, supervision, loss, optimizer, scheduler, and seed
settings from `configs/pa_hydrokan.xml`; only the listed architecture switches
are disabled. The full reference is `configs/pa_hydrokan.xml` itself and is not
duplicated under `configs/ablation/`.

| Variant | Configuration | Disabled modules |
|---|---|---|
| PA-HydroKAN | `configs/pa_hydrokan.xml` | none |
| PA-HydroKAN w/o RCP | `configs/ablation/pa_hydrokan_wo_rcp.xml` | RCP |
| PA-HydroKAN w/o TCF | `configs/ablation/pa_hydrokan_wo_tcf.xml` | TCF |
| PA-HydroKAN w/o TAE-KAN | `configs/ablation/pa_hydrokan_wo_tae_kan.xml` | TAE-KAN |
| PA-HydroKAN w/o RCP + TCF | `configs/ablation/pa_hydrokan_wo_rcp_tcf.xml` | RCP, TCF |
| PA-HydroKAN w/o RCP + TAE-KAN | `configs/ablation/pa_hydrokan_wo_rcp_tae_kan.xml` | RCP, TAE-KAN |
| PA-HydroKAN w/o TCF + TAE-KAN | `configs/ablation/pa_hydrokan_wo_tcf_tae_kan.xml` | TCF, TAE-KAN |
| PA-HydroKAN w/o RCP + TCF + TAE-KAN | `configs/ablation/pa_hydrokan_wo_rcp_tcf_tae_kan.xml` | RCP, TCF, TAE-KAN |

LCA remains documented as an internal mechanism of TAE-KAN, not an independent
ablation factor. Disabling TAE-KAN bypasses and freezes its entire graph path,
including LCA. Treating LCA as a fourth peer module would misrepresent the model
hierarchy and introduce redundant interventions.

The top-level intervention boundaries are:

- `w/o RCP` supplies zero reliability-conditioning features and zeroes the
  reliability channels seen by the internal CDH global calibrator, while
  retaining S1 observations and spatial validity masks. Thus reliability
  metadata has no residual route to the prediction.
- `w/o TCF` removes terrain/proxy-conditioned additive fusion paths while TPP
  remains available to the decoder and TAE-KAN.
- `w/o TAE-KAN` bypasses and freezes the complete topographic-affinity graph
  update, including its latent compatibility calculation.

Every combination must be independently retrained from scratch. The resulting
eight-cell design (full plus seven removals) supports both individual-module and
interaction analysis; a same-checkpoint functional probe is not a substitute
for these retrained experiments.

Example:

```bash
python train.py configs/ablation/pa_hydrokan_wo_rcp_tae_kan.xml
```

The shared runtime policy is in `configs/base/base.xml`. Standalone runs are
stored under `runs/flooddepthnet_s1_terrain/ablation/<variant>/<started-at>/`.
`validate_k_fold.py` automatically includes the main model and all seven
ablation configurations once each.

## Lightweight functional probe

`tools/run_ablation_probe.py` remains an internal same-checkpoint sensitivity
diagnostic. Its output must be labelled as a functional probe, never as the
formal ablation result.
