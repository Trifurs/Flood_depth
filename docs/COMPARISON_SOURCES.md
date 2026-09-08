# Learned-comparison sources and adaptation record

## Scope and selection rule

This register records the sources supplied for the learned flood-depth
comparators and the exact adaptation made for this project. A model is retained
only when it can consume the audited complete FloodDepthNet S1/terrain release, can use
`valid_depth_mask` directly as the common flood range, and has enough public
architectural information for a traceable implementation. Every selected model
uses the same train/validation split and reports depth only on
`valid_depth_mask_and_output_valid`.

`valid_depth_mask` is a requested range oracle for comparison models, not a
predicted product, threshold, or separately trained extent model. It is never
an input to PA-HydroKAN.

## Selected models

| Identifier | Source role | Public source | Local implementation and controlled adaptation |
|---|---|---|---|
| `dlsim_attention_unet` | Direct method/architecture source | Yokoya et al., *Breaking Limits of Remote Sensing by Deep Learning From Simulated Data for Flood and Debris-Flow Mapping*, [IEEE TGRS](https://doi.org/10.1109/TGRS.2020.3035469); [official DLSIM code](https://github.com/nyokoya/dlsim) | `compare/deep_learning/dlsim_attention_unet.py` follows the DLSIM water-level regression branch’s Attention U-Net family. The simulated binary change map is replaced by the three normalized observed S1-change bands; normalized DSM and direct `valid_depth_mask` are appended. |
| `dlsim_linknet` | Direct method/architecture source | Same [DLSIM paper](https://doi.org/10.1109/TGRS.2020.3035469) and [official code](https://github.com/nyokoya/dlsim) | `compare/deep_learning/dlsim_linknet.py` follows the DLSIM LinkNet alternative with the same five-channel DLSIM-adapted input. |
| `unet_depth_regression` | Architecture source; flood-depth task context | [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) documents the U-Net architecture API; Blay & Hashemi-Beni, [Remote Sensing 2026, 18, 60](https://doi.org/10.3390/rs18010060) is the supplied flood-depth task reference. | `compare/deep_learning/unet_depth_regression.py` is a local, dependency-free U-Net regression implementation. It receives all normalized S1 T1/T2/change bands, DSM/slope, and direct `valid_depth_mask`. The Remote Sensing paper is not claimed to be a U-Net implementation. |
| `resnet18_depth_regression` | Direct task reference; encoder implementation source | Blay & Hashemi-Beni, [Remote Sensing 2026, 18, 60](https://doi.org/10.3390/rs18010060); [Torchvision ResNet documentation](https://pytorch.org/vision/stable/models/resnet.html) | `compare/deep_learning/resnet18_depth_regression.py` uses a randomly initialized Torchvision ResNet18 encoder and a local full-resolution decoder. It uses the common ten-channel input and direct range. No ImageNet weights are used. |
| `unetplusplus_depth_regression` | Architecture source; data-context reference | [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) documents U-Net++; Blay et al., *Inundation2Depth*, [Data in Brief 2025, 112347](https://doi.org/10.1016/j.dib.2025.112347) is the supplied inundation/depth data reference. | `compare/deep_learning/unetplusplus_depth_regression.py` is a local U-Net++-style nested decoder with the common ten-channel input and direct range. The Data in Brief article is a dataset paper, not evidence that it trained this U-Net++ configuration; it is retained only as task/data context. |

All five local implementations use a conditional-positive (`softplus`) depth
head and set depths outside the supplied range to zero. They intentionally use
the same 0.50 m Huber + 0.05 log-depth objective so results compare
architectures rather than loss engineering. Their constant uncertainty scale is
only a schema-compatible placeholder; uncertainty metrics are not a model claim
for these baselines.

## Considered but not selected

| Candidate | Source | Reason it is not in the current comparable set |
|---|---|---|
| Swin U-Net flood depth | [Swin-Unet repository](https://github.com/HuCaoFighting/Swin-Unet); supplied [Remote Sensing task DOI](https://doi.org/10.3390/rs18010060) | The public repository targets medical-image segmentation and has a medical-data/pretrained-Swin setup. It does not provide a flood-depth/S1 implementation. Adding a transformer with substantial new training and positional-embedding choices would not be a faithful, minimal adaptation. |
| SWOT Attention flood-depth network | Li et al., [Journal of Hydrology 2026, 136000](https://doi.org/10.1016/j.jhydrol.2026.136000) | The paper is relevant, but no official training implementation is public and the accessible article record does not expose enough layer-level specification for an auditable reproduction. It is deferred until the full architectural and training specification can be verified. |
| U-FLOOD adapted | Löwe et al., [Journal of Hydrology 2021, 126898](https://doi.org/10.1016/j.jhydrol.2021.126898); [official DTU code/data archive](https://data.dtu.dk/articles/code/U-FLOOD_-_computer_code_and_data_associated_with_the_article_U-FLOOD_topographic_deep_learning_for_predicting_urban_pluvial_flood_water_depth_/14206838) | U-FLOOD learns urban **pluvial** depths from hydrodynamic-simulation/topographic inputs. This release has observed S1/terrain inputs but not the rainfall and simulation forcings required to reproduce that task. It is therefore not a like-for-like baseline here. |

## Reference-manager interchange file

Machine-importable records are in
[`comparison_sources.enw`](comparison_sources.enw). The file includes all
selected and deferred user-supplied sources, their DOI/official code links, and
the source role above. It does not claim that an architecture-only or
dataset-only citation is a direct flood-depth implementation.
