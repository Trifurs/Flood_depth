# Optical-free model architecture

The active architecture is documented in
`docs/HYDROKAN_S1_V15_REFACTOR_REPORT.md`.

At a high level, `pa_hydrokan_s1_v15` contains:

1. separate Sentinel-1 pre-event/event/change encoders with validity masking;
2. incidence-angle conditioning and SAR-first terrain/reliability residual fusion;
3. multi-scale hydrologic context and DSM-derived physical features;
4. a terrain-conditioned Edge-KAN residual block;
5. an independent-gate decoder with conditional positive-depth and uncertainty
   heads.

There is no Sentinel-2 or optical encoder, fusion gate, or optical-derived feature
in this active path.
