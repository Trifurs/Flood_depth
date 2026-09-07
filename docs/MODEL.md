# Model

The model uses a radar-first information flow:

- Separate pre-event state, event state, and externally supplied change branches.
- Branch-validity fractions prevent raster fill values from becoming evidence.
- Acquisition reliability is encoded once and reused at each spatial scale.
- Incidence-angle conditioning is initialized as an identity residual.
- Terrain is fused as a bounded residual rather than replacing the SAR stream.
- An eight-neighbour Edge-KAN module learns terrain-conditioned latent affinity.
- The decoder predicts positive conditional depth and a detached uncertainty scale.

The graph is a learned spatial prior, not a hydraulic simulator. Physical losses
are optional and disabled in the default training configuration.
