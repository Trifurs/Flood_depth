# Scientific assumptions and claim boundaries

1. `depth_m` is an event-interval reconstructed/aggregated JRC reference. It is not
   assumed to be a contemporaneous in-situ depth survey.
2. S1/T2 and S2/T2 are event-period per-pixel composites, not single acquisitions.
   Pixels in one patch can select different dates.
3. Sentinel-1 and Sentinel-2 event composites are generally asynchronous. The model
   uses selected-day distance as reliability information and does not force their
   feature representations to be equal.
4. Elevation is DSM. It is never called bare-earth DTM or assumed to be the riverbed.
5. `valid_depth_mask` is a reliable positive-depth set. In subset150, `flood_mask`
   equals this set and supplies no reliable complete dry negatives. Unknown,
   permanent-water, extreme-high, and nodata pixels are not supervised as 0 m.
6. Dataset admission used label-dependent post-export QC and selected scenes with
   clear changes. This creates a selection bias toward high-quality/easier cases even
   though the exported imagery algorithm itself is label-independent.
7. PA-HydroKAN is physics-guided, not a strict shallow-water-equation PINN. The active
   Hydro-v9 retains Hydro-v6's local, one-sided terrain--depth ordering penalty on reliable
   positive neighbour pairs and is introduced only after a delayed warm-up. It does
   not require a globally flat water surface, and there is no mass-conservation,
   velocity, discharge, or flux claim. Because `z_hyd` is DSM-derived, even this weak
   ordering can be imperfect near buildings, canopy, levees, or unresolved channels.
   Its additional 33/65-pixel relative-height, depression, and relief channels remain
   DSM-derived context proxies; they are not HAND, flow accumulation, or catchment
   connectivity.
8. The Terrain Graph-KAN gate represents learned latent connectivity conditioned on
   terrain and reliability. It is not physical discharge, velocity, or hydrodynamic
   flux.
9. The model directly predicts continuous depth. It does not first threshold a flood
   mask and interpolate depth from DEM.
10. Main scientific conclusions must rely on the future complete dataset and
    independent external validation, not only the 150-sample engineering subset.
