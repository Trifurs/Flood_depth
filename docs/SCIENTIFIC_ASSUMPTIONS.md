# Scientific assumptions

1. The target is an event-aggregated reconstructed flood-depth reference, not a
   direct instantaneous water-depth observation.
2. Sentinel-1 event composites are treated as partially observed states; they are
   not assumed to form a regular time series.
3. DSM elevation is a surface model rather than a bare-earth hydrologic DEM.
4. The conditional positive-depth head estimates depth on the reliable positive
   supervision region. It is not a calibrated flood-extent probability.
5. Terrain Graph-KAN features are latent topographic connectivity priors, not a
   shallow-water solver and not a claim of mass conservation.
6. Validity masks gate inputs and metric aggregation. Invalid/unknown target
   pixels are not treated as zero-depth supervision.

The active model contains no optical/Sentinel-2 input branch.
