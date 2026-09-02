# PA-HydroKAN architecture

## Data flow

```text
S1 T1/T2 shared encoder + S1 change stem ─┐
                                          ├─ per-scale masked reliability softmax ─┐
S2 T1/T2 shared encoder + S2 change stem ─┘                                        │
DSM/slope → mask-aware terrain proxies → terrain pyramid ──────────────────────────┤
                                                                                   │
1/8 fused feature → 8-neighbour Terrain Graph-KAN → U-Net/FPN decoder → 3 heads ──┘
```

For a 256×256 input and channels `[32,64,128,256]`, encoder/fusion tensors are
`[B,32,256,256]`, `[B,64,128,128]`, `[B,128,64,64]`, and
`[B,256,32,32]`. The Graph-KAN therefore operates on 1,024 nodes rather than a Python
graph over all 65,536 source pixels. The decoder returns `[B,32,256,256]` and every
head returns `[B,1,256,256]`.

Within modality (m), each scale fuses

\[
[F_{pre}, F_{event}, F_{event}-F_{pre}, |F_{event}-F_{pre}|, F_{change}].
\]

S1 and S2 do not share weights. S1 incidence angle remains an input and also provides
a bounded FiLM modulation.

## Asynchronous fusion

QA-derived logits are masked by local sensor availability:

\[
(w_{S1},w_{S2})=\mathrm{masked\_softmax}(l_{S1},l_{S2}),
\]

\[
F_{fused}=w_{S1}P_{S1}(F_{S1})+w_{S2}P_{S2}(F_{S2})+P_z(F_z).
\]

One valid modality receives weight one; both invalid modalities receive zero. Terrain
remains available. Modality dropout updates the same availability mask.

## Terrain Graph-KAN

Mask-aware DSM operations construct 9-pixel low-frequency `z_hyd`, relative height,
barrier, gradients, local relief, and slope. Historical Hydro-v9 additionally exposed
33- and 65-pixel context, but those features worsened pixel accuracy and are disabled
in Hydro-v12. These online DSM products are not HAND or bare-earth DTM. For each of
eight fixed grid directions, an edge
descriptor contains relative/absolute low-frequency elevation difference, mean slope,
barrier, sensor reliability, selected-day difference, and signed/absolute projected
feature difference.

`KANLinear` bounds layer-normalized inputs with `tanh` and evaluates learnable cubic
B-spline coefficients on a fixed open-uniform grid, with a parallel SiLU linear path.
No third-party KAN library or dynamic knot rearrangement is used. Messages are

\[
c_{ij}=\sigma(\mathrm{KAN}_{edge}(e_{ij})),\qquad
m_{ij}=c_{ij}W(F_j-F_i).
\]

Eight direction tensors are aggregated and passed through 1×1 projection,
GroupNorm, SiLU, dropout, and a residual connection. (c_{ij}) is latent connectivity,
not real hydraulic flux.

## Output and loss

\[
p=\sigma(l),\quad d_{cond}=\mathrm{softplus}(r_d),\quad
b=\mathrm{softplus}(r_b)+\epsilon,\quad d_{weighted}=p\,d_{cond}.
\]

Hydro-v2 through Hydro-v12 use `d_cond` as the primary continuous depth because supervision is
conditional on a reliable positive flood label. `d_weighted` is exported separately
and is not used to shrink the positive-depth target. The support output remains a PU
evidence score; the observed-label prior proxy is insufficient to claim complete
wet/dry probability calibration. A semantics field in the resolved model config keeps
pre-v2 checkpoints on the legacy `d=d_weighted` behavior.

Only reliable positives (P) enter linear/log SmoothL1 and Laplace NLL. Hydro-v12 uses
frozen train-only boundaries at 0.10, 0.23, 0.48, 0.83, 1.22, and 2.14 m. Errors are
averaged within every non-empty depth stratum of each raster, then across strata and
rasters. Source-event IDs are deliberately discarded before loss computation. This
keeps rare depth ranges visible without making the deployed estimator event-specific.
The
unlabeled candidate set (U) excludes positives, permanent water, extreme-high
labels, invalid DEM, and pixels without either sensor. Stable nnPU logistic risk is

\[
R_+=\pi E_P[\mathrm{softplus}(-l)],\quad
R_-=E_U[\mathrm{softplus}(l)]-\pi E_P[\mathrm{softplus}(l)],
\]

\[
L_{nnPU}=R_+ + \max(0,R_-).
\]

The active Hydro-v12 physical term applies a weak local curvature penalty to predicted
water-surface elevation `w=z_hyd+d` on supported positive neighbourhoods:

\[
L_{wse}=\operatorname{mean}\left|4w_{ij}-w_{i-1,j}-w_{i+1,j}-w_{i,j-1}-w_{i,j+1}\right|.
\]

Asynchronous sensor-day differences downweight the contribution. The term starts at
zero-based epoch 5 and reaches weight 0.02 after a 15-epoch warm-up. It is retained as
a weak local inductive prior because the pixel-first validation baseline benefited
from it; it is not evidence of a flat water surface, mass conservation, or solved
hydrodynamics. Terrain-order and reference-gradient alternatives remain implemented
for reproducible historical ablation only.
