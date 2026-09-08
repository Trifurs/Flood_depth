# Data contract

The active configuration uses the complete rebalanced FloodDepthNet v3 release
at `/media/whu/0d7bb559-7b14-4875-843f-08befb3ca56b/myData/FloodDepthNet`.
Its event-chain-independent split has 5,323 train, 258 validation, and 254 test
patches.

`assets/flooddepthnet_s1_terrain/dataset_contract.json` binds the full release
manifest, active raster descriptions, split counts, and provenance-file
fingerprints. Its matching
`assets/flooddepthnet_s1_terrain/train_stats.json` contains train-only robust
normalization statistics. The active loader opens only Sentinel-1 T1/T2/change,
S1 QA, DEM, masks, and labels. Sentinel-2 paths may remain in the release
manifest as metadata, but they are not registered as an input group or read at
runtime.

Rebuild both assets with
`tools/prepare_flooddepthnet_s1_terrain_assets.py` if the release, manifest,
selected S1 bands, or preprocessing semantics change. The detailed formal
training workflow is in [FULL_DATA_TRAINING.md](FULL_DATA_TRAINING.md).
