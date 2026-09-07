# Data contract

The active configuration points to the subset stored at
`/home/whu/桌面/myData/Flood_depth/subset1000`.

`assets/dataset_contract.json` binds the manifest, selected bands, raster
descriptions, and source file fingerprints. `assets/train_stats.json` contains
normalization values computed from valid training pixels only. The loader reads
only Sentinel-1 state, event, change, quality, terrain, masks, and labels.

Do not modify either asset in place when retraining on the same dataset. Build a
new audited contract and matching normalization statistics when the data source,
manifest, bands, or preprocessing change.
