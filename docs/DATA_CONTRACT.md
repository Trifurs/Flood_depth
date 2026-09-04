# S1-only data contract

The active depth model uses the audited contract at
`artifacts/dataset_audit/subset1000_contract.json` with
`input_mode=s1_terrain`.

Active raster groups are:

- S1 pre-event, event, change, and QA;
- DSM elevation and slope;
- label and masks for loss/evaluation only.

`datasets/model_input_spec.py` defines the active group boundary and
`datasets/band_selection.py` resolves selected channels by exact audited band
names. The loader opens only active groups for the S1-only model path and records
the opened-file/band profile in sample metadata. No source raster is modified.

The audited source contract may retain optional historical raster groups for
provenance, but they are inactive and are not part of the active model input.
