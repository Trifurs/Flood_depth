from __future__ import annotations

import json

import rasterio

from datasets.contract import DatasetContract


def test_audited_contract_matches_real_rasters(config: dict) -> None:
    contract = DatasetContract.load(config["dataset"]["contract"])
    contract.verify_fingerprints(include_normalization=True)
    assert contract.main_input_channels == 27
    assert contract.payload["sample_counts"] == {"test": 83, "train": 825, "val": 89}
    audit = json.loads(
        (config["project_root"] / "artifacts/dataset_audit/subset1000_audit.json").read_text()
    )
    assert audit["status"] == "ready"
    assert not audit["errors"]
    with contract.manifest_path.open(encoding="utf-8-sig") as handle:
        header, first = handle.readline(), handle.readline()
    assert "sample_id" in header and first
    import csv

    row = next(csv.DictReader(contract.manifest_path.open(encoding="utf-8-sig")))
    for group in ("s1_t1", "s1_t2", "s1_change", "s1_qa", "terrain", "label", "masks"):
        specification = contract.group(group)
        with rasterio.open(contract.dataset_root / row[specification["path_column"]]) as dataset:
            assert list(dataset.descriptions) == specification["band_descriptions"]
            assert dataset.count == specification["band_count"]
