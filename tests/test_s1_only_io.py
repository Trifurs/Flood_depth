from __future__ import annotations

import json
from pathlib import Path

from datasets.band_selection import BandSpec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _s1_dataset(contract_path: Path) -> FloodDepthDataset:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v14.xml")
    input_spec = ModelInputSpec.from_config(config)
    contract = DatasetContract.load(contract_path)
    bands = BandSpec.resolve(
        contract,
        config["dataset"]["model_bands"],
        input_spec.continuous_groups,
    )
    return FloodDepthDataset(
        contract_path,
        config["dataset"]["train_stats"],
        "val",
        input_spec=input_spec,
        band_spec=bands,
    )


def test_s1_only_reads_only_active_groups_and_exposes_no_s2_fields(monkeypatch) -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v14.xml")
    contract_path = Path(config["dataset"]["contract"])
    dataset = _s1_dataset(contract_path)
    original_read = dataset._read
    opened: list[str] = []

    def audited_read(row, group, indexes=None):
        opened.append(group)
        assert not group.startswith("s2_"), f"S2 group was opened: {group}"
        return original_read(row, group, indexes)

    monkeypatch.setattr(dataset, "_read", audited_read)
    sample = dataset[0]
    assert not any(group.startswith("s2_") for group in opened)
    assert set(opened) == set(ModelInputSpec.from_mode("s1_terrain").active_groups)
    assert not any(key.startswith("s2_") for key in sample)
    assert not any(key.startswith("s2_") for key in sample["validity"])
    assert "S2_event_composite_valid_mask" not in sample["masks"]
    model_inputs = prepare_model_inputs(sample, ModelInputSpec.from_mode("s1_terrain"))
    assert not any(key.startswith("s2_") for key in model_inputs)
    assert tuple(model_inputs["reliability_names"]) == (
        "s1_event_observation_count_z",
        "s1_event_day_z",
        "s1_available",
        "dem_available",
        "event_duration_log_scaled",
        "s1_day_missing",
    )


def test_s1_only_contract_still_works_when_optional_s2_groups_are_absent(tmp_path) -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v14.xml")
    source = Path(config["dataset"]["contract"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    for group in ("s2_t1", "s2_t2", "s2_change", "s2_qa"):
        payload["raster_groups"].pop(group, None)
    contract_path = tmp_path / "s1_only_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = _s1_dataset(contract_path)
    sample = dataset[0]
    assert sample["metadata"]["model_input_spec"]["mode"] == "s1_terrain"
    assert sample["metadata"]["io_profile"]["read_band_counts"]["s1_t2"] == 3
