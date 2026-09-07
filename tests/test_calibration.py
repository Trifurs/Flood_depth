from __future__ import annotations

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from tools.train_pa_hydrokan import apply_train_only_calibration, prepare_train_only_calibration


def test_train_calibration_builds_a_frozen_weight_curve(production_config) -> None:
    contract = DatasetContract.load(production_config["dataset"]["contract"])
    dataset = FloodDepthDataset(
        production_config["dataset"]["contract"],
        production_config["dataset"]["train_stats"],
        "train",
        band_spec=resolve_band_spec(production_config, contract),
        input_spec=ModelInputSpec.from_config(production_config),
        minimum_event_band_fraction=float(
            production_config["dataset"]["minimum_event_band_fraction"]
        ),
        s1_qa_names=production_config["dataset"].get("model_s1_qa_names"),
    )
    payload = prepare_train_only_calibration(production_config, dataset)
    balance = apply_train_only_calibration(production_config, payload)
    assert balance is not None
    assert balance.train_positive_pixels > 0
    assert production_config["model"]["depth_initialization_bias"] < 0
