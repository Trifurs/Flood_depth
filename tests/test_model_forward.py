from __future__ import annotations

import torch
from torch.utils.data import default_collate

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from utils.registry import build_model


def test_model_forward_on_audited_sample(production_config) -> None:
    contract = DatasetContract.load(production_config["dataset"]["contract"])
    spec = ModelInputSpec.from_config(production_config)
    dataset = FloodDepthDataset(
        production_config["dataset"]["contract"],
        production_config["dataset"]["train_stats"],
        "train",
        band_spec=resolve_band_spec(production_config, contract),
        input_spec=spec,
        minimum_event_band_fraction=float(
            production_config["dataset"]["minimum_event_band_fraction"]
        ),
        s1_qa_names=production_config["dataset"].get("model_s1_qa_names"),
    )
    batch = default_collate([dataset[0]])
    model = build_model(production_config).eval()
    with torch.no_grad():
        outputs = model(prepare_model_inputs(batch, spec))
    assert outputs["depth"].shape == batch["label"].shape
    assert torch.isfinite(outputs["depth"]).all()
    assert torch.isfinite(outputs["uncertainty_scale"]).all()
    assert torch.all(outputs["depth"] > 0)
