from __future__ import annotations

import torch

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec


def test_each_split_loads_real_finite_sample(config: dict) -> None:
    expected = {"train": 825, "val": 89, "test": 83}
    shapes = {
        "s1_t1": (2, 256, 256),
        "s1_t2": (2, 256, 256),
        "s1_change": (3, 256, 256),
        "s1_qa": (5, 256, 256),
        "terrain": (2, 256, 256),
        "label": (1, 256, 256),
        "reliability": (6, 256, 256),
    }
    contract = DatasetContract.load(config["dataset"]["contract"])
    input_spec = ModelInputSpec.from_config(config)
    band_spec = resolve_band_spec(config, contract)
    for split in ("train", "val", "test"):
        dataset = FloodDepthDataset(
            config["dataset"]["contract"], config["dataset"]["train_stats"], split,
            band_spec=band_spec, input_spec=input_spec,
        )
        assert len(dataset) == expected[split]
        sample = dataset[0]
        for key, shape in shapes.items():
            assert sample[key].shape == shape
            assert sample[key].dtype == torch.float32
            assert torch.isfinite(sample[key]).all()
        assert set(sample["masks"]) == {
            "valid_depth_mask",
            "flood_mask",
            "unknown_mask",
            "permanent_water_mask",
            "extreme_high_mask",
            "DEM_valid_mask",
            "slope_valid_mask",
            "persistent_water",
            "S1_event_composite_valid_mask",
        }
        assert sample["metadata"]["split"] == split
        assert torch.equal(
            sample["masks"]["valid_depth_mask"], sample["masks"]["flood_mask"]
        )
        model_inputs = prepare_model_inputs(sample, input_spec)
        assert "label" not in model_inputs
        assert "masks" not in model_inputs
        assert not any(key.startswith("s2_") for key in model_inputs)
