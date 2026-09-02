from __future__ import annotations

import torch

from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs


def test_each_split_loads_real_finite_sample(config: dict) -> None:
    expected = {"train": 105, "val": 23, "test": 22}
    shapes = {
        "s1_t1": (3, 256, 256),
        "s1_t2": (3, 256, 256),
        "s1_change": (4, 256, 256),
        "s1_qa": (5, 256, 256),
        "s2_t1": (6, 256, 256),
        "s2_t2": (6, 256, 256),
        "s2_change": (3, 256, 256),
        "s2_qa": (3, 256, 256),
        "terrain": (2, 256, 256),
        "label": (1, 256, 256),
        "reliability": (12, 256, 256),
    }
    for split in ("train", "val", "test"):
        dataset = FloodDepthDataset(
            config["dataset"]["contract"], config["dataset"]["train_stats"], split
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
            "S2_event_composite_valid_mask",
        }
        assert sample["metadata"]["split"] == split
        assert torch.equal(
            sample["masks"]["valid_depth_mask"], sample["masks"]["flood_mask"]
        )
        assert "label" not in prepare_model_inputs(sample)
        assert "masks" not in prepare_model_inputs(sample)
