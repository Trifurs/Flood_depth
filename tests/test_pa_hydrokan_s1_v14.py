from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.flooddepth_dataset import prepare_model_inputs
from datasets.model_input_spec import ModelInputSpec
from models.sar_hydro_decoder import SARHydroDecoder
from utils.config import load_config
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_s1_model_real_raster_forward_backward_and_no_support_branch() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v14.xml")
    from tools.train import create_dataloaders

    _, loader, _, _ = create_dataloaders(config)
    batch = next(iter(loader))
    model = build_model(config)
    outputs = model(prepare_model_inputs(batch, ModelInputSpec.from_config(config)))
    assert outputs["depth"].shape == batch["label"].shape
    assert outputs["uncertainty_scale"].shape == batch["label"].shape
    assert len(outputs["auxiliary_depths"]) == 1
    assert "support_probability" not in outputs
    assert "modality_weights" not in outputs
    assert "fusion_entropy" not in outputs
    outputs["depth"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_s1_model_rejects_non_s1_inputs() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_minimal.xml")
    model = build_model(config)
    required = {
        "s1_t1": torch.zeros(1, 2, 32, 32),
        "s1_t2": torch.zeros(1, 2, 32, 32),
        "s1_change": torch.zeros(1, 3, 32, 32),
        "s1_qa": torch.zeros(1, 5, 32, 32),
        "terrain": torch.zeros(1, 2, 32, 32),
        "terrain_raw": torch.ones(1, 2, 32, 32),
        "reliability": torch.zeros(1, 6, 32, 32),
        "s1_valid": torch.ones(1, 1, 32, 32),
        "s1_event_support": torch.ones(1, 1, 32, 32),
        "dem_valid": torch.ones(1, 1, 32, 32),
        "s1_conditioning": torch.zeros(1, 2, 32, 32),
        "branch_validity": {},
    }
    with pytest.raises(ValueError, match="non-S1"):
        model({**required, "s2_t1": torch.zeros(1, 1, 32, 32)})


def test_s1_decoder_supports_odd_spatial_sizes_and_one_quarter_auxiliary() -> None:
    decoder = SARHydroDecoder([32, 64, 128, 192], widths=[96, 64, 48, 32], auxiliary_count=1)
    sizes = [(65, 67), (33, 34), (17, 17), (9, 9)]
    sar = [torch.randn(2, channels, *size) for channels, size in zip((32, 64, 128, 192), sizes)]
    terrain = [torch.randn_like(value) for value in sar]
    fractions = [torch.ones(2, 1, *size) for size in sizes]
    output, auxiliaries, _ = decoder(
        sar[-1], sar, terrain, fractions, torch.ones(2, 1, 65, 67), sar
    )
    assert output.shape == (2, 32, 65, 67)
    assert len(auxiliaries) == 1
    assert auxiliaries[0].shape[-2:] == (17, 17)
