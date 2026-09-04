from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from utils.misc import move_to_device
from utils.registry import build_model


def test_real_sample_model_forward_and_backward(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = DatasetContract.load(config["dataset"]["contract"])
    input_spec = ModelInputSpec.from_config(config)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "train",
        band_spec=resolve_band_spec(config, contract), input_spec=input_spec,
    )
    batch = move_to_device(next(iter(DataLoader(dataset, batch_size=1))), device)
    model = build_model(config).to(device)
    outputs = model(prepare_model_inputs(batch, input_spec))
    for name in (
        "depth",
        "support_logits",
        "support_probability",
        "conditional_depth",
        "positive_depth",
        "expected_depth",
        "uncertainty_scale",
    ):
        assert outputs[name].shape == (1, 1, 256, 256)
        assert torch.isfinite(outputs[name]).all()
    torch.testing.assert_close(outputs["depth"], outputs["conditional_depth"])
    torch.testing.assert_close(outputs["positive_depth"], outputs["conditional_depth"])
    assert torch.isfinite(outputs["expected_depth"]).all()
    if config["model"].get("event_depth_scale_enabled", False):
        assert outputs["event_depth_scale"].shape == (1, 1, 1, 1)
        assert outputs["event_log_depth_scale"].shape == (1, 1, 1, 1)
        torch.testing.assert_close(
            outputs["event_depth_scale"],
            torch.ones_like(outputs["event_depth_scale"]),
        )
    else:
        assert "event_depth_scale" not in outputs
    assert torch.all(outputs["depth"] >= 0)
    assert torch.all(outputs["uncertainty_scale"] > 0)
    loss = outputs["depth"].mean() + outputs["uncertainty_scale"].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
