import torch

from datasets.flooddepth_dataset import prepare_model_inputs
from tools.train import create_dataloaders
from utils.config import load_config
from utils.registry import build_model


def test_v13_2_real_raster_cpu_forward_backward():
    config = load_config("configs/pa_hydrokan/subset150_v13_2_loss_simple.xml")
    config["training"]["num_workers"] = 0
    config["training"]["persistent_workers"] = False
    loader, _, _, _ = create_dataloaders(config)
    batch = next(iter(loader))
    model = build_model(config)
    outputs = model(prepare_model_inputs(batch))
    loss = outputs["depth"].mean() + outputs["graph_diagnostics"]["kan_coefficient_smoothness"]
    loss.backward()
    assert outputs["depth"].shape == batch["label"].shape
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
