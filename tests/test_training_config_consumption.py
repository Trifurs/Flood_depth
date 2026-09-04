from pathlib import Path
import pytest
from utils.config import load_config
from utils.optim import build_optimizer, build_scheduler
from tools.train import infer_legacy_patience
import torch


def test_optimizer_and_scheduler_names_are_consumed() -> None:
    config = load_config(Path("configs/pa_hydrokan/subset1000_s1_v15.xml"))
    model = torch.nn.Linear(2, 1); optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, 10, 2)
    assert isinstance(optimizer, torch.optim.AdamW)
    config["optimizer"]["name"] = "ignored"
    with pytest.raises(ValueError): build_optimizer(model, config)


def test_final_runtime_controls_are_resolved() -> None:
    config = load_config(Path("configs/pa_hydrokan/subset1000_s1_v15_gpu_precision.xml"))
    assert config["training"]["minimum_epochs"] == 35
    assert config["training"]["num_workers"] == 4
    assert config["training"]["persistent_workers"] is True
    assert config["checkpoint"]["save_last"] is True
    assert config["checkpoint"]["save_best"] is True


def test_legacy_patience_is_recovered_from_epoch_log(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text(
        "epoch,val_ema_pixel_micro_mae\n0,0.6\n1,0.5\n2,0.55\n3,0.56\n",
        encoding="utf-8",
    )
    assert infer_legacy_patience(path, 3, "pixel_micro_mae", "ema", 0.0) == 2
