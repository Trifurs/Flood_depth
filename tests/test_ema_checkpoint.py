from pathlib import Path
import pytest
import torch
from utils.checkpoint import (
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    training_identity_sha256,
)
from utils.ema import ModelEMA


def test_ema_update_checkpoint_and_restore(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1); ema = ModelEMA(model, .9)
    with torch.no_grad(): model.weight.add_(1)
    ema.update(model)
    path = tmp_path / "ema.pth"
    save_checkpoint(path, model, None, None, None, 0, 1., {"model": {}, "loss": {}, "optimizer": {}, "scheduler": {}}, {}, ema=ema)
    loaded = load_checkpoint(path, torch.nn.Linear(2, 1))
    restored = ModelEMA(model, .5); restored.load_state_dict(loaded["ema"])
    assert restored.updates == 1 and loaded["ema_model"] is not None


def test_resume_rejects_changed_training_identity(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    config = {
        "model": {"name": "test"},
        "loss": {"beta": 0.25},
        "optimizer": {"name": "adamw"},
        "scheduler": {"name": "cosine"},
        "dataset": {
            "resolved_model_bands": {"s1_t1": ["VV_pre_db"]},
            "resolved_reliability_schema": ["s1_available"],
        },
    }
    fingerprint = {"contract_sha256": "contract"}
    path = tmp_path / "identity.pth"
    save_checkpoint(
        path, model, None, None, None, 0, 1.0, config, fingerprint
    )
    changed = {**config, "loss": {"beta": 0.50}}
    with pytest.raises(CheckpointError, match="Training identity differs"):
        load_checkpoint(
            path,
            torch.nn.Linear(2, 1),
            expected_training_identity_sha256=training_identity_sha256(
                changed, fingerprint
            ),
            expected_legacy_training_identity_sha256=training_identity_sha256(
                changed, fingerprint, version=1
            ),
        )
