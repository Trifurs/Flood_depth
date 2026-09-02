from __future__ import annotations

from pathlib import Path

import torch

from datasets.flooddepth_dataset import FloodDepthDataset, prepare_model_inputs
from extent.ai4g_mobilenet_unet import AI4GFloodExtentNet
from extent.losses import masked_soft_iou_loss
from extent.protocol import build_ai4g_change_features, postprocess_extent
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _batch() -> dict:
    pre = torch.tensor(
        [[[[ -10.0, -10.0], [-40.0, -10.0]], [[-15.0, -15.0], [-15.0, -15.0]], [[35.0, 35.0], [35.0, 35.0]]]]
    )
    event = torch.tensor(
        [[[[ -20.0, -15.0], [-20.0, -24.0]], [[-25.0, -20.0], [-25.0, -25.0]], [[35.0, 35.0], [35.0, 35.0]]]]
    )
    ones = torch.ones(1, 1, 2, 2)
    return {
        "extent_inputs": {
            "s1_t1_db": pre,
            "s1_t2_db": event,
            "s1_pair_valid": ones,
        },
        "validity": {"s1_valid": ones, "output_valid": ones},
    }


def test_published_change_thresholds_are_reproduced() -> None:
    features, valid = build_ai4g_change_features(_batch())
    assert valid.all()
    expected_vv = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    expected_vh = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    torch.testing.assert_close(features[:, 0:1], expected_vv)
    torch.testing.assert_close(features[:, 1:2], expected_vh)


def test_extent_postprocessing_applies_four_pixel_buffer_and_validity() -> None:
    probability = torch.zeros(1, 1, 15, 15)
    probability[0, 0, 7, 7] = 0.9
    valid = torch.ones_like(probability)
    valid[:, :, 0] = 0.0
    raw, buffered = postprocess_extent(
        probability, valid, probability_threshold=0.5, buffer_pixels=4
    )
    assert int(raw.sum()) == 1
    assert int(buffered.sum()) == 81
    assert not buffered[:, :, 0].any()


def test_extent_model_forward_contract() -> None:
    model = AI4GFloodExtentNet((64, 48, 32, 24, 16)).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 2, 64, 64))
    assert output.shape == (2, 1, 64, 64)
    assert torch.isfinite(output).all()


def test_dataset_exposes_raw_s1_only_in_extent_namespace() -> None:
    config = load_config(PROJECT_ROOT / "configs/extent/subset150_ai4g_mobilenet_iou.xml")
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "val"
    )
    sample = dataset[0]
    assert set(sample["extent_inputs"]) == {"s1_t1_db", "s1_t2_db", "s1_pair_valid"}
    assert sample["extent_inputs"]["s1_t1_db"].shape == (3, 256, 256)
    assert "extent_inputs" not in prepare_model_inputs(sample)


def test_soft_iou_is_the_extent_optimization_objective() -> None:
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    valid = torch.ones_like(target)
    perfect_logits = torch.tensor([[[[20.0, -20.0], [20.0, -20.0]]]])
    wrong_logits = -perfect_logits
    perfect_loss, perfect_iou = masked_soft_iou_loss(perfect_logits, target, valid)
    wrong_loss, wrong_iou = masked_soft_iou_loss(wrong_logits, target, valid)
    assert perfect_iou > 0.999
    assert perfect_loss < 0.001
    assert wrong_iou < perfect_iou
    assert wrong_loss > perfect_loss


def test_soft_iou_ignores_invalid_pixels() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 0.0]]]])
    first, _ = masked_soft_iou_loss(torch.tensor([[[[3.0, -20.0]]]]), target, valid)
    second, _ = masked_soft_iou_loss(torch.tensor([[[[3.0, 20.0]]]]), target, valid)
    torch.testing.assert_close(first, second)
