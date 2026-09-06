"""Regression coverage for V15.2 correctness fixes before candidate training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from losses.frozen_soft_depth_balance import FrozenSoftDepthBalance
from losses.task_adaptive_depth_loss import task_adaptive_positive_depth_loss
from models.s1_hydrology_backbone_v15 import SARReliabilityConditioner
from models.s1_hydrology_backbone_v15_1 import SARHydrologyEncoderV15_1Simple
from tools.train import _inverse_softplus, apply_train_only_calibration
from utils.config import load_config
from utils.ema import ModelEMA, restore_ema_after_checkpoint_load
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _frozen_balance() -> FrozenSoftDepthBalance:
    train_depths = torch.tensor(
        [.1, .1, .1, .15, .2, .24, .3, .5, .75, 1., 1.5, 2., 3., 4., 5.]
    )
    return FrozenSoftDepthBalance.from_train_depths(
        train_depths,
        [.1, .24, .5, 1., 2., 3.5, 5.],
        minimum=.5,
        maximum=3.,
        alpha=.5,
        tau=10.,
    )


def test_depth_weight_is_batch_invariant() -> None:
    balance = _frozen_balance()

    def target_weight(values: list[float]) -> float:
        target = torch.tensor(values).view(1, 1, 1, -1)
        weights = balance.weights(target, torch.ones_like(target))
        return float(weights[0, 0, 0, values.index(1.0)])

    weights = [
        target_weight([1.0]),
        target_weight([.1, 1.0]),
        target_weight([.1, .2, .3, 1.0]),
        target_weight([1.0, 2.0, 4.0]),
    ]
    assert max(weights) - min(weights) < 1.0e-7
    train = torch.tensor([.1, .1, .1, .15, .2, .24, .3, .5, .75, 1., 1.5, 2., 3., 4., 5.])
    train_weights = balance.weights(train, torch.ones_like(train))
    assert float(train_weights.min()) >= .5
    assert float(train_weights.max()) <= 3.
    assert abs(float(train_weights.mean()) - 1.0) < 1.0e-6


def test_frozen_weighted_pixel_loss_is_partition_additive() -> None:
    balance = _frozen_balance()
    feature = torch.tensor([.2, .5, .8, 1.2], dtype=torch.float32)
    target = torch.tensor([.1, .4, 1., 3.], dtype=torch.float32)

    full_parameter = torch.tensor(1.3, requires_grad=True)
    full_prediction = full_parameter * feature
    full_pixels = F.smooth_l1_loss(full_prediction, target, reduction="none", beta=.5)
    full_loss = (full_pixels * balance.weights(target)).mean()
    full_loss.backward()

    partition_parameter = torch.tensor(1.3, requires_grad=True)
    numerator = partition_parameter.new_zeros(())
    for index in (slice(0, 2), slice(2, 4)):
        prediction = partition_parameter * feature[index]
        pixels = F.smooth_l1_loss(prediction, target[index], reduction="none", beta=.5)
        numerator = numerator + (pixels * balance.weights(target[index])).sum()
    (numerator / target.numel()).backward()
    torch.testing.assert_close(partition_parameter.grad, full_parameter.grad, rtol=1.0e-6, atol=1.0e-7)


def test_task_adaptive_log_huber_beta_changes_loss_and_rejects_invalid_value() -> None:
    prediction = torch.tensor([[[[.1, .4, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[[.3, 1.2, 4.0]]]])
    positive = torch.ones_like(target)
    low = task_adaptive_positive_depth_loss(
        prediction, target, positive, [.1, .5, 5.], balance=False, log_beta=.05
    )
    high = task_adaptive_positive_depth_loss(
        prediction, target, positive, [.1, .5, 5.], balance=False, log_beta=1.0
    )
    assert not torch.isclose(low["depth_log"], high["depth_log"])
    with pytest.raises(ValueError, match="log_beta"):
        task_adaptive_positive_depth_loss(
            prediction, target, positive, [.1, .5, 5.], balance=False, log_beta=0.0
        )


def _temporal_encoder() -> tuple[SARHydrologyEncoderV15_1Simple, SARReliabilityConditioner]:
    widths = [8, 16, 24, 32]
    return (
        SARHydrologyEncoderV15_1Simple(
            2, 3, 0, 6, widths, dropout=.4, groups=8,
            deduplicated_reliability=True,
        ),
        SARReliabilityConditioner(6, widths, groups=8),
    )


def _temporal_inputs(size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    pre = torch.randn(1, 2, size, size)
    change = torch.randn(1, 3, size, size)
    valid = torch.ones(1, 1, size, size)
    branch = {"s1_t1": valid, "s1_t2": valid, "s1_change": valid}
    return pre, change, torch.randn(1, 6, size, size), valid, branch


def test_shared_temporal_encoder_does_not_create_false_change_in_train_mode() -> None:
    encoder, conditioner = _temporal_encoder()
    encoder.train()
    pre, _, reliability, valid, branch = _temporal_inputs()
    features = conditioner(reliability, branch)
    _, diagnostics = encoder(
        pre, pre.clone(), torch.zeros(1, 3, 32, 32), torch.empty(1, 0, 32, 32),
        reliability, valid, branch_validity=branch, reliability_features=features,
    )
    assert all(float(value.detach().abs().max()) < 1.0e-6 for value in diagnostics["internal_change"])


def test_masked_temporal_change_paths_preserve_zero_evidence_contract() -> None:
    encoder, conditioner = _temporal_encoder()
    encoder.train()
    pre, change, reliability, valid, branch = _temporal_inputs()

    pre_invalid = {**branch, "s1_t1": torch.zeros_like(valid)}
    _, diagnostics = encoder(
        pre, torch.randn_like(pre), change, torch.empty(1, 0, 32, 32), reliability,
        valid, branch_validity=pre_invalid, reliability_features=conditioner(reliability, pre_invalid),
    )
    assert all(float(value.detach().abs().max()) == 0.0 for value in diagnostics["internal_change"])

    external_invalid = {**branch, "s1_change": torch.zeros_like(valid)}
    _, diagnostics = encoder(
        pre, torch.randn_like(pre), change, torch.empty(1, 0, 32, 32), reliability,
        valid, branch_validity=external_invalid,
        reliability_features=conditioner(reliability, external_invalid),
    )
    assert all(float(value.detach().abs().max()) == 0.0 for value in diagnostics["external_change_weights"])

    no_evidence = {
        **branch,
        "s1_t1": torch.zeros_like(valid),
        "s1_t2": torch.zeros_like(valid),
        "s1_change": torch.zeros_like(valid),
    }
    _, diagnostics = encoder(
        pre, torch.randn_like(pre), change, torch.empty(1, 0, 32, 32), reliability,
        valid, branch_validity=no_evidence, reliability_features=conditioner(reliability, no_evidence),
    )
    assert all(float(value.detach().abs().max()) == 0.0 for value in diagnostics["change_evidence"])


def test_legacy_checkpoint_ema_initializes_from_loaded_model() -> None:
    loaded = torch.nn.Linear(2, 1)
    with torch.no_grad():
        loaded.weight.fill_(2.0)
        loaded.bias.fill_(-.5)
    stale = torch.nn.Linear(2, 1)
    with torch.no_grad():
        stale.weight.fill_(-7.0)
        stale.bias.fill_(7.0)
    ema = ModelEMA(stale, .9)
    restored = restore_ema_after_checkpoint_load(ema, {"ema": None}, loaded)
    assert not restored and ema.updates == 0
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(ema.shadow[name], value)


def test_train_median_depth_bias_and_zero_qa_reliability_conditioner() -> None:
    median = .42
    assert torch.isclose(F.softplus(torch.tensor(_inverse_softplus(median))), torch.tensor(median))
    config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_simple_fixed.xml"
    )
    frozen = _frozen_balance().to_dict()
    apply_train_only_calibration(
        config,
        {
            "frozen_depth_balance": frozen,
            "depth_initialization": {
                "source": "canonical_train_positive_depth_median",
                "train_positive_depth_median_m": median,
                "depth_initialization_bias": _inverse_softplus(median),
            },
        },
    )
    model = build_model(config)
    assert model.reliability_conditioner is not None
    assert model.sar_encoder.conditioner is None
    assert model.fusion.reliability_projection is None
    assert config["model"]["s1_qa_channels"] == 0


def test_v15_2_dataset_exposes_no_duplicated_qa_tensor_and_never_s2() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_simple_fixed.xml"
    )
    contract = DatasetContract.load(config["dataset"]["contract"])
    sample = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "val",
        input_spec=ModelInputSpec.from_config(config),
        band_spec=resolve_band_spec(config, contract),
        minimum_event_band_fraction=config["dataset"]["minimum_event_band_fraction"],
        s1_qa_names=config["dataset"]["model_s1_qa_names"],
    )[0]
    assert sample["s1_qa"].shape[0] == 0
    # The two source bands are read only to construct the single reliability
    # tensor; no redundant QA channels are exposed to the model.
    assert sample["metadata"]["io_profile"]["read_band_counts"]["s1_qa"] == 2
    assert sample["metadata"]["model_s1_qa_names"] == ()
    assert not any(key.startswith("s2_") for key in sample)
    assert not any(key.startswith("s2_") for key in sample["validity"])
