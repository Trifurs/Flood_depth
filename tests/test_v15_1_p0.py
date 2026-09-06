from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from datasets.supervision_masks import (
    CANONICAL_POSITIVE_MASK,
    canonical_positive_mask_from_batch,
    s1_output_valid_mask,
    supervision_mask_counts,
)
from losses.composite_loss import CompositeFloodDepthLoss
from models.hydro_edge_kan_s1 import HydroEdgeKANS1
from models.kan_layers import KANLinear
from models.s1_hydrology_backbone_v15 import masked_two_way_change_mixer
from tools.evaluate import evaluate_loader
from utils.config import load_config
from utils.registry import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _batch() -> dict[str, object]:
    label = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    ones = torch.ones_like(label)
    zeros = torch.zeros_like(label)
    return {
        "label": label,
        "masks": {
            "valid_depth_mask": torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]]),
            "permanent_water_mask": zeros,
            "extreme_high_mask": zeros,
        },
        "validity": {
            "output_valid": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]]),
            "s1_valid": ones,
            "s1_event_support": ones,
            "dem_valid": ones,
            "s1_t1_valid_fraction": ones,
            "s1_t2_valid_fraction": ones,
            "s1_change_valid_fraction": ones,
            "terrain_valid_fraction": ones,
        },
        "s1_t1": ones.repeat(1, 2, 1, 1),
        "s1_t2": ones.repeat(1, 2, 1, 1),
        "s1_change": ones.repeat(1, 3, 1, 1),
        "s1_qa": ones.repeat(1, 5, 1, 1),
        "terrain": ones.repeat(1, 2, 1, 1),
        "terrain_raw": ones.repeat(1, 2, 1, 1),
        "reliability": ones.repeat(1, 6, 1, 1),
        "reliability_names": (
            "s1_event_observation_count_z",
            "s1_event_day_z",
            "s1_available",
            "dem_available",
            "event_duration_log_scaled",
            "s1_day_missing",
        ),
        "metadata": {
            "sample_id": ["unit"],
            "source_event_id": ["event"],
            "crs": ["EPSG:3857"],
            "transform": [(1.0, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0, 1.0)],
        },
    }


def test_canonical_mask_is_output_aware_and_loss_logs_its_counts() -> None:
    batch = _batch()
    mask = canonical_positive_mask_from_batch(batch)
    assert torch.equal(mask, torch.tensor([[[[True, False], [False, True]]]]))
    counts = supervision_mask_counts(batch)
    assert counts["valid_depth_mask_pixels"].item() == 3
    assert counts["output_valid_pixels"].item() == 3
    assert counts["positive_supervision_pixels"].item() == 2
    assert counts["positive_excluded_by_output_valid_pixels"].item() == 1

    config = {
        "lambda_depth": 1.0,
        "lambda_log": 0.0,
        "lambda_final": 0.0,
        "lambda_depth_bias": 0.0,
        "lambda_depth_exceedance": 0.0,
        "lambda_pu": 0.0,
        "lambda_unc": 0.0,
        "lambda_gradient": 0.0,
        "lambda_auxiliary": 0.0,
        "lambda_kan": 0.0,
        "lambda_wse": 0.0,
        "wse_start_epoch": 0,
        "wse_warmup_epochs": 1,
    }
    outputs = {
        "depth": batch["label"],
        "positive_depth": batch["label"],
        "conditional_depth": batch["label"],
        "uncertainty_scale": torch.ones_like(batch["label"]),
        "physical_features": {"z_hyd": torch.zeros_like(batch["label"])},
    }
    _, terms = CompositeFloodDepthLoss(config, 0.1)(outputs, batch, 0)
    assert terms["positive_pixels"].item() == 2
    assert terms["positive_excluded_by_output_valid_pixels"].item() == 1


def test_event_support_threshold_and_dem_output_support_are_explicit() -> None:
    available = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    fraction = torch.tensor([[[[1.0, 0.5, 1.0]]]])
    dem = torch.tensor([[[[1.0, 1.0, 1.0]]]])
    support, output = s1_output_valid_mask(available, fraction, dem, 1.0)
    assert torch.equal(support, torch.tensor([[[[True, False, False]]]]))
    assert torch.equal(output, support)
    support_half, _ = s1_output_valid_mask(available, fraction, dem, 0.5)
    assert torch.equal(support_half, torch.tensor([[[[True, True, False]]]]))
    _, no_dem_output = s1_output_valid_mask(
        available, fraction, torch.tensor([[[[1.0, 0.0, 1.0]]]]), 0.5
    )
    assert torch.equal(no_dem_output, torch.tensor([[[[True, False, False]]]]))


def test_masked_change_mixer_has_exact_single_branch_weights_and_zero_no_evidence() -> None:
    internal = torch.full((1, 2, 2, 2), 2.0)
    external = torch.full((1, 2, 2, 2), 3.0)
    logits = torch.zeros_like(internal)
    internal_valid = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    external_valid = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    mixed, internal_weight, external_weight = masked_two_way_change_mixer(
        internal, external, internal_valid, external_valid, logits
    )
    assert torch.all(internal_weight[:, :, 0, 0] == 1.0)
    assert torch.all(external_weight[:, :, 0, 0] == 0.0)
    assert torch.all(internal_weight[:, :, 0, 1] == 0.0)
    assert torch.all(external_weight[:, :, 0, 1] == 1.0)
    assert torch.allclose(mixed[:, :, 0, 0], torch.full((1, 2), 2.0))
    assert torch.allclose(mixed[:, :, 0, 1], torch.full((1, 2), 3.0))
    assert torch.allclose(mixed[:, :, 1, 0], torch.full((1, 2), 2.5))
    assert torch.all(mixed[:, :, 1, 1] == 0.0)
    assert torch.isfinite(mixed).all()


def test_prebounded_kan_does_not_apply_a_second_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = KANLinear(
        1,
        1,
        grid_size=4,
        spline_order=3,
        normalization="explicit_fixed_scaling",
        input_bounding="prebounded",
        base_path="none",
    )
    seen: dict[str, object] = {}
    original = layer.b_spline_basis

    def spy(values: torch.Tensor, *, clamp_inputs: bool = True) -> torch.Tensor:
        seen["values"] = values.detach().clone()
        seen["clamp_inputs"] = clamp_inputs
        return original(values, clamp_inputs=clamp_inputs)

    monkeypatch.setattr(layer, "b_spline_basis", spy)
    inputs = torch.tensor([[0.25]])
    layer.forward_with_contributions(inputs)
    assert seen["clamp_inputs"] is False
    torch.testing.assert_close(seen["values"], inputs)


def test_graph_stride_metadata_uses_true_node_spacing() -> None:
    graph4 = HydroEdgeKANS1(channels=8, heads=2, graph_feature_stride=4)
    graph8 = HydroEdgeKANS1(channels=8, heads=2, graph_feature_stride=8)
    assert graph4.graph_identity()["graph_node_spacing_m"] == 80.0
    assert graph8.graph_identity()["graph_node_spacing_m"] == 160.0
    assert graph4.graph_identity()["diagonal_neighbour_distance_m"] == pytest.approx(80.0 * 2**0.5)
    assert graph8.graph_identity()["diagonal_neighbour_distance_m"] == pytest.approx(160.0 * 2**0.5)


def test_v15_config_fields_are_consumed_and_mismatch_fails() -> None:
    config = load_config(PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15.xml")
    config["model"]["p0_corrections_enabled"] = True
    disabled = deepcopy(config)
    disabled["model"]["context_enabled"] = False
    disabled_model = build_model(disabled)
    assert isinstance(disabled_model.context, nn.Identity)

    wide = deepcopy(config)
    wide["model"]["decoder_widths"] = [128, 96, 64, 32]
    wide_model = build_model(wide)
    assert wide_model.decoder.widths == [128, 96, 64, 32]
    torch.testing.assert_close(
        wide_model.fusion.terrain_mix,
        torch.full_like(wide_model.fusion.terrain_mix, 0.30),
        atol=1e-6,
        rtol=0.0,
    )

    invalid = deepcopy(config)
    invalid["model"]["graph_feature_stride"] = 4
    with pytest.raises(ValueError, match="1/8 bottleneck"):
        build_model(invalid)


class _EvaluationDummy(nn.Module):
    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        label_shape = inputs["s1_t1"][:, :1]
        depth = label_shape.new_full(label_shape.shape, 1.0)
        return {
            "depth": depth,
            "conditional_depth": depth,
            "positive_depth": depth,
            "expected_depth": depth,
            "uncertainty_scale": torch.ones_like(depth),
            "physical_features": {
                "z_hyd": torch.zeros_like(depth),
                "local_relief": torch.zeros_like(depth),
            },
        }


def test_evaluation_defaults_to_the_canonical_mask() -> None:
    summary, _, _, _ = evaluate_loader(
        _EvaluationDummy(),
        [_batch()],
        torch.device("cpu"),
        [0.1, 0.5, 5.0],
        progress=False,
    )
    assert summary["evaluation_validity_mask"] == CANONICAL_POSITIVE_MASK
    assert summary["pixel_micro_pixels"] == 2
    assert summary["positive_supervision_pixels"] == 2
    assert summary["positive_excluded_by_output_valid_pixels"] == 1
