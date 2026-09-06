"""Unit coverage for the two explicitly local V15.2 weak-physics candidates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from datasets.model_input_spec import ModelInputSpec
from losses.composite_loss import CompositeFloodDepthLoss
from losses.physics_losses import weak_physics_pair_loss
from tools.train import _physics_output_gradient_diagnostics
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pair_inputs(width: int = 3) -> dict[str, torch.Tensor]:
    elevation = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width) * 0.1
    return {
        "physics_elevation": elevation,
        "dsm_elevation": elevation.clone(),
        "ground_proxy": elevation.clone(),
        "local_relief": torch.zeros_like(elevation),
        "positive": torch.ones_like(elevation),
        "dem_valid": torch.ones_like(elevation),
        "s1_valid": torch.ones_like(elevation),
        "s1_change": torch.zeros(1, 3, 1, width),
    }


def _physics(
    depth: torch.Tensor,
    values: dict[str, torch.Tensor],
    mode: str,
    **overrides: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    defaults: dict[str, float | str] = {
        "mode": mode,
        "elevation_tolerance_m": 0.05,
        "maximum_elevation_step_m": 0.75,
        "depth_tolerance_m": 0.02,
        "order_softplus_temperature_m": 0.05,
        "allowed_depth_jump_m": 0.25,
        "barrier_sigma_m": 0.25,
        "complexity_sigma_m": 4.0,
        "maximum_complexity_m": 100.0,
        "sar_compatibility_sigma": 0.5,
    }
    defaults.update(overrides)
    return weak_physics_pair_loss(
        depth,
        values["physics_elevation"],
        values["dsm_elevation"],
        values["ground_proxy"],
        values["local_relief"],
        values["positive"],
        values["dem_valid"],
        values["s1_valid"],
        values["s1_change"],
        **defaults,
    )


def test_terrain_order_margin_only_acts_on_eligible_uphill_pairs() -> None:
    values = _pair_inputs()
    flat = torch.ones_like(values["physics_elevation"])
    uphill = torch.tensor([[[[1.0, 1.2, 1.4]]]], requires_grad=True)
    flat_loss, _ = _physics(flat, values, "terrain_order_margin")
    uphill_loss, diagnostics = _physics(uphill, values, "terrain_order_margin")
    assert uphill_loss > flat_loss
    assert diagnostics["active_pair_count"] == 2
    assert diagnostics["active_pair_fraction"] == 1
    assert diagnostics["mean_violation_m"] > 0
    assert diagnostics["p90_violation_m"] > 0
    uphill_loss.backward()
    assert uphill.grad is not None
    assert torch.isfinite(uphill.grad).all() and uphill.grad.abs().sum() > 0


def test_terrain_order_ignores_an_invalid_endpoint_exactly() -> None:
    values = _pair_inputs()
    baseline_depth = torch.tensor([[[[1.0, 1.1, 1.2]]]])
    values["positive"][..., -1] = 0.0
    baseline, _ = _physics(baseline_depth, values, "terrain_order_margin")
    altered = baseline_depth.clone()
    altered[..., -1] = 1_000.0
    changed, diagnostics = _physics(altered, values, "terrain_order_margin")
    torch.testing.assert_close(changed, baseline, rtol=0.0, atol=0.0)
    assert diagnostics["active_pair_count"] == 1


def test_barrier_complexity_and_sar_incompatibility_downweight_large_jumps() -> None:
    values = _pair_inputs()
    depth = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    low_barrier, _ = _physics(depth, values, "barrier_consistency")

    barrier_values = _pair_inputs()
    barrier_values["dsm_elevation"][..., 0] += 2.0
    high_barrier, barrier_diag = _physics(depth, barrier_values, "barrier_consistency")
    assert high_barrier < low_barrier
    assert barrier_diag["mean_barrier_weight"] < 1.0

    complexity_values = _pair_inputs()
    complexity_values["local_relief"][..., 0] = 20.0
    high_complexity, complexity_diag = _physics(
        depth, complexity_values, "barrier_consistency"
    )
    assert high_complexity < low_barrier
    assert complexity_diag["mean_complexity_weight"] < 1.0

    sar_values = _pair_inputs()
    sar_values["s1_change"][..., 0] = 10.0
    incompatible, sar_diag = _physics(depth, sar_values, "barrier_consistency")
    assert incompatible < low_barrier
    assert sar_diag["mean_sar_compatibility"] < 1.0


def test_v15_3_wse_consistency_is_tolerant_and_low_barrier_gated() -> None:
    values = _pair_inputs()
    flat_wse_depth = torch.zeros_like(values["physics_elevation"])
    varying_wse_depth = torch.tensor([[[[0.0, 0.4, 0.8]]]])
    flat_loss, flat_diag = _physics(
        flat_wse_depth,
        values,
        "wse_consistency",
        wse_consistency_tolerance_m=0.05,
        wse_consistency_softplus_temperature_m=0.05,
        wse_consistency_maximum_barrier_m=1.0,
    )
    varying_loss, varying_diag = _physics(
        varying_wse_depth,
        values,
        "wse_consistency",
        wse_consistency_tolerance_m=0.05,
        wse_consistency_softplus_temperature_m=0.05,
        wse_consistency_maximum_barrier_m=1.0,
    )
    assert varying_loss > flat_loss
    assert varying_diag["active_pair_count"] == 2
    assert flat_diag["active_pair_count"] == 2

    blocked = _pair_inputs()
    blocked["dsm_elevation"][..., 1] += 2.0
    blocked_loss, blocked_diag = _physics(
        varying_wse_depth,
        blocked,
        "wse_consistency",
        wse_consistency_tolerance_m=0.05,
        wse_consistency_maximum_barrier_m=1.0,
    )
    assert blocked_diag["active_pair_count"] < varying_diag["active_pair_count"]
    assert blocked_loss <= varying_loss


def _composite_batch(depth: torch.Tensor) -> tuple[dict[str, object], dict[str, object]]:
    values = _pair_inputs(width=4)
    zeros = torch.zeros_like(depth)
    ones = torch.ones_like(depth)
    outputs: dict[str, object] = {
        "depth": depth,
        "positive_depth": depth,
        "conditional_depth": depth,
        "uncertainty_scale": ones,
        "physical_features": {
            "physics_elevation": values["physics_elevation"],
            "dsm_elevation": values["dsm_elevation"],
            "z_ground_proxy": values["ground_proxy"],
            "local_relief": values["local_relief"],
            "z_hyd": values["physics_elevation"],
        },
    }
    batch: dict[str, object] = {
        "label": depth.detach() + 0.05,
        "s1_change": values["s1_change"],
        "masks": {
            "valid_depth_mask": ones,
            "permanent_water_mask": zeros,
            "extreme_high_mask": zeros,
        },
        # Deliberately no S2 key: the new physics path must remain S1-only.
        "validity": {
            "output_valid": ones,
            "s1_valid": values["s1_valid"],
            "dem_valid": values["dem_valid"],
        },
        "reliability": torch.empty(1, 0, 1, 4),
    }
    return outputs, batch


def test_composite_physics_is_delayed_s1_only_and_has_nonzero_gradient() -> None:
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
        "lambda_phys": 0.0025,
        "phys_start_epoch": 15,
        "phys_warmup_epochs": 10,
        "physics_mode": "terrain_order_margin",
    }
    objective = CompositeFloodDepthLoss(config, 0.1)
    assert objective.physics_weight(14) == 0.0
    assert objective.physics_weight(15) == pytest.approx(0.00025)
    assert objective.physics_weight(24) == pytest.approx(0.0025)

    depth = torch.tensor([[[[1.0, 1.2, 1.4, 1.6]]]], requires_grad=True)
    outputs, batch = _composite_batch(depth)
    total, terms = objective(outputs, batch, 15)
    assert terms["physics_effective_weight"] == pytest.approx(0.00025)
    assert terms["physics_active_pair_count"] > 0
    assert terms["physics"].requires_grad
    total.backward()
    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum() > 0


def test_physics_output_gradient_diagnostics_are_finite_and_nonzero() -> None:
    depth = torch.tensor([[[[0.4, 0.9, 1.5]]]], requires_grad=True)
    physics = (depth[..., 1:] - depth[..., :-1]).square().mean()
    supervised = (depth - 0.5).square().mean()
    diagnostics = _physics_output_gradient_diagnostics(
        {"conditional_depth": depth},
        {
            "physics": physics,
            "physics_effective_weight": depth.new_tensor(0.0025),
            "depth": supervised,
        },
    )
    assert diagnostics["physics_gradient_measured"] == 1.0
    assert diagnostics["physics_gradient_norm"] > 0.0
    assert diagnostics["depth_gradient_norm"] > 0.0
    assert -1.0 <= diagnostics["physics_depth_gradient_cosine_similarity"] <= 1.0


def test_physics_candidate_configs_change_only_the_local_prior() -> None:
    baseline = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_simple_fixed.xml"
    )
    for filename, expected_mode, expected_changes in (
        (
            "subset1000_s1_v15_2_physics_order.xml",
            "terrain_order_margin",
            {"lambda_phys"},
        ),
        (
            "subset1000_s1_v15_2_physics_barrier.xml",
            "barrier_consistency",
            {"lambda_phys", "physics_mode"},
        ),
    ):
        candidate = load_config(PROJECT_ROOT / "configs/pa_hydrokan" / filename)
        assert candidate["seed"] == baseline["seed"]
        for section in ("training", "optimizer", "scheduler", "dataset", "supervision"):
            assert candidate[section] == baseline[section]
        assert candidate["model"] == baseline["model"]
        changed_loss_keys = {
            name
            for name in set(candidate["loss"]).union(baseline["loss"])
            if candidate["loss"].get(name) != baseline["loss"].get(name)
        }
        assert changed_loss_keys == expected_changes
        assert candidate["loss"]["lambda_phys"] == pytest.approx(0.0025)
        assert candidate["loss"]["physics_mode"] == expected_mode
        assert ModelInputSpec.from_config(candidate).is_s1_only


def test_combined_alias_and_raw_shortcut_configs_preserve_the_selected_physics() -> None:
    order = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_physics_order.xml"
    )
    combined = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_combined.xml"
    )
    shortcut = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_raw_sar_shortcut.xml"
    )
    for section in ("training", "optimizer", "scheduler", "dataset", "supervision", "loss"):
        assert combined[section] == order[section]
    assert combined["model"] == order["model"]
    assert ModelInputSpec.from_config(combined).is_s1_only
    assert combined["loss"]["lambda_phys"] > 0.0

    for section in ("training", "optimizer", "scheduler", "dataset", "supervision", "loss"):
        assert shortcut[section] == combined[section]
    changed_model_keys = {
        key
        for key in set(shortcut["model"]).union(combined["model"])
        if shortcut["model"].get(key) != combined["model"].get(key)
    }
    assert changed_model_keys == {"absolute_sar_shortcut_enabled"}
    assert shortcut["model"]["absolute_sar_shortcut_enabled"] is True
    assert ModelInputSpec.from_config(shortcut).is_s1_only


def test_augmentation_off_changes_only_sar_flip_probabilities() -> None:
    combined = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_combined.xml"
    )
    candidate = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_augmentation_off.xml"
    )
    for section in ("training", "optimizer", "scheduler", "supervision", "loss", "model"):
        assert candidate[section] == combined[section]
    expected_dataset = deepcopy(combined["dataset"])
    expected_dataset["augmentation"]["horizontal_flip_probability"] = 0.0
    expected_dataset["augmentation"]["vertical_flip_probability"] = 0.0
    assert candidate["dataset"] == expected_dataset
    assert ModelInputSpec.from_config(candidate).is_s1_only


def test_loss_beta025_changes_only_the_linear_depth_huber_transition() -> None:
    combined = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_combined.xml"
    )
    candidate = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_loss_beta025.xml"
    )
    for section in ("training", "optimizer", "scheduler", "dataset", "supervision", "model"):
        assert candidate[section] == combined[section]
    changed_loss_keys = {
        key
        for key in set(candidate["loss"]).union(combined["loss"])
        if candidate["loss"].get(key) != combined["loss"].get(key)
    }
    assert changed_loss_keys == {"depth_huber_beta_m"}
    assert candidate["loss"]["depth_huber_beta_m"] == pytest.approx(0.25)
    assert candidate["loss"]["lambda_phys"] == pytest.approx(0.0025)
    assert ModelInputSpec.from_config(candidate).is_s1_only


def test_primary_huber_beta_does_not_change_auxiliary_huber_beta() -> None:
    depth = torch.full((1, 1, 1, 4), 1.0)
    outputs, batch = _composite_batch(depth)
    auxiliary_prediction = torch.zeros((1, 1, 1, 1), requires_grad=True)
    outputs["auxiliary_depths"] = (auxiliary_prediction,)
    objective = CompositeFloodDepthLoss(
        {
            "lambda_depth": 0.0,
            "lambda_log": 0.0,
            "lambda_final": 0.0,
            "lambda_depth_bias": 0.0,
            "lambda_depth_exceedance": 0.0,
            "lambda_pu": 0.0,
            "lambda_unc": 0.0,
            "lambda_gradient": 0.0,
            "lambda_auxiliary": 1.0,
            "auxiliary_depth_weights": [1.0],
            "auxiliary_huber_beta_m": 0.50,
            "depth_huber_beta_m": 0.25,
            "lambda_kan": 0.0,
            "lambda_wse": 0.0,
            "wse_start_epoch": 0,
            "wse_warmup_epochs": 0,
            "lambda_phys": 0.0,
        },
        0.1,
    )
    total, terms = objective(outputs, batch, 0)
    expected = torch.nn.functional.smooth_l1_loss(
        auxiliary_prediction,
        torch.full_like(auxiliary_prediction, 1.05),
        beta=0.50,
    )
    torch.testing.assert_close(terms["auxiliary"], expected)
    torch.testing.assert_close(total, expected)


def test_loss_log010_changes_only_the_log_depth_weight() -> None:
    combined = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_combined.xml"
    )
    candidate = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_loss_log010.xml"
    )
    for section in ("training", "optimizer", "scheduler", "dataset", "supervision", "model"):
        assert candidate[section] == combined[section]
    changed_loss_keys = {
        key
        for key in set(candidate["loss"]).union(combined["loss"])
        if candidate["loss"].get(key) != combined["loss"].get(key)
    }
    assert changed_loss_keys == {"lambda_log"}
    assert candidate["loss"]["lambda_log"] == pytest.approx(0.10)
    assert candidate["loss"]["depth_huber_beta_m"] == pytest.approx(0.50)
    assert candidate["loss"]["auxiliary_huber_beta_m"] == pytest.approx(0.50)
    assert candidate["loss"]["lambda_phys"] == pytest.approx(0.0025)
    assert ModelInputSpec.from_config(candidate).is_s1_only


def test_tail_candidates_are_high_depth_only_and_change_only_alpha() -> None:
    combined = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_combined.xml"
    )
    for filename, alpha in (
        ("subset1000_s1_v15_2_tail_alpha010.xml", 0.10),
        ("subset1000_s1_v15_2_tail_alpha015.xml", 0.15),
    ):
        candidate = load_config(PROJECT_ROOT / "configs/pa_hydrokan" / filename)
        for section in ("training", "optimizer", "scheduler", "dataset", "supervision", "model"):
            assert candidate[section] == combined[section]
        changed_loss_keys = {
            key
            for key in set(candidate["loss"]).union(combined["loss"])
            if candidate["loss"].get(key) != combined["loss"].get(key)
        }
        assert changed_loss_keys == {"tail_underprediction_alpha"}
        assert candidate["loss"]["tail_underprediction_alpha"] == pytest.approx(alpha)
        assert candidate["loss"]["depth_underprediction_min_m"] == pytest.approx(0.50)
        assert candidate["loss"]["lambda_depth_bias"] == 0.0
        assert candidate["loss"]["lambda_depth_exceedance"] == 0.0
        assert candidate["loss"]["lambda_tail"] == 0.0
        assert candidate["loss"]["lambda_phys"] == pytest.approx(0.0025)
        assert ModelInputSpec.from_config(candidate).is_s1_only


def test_final_candidate_and_matched_v15_use_a_strictly_shared_protocol() -> None:
    """Only the selected model/prior may differ in the final seed comparison."""
    candidate = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_final.xml"
    )
    baseline = load_config(
        PROJECT_ROOT / "configs/pa_hydrokan/subset1000_s1_v15_2_final_matched_v15.xml"
    )
    for section in ("training", "optimizer", "scheduler", "dataset", "supervision"):
        assert candidate[section] == baseline[section]
    assert candidate["training"]["epochs"] == 100
    assert candidate["training"]["minimum_epochs"] == 35
    assert candidate["training"]["early_stop_patience"] == 20
    assert candidate["training"]["batch_size"] == 12
    assert candidate["training"]["gradient_accumulation_steps"] == 1
    assert candidate["training"]["amp_dtype"] == "bfloat16"
    assert candidate["training"]["best_weights"] == "raw"
    assert candidate["loss"]["schedule_total_epochs"] == 100
    assert baseline["loss"]["schedule_total_epochs"] == 100
    assert candidate["loss"]["lambda_phys"] == pytest.approx(0.0025)
    assert baseline["loss"]["lambda_phys"] == 0.0
    assert ModelInputSpec.from_config(candidate).is_s1_only
    assert ModelInputSpec.from_config(baseline).is_s1_only
