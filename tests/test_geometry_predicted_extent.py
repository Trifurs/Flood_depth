from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from compare.geometry import GEOMETRY_METHODS, run_geometry_method
from datasets.contract import sha256_file
from datasets.flooddepth_dataset import FloodDepthDataset
from tools.evaluate_geometry import _load_extent_product
from utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = {
    "fwdet_v21_dsm_extent",
    "ricorde_local_hand_dsm_extent",
    "flexth_method_a_dsm_extent",
}


def synthetic_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[:40, :44]
    dsm = (10.0 + 0.035 * x + 0.020 * y + 0.15 * np.sin(x / 5.0)).astype(
        np.float32
    )
    slope = np.full_like(dsm, 2.0)
    extent = np.zeros_like(dsm, dtype=bool)
    extent[5:35, 6:38] = True
    extent[18:21, 20:23] = False
    dem_valid = np.ones_like(extent)
    return dsm, slope, extent, dem_valid


@pytest.mark.parametrize("method", sorted(EXPECTED_METHODS))
def test_geometry_methods_preserve_predicted_support_and_depth_semantics(method: str) -> None:
    dsm, slope, extent, dem_valid = synthetic_geometry()
    result = run_geometry_method(
        method,
        dsm=dsm,
        slope_degrees=slope,
        extent=extent,
        dem_valid=dem_valid,
        pixel_size=(20.0, 20.0),
    )
    assert result.depth.shape == dsm.shape
    assert result.depth.dtype == np.float32
    assert np.array_equal(result.support, extent)
    assert np.all(np.isfinite(result.depth))
    assert np.all(result.depth >= 0.0)
    assert np.all(result.depth[~extent] == 0.0)
    np.testing.assert_allclose(
        result.water_surface_elevation[extent],
        dsm[extent] + result.depth[extent],
        rtol=0.0,
        atol=2e-6,
    )


def test_flexth_minimum_depth_matches_declared_default() -> None:
    dsm = np.full((24, 24), 12.0, dtype=np.float32)
    slope = np.zeros_like(dsm)
    extent = np.zeros_like(dsm, dtype=bool)
    extent[4:20, 4:20] = True
    result = run_geometry_method(
        "flexth_method_a_dsm_extent",
        dsm=dsm,
        slope_degrees=slope,
        extent=extent,
        dem_valid=np.ones_like(extent),
        pixel_size=20.0,
    )
    np.testing.assert_allclose(result.depth[extent], 0.10, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("method", sorted(EXPECTED_METHODS))
def test_geometry_methods_fill_terrain_void_without_dropping_extent(method: str) -> None:
    dsm, slope, extent, dem_valid = synthetic_geometry()
    dem_valid[15:18, 15:18] = False
    dsm[~dem_valid] = 0.0
    slope[~dem_valid] = 0.0
    result = run_geometry_method(
        method,
        dsm=dsm,
        slope_degrees=slope,
        extent=extent,
        dem_valid=dem_valid,
        pixel_size=(20.0, 20.0),
    )
    assert np.array_equal(result.support, extent)
    assert result.diagnostics["terrain_void_pixels_imputed_inside_extent"] == 9
    assert np.all(np.isfinite(result.depth[extent]))


def test_geometry_api_has_no_target_depth_argument() -> None:
    for method in GEOMETRY_METHODS.values():
        parameters = inspect.signature(method).parameters
        assert "label" not in parameters
        assert "target" not in parameters
        assert "depth" not in parameters


def test_geometry_rejects_grid_mismatch() -> None:
    dsm, slope, extent, dem_valid = synthetic_geometry()
    with pytest.raises(ValueError, match="share one grid"):
        run_geometry_method(
            "fwdet_v21_dsm_extent",
            dsm=dsm,
            slope_degrees=slope[:-1],
            extent=extent,
            dem_valid=dem_valid,
            pixel_size=20.0,
        )


def test_geometry_config_requires_shared_predicted_extent() -> None:
    config = load_config(
        PROJECT_ROOT / "configs/compare/subset150_geometry_predicted_extent.xml"
    )
    assert (
        config["geometry"]["extent_source"]
        == "ai4g_mobilenet_v2_unet_iou_frozen_prediction"
    )
    assert set(config["geometry"]["methods"]) == EXPECTED_METHODS
    assert set(config["geometry"]["parameters"]) == EXPECTED_METHODS


def test_geometry_evaluator_rejects_label_derived_extent_manifest(tmp_path: Path) -> None:
    config = load_config(
        PROJECT_ROOT / "configs/compare/subset150_geometry_predicted_extent.xml"
    )
    dataset = FloodDepthDataset(
        config["dataset"]["contract"], config["dataset"]["train_stats"], "val"
    )
    product = {
        "dataset_fingerprint": {
            "contract_sha256": dataset.contract.hash,
            "manifest_sha256": sha256_file(dataset.contract.manifest_path),
            "normalization_sha256": sha256_file(Path(config["dataset"]["train_stats"])),
        },
        "prediction_uses_valid_depth_mask": True,
        "splits": {"val": {}},
    }
    (tmp_path / "extent_product.json").write_text(json.dumps(product), encoding="utf-8")
    with pytest.raises(RuntimeError, match="label-independent inference"):
        _load_extent_product(tmp_path, dataset, config, "val")
