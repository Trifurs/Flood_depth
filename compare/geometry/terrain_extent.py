"""Patch-local terrain-geometry baselines conditioned on a supplied flood extent.

These methods accept one externally predicted inundation-support raster. Depth
values and label-derived masks are never passed into this module.

The implementations are clean-room, NumPy/SciPy adaptations of the documented
FwDET v2.1, FLEXTH method A, and RICorDE workflows.  They are not drop-in copies of
the ArcPy, OpenCV, QGIS, GRASS, or WhiteboxTools implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryPrediction:
    """Depth and water-surface products on the supplied inundation support."""

    depth: np.ndarray
    water_surface_elevation: np.ndarray
    support: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _PreparedTerrain:
    dsm: np.ndarray
    slope_degrees: np.ndarray
    extent: np.ndarray
    pixel_size_yx: tuple[float, float]
    imputed_pixel_count: int


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D or [1,H,W], got {array.shape}")
    return array


def _pixel_size_yx(pixel_size: float | tuple[float, float] | list[float]) -> tuple[float, float]:
    if np.isscalar(pixel_size):
        size = abs(float(pixel_size))
        result = (size, size)
    else:
        values = tuple(abs(float(value)) for value in pixel_size)
        if len(values) != 2:
            raise ValueError(f"pixel_size must have two entries, got {values}")
        # Raster metadata stores resolution as (x, y); array coordinates are (y, x).
        result = (values[1], values[0])
    if not all(np.isfinite(value) and value > 0.0 for value in result):
        raise ValueError(f"pixel_size must be finite and positive, got {result}")
    return result


def _nearest_valid_fill(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if np.all(valid):
        return array.astype(np.float64, copy=True)
    if not np.any(valid):
        raise ValueError("Terrain has no finite valid pixel for nearest-neighbour fill")
    nearest = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return array[tuple(nearest)].astype(np.float64, copy=False)


def _prepare_terrain(
    dsm: np.ndarray,
    slope_degrees: np.ndarray,
    extent: np.ndarray,
    dem_valid: np.ndarray,
    pixel_size: float | tuple[float, float] | list[float],
) -> _PreparedTerrain:
    dsm_2d = _as_2d("dsm", dsm).astype(np.float64, copy=False)
    slope_2d = _as_2d("slope_degrees", slope_degrees).astype(np.float64, copy=False)
    extent_2d = _as_2d("extent", extent).astype(bool, copy=False)
    valid_2d = _as_2d("dem_valid", dem_valid).astype(bool, copy=False)
    if not (dsm_2d.shape == slope_2d.shape == extent_2d.shape == valid_2d.shape):
        raise ValueError(
            "dsm, slope_degrees, extent, and dem_valid must share one grid: "
            f"{dsm_2d.shape}, {slope_2d.shape}, {extent_2d.shape}, {valid_2d.shape}"
        )
    terrain_valid = valid_2d & np.isfinite(dsm_2d) & np.isfinite(slope_2d)
    return _PreparedTerrain(
        dsm=_nearest_valid_fill(dsm_2d, terrain_valid),
        slope_degrees=_nearest_valid_fill(slope_2d, terrain_valid),
        extent=extent_2d.copy(),
        pixel_size_yx=_pixel_size_yx(pixel_size),
        imputed_pixel_count=int(np.count_nonzero(extent_2d & ~terrain_valid)),
    )


def _masked_mean(values: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"Mean-filter size must be a positive odd integer, got {size}")
    weights = mask.astype(np.float64)
    numerator = ndimage.uniform_filter(
        np.where(mask, values, 0.0), size=size, mode="constant", cval=0.0
    )
    denominator = ndimage.uniform_filter(weights, size=size, mode="constant", cval=0.0)
    result = np.asarray(values, dtype=np.float64).copy()
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def _inner_boundary(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    elif connectivity == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        raise ValueError(f"Unsupported connectivity: {connectivity}")
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask & ~eroded


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))


def _scaled_coordinates(mask: np.ndarray, pixel_size_yx: tuple[float, float]) -> np.ndarray:
    coordinates = np.argwhere(mask).astype(np.float64)
    coordinates[:, 0] *= pixel_size_yx[0]
    coordinates[:, 1] *= pixel_size_yx[1]
    return coordinates


def _idw(
    sample_mask: np.ndarray,
    sample_values: np.ndarray,
    query_mask: np.ndarray,
    pixel_size_yx: tuple[float, float],
    *,
    neighbours: int,
    power: float,
    chunk_size: int = 8192,
) -> np.ndarray:
    if neighbours <= 0:
        raise ValueError("IDW neighbours must be positive")
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("IDW power must be finite and positive")
    sample_values_flat = np.asarray(sample_values, dtype=np.float64)[sample_mask]
    if sample_values_flat.size == 0:
        raise ValueError("IDW requires at least one sample pixel")
    sample_coordinates = _scaled_coordinates(sample_mask, pixel_size_yx)
    query_coordinates = _scaled_coordinates(query_mask, pixel_size_yx)
    output = np.zeros(query_mask.shape, dtype=np.float64)
    if query_coordinates.size == 0:
        return output
    tree = cKDTree(sample_coordinates)
    k = min(int(neighbours), sample_values_flat.size)
    values = np.empty(query_coordinates.shape[0], dtype=np.float64)
    for start in range(0, query_coordinates.shape[0], chunk_size):
        stop = min(start + chunk_size, query_coordinates.shape[0])
        distances, indices = tree.query(query_coordinates[start:stop], k=k, workers=1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        exact = distances <= np.finfo(np.float64).eps
        safe_distances = np.maximum(distances, np.finfo(np.float64).eps)
        weights = np.power(safe_distances, -power)
        interpolated = np.sum(sample_values_flat[indices] * weights, axis=1) / np.sum(
            weights, axis=1
        )
        exact_rows = np.any(exact, axis=1)
        if np.any(exact_rows):
            first_exact = np.argmax(exact[exact_rows], axis=1)
            exact_indices = indices[exact_rows, first_exact]
            interpolated[exact_rows] = sample_values_flat[exact_indices]
        values[start:stop] = interpolated
    output[query_mask] = values
    return output


def _fill_small_internal_holes(mask: np.ndarray, maximum_area_pixels: int) -> np.ndarray:
    if maximum_area_pixels <= 0:
        return mask.copy()
    complement_labels, count = _component_labels(~mask)
    if count == 0:
        return mask.copy()
    border_labels = np.unique(
        np.concatenate(
            (
                complement_labels[0],
                complement_labels[-1],
                complement_labels[:, 0],
                complement_labels[:, -1],
            )
        )
    )
    areas = np.bincount(complement_labels.reshape(-1))
    fill_labels = [
        label
        for label in range(1, count + 1)
        if label not in border_labels and areas[label] <= maximum_area_pixels
    ]
    return mask | np.isin(complement_labels, fill_labels)


def _finalize_prediction(
    terrain: _PreparedTerrain,
    depth: np.ndarray,
    diagnostics: dict[str, Any],
) -> GeometryPrediction:
    clipped = np.where(terrain.extent, np.maximum(depth, 0.0), 0.0)
    if not np.all(np.isfinite(clipped)):
        raise RuntimeError("Geometry method produced non-finite depth")
    water_surface = terrain.dsm + clipped
    diagnostics = {
        **diagnostics,
        "input_extent_pixel_count": int(np.count_nonzero(terrain.extent)),
        "terrain_void_pixels_imputed_inside_extent": terrain.imputed_pixel_count,
    }
    return GeometryPrediction(
        depth=clipped.astype(np.float32),
        water_surface_elevation=water_surface.astype(np.float32),
        support=terrain.extent.copy(),
        diagnostics=diagnostics,
    )


def fwdet_v21_dsm_extent(
    dsm: np.ndarray,
    slope_degrees: np.ndarray,
    extent: np.ndarray,
    dem_valid: np.ndarray,
    pixel_size: float | tuple[float, float] | list[float],
    *,
    boundary_smoothing_iterations: int = 10,
    boundary_smoothing_window: int = 5,
    minimum_boundary_slope_percent: float = 0.5,
    depth_filter_window: int = 3,
) -> GeometryPrediction:
    """Adapt FwDET v2.1 boundary-WSE allocation to a DSM raster patch.

    The upstream default unit-cost allocation becomes nearest-boundary allocation.
    Its non-positive-elevation ocean filter is intentionally omitted because the
    local DSM vertical datum and coastal status are not encoded in the contract.
    """

    terrain = _prepare_terrain(dsm, slope_degrees, extent, dem_valid, pixel_size)
    if boundary_smoothing_iterations < 0:
        raise ValueError("boundary_smoothing_iterations cannot be negative")
    labels, component_count = _component_labels(terrain.extent)
    depth = np.zeros_like(terrain.dsm)
    fallback_components = 0
    retained_boundary_pixels = 0
    slope_percent = np.tan(np.deg2rad(np.clip(terrain.slope_degrees, 0.0, 89.9))) * 100.0
    for label in range(1, component_count + 1):
        component = labels == label
        boundary = _inner_boundary(component, connectivity=8)
        selected_boundary = boundary & (slope_percent > minimum_boundary_slope_percent)
        if not np.any(selected_boundary):
            selected_boundary = boundary
            fallback_components += 1
        retained_boundary_pixels += int(np.count_nonzero(selected_boundary))
        boundary_values = terrain.dsm.copy()
        for _ in range(int(boundary_smoothing_iterations)):
            smoothed = _masked_mean(
                boundary_values, selected_boundary, int(boundary_smoothing_window)
            )
            boundary_values[selected_boundary] = smoothed[selected_boundary]
        allocated_wse = _idw(
            selected_boundary,
            boundary_values,
            component,
            terrain.pixel_size_yx,
            neighbours=1,
            power=1.0,
        )
        raw_depth = np.maximum(allocated_wse - terrain.dsm, 0.0)
        positive = component & (raw_depth > 0.0)
        if np.any(positive):
            filtered = _masked_mean(raw_depth, positive, int(depth_filter_window))
            depth[component] = np.maximum(filtered[component], 0.0)
    return _finalize_prediction(
        terrain,
        depth,
        {
            "method": "fwdet_v21_dsm_extent",
            "adaptation": "unit_cost_nearest_boundary_on_dsm_patch",
            "component_count": component_count,
            "retained_boundary_pixel_count": retained_boundary_pixels,
            "slope_filter_fallback_component_count": fallback_components,
        },
    )


def flexth_method_a_dsm_extent(
    dsm: np.ndarray,
    slope_degrees: np.ndarray,
    extent: np.ndarray,
    dem_valid: np.ndarray,
    pixel_size: float | tuple[float, float] | list[float],
    *,
    slope_ratio_threshold: float = 0.05,
    gap_close_area_km2: float = 0.05,
    morphology_closing_iterations: int = 2,
    minimum_boundary_pixels: int = 100,
    inner_elevation_quantile: float = 0.98,
    idw_neighbours: int = 100,
    idw_power: float = 2.0,
    minimum_depth_m: float = 0.10,
) -> GeometryPrediction:
    """Adapt FLEXTH method A to DSM and retain only the supplied predicted support.

    FLEXTH's outward flood expansion is disabled because subset150 has isolated
    patches and no complete evaluation target outside the reliable positive labels.
    Morphological closing and small-hole filling remain part of extent preprocessing.
    """

    terrain = _prepare_terrain(dsm, slope_degrees, extent, dem_valid, pixel_size)
    if not 0.0 <= inner_elevation_quantile <= 1.0:
        raise ValueError("inner_elevation_quantile must be in [0, 1]")
    if minimum_depth_m < 0.0:
        raise ValueError("minimum_depth_m cannot be negative")
    cross = ndimage.generate_binary_structure(2, 1)
    processing_extent = ndimage.binary_closing(
        terrain.extent,
        structure=cross,
        iterations=int(morphology_closing_iterations),
        border_value=0,
    )
    pixel_area = terrain.pixel_size_yx[0] * terrain.pixel_size_yx[1]
    maximum_hole_pixels = int(max(0.0, gap_close_area_km2) * 1_000_000.0 / pixel_area)
    processing_extent = _fill_small_internal_holes(processing_extent, maximum_hole_pixels)
    labels, component_count = _component_labels(processing_extent)
    water_level = terrain.dsm.copy()
    idw_components = 0
    quantile_fallback_components = 0
    boundary_pixel_count = 0
    slope_ratio = np.tan(np.deg2rad(np.clip(terrain.slope_degrees, 0.0, 89.9)))
    for label in range(1, component_count + 1):
        component = labels == label
        boundary = _inner_boundary(component, connectivity=4)
        mild_boundary = boundary & (slope_ratio < float(slope_ratio_threshold))
        boundary_pixel_count += int(np.count_nonzero(mild_boundary))
        if np.count_nonzero(mild_boundary) < int(minimum_boundary_pixels):
            level = float(np.quantile(terrain.dsm[component], inner_elevation_quantile))
            water_level[component] = level
            quantile_fallback_components += 1
            continue
        smoothed_boundary_dsm = _masked_mean(terrain.dsm, mild_boundary, 3)
        interpolated = _idw(
            mild_boundary,
            smoothed_boundary_dsm,
            component,
            terrain.pixel_size_yx,
            neighbours=int(idw_neighbours),
            power=float(idw_power),
        )
        water_level[component] = interpolated[component]
        idw_components += 1
    raw_depth = np.maximum(water_level - terrain.dsm, 0.0)
    depth = np.where(terrain.extent, raw_depth + float(minimum_depth_m), 0.0)
    return _finalize_prediction(
        terrain,
        depth,
        {
            "method": "flexth_method_a_dsm_extent",
            "adaptation": "method_a_without_outward_expansion_or_optional_masks",
            "component_count_after_extent_closing": component_count,
            "idw_component_count": idw_components,
            "quantile_fallback_component_count": quantile_fallback_components,
            "usable_boundary_pixel_count": boundary_pixel_count,
            "processing_extent_pixel_count": int(np.count_nonzero(processing_extent)),
            "filled_gap_area_threshold_pixels": maximum_hole_pixels,
        },
    )


def _local_range(values: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    maximum = ndimage.maximum_filter(
        np.where(mask, values, -np.inf), size=size, mode="constant", cval=-np.inf
    )
    minimum = ndimage.minimum_filter(
        np.where(mask, values, np.inf), size=size, mode="constant", cval=np.inf
    )
    result = maximum - minimum
    return np.where(mask & np.isfinite(result), result, 0.0)


def ricorde_local_hand_dsm_extent(
    dsm: np.ndarray,
    slope_degrees: np.ndarray,
    extent: np.ndarray,
    dem_valid: np.ndarray,
    pixel_size: float | tuple[float, float] | list[float],
    *,
    pseudo_drainage_quantile: float = 0.05,
    lower_hand_quantile: float = 0.10,
    upper_hand_quantile: float = 0.90,
    lower_hand_floor_m: float = 0.50,
    upper_hand_cap_m: float = 7.0,
    idw_neighbours: int = 5,
    idw_power: float = 2.0,
    initial_smoothing_window: int = 7,
    maximum_hand_grade: float = 0.10,
    smoothing_resolution_factor: float = 3.0,
    maximum_smoothing_iterations: int = 5,
    hand_precision_m: float = 0.10,
) -> GeometryPrediction:
    """Patch-local RICorDE-style rolling-HAND adaptation.

    Full RICorDE needs a hydrologically conditioned DEM and permanent-water/drainage
    network.  Neither exists locally, so the lowest DSM quantile within each connected
    predicted extent is declared as a pseudo-drainage support. This substitution is
    intentionally exposed in the method name and diagnostics.
    """

    terrain = _prepare_terrain(dsm, slope_degrees, extent, dem_valid, pixel_size)
    for name, value in (
        ("pseudo_drainage_quantile", pseudo_drainage_quantile),
        ("lower_hand_quantile", lower_hand_quantile),
        ("upper_hand_quantile", upper_hand_quantile),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if hand_precision_m <= 0.0:
        raise ValueError("hand_precision_m must be positive")
    labels, component_count = _component_labels(terrain.extent)
    if component_count == 0:
        return _finalize_prediction(
            terrain,
            np.zeros_like(terrain.dsm),
            {
                "method": "ricorde_local_hand_dsm_extent",
                "adaptation": "pseudo_drainage_rolling_hand_on_dsm_patch",
                "component_count": 0,
            },
        )

    component_state: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_boundary_hand: list[np.ndarray] = []
    pseudo_drainage_pixels = 0
    for label in range(1, component_count + 1):
        component = labels == label
        elevations = terrain.dsm[component]
        drainage_threshold = float(np.quantile(elevations, pseudo_drainage_quantile))
        drainage = component & (terrain.dsm <= drainage_threshold)
        if not np.any(drainage):
            component_coordinates = np.argwhere(component)
            minimum_index = int(np.argmin(elevations))
            y, x = component_coordinates[minimum_index]
            drainage[y, x] = True
        pseudo_drainage_pixels += int(np.count_nonzero(drainage))
        nearest_drainage_elevation = _idw(
            drainage,
            terrain.dsm,
            component,
            terrain.pixel_size_yx,
            neighbours=1,
            power=1.0,
        )
        hand = np.zeros_like(terrain.dsm)
        hand[component] = np.maximum(
            terrain.dsm[component] - nearest_drainage_elevation[component], 0.0
        )
        boundary = _inner_boundary(component, connectivity=8)
        all_boundary_hand.append(hand[boundary])
        component_state.append((component, boundary, hand))

    boundary_hand = np.concatenate(all_boundary_hand)
    lower_bound = max(
        float(np.quantile(boundary_hand, lower_hand_quantile)), float(lower_hand_floor_m)
    )
    upper_bound = min(
        float(np.quantile(boundary_hand, upper_hand_quantile)), float(upper_hand_cap_m)
    )
    upper_bound = max(upper_bound, lower_bound)
    depth = np.zeros_like(terrain.dsm)
    range_threshold = min(
        float(maximum_hand_grade)
        * max(terrain.pixel_size_yx)
        * float(smoothing_resolution_factor),
        2.0,
    )
    total_smoothing_iterations = 0
    for component, boundary, hand in component_state:
        capped_hand = np.clip(hand, lower_bound, upper_bound)
        rolling_hand = _idw(
            boundary,
            capped_hand,
            component,
            terrain.pixel_size_yx,
            neighbours=int(idw_neighbours),
            power=float(idw_power),
        )
        rolling_hand = _masked_mean(
            rolling_hand, component, int(initial_smoothing_window)
        )
        for _ in range(int(maximum_smoothing_iterations)):
            local_range = _local_range(rolling_hand, component, 3)
            if float(np.max(local_range[component])) <= range_threshold:
                break
            failing = component & (local_range > range_threshold * 0.75)
            smoothed = _masked_mean(rolling_hand, component, 5)
            rolling_hand[failing] = smoothed[failing]
            total_smoothing_iterations += 1
        rolling_hand[component] = (
            np.round(rolling_hand[component] / hand_precision_m) * hand_precision_m
        )
        depth[component] = np.maximum(
            rolling_hand[component] - hand[component], 0.0
        )
    return _finalize_prediction(
        terrain,
        depth,
        {
            "method": "ricorde_local_hand_dsm_extent",
            "adaptation": "pseudo_drainage_rolling_hand_on_dsm_patch",
            "component_count": component_count,
            "pseudo_drainage_pixel_count": pseudo_drainage_pixels,
            "boundary_hand_lower_bound_m": lower_bound,
            "boundary_hand_upper_bound_m": upper_bound,
            "rolling_hand_range_threshold_m": range_threshold,
            "total_conditional_smoothing_iterations": total_smoothing_iterations,
        },
    )


GeometryMethod = Callable[..., GeometryPrediction]

GEOMETRY_METHODS: dict[str, GeometryMethod] = {
    "fwdet_v21_dsm_extent": fwdet_v21_dsm_extent,
    "flexth_method_a_dsm_extent": flexth_method_a_dsm_extent,
    "ricorde_local_hand_dsm_extent": ricorde_local_hand_dsm_extent,
}


def run_geometry_method(
    name: str,
    *,
    dsm: np.ndarray,
    slope_degrees: np.ndarray,
    extent: np.ndarray,
    dem_valid: np.ndarray,
    pixel_size: float | tuple[float, float] | list[float],
    parameters: Mapping[str, Any] | None = None,
) -> GeometryPrediction:
    """Run one registered extent-conditioned method with geometry-only inputs."""

    try:
        method = GEOMETRY_METHODS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown geometry method {name!r}; available={sorted(GEOMETRY_METHODS)}"
        ) from exc
    return method(
        dsm,
        slope_degrees,
        extent,
        dem_valid,
        pixel_size,
        **dict(parameters or {}),
    )
