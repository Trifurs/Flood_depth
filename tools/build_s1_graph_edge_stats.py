#!/usr/bin/env python3
"""Build train-only robust statistics for V15.1/V15.2 topographic graph edges."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from models.hydro_edge_kan_v15_1 import (
    DEFAULT_EDGE_FEATURE_NAMES,
    HydroEdgeKANV15_1,
    edge_feature_schema,
)
from models.hydro_edge_kan_v15_2 import HydroEdgeKANV15_2
from models.terrain_features_v14 import _masked_mean, ground_like_proxy
from utils.config import load_config
from utils.misc import atomic_write_json, move_to_device
from utils.logging import write_rows


def _physical_from_raw(
    terrain_raw: torch.Tensor,
    dem_valid: torch.Tensor,
    terrain_names: tuple[str, ...],
    ground_proxy_kernel_size: int,
) -> dict[str, torch.Tensor]:
    elevation_index = terrain_names.index("elevation_m_DSM")
    valid = (dem_valid > 0.5).to(terrain_raw.dtype)
    elevation = terrain_raw[:, elevation_index : elevation_index + 1]
    local_mean = _masked_mean(elevation, valid, 9)
    ground = ground_like_proxy(elevation, valid, ground_proxy_kernel_size)
    second = _masked_mean(elevation.square(), valid, 9)
    relief = (second - local_mean.square()).clamp_min(0.0).sqrt() * valid
    return {
        "dem_valid": valid,
        "dsm_elevation": elevation,
        "z_ground_proxy": ground,
        "physics_elevation": ground,
        "local_relief": relief,
    }


def _feature_summary(values: np.ndarray, grid_size: int, temperature: float) -> tuple[dict[str, float | int | list[int]], list[dict[str, Any]]]:
    if values.size == 0:
        raise RuntimeError("No valid training graph edges were observed")
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    q01, q05, q25, q50, q75, q95, q99 = (float(value) for value in quantiles)
    iqr = max(q75 - q25, 1.0e-6)
    bounded = np.tanh((values - q50) / iqr / float(temperature))
    standardized_abs = np.abs((values - q50) / iqr)
    boundary_normalized = float(np.arctanh(0.98))
    knots = np.linspace(-1.0, 1.0, grid_size + 1)
    occupancy, _ = np.histogram(bounded, bins=knots)
    rows = [
        {
            "knot_interval": index,
            "lower": float(knots[index]),
            "upper": float(knots[index + 1]),
            "count": int(count),
            "fraction": float(count / max(values.size, 1)),
        }
        for index, count in enumerate(occupancy)
    ]
    return (
        {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p01": q01,
            "p05": q05,
            "p25": q25,
            "p50": q50,
            "p75": q75,
            "p95": q95,
            "p99": q99,
            "median": q50,
            "iqr": iqr,
            "recommended_center": q50,
            "recommended_scale": iqr,
            "mapping_temperature": float(temperature),
            "bounded_boundary_saturation_fraction": float(np.mean(np.abs(bounded) >= 0.98)),
            "temperature_for_train_saturation_05": float(
                np.quantile(standardized_abs, 0.95) / boundary_normalized
            ),
            "temperature_for_train_saturation_10": float(
                np.quantile(standardized_abs, 0.90) / boundary_normalized
            ),
            "knot_interval_occupancy": [int(value) for value in occupancy],
        },
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--occupancy-output", type=Path)
    parser.add_argument("--feature-stats-output", type=Path)
    parser.add_argument("--schema-output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("V15.1 graph edge statistics require dataset.input_mode='s1_terrain'")
    model_config = config["model"]
    names = tuple(model_config.get("graph_edge_feature_names", DEFAULT_EDGE_FEATURE_NAMES))
    stride = int(model_config.get("graph_feature_stride", 4))
    grid_size = int(model_config.get("kan_grid_size", 4))
    temperature = float(model_config.get("graph_mapping_temperature", 1.0))
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        "train",
        band_spec=band_spec,
        input_spec=input_spec,
        minimum_event_band_fraction=float(
            config["dataset"].get("minimum_event_band_fraction", 1.0)
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
        if args.device == "auto"
        else args.device
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    terrain_names = tuple(str(name) for name in contract.group("terrain")["band_descriptions"])
    is_v15_2 = str(model_config.get("name")) == "pa_hydrokan_s1_v15_2"
    if is_v15_2:
        extractor = HydroEdgeKANV15_2(
            channels=64,
            heads=2,
            grid_size=grid_size,
            spline_order=int(model_config.get("kan_spline_order", 3)),
            graph_feature_stride=stride,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            edge_feature_names=names,
            mapping_temperature=temperature,
            mapping_temperatures=model_config.get("graph_mapping_temperatures"),
            edge_minimum_dem_fraction=float(
                model_config.get("edge_minimum_dem_fraction", 0.5)
            ),
            edge_minimum_sar_fraction=float(
                model_config.get("edge_minimum_sar_fraction", 0.5)
            ),
            edge_minimum_barrier_valid_fraction=float(
                model_config.get("edge_minimum_barrier_valid_fraction", 0.5)
            ),
        ).to(device).eval()
    else:
        extractor = HydroEdgeKANV15_1(
            channels=64,
            heads=2,
            grid_size=grid_size,
            spline_order=int(model_config.get("kan_spline_order", 3)),
            graph_feature_stride=stride,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            edge_feature_names=names,
            mapping_temperature=temperature,
        ).to(device).eval()
    temperatures = {
        name: float(extractor.feature_temperatures[0, 0, index, 0, 0].cpu())
        if is_v15_2
        else temperature
        for index, name in enumerate(names)
    }
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    total_edges = 0
    feature_sum = np.zeros(len(names), dtype=np.float64)
    feature_cross = np.zeros((len(names), len(names)), dtype=np.float64)
    with torch.no_grad():
        for batch_index, cpu_batch in enumerate(tqdm(loader, desc="graph edge stats")):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = move_to_device(cpu_batch, device)
            physical = _physical_from_raw(
                batch["terrain_raw"],
                batch["validity"]["dem_valid"],
                terrain_names,
                int(model_config.get("ground_proxy_kernel_size", 9)),
            )
            size = tuple(
                max(1, int(dimension // stride))
                for dimension in batch["terrain_raw"].shape[-2:]
            )
            raw, _, valid_edge, _ = extractor._edge_descriptors(
                physical, batch["validity"]["s1_event_support"], size
            )
            selected = valid_edge.squeeze(2) > 0.5
            total_edges += int(selected.sum().item())
            matrix = raw.permute(0, 1, 3, 4, 2)[selected].detach().float().cpu().numpy()
            if matrix.size:
                matrix64 = matrix.astype(np.float64, copy=False)
                feature_sum += matrix64.sum(axis=0)
                feature_cross += matrix64.T @ matrix64
            for feature_index, name in enumerate(names):
                feature_values = raw[:, :, feature_index][selected]
                if feature_values.numel():
                    values[name].append(feature_values.detach().float().cpu().numpy())
    summaries: dict[str, dict[str, Any]] = {}
    occupancy_rows: list[dict[str, Any]] = []
    for name in names:
        feature_values = np.concatenate(values[name]) if values[name] else np.empty(0, dtype=np.float32)
        summary, rows = _feature_summary(feature_values, grid_size, temperatures[name])
        summaries[name] = summary
        for row in rows:
            occupancy_rows.append({"feature": name, **row})
    feature_mean = feature_sum / max(total_edges, 1)
    covariance = feature_cross / max(total_edges, 1) - np.outer(feature_mean, feature_mean)
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), 1.0e-12, None))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    feature_stats_rows = [
        {"feature": name, **summaries[name]}
        for name in names
    ]
    payload = {
        "scope": "train split only",
        "config": str(args.config.resolve()),
        "dataset_contract": str(config["dataset"]["contract"]),
        "graph_feature_stride": stride,
        "terrain_pixel_size_m": float(model_config.get("terrain_pixel_size_m", 20.0)),
        "graph_node_spacing_m": float(model_config.get("terrain_pixel_size_m", 20.0)) * stride,
        "grid_size": grid_size,
        "spline_order": int(model_config.get("kan_spline_order", 3)),
        "feature_schema": edge_feature_schema(names),
        "features": summaries,
        "valid_edge_observations": total_edges,
        "feature_correlation_matrix": {
            "feature_order": list(names),
            "values": correlation.tolist(),
        },
        "graph_identity": extractor.graph_identity(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    occupancy_output = args.occupancy_output or args.output.with_name("kan_feature_occupancy.csv")
    feature_stats_output = args.feature_stats_output or args.output.with_name("kan_feature_stats.csv")
    schema_output = args.schema_output or args.output.with_name("graph_edge_feature_schema.json")
    write_rows(occupancy_output, occupancy_rows)
    write_rows(feature_stats_output, feature_stats_rows)
    atomic_write_json(schema_output, edge_feature_schema(names))
    print({"output": str(args.output), "valid_edge_observations": total_edges})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
