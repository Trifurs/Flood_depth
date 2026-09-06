"""PA-HydroKAN-S1-V15.2: V15.1-simple with the audited V15.2 graph input map."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets.band_selection import BandSpec, resolve_band_spec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from models.hydro_edge_kan_v15_1 import DEFAULT_EDGE_FEATURE_NAMES
from models.hydro_edge_kan_v15_2 import HydroEdgeKANV15_2
from models.pa_hydrokan_s1_v15_1 import PAHydroKANS1V15_1


class PAHydroKANS1V15_2(PAHydroKANS1V15_1):
    """Keep the V15.1 encoder/decoder fixed while replacing only its graph map."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        band_spec: BandSpec,
        raw_terrain_names: tuple[str, ...],
        input_spec: ModelInputSpec,
    ) -> None:
        if str(model_config.get("variant", "simple")) not in {"kan", "simple"}:
            raise ValueError("PA-HydroKAN-S1-V15.2 requires the kan or simple graph variant")
        super().__init__(model_config, band_spec, raw_terrain_names, input_spec)
        graph_stride = int(model_config.get("graph_feature_stride", 4))
        self.graph = HydroEdgeKANV15_2(
            int(model_config.get("graph_channels", 64)),
            heads=int(model_config.get("graph_heads", 2)),
            grid_size=int(model_config.get("kan_grid_size", 4)),
            spline_order=int(model_config.get("kan_spline_order", 3)),
            graph_feature_stride=graph_stride,
            terrain_pixel_size_m=float(model_config.get("terrain_pixel_size_m", 20.0)),
            edge_feature_names=tuple(
                model_config.get("graph_edge_feature_names", DEFAULT_EDGE_FEATURE_NAMES)
            ),
            edge_stats_path=model_config.get("graph_edge_stats"),
            mapping_temperature=float(model_config.get("graph_mapping_temperature", 1.0)),
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
            gamma_init=float(model_config.get("kan_gamma_init_effective", 0.03)),
            gamma_max=float(model_config.get("kan_gamma_max", 0.25)),
            base_scale_init=float(model_config.get("kan_base_scale_init", 0.35)),
            spline_scale_init=float(model_config.get("kan_spline_scale_init", 1.0)),
            latent_scale_init=float(model_config.get("kan_latent_scale_init", 0.25)),
            symmetric_static_prior_enabled=bool(
                model_config.get("symmetric_static_prior_enabled", False)
            ),
            diagnostics_enabled=bool(
                model_config.get(
                    "diagnostics_enabled", model_config.get("diagnostic_mode", False)
                )
            ),
            regularization_enabled=bool(
                model_config.get("kan_regularization_enabled", True)
            ),
        )


def build_pa_hydrokan_s1_v15_2(config: Mapping[str, Any]) -> PAHydroKANS1V15_2:
    """Build the strict Sentinel-1-plus-terrain V15.2 graph-map candidate."""

    if "model" not in config:
        raise ValueError("PA-HydroKAN-S1-V15.2 builder requires the full resolved config")
    input_spec = ModelInputSpec.from_config(config)
    if not input_spec.is_s1_only:
        raise ValueError("PA-HydroKAN-S1-V15.2 requires dataset.input_mode='s1_terrain'")
    contract = DatasetContract.load(config["dataset"]["contract"])
    band_spec = resolve_band_spec(config, contract)
    raw_names = tuple(str(value) for value in contract.group("terrain")["band_descriptions"])
    model_config = dict(config["model"])
    if "kan_regularization_enabled" not in model_config:
        model_config["kan_regularization_enabled"] = float(
            config.get("loss", {}).get("lambda_kan", 0.0)
        ) != 0.0
    return PAHydroKANS1V15_2(model_config, band_spec, raw_names, input_spec)
