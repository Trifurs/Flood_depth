"""PA-HydroKAN: partial-label asynchronous hydro-topographic KAN."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from models.asynchronous_fusion import AsynchronousFusionPyramid
from models.decoder import HydroDecoder
from models.encoders import ModalityTemporalEncoder
from models.heads import FloodDepthHeads, GlobalEventDepthScale
from models.terrain_features import TerrainFeaturePyramid
from models.terrain_graph_kan import TerrainGraphKAN


REQUIRED_INPUTS = {
    "s1_t1",
    "s1_t2",
    "s1_change",
    "s2_t1",
    "s2_t2",
    "s2_change",
    "terrain",
    "terrain_raw",
    "reliability",
    "s1_valid",
    "s2_valid",
    "dem_valid",
}
FORBIDDEN_INPUTS = {"label", "masks", "flood_mask", "valid_depth_mask", "split", "sample_origin"}


class PAHydroKAN(nn.Module):
    """Event-scale depth estimator; physics-guided, not an SWE/PINN solver."""

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        channels = [int(value) for value in model_config["channels"]]
        if len(channels) != 4:
            raise ValueError("PA-HydroKAN currently requires four encoder levels")
        dropout = float(model_config["dropout"])
        groups = int(model_config["group_norm_groups"])
        self.s1_encoder = ModalityTemporalEncoder(
            3, 4, channels, dropout, groups, incidence_film=True
        )
        self.s2_encoder = ModalityTemporalEncoder(
            6, 3, channels, dropout, groups, incidence_film=False
        )
        self.terrain = TerrainFeaturePyramid(
            channels,
            dropout,
            groups,
            model_config.get("terrain_context_kernel_sizes", (9,)),
        )
        self.fusion = AsynchronousFusionPyramid(channels, 12, dropout, groups)
        self.graph = TerrainGraphKAN(
            channels[-1],
            int(model_config["kan_grid_size"]),
            int(model_config["kan_spline_order"]),
            dropout,
            groups,
        )
        self.decoder = HydroDecoder(channels, dropout, groups)
        self.heads = FloodDepthHeads(
            channels[0],
            float(model_config["uncertainty_epsilon"]),
            float(model_config["uncertainty_maximum"]),
            str(model_config.get("depth_output_semantics", "probability_weighted_v1")),
        )
        self.event_depth_scale = (
            GlobalEventDepthScale(
                channels[-1],
                int(model_config.get("event_depth_scale_hidden_channels", 64)),
                float(
                    model_config.get(
                        "event_depth_scale_max_log_abs", 0.6931471805599453
                    )
                ),
            )
            if bool(model_config.get("event_depth_scale_enabled", False))
            else None
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        forbidden = FORBIDDEN_INPUTS.intersection(inputs)
        if forbidden:
            raise ValueError(f"Label-derived/provenance fields were passed to model.forward: {sorted(forbidden)}")
        missing = REQUIRED_INPUTS.difference(inputs)
        if missing:
            raise KeyError(f"Missing PA-HydroKAN inputs: {sorted(missing)}")
        s1_features = self.s1_encoder(inputs["s1_t1"], inputs["s1_t2"], inputs["s1_change"])
        s2_features = self.s2_encoder(inputs["s2_t1"], inputs["s2_t2"], inputs["s2_change"])
        terrain_features, physical = self.terrain(
            inputs["terrain"], inputs["terrain_raw"], inputs["dem_valid"]
        )
        fused, weights = self.fusion(
            s1_features,
            s2_features,
            terrain_features,
            inputs["reliability"],
            inputs["s1_valid"],
            inputs["s2_valid"],
        )
        sensor_valid = torch.maximum(inputs["s1_valid"], inputs["s2_valid"])
        bottleneck, graph_diagnostics = self.graph(
            fused[-1], physical, inputs["reliability"], weights[-1], sensor_valid
        )
        decoded = self.decoder(bottleneck, fused)
        outputs: dict[str, Any] = self.heads(decoded)
        if self.event_depth_scale is not None:
            output_valid = inputs["dem_valid"] * sensor_valid
            depth_scale, log_depth_scale = self.event_depth_scale(
                bottleneck, output_valid
            )
            conditional_depth = outputs["conditional_depth"] * depth_scale
            expected_depth = outputs["support_probability"] * conditional_depth
            outputs["conditional_depth"] = conditional_depth
            outputs["positive_depth"] = conditional_depth
            outputs["expected_depth"] = expected_depth
            outputs["depth"] = (
                conditional_depth
                if self.heads.depth_output_semantics == "conditional_positive_v2"
                else expected_depth
            )
            outputs["event_depth_scale"] = depth_scale
            outputs["event_log_depth_scale"] = log_depth_scale
        outputs["modality_weights"] = weights
        outputs["graph_diagnostics"] = graph_diagnostics
        outputs["physical_features"] = physical
        return outputs


def build_pa_hydrokan(config: Mapping[str, Any]) -> PAHydroKAN:
    model_config = config["model"] if "model" in config else config
    return PAHydroKAN(model_config)
