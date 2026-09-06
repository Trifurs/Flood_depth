"""Scientific, S1-only terrain-conditioned spatial compatibility graph.

This module is deliberately not a hydraulic solver.  Its KAN receives only four
low-dimensional topographic edge descriptors; SAR latent similarity and
observation completeness are separate factors in the final graph gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.kan_layers import KANLinear
from models.terrain_features_v14 import path_barrier_proxy
from models.terrain_graph_kan import DIRECTIONS, _masked_pool, _roll_with_boundary_mask


DEFAULT_EDGE_FEATURE_NAMES = (
    "signed_grade",
    "absolute_grade",
    "barrier_magnitude",
    "local_surface_complexity",
)


COUNTERFACTUAL_MODES = (
    "full",
    "graph_off",
    "spline_off",
    "base_off",
    "constant_gate",
    "shuffled_terrain",
)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(float(value), 1.0e-8)))


def edge_feature_schema(
    feature_names: Sequence[str] = DEFAULT_EDGE_FEATURE_NAMES,
) -> dict[str, Any]:
    """Return a machine-readable definition of the low-dimensional KAN input."""

    definitions = {
        "signed_grade": {
            "formula": "(z_neighbor - z_center) / physical_distance_m",
            "units": "m_per_m",
            "role": "learned directional terrain relation; no fixed sign prior",
        },
        "absolute_grade": {
            "formula": "abs(z_neighbor - z_center) / physical_distance_m",
            "units": "m_per_m",
            "role": "symmetric terrain steepness",
        },
        "barrier_magnitude": {
            "formula": "max(DSM path) - max(ground_proxy_center, ground_proxy_neighbor), clipped at zero",
            "units": "m",
            "role": "DSM-derived path obstruction proxy",
        },
        "local_surface_complexity": {
            "formula": "mean(local_relief_center, local_relief_neighbor)",
            "units": "m",
            "role": "symmetric local DSM relief proxy",
        },
    }
    names = tuple(str(name) for name in feature_names)
    unknown = set(names).difference(definitions)
    if unknown:
        raise ValueError(f"Unknown HydroEdgeKAN V15.1 feature(s): {sorted(unknown)}")
    return {
        "name": "terrain_conditioned_spatial_compatibility",
        "feature_order": list(names),
        "features": {name: definitions[name] for name in names},
        "notes": [
            "KAN receives no high-dimensional SAR latent feature, availability, date, uncertainty, or reliability embedding.",
            "The graph is not interpreted as water flow, routing, or mass transport.",
        ],
    }


def _load_edge_stats(
    path: str | Path | None,
    feature_names: Sequence[str],
) -> tuple[list[float], list[float], str | None]:
    names = tuple(str(name) for name in feature_names)
    if path is None:
        return [0.0] * len(names), [1.0] * len(names), None
    source = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    features = payload.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("edge stats must contain a features mapping")
    centers, scales = [], []
    for name in names:
        item = features.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"edge stats are missing feature {name!r}")
        center = float(item.get("recommended_center", item.get("median")))
        scale = float(item.get("recommended_scale", item.get("iqr")))
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"edge stats for {name!r} have invalid center/scale")
        centers.append(center)
        scales.append(max(scale, 1.0e-6))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return centers, scales, digest


class HydroEdgeKANV15_1(nn.Module):
    """Eight-neighbour graph with one multi-head topographic KAN evaluation."""

    def __init__(
        self,
        channels: int,
        *,
        heads: int = 2,
        grid_size: int = 4,
        spline_order: int = 3,
        graph_feature_stride: int = 4,
        terrain_pixel_size_m: float = 20.0,
        edge_feature_names: Sequence[str] = DEFAULT_EDGE_FEATURE_NAMES,
        edge_stats_path: str | Path | None = None,
        mapping_temperature: float = 1.0,
        gamma_init: float = 0.03,
        gamma_max: float = 0.25,
        base_scale_init: float = 0.35,
        spline_scale_init: float = 1.0,
        latent_scale_init: float = 0.25,
        symmetric_static_prior_enabled: bool = False,
        diagnostics_enabled: bool = False,
        regularization_enabled: bool = False,
        message_mode: str = "linear",
        message_scale: float = 1.0,
        extreme_preservation_enabled: bool = False,
        extreme_preservation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads:
            raise ValueError("channels must be positive and divisible by graph heads")
        if graph_feature_stride not in {4, 8}:
            raise ValueError("graph_feature_stride must be 4 or 8")
        if terrain_pixel_size_m <= 0 or mapping_temperature <= 0:
            raise ValueError("terrain pixel size and mapping temperature must be positive")
        if not 0 < gamma_init < gamma_max:
            raise ValueError("gamma_init must lie strictly between zero and gamma_max")
        if base_scale_init <= 0 or spline_scale_init <= 0 or latent_scale_init < 0:
            raise ValueError("KAN scales must be positive and latent scale nonnegative")
        if message_mode not in {"linear", "tanh", "rational"}:
            raise ValueError("message_mode must be linear, tanh, or rational")
        if message_scale <= 0 or extreme_preservation_scale <= 0:
            raise ValueError("message and extreme-preservation scales must be positive")
        names = tuple(str(name) for name in edge_feature_names)
        if len(names) != len(set(names)) or not names:
            raise ValueError("edge_feature_names must be a non-empty unique sequence")
        edge_feature_schema(names)
        centers, scales, digest = _load_edge_stats(edge_stats_path, names)

        self.channels = int(channels)
        self.heads = int(heads)
        self.head_channels = self.channels // self.heads
        self.graph_feature_stride = int(graph_feature_stride)
        self.graph_node_spacing_m = float(terrain_pixel_size_m) * self.graph_feature_stride
        self.gamma_max = float(gamma_max)
        self.mapping_temperature = float(mapping_temperature)
        self.edge_feature_names = names
        self.edge_stats_path = str(Path(edge_stats_path).resolve()) if edge_stats_path is not None else None
        self.edge_stats_sha256 = digest
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.regularization_enabled = bool(regularization_enabled)
        self.message_mode = str(message_mode)
        self.message_scale = float(message_scale)
        self.extreme_preservation_enabled = bool(extreme_preservation_enabled)
        self.extreme_preservation_scale = float(extreme_preservation_scale)
        self.symmetric_static_prior_enabled = bool(symmetric_static_prior_enabled)
        # These switches are deliberately plain Python state rather than model
        # parameters/buffers: they are evaluation-only causal interventions and
        # must not alter a checkpoint's learned state or default forward path.
        self.counterfactual_mode = "full"
        self.counterfactual_constant_gate = 1.0
        self.counterfactual_shuffle_seed = 20260904
        self._symmetric_feature_indices = tuple(
            names.index(name)
            for name in (
                "absolute_grade",
                "barrier_magnitude",
                "local_surface_complexity",
            )
            if name in names
        )
        self.register_buffer(
            "feature_centers",
            torch.tensor(centers, dtype=torch.float32).view(1, 1, len(names), 1, 1),
        )
        self.register_buffer(
            "feature_scales",
            torch.tensor(scales, dtype=torch.float32).view(1, 1, len(names), 1, 1),
        )

        # One KAN evaluation yields all heads.  Each output head retains separate
        # feature-wise base and spline coefficients inside KANLinear.
        self.edge_kan = KANLinear(
            len(names),
            self.heads,
            grid_size=grid_size,
            spline_order=spline_order,
            normalization="explicit_fixed_scaling",
            input_bounding="prebounded",
            base_path="linear",
            base_scale_init=base_scale_init,
            spline_scale_init=spline_scale_init,
            learnable_base_scale=True,
            learnable_spline_scale=True,
        )
        self.latent_projection = nn.ModuleList(
            [nn.Conv2d(self.channels, self.head_channels, 1, bias=False) for _ in range(self.heads)]
        )
        self.latent_compatibility = nn.ModuleList(
            [nn.Conv2d(3, 1, 1) for _ in range(self.heads)]
        )
        for projection in self.latent_compatibility:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.message_projection = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        self.output_projection = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        nn.init.orthogonal_(self.output_projection.weight)
        self.raw_gamma = nn.Parameter(
            torch.full(
                (self.heads,),
                _inverse_sigmoid(float(gamma_init) / float(gamma_max)),
            )
        )
        self.raw_latent_scale = nn.Parameter(
            torch.full((self.heads,), _inverse_softplus(latent_scale_init))
        )
        if self.symmetric_static_prior_enabled:
            # Only symmetric variables (absolute grade, barrier, complexity) are
            # eligible for this optional weak suppression.
            if not self._symmetric_feature_indices:
                raise ValueError(
                    "symmetric_static_prior_enabled requires at least one symmetric "
                    "edge feature"
                )
            self.raw_symmetric_prior = nn.Parameter(
                torch.full(
                    (len(self._symmetric_feature_indices),), _inverse_softplus(0.02)
                )
            )
        else:
            self.register_parameter("raw_symmetric_prior", None)

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.sigmoid(self.raw_gamma)

    @property
    def latent_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_latent_scale)

    @property
    def neighbour_distances_m(self) -> tuple[float, ...]:
        return tuple(
            self.graph_node_spacing_m * math.sqrt(dx * dx + dy * dy)
            for dy, dx in DIRECTIONS
        )

    def graph_identity(self, feature_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        return {
            "graph_feature_stride": self.graph_feature_stride,
            "graph_node_spacing_m": self.graph_node_spacing_m,
            "graph_feature_shape": list(feature_shape) if feature_shape else None,
            "orthogonal_neighbour_distance_m": self.graph_node_spacing_m,
            "diagonal_neighbour_distance_m": self.graph_node_spacing_m * math.sqrt(2.0),
            "edge_stats_sha256": self.edge_stats_sha256,
            "message_mode": self.message_mode,
            "message_scale": self.message_scale,
            "extreme_preservation_enabled": self.extreme_preservation_enabled,
            "extreme_preservation_scale": self.extreme_preservation_scale,
        }

    def set_counterfactual_mode(
        self,
        mode: str = "full",
        *,
        constant_gate: float = 1.0,
        shuffle_seed: int = 20260904,
    ) -> None:
        """Set a deterministic evaluation-only Graph/KAN intervention.

        ``graph_off`` is an exact identity intervention (equivalent to zeroing
        the graph residual scale); the remaining modes preserve graph placement
        and only change the named causal contribution.
        """

        mode = str(mode)
        if mode not in COUNTERFACTUAL_MODES:
            raise ValueError(
                f"Unknown HydroEdgeKAN counterfactual {mode!r}; "
                f"expected one of {COUNTERFACTUAL_MODES}"
            )
        if not math.isfinite(float(constant_gate)) or not 0.0 <= float(constant_gate) <= 1.0:
            raise ValueError("constant_gate must be finite and lie in [0, 1]")
        self.counterfactual_mode = mode
        self.counterfactual_constant_gate = float(constant_gate)
        self.counterfactual_shuffle_seed = int(shuffle_seed)

    def _shuffle_terrain_descriptors(
        self, raw: torch.Tensor, bounded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministically break terrain-to-edge alignment within each sample.

        Latent SAR features, validity and observation-amplitude terms are left
        untouched.  A separate permutation is used for every sample/direction.
        """

        batch, directions, features, height, width = raw.shape
        shuffled_raw = torch.empty_like(raw)
        shuffled_bounded = torch.empty_like(bounded)
        for sample_index in range(batch):
            for direction_index in range(directions):
                generator = torch.Generator(device=raw.device)
                generator.manual_seed(
                    self.counterfactual_shuffle_seed
                    + sample_index * 10_007
                    + direction_index * 101
                )
                permutation = torch.randperm(
                    height * width, device=raw.device, generator=generator
                )
                shuffled_raw[sample_index, direction_index] = raw[
                    sample_index, direction_index
                ].reshape(features, -1).index_select(-1, permutation).reshape(
                    features, height, width
                )
                shuffled_bounded[sample_index, direction_index] = bounded[
                    sample_index, direction_index
                ].reshape(features, -1).index_select(-1, permutation).reshape(
                    features, height, width
                )
        return shuffled_raw, shuffled_bounded

    @staticmethod
    def _identity_diagnostics(features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scalar-compatible diagnostics for the exact graph identity mode."""

        zero = features.sum() * 0.0
        return {
            "topographic_kan_logit_mean": zero,
            "topographic_affinity_mean": zero,
            "observation_amplitude_mean": zero,
            "latent_compatibility_mean": zero,
            "latent_compatibility_logit_mean": zero,
            "final_graph_gate_mean": zero,
            "gate_mean": zero,
            "gate_std": zero,
            "valid_edge_fraction": zero,
            "graph_gamma_mean": zero,
            "gamma_mean": zero,
            "graph_input_rms": features.float().square().mean().sqrt(),
            "graph_update_rms": zero,
            "graph_update_rms_ratio": zero,
            "spline_output_rms": zero,
            "base_output_rms": zero,
            "spline_base_rms_ratio": zero,
            "knot_boundary_saturation_fraction": zero,
            "kan_coefficient_magnitude": zero,
            "kan_coefficient_smoothness": zero,
            "kan_monotonicity": zero,
            "kan_curve_smoothness": zero,
            "static_topographic_affinity_mean": zero,
            "observation_confidence_mean": zero,
            "valid_edge_gate_mean": zero,
        }

    @staticmethod
    def _stack_roll(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rolled, boundaries = zip(
            *[_roll_with_boundary_mask(value, dy, dx) for dy, dx in DIRECTIONS]
        )
        return torch.stack(rolled, dim=1), torch.stack(boundaries, dim=1)

    def _edge_descriptors(
        self,
        physical: Mapping[str, torch.Tensor],
        sensor_valid: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dem_full = physical["dem_valid"]
        dem_fraction = F.adaptive_avg_pool2d(dem_full, size)
        sensor_fraction = F.adaptive_avg_pool2d(sensor_valid, size)
        node_valid = (
            (dem_fraction > 0.5).to(sensor_fraction.dtype)
            * (sensor_fraction > 0.0).to(sensor_fraction.dtype)
        )
        elevation = _masked_pool(physical["physics_elevation"], dem_full, size)
        complexity = _masked_pool(physical["local_relief"], dem_full, size)
        neighbour_elevation, boundary = self._stack_roll(elevation)
        neighbour_complexity, _ = self._stack_roll(complexity)
        distance = elevation.new_tensor(self.neighbour_distances_m).view(
            1, len(DIRECTIONS), 1, 1, 1
        )
        signed_grade = (neighbour_elevation - elevation.unsqueeze(1)) / distance.clamp_min(1.0e-6)
        absolute_grade = signed_grade.abs()
        barrier_full, barrier_valid = path_barrier_proxy(
            physical["dsm_elevation"],
            dem_full,
            self.graph_feature_stride,
            physical["z_ground_proxy"],
            statistic="max",
        )
        barrier = torch.stack(
            [F.adaptive_avg_pool2d(barrier_full[:, direction], size) for direction in range(len(DIRECTIONS))],
            dim=1,
        )
        barrier_valid = torch.stack(
            [F.adaptive_avg_pool2d(barrier_valid[:, direction], size) for direction in range(len(DIRECTIONS))],
            dim=1,
        )
        local_complexity = 0.5 * (complexity.unsqueeze(1) + neighbour_complexity)
        raw_by_name = {
            "signed_grade": signed_grade,
            "absolute_grade": absolute_grade,
            "barrier_magnitude": barrier,
            "local_surface_complexity": local_complexity,
        }
        raw = torch.cat([raw_by_name[name] for name in self.edge_feature_names], dim=2)
        neighbour_node, _ = self._stack_roll(node_valid)
        valid_edge = node_valid.unsqueeze(1) * neighbour_node * boundary * barrier_valid
        neighbour_sensor, _ = self._stack_roll(sensor_fraction)
        neighbour_dem, _ = self._stack_roll(dem_fraction)
        sar_pair_fraction = torch.sqrt(
            (sensor_fraction.unsqueeze(1).clamp(0.0, 1.0) * neighbour_sensor.clamp(0.0, 1.0)).clamp_min(0.0)
        )
        dem_pair_fraction = torch.sqrt(
            (dem_fraction.unsqueeze(1).clamp(0.0, 1.0) * neighbour_dem.clamp(0.0, 1.0)).clamp_min(0.0)
        )
        observation_amplitude = sar_pair_fraction * dem_pair_fraction
        robust = (raw - self.feature_centers.to(raw.dtype)) / self.feature_scales.to(raw.dtype)
        bounded = torch.tanh(robust / self.mapping_temperature)
        return raw, bounded, valid_edge, observation_amplitude

    def spline_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.edge_kan.spline_coefficients
        magnitude = (coefficients.square().mean() + 1.0e-12).sqrt()
        if coefficients.shape[-1] < 3:
            return magnitude, coefficients.sum() * 0.0
        curvature = (
            coefficients[..., 2:]
            - 2.0 * coefficients[..., 1:-1]
            + coefficients[..., :-2]
        ).square().mean()
        return magnitude, curvature

    def _latent_compatibility(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, compatibility = [], []
        for head, projection in enumerate(self.latent_projection):
            latent = projection(features)
            neighbour, _ = self._stack_roll(latent)
            center = latent.unsqueeze(1)
            signed_difference = (neighbour - center).mean(dim=2, keepdim=True)
            absolute_difference = (neighbour - center).abs().mean(dim=2, keepdim=True)
            cosine = F.cosine_similarity(neighbour, center.expand_as(neighbour), dim=2).unsqueeze(2)
            descriptor = torch.cat((signed_difference, absolute_difference, cosine), dim=2)
            value = self.latent_compatibility[head](
                descriptor.reshape(-1, 3, *features.shape[-2:])
            ).reshape(features.shape[0], len(DIRECTIONS), 1, *features.shape[-2:])
            logits.append(value)
            compatibility.append(torch.sigmoid(value))
        return torch.cat(logits, dim=2), torch.cat(compatibility, dim=2)

    def forward(
        self,
        features: torch.Tensor,
        physical: Mapping[str, torch.Tensor],
        sensor_valid: torch.Tensor,
        *,
        feature_stride: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if int(feature_stride) != self.graph_feature_stride:
            raise ValueError(
                "HydroEdgeKANV15_1 feature-stride mismatch: configured "
                f"{self.graph_feature_stride}, received {feature_stride}"
            )
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("features must be BCHW with the configured graph channel count")
        if self.counterfactual_mode == "graph_off":
            return features, self._identity_diagnostics(features)
        size = features.shape[-2:]
        raw, bounded, valid_edge, observation_amplitude = self._edge_descriptors(
            physical, sensor_valid, size
        )
        if self.counterfactual_mode == "shuffled_terrain":
            raw, bounded = self._shuffle_terrain_descriptors(raw, bounded)
        flat = bounded.permute(0, 1, 3, 4, 2).reshape(-1, len(self.edge_feature_names))
        total, base, spline = self.edge_kan.forward_with_contributions(flat)
        kan_logit = total.reshape(features.shape[0], len(DIRECTIONS), *size, self.heads).permute(0, 1, 4, 2, 3)
        base = base.reshape(features.shape[0], len(DIRECTIONS), *size, self.heads).permute(0, 1, 4, 2, 3)
        spline = spline.reshape(features.shape[0], len(DIRECTIONS), *size, self.heads).permute(0, 1, 4, 2, 3)
        if self.counterfactual_mode == "spline_off":
            kan_logit = base
        elif self.counterfactual_mode == "base_off":
            kan_logit = spline
        latent_logit, latent_compatibility = self._latent_compatibility(features)
        static_logit = torch.zeros_like(kan_logit)
        if self.raw_symmetric_prior is not None:
            symmetric = raw[:, :, self._symmetric_feature_indices].abs()
            scales = F.softplus(self.raw_symmetric_prior).view(
                1, 1, 1, 1, 1, len(self._symmetric_feature_indices)
            )
            # The optional prior is symmetric by construction and never touches
            # signed grade.  It is disabled by default.
            static_logit = -(
                symmetric.permute(0, 1, 3, 4, 2).unsqueeze(2) * scales
            ).sum(dim=-1)
        compatibility_logit = (
            kan_logit
            + self.latent_scale.view(1, 1, self.heads, 1, 1) * latent_logit
            + static_logit
        )
        compatibility = torch.sigmoid(compatibility_logit)
        if self.counterfactual_mode == "constant_gate":
            final_gate = valid_edge.expand(-1, -1, self.heads, -1, -1) * (
                features.new_tensor(self.counterfactual_constant_gate)
            )
        else:
            final_gate = (
                valid_edge.expand(-1, -1, self.heads, -1, -1)
                * observation_amplitude.expand(-1, -1, self.heads, -1, -1)
                * compatibility
            )

        projected = self.message_projection(features).reshape(
            features.shape[0], self.heads, self.head_channels, *size
        )
        neighbour_projected, _ = self._stack_roll(projected)
        message = neighbour_projected - projected.unsqueeze(1)
        raw_message_rms = message.float().square().mean().sqrt()
        if self.message_mode == "tanh":
            message = self.message_scale * torch.tanh(message / self.message_scale)
        elif self.message_mode == "rational":
            message = message / (1.0 + message.abs() / self.message_scale)
        if self.extreme_preservation_enabled:
            # High local contrast is a signal to avoid blindly averaging the
            # neighbourhood.  This gate acts on the message confidence, not on
            # the SAR identity feature, so it cannot erase an isolated extreme.
            contrast = message.detach().abs().mean(dim=3)
            contrast_gate = 1.0 / (
                1.0 + contrast / self.extreme_preservation_scale
            )
            final_gate = final_gate * contrast_gate
        else:
            contrast_gate = torch.ones_like(final_gate[:, :, :1])
        denominator = final_gate.sum(dim=1).clamp_min(1.0e-6)
        weighted_mean = (final_gate.unsqueeze(3) * message).sum(dim=1) / denominator.unsqueeze(2)
        valid_count = valid_edge.sum(dim=1).clamp_min(1.0).expand_as(denominator)
        confidence_amplitude = final_gate.sum(dim=1) / valid_count
        update_heads = (
            weighted_mean
            * confidence_amplitude.unsqueeze(2)
            * self.gamma.view(1, self.heads, 1, 1, 1)
        )
        update = self.output_projection(update_heads.reshape(features.shape[0], self.channels, *size))
        output = features + update

        zero = features.sum() * 0.0
        if self.regularization_enabled:
            coefficient_magnitude, coefficient_curvature = self.spline_regularization()
        else:
            coefficient_magnitude, coefficient_curvature = zero, zero
        valid_heads = valid_edge.expand_as(final_gate)
        graph_input_rms = features.float().square().mean().sqrt()
        graph_update_rms = update.float().square().mean().sqrt()
        spline_rms = spline.float().square().mean().sqrt()
        base_rms = base.float().square().mean().sqrt()
        diagnostics: dict[str, Any] = {
            "topographic_kan_logit_mean": kan_logit.mean(),
            "topographic_affinity_mean": torch.sigmoid(kan_logit).mean(),
            "observation_amplitude_mean": observation_amplitude.mean(),
            "latent_compatibility_mean": latent_compatibility.mean(),
            "latent_compatibility_logit_mean": latent_logit.mean(),
            "final_graph_gate_mean": final_gate.mean(),
            "gate_mean": final_gate.mean(),
            "gate_std": final_gate.float().std(unbiased=False),
            "valid_edge_fraction": valid_edge.mean(),
            "graph_gamma_mean": self.gamma.mean(),
            "gamma_mean": self.gamma.mean(),
            "graph_input_rms": graph_input_rms,
            "graph_update_rms": graph_update_rms,
            "graph_update_rms_ratio": graph_update_rms / graph_input_rms.clamp_min(1.0e-6),
            "spline_output_rms": spline_rms,
            "base_output_rms": base_rms,
            "spline_base_rms_ratio": spline_rms / base_rms.clamp_min(1.0e-8),
            "knot_boundary_saturation_fraction": (
                bounded.abs() >= 0.98
            ).to(features.dtype).mean(),
            "kan_coefficient_magnitude": coefficient_magnitude,
            "kan_coefficient_smoothness": coefficient_curvature,
            "kan_monotonicity": zero,
            "kan_curve_smoothness": coefficient_curvature,
            "static_topographic_affinity_mean": torch.sigmoid(kan_logit).mean(),
            "observation_confidence_mean": observation_amplitude.mean(),
            "valid_edge_gate_mean": (
                (final_gate * valid_heads).sum() / valid_heads.sum().clamp_min(1.0)
            ),
            "message_raw_rms": raw_message_rms,
            "message_rms": message.float().square().mean().sqrt(),
            "extreme_preservation_gate_mean": contrast_gate.mean(),
        }
        if self.diagnostics_enabled:
            diagnostics.update(
                {
                    "edge_descriptors": bounded.detach(),
                    "raw_edge_descriptors": raw.detach(),
                    "valid_edges": valid_edge.detach(),
                    "observation_amplitude": observation_amplitude.detach(),
                    "topographic_kan_logits": kan_logit.detach(),
                    "latent_compatibility": latent_compatibility.detach(),
                    "final_graph_gates": final_gate.detach(),
                    "kan_base_output": base.detach(),
                    "kan_spline_output": spline.detach(),
                }
            )
        return output, diagnostics
