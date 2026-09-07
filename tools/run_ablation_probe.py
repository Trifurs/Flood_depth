#!/usr/bin/env python3
"""Run same-checkpoint functional probes for PA-HydroKAN ablation variants.

This tool deliberately does not substitute for a retrained ablation study.  It
loads one full-model checkpoint into state-schema-compatible variants, bypasses
one named module at a time, and evaluates every variant on exactly the same
validation batches.  The resulting deltas establish functional sensitivity and
configuration validity; final claims require matched retraining across seeds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from datasets.flooddepth_dataset import FloodDepthDataset
from datasets.model_input_spec import ModelInputSpec
from datasets.preprocessing import RobustNormalizer, resolve_depth_stratification_bins
from tools.evaluate_pa_hydrokan import (
    dataset_fingerprint,
    embed_source_fingerprints,
    evaluate_loader,
)
from utils.checkpoint import load_checkpoint
from utils.config import jsonable_config, load_config
from utils.logging import write_rows
from utils.misc import atomic_write_json, atomic_write_text
from utils.registry import build_model


DEFAULT_CONFIGS = (
    Path("configs/ablation/pa_hydrokan_full.xml"),
    Path("configs/ablation/pa_hydrokan_no_reliability_conditioning.xml"),
    Path("configs/ablation/pa_hydrokan_no_terrain_conditioned_fusion.xml"),
    Path("configs/ablation/pa_hydrokan_no_topographic_affinity_edge_kan.xml"),
    Path("configs/ablation/pa_hydrokan_no_latent_compatibility.xml"),
)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _loader(
    config: Mapping[str, Any], split: str, batch_size: int, num_workers: int
) -> tuple[DataLoader, ModelInputSpec, RobustNormalizer]:
    contract = DatasetContract.load(config["dataset"]["contract"])
    input_spec = ModelInputSpec.from_config(config)
    dataset = FloodDepthDataset(
        config["dataset"]["contract"],
        config["dataset"]["train_stats"],
        split,
        band_spec=resolve_band_spec(config, contract),
        input_spec=input_spec,
        minimum_event_band_fraction=float(
            config["dataset"].get("minimum_event_band_fraction", 1.0)
        ),
        s1_qa_names=config["dataset"].get("model_s1_qa_names"),
    )
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        ),
        input_spec,
        RobustNormalizer(Path(config["dataset"]["train_stats"]), contract),
    )


def _ablation_metadata(config: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    ablation = config.get("ablation")
    if not isinstance(ablation, Mapping):
        raise KeyError(f"Ablation configuration {path} requires an <ablation> section")
    required = ("variant_id", "display_name", "removed_module", "training_rule")
    missing = [key for key in required if key not in ablation]
    if missing:
        raise KeyError(f"Ablation configuration {path} is missing {missing}")
    return ablation


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "variant_id",
        "display_name",
        "removed_module",
        "trainable_parameters",
        "pixel_micro_mae",
        "pixel_micro_mae_delta_from_full",
        "pixel_micro_rmse",
        "event_macro_mae",
    )
    header = "| " + " | ".join(field.replace("_", " ") for field in fields) + " |\n"
    separator = "|" + "|".join("---" for _ in fields) + "|\n"
    body = "".join(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n"
        for row in rows
    )
    notice = (
        "# PA-HydroKAN frozen-weight ablation probe\n\n"
        "These values are same-checkpoint functional interventions, not retrained "
        "ablation results. Retrain each configuration with matched seeds before "
        "making an effectiveness claim.\n\n"
    )
    return notice + header + separator + body


def run_probe(
    config_paths: Sequence[Path], checkpoint: Path, split: str, device: torch.device,
    batch_size: int, num_workers: int, max_batches: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate every named intervention on the same checkpoint and split."""

    rows: list[dict[str, Any]] = []
    resolved_configs: list[dict[str, Any]] = []
    for path in config_paths:
        config = embed_source_fingerprints(load_config(path))
        ablation = _ablation_metadata(config, path)
        loader, input_spec, normalizer = _loader(
            config, split, batch_size, num_workers
        )
        model = build_model(config).to(device)
        load_checkpoint(
            checkpoint,
            model,
            expected_fingerprint=dataset_fingerprint(config),
            map_location=device,
        )
        depth_bins = resolve_depth_stratification_bins(config["loss"], normalizer)
        summary, _, _, _ = evaluate_loader(
            model,
            loader,
            device,
            depth_bins,
            primary_depth_bins=normalizer.train_depth_bins,
            max_batches=max_batches,
            progress=False,
            input_spec=input_spec,
        )
        total_parameters, trainable_parameters = _count_parameters(model)
        flags = getattr(model, "component_flags")()
        rows.append(
            {
                "variant_id": str(ablation["variant_id"]),
                "display_name": str(ablation["display_name"]),
                "removed_module": str(ablation["removed_module"]),
                "training_rule": str(ablation["training_rule"]),
                "config": str(path),
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "frozen_parameters": total_parameters - trainable_parameters,
                "reliability_conditioning_enabled": flags["reliability_conditioning_enabled"],
                "terrain_conditioned_fusion_enabled": flags[
                    "terrain_conditioned_fusion_enabled"
                ],
                "topographic_affinity_enabled": flags["topographic_affinity_enabled"],
                "latent_compatibility_enabled": flags["latent_compatibility_enabled"],
                "pixel_micro_mae": float(summary["pixel_micro_mae"]),
                "pixel_micro_rmse": float(summary["pixel_micro_rmse"]),
                "event_macro_mae": float(summary["event_macro_mae"]),
                "positive_supervision_pixels": int(summary["pixel_micro_pixels"]),
            }
        )
        resolved_configs.append(jsonable_config(config))

    full = next((row for row in rows if row["variant_id"] == "full"), None)
    if full is None:
        raise ValueError("A frozen-weight probe requires the full ablation configuration")
    full_mae = float(full["pixel_micro_mae"])
    for row in rows:
        row["pixel_micro_mae_delta_from_full"] = float(row["pixel_micro_mae"]) - full_mae
    return rows, resolved_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers must be non-negative")
    config_paths = tuple(args.config) if args.config else DEFAULT_CONFIGS
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows, resolved_configs = run_probe(
        config_paths,
        args.checkpoint.resolve(),
        args.split,
        _resolve_device(args.device),
        args.batch_size,
        args.num_workers,
        args.max_batches,
    )
    report = _markdown(rows)
    write_rows(output / "ablation_probe.csv", rows)
    atomic_write_json(
        output / "ablation_probe.json",
        {
            "protocol": "same_checkpoint_frozen_weight_functional_probe",
            "checkpoint": str(args.checkpoint.resolve()),
            "split": args.split,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "rows": rows,
            "resolved_configs": resolved_configs,
        },
    )
    atomic_write_text(output / "ablation_probe.md", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
