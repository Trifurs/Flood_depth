#!/usr/bin/env python3
"""Export one-dimensional HydroEdgeKAN base/spline response curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint import load_checkpoint
from utils.config import load_config
from tools.evaluate import embed_source_fingerprints
from utils.registry import build_model


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=Path("artifacts/optimization/hydrov13_2/kan_curves")); parser.add_argument("--points", type=int, default=65); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    config = embed_source_fingerprints(load_config(args.config)); model = build_model(config).to(args.device); load_checkpoint(args.checkpoint, model, map_location=args.device); model.eval(); kan = model.graph.edge_kan
    v14_edge_kan = hasattr(model.graph, "prior_scales") and kan.out_features == 1
    xs = torch.linspace(-1.0, 1.0, args.points, device=args.device); rows = []
    centers = model.graph.feature_centers.flatten().detach().cpu(); scales = model.graph.feature_scales.flatten().detach().cpu()
    with torch.no_grad():
        for head in range(model.graph.heads):
            for feature, name in enumerate(model.graph.edge_feature_names):
                descriptor = torch.zeros(args.points, len(model.graph.edge_feature_names), device=args.device); descriptor[:, feature] = xs
                base_terms, spline_terms = kan.featurewise_contributions(descriptor)
                if v14_edge_kan:
                    base = base_terms[:, feature, 0]; spline = spline_terms[:, feature, 0]
                    prior = -model.graph.prior_scales[feature] * (centers[feature] + scales[feature] * xs)
                    total = model.graph.prior_bias[head] + prior + base + spline
                    prior_value = prior + model.graph.prior_bias[head]
                else:
                    base = base_terms[:, feature, head]; spline = spline_terms[:, feature, head]; total = base + spline
                    prior_value = torch.zeros_like(total)
                first = torch.gradient(total, spacing=(xs,))[0]; second = torch.gradient(first, spacing=(xs,))[0]
                for i in range(args.points):
                    rows.append({"head": head, "feature": name, "feature_index": feature, "standardized_input": float(xs[i]), "physical_input": float(centers[feature] + scales[feature] * xs[i]), "prior_contribution": float(prior_value[i]), "base_contribution": float(base[i]), "spline_contribution": float(spline[i]), "total_contribution": float(total[i]), "first_difference": float(first[i]), "second_difference": float(second[i])})
    with (args.output_dir / "curves.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"config": str(args.config), "checkpoint": str(args.checkpoint), "heads": model.graph.heads, "features": list(model.graph.edge_feature_names), "points_per_curve": args.points, "curves": model.graph.heads * len(model.graph.edge_feature_names), "max_abs_second_difference": max(abs(row["second_difference"]) for row in rows), "mean_abs_spline_contribution": sum(abs(row["spline_contribution"]) for row in rows) / len(rows)}
    (args.output_dir / "curve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
