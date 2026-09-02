#!/usr/bin/env python3
"""Create a compact panel from already-exported prediction GeoTIFFs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = (
        "conditional_depth_m.tif",
        "support_weighted_depth_m.tif",
        "support_probability.tif",
        "uncertainty_scale_m.tif",
    )
    arrays = []
    for name in names:
        with rasterio.open(args.prediction_dir / name) as dataset:
            arrays.append(dataset.read(1, masked=True))
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    titles = (
        "Conditional depth (m)",
        "Support-weighted depth (m)",
        "Support score",
        "Uncertainty scale (m)",
    )
    cmaps = ("Blues", "Blues", "viridis", "magma")
    for axis, array, title, cmap in zip(axes, arrays, titles, cmaps):
        shown = axis.imshow(array, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis)
    output = args.output or args.prediction_dir / "prediction_products.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
