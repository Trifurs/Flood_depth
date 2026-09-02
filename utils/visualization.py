"""Non-interactive prediction panels."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_prediction_panel(
    path: Path,
    *,
    s1_change: np.ndarray,
    s2_change: np.ndarray,
    dsm: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    valid_label: np.ndarray,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_view = np.where(valid_label, target, np.nan)
    error_view = np.where(valid_label, np.abs(prediction - target), np.nan)
    panels = [
        (s1_change, "S1 representative change", "coolwarm"),
        (s2_change, "S2 representative change", "coolwarm"),
        (dsm, "DSM (m)", "terrain"),
        (target_view, "Target depth (m)", "Blues"),
        (prediction, "Predicted depth (m)", "Blues"),
        (uncertainty, "Uncertainty scale (m)", "magma"),
        (error_view, "Absolute error on valid labels", "inferno"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        shown = axis.imshow(np.asarray(image).squeeze(), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    axes.flat[-1].axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
