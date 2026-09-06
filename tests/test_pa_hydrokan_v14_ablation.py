from __future__ import annotations

import torch

from utils.config import load_config
from utils.registry import build_model


def test_no_s2_ablation_is_explicit_and_masks_all_s2_dependence() -> None:
    config = load_config("configs/pa_hydrokan/subset1000_v14_no_s2.xml")
    model = build_model(config).eval()
    shape = (1, 64, 64)
    inputs = {
        "s1_t1": torch.randn(1, 2, *shape[1:]), "s1_t2": torch.randn(1, 2, *shape[1:]),
        "s1_change": torch.randn(1, 3, *shape[1:]), "s1_conditioning": torch.randn(1, 2, *shape[1:]),
        "s2_t1": torch.randn(1, 3, *shape[1:]), "s2_t2": torch.randn(1, 3, *shape[1:]),
        "s2_change": torch.randn(1, 2, *shape[1:]), "terrain": torch.randn(1, 2, *shape[1:]),
        "terrain_raw": torch.randn(1, 2, *shape[1:]), "reliability": torch.randn(1, 12, *shape[1:]),
        "s1_valid": torch.ones(1, 1, *shape[1:]), "s2_valid": torch.ones(1, 1, *shape[1:]),
        "dem_valid": torch.ones(1, 1, *shape[1:]),
        "branch_validity": {
            key: torch.ones(1, 1, *shape[1:]) for key in
            ("s1_t1", "s1_t2", "s1_change", "s2_t1", "s2_t2", "s2_change")
        },
    }
    changed = {key: value.clone() if torch.is_tensor(value) else dict(value) for key, value in inputs.items()}
    for key in ("s2_t1", "s2_t2", "s2_change"):
        changed[key].normal_()
    changed["reliability"][:, (2, 3, 4, 6, 9, 11)] = torch.randn(1, 6, *shape[1:]) * 100.0
    with torch.no_grad():
        first = model(inputs)
        second = model(changed)
    torch.testing.assert_close(first["conditional_depth"], second["conditional_depth"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first["support_probability"], second["support_probability"], atol=1e-6, rtol=1e-6)
