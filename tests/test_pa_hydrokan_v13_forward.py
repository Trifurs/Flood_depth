from pathlib import Path
import torch

from datasets.band_selection import resolve_band_spec
from datasets.contract import DatasetContract
from models.pa_hydrokan_v13 import PAHydroKANV13
from utils.config import load_config


def synthetic_inputs(spec, size=32):
    tensor = lambda channels: torch.randn(1, channels, size, size)
    result = {group: tensor(spec.channels(group)) for group in (
        "s1_t1", "s1_t2", "s1_change", "s2_t1", "s2_t2", "s2_change", "terrain"
    )}
    result.update({"terrain_raw": tensor(2), "reliability": tensor(12),
                   "s1_valid": torch.ones(1, 1, size, size), "s2_valid": torch.ones(1, 1, size, size),
                   "dem_valid": torch.ones(1, 1, size, size)})
    if spec.channels("s1_conditioning"):
        result["s1_conditioning"] = tensor(spec.channels("s1_conditioning"))
    return result


def test_core_and_compact_forward_backward_are_finite() -> None:
    for path in ("configs/pa_hydrokan/subset150_v13_core.xml", "configs/pa_hydrokan/subset150_v13_compact.xml"):
        config = load_config(Path(path)); contract = DatasetContract.load(config["dataset"]["contract"])
        spec = resolve_band_spec(config, contract); model = PAHydroKANV13(config["model"], spec)
        outputs = model(synthetic_inputs(spec))
        assert outputs["depth"].shape == (1, 1, 32, 32)
        loss = outputs["depth"].mean() + outputs["uncertainty_scale"].mean()
        loss.backward()
        assert torch.isfinite(loss)

