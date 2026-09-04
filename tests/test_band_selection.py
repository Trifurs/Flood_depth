from pathlib import Path
import pytest

from datasets.band_selection import BandSpec
from datasets.contract import DatasetContract
from datasets.model_input_spec import ModelInputSpec
from utils.config import load_config


def contract() -> DatasetContract:
    config = load_config(Path("configs/pa_hydrokan/subset1000_s1_v15.xml"))
    return DatasetContract.load(config["dataset"]["contract"])


def s1_groups() -> tuple[str, ...]:
    return ModelInputSpec.from_mode("s1_terrain").continuous_groups


def test_exact_selection_preserves_arbitrary_order_and_resolves_indexes() -> None:
    spec = BandSpec.resolve(contract(), {
        "s1_t1": ["VH_pre_db", "VV_pre_db"], "s1_t2": ["VH_event_db", "VV_event_db"],
        "s1_change": ["anomaly_raw"], "s1_conditioning": ["angle_event_deg", "angle_pre_deg"],
        "terrain": ["slope_deg", "elevation_m_DSM"],
    }, s1_groups())
    assert spec.names("s1_t1") == ("VH_pre_db", "VV_pre_db")
    assert spec.indexes("s1_t1") == (1, 0)
    assert spec.conditioning_sources(contract()) == (("s1_t2", 2), ("s1_t1", 2))


def test_unknown_band_fails_and_legacy_uses_every_channel() -> None:
    with pytest.raises(ValueError, match="Unknown band"):
        BandSpec.resolve(contract(), {"s1_t1": ["not-a-band"]}, s1_groups())
    legacy = BandSpec.resolve(contract(), None, s1_groups())
    assert legacy.legacy_full_bands
    assert legacy.channels("s1_t1") == 3


def test_temporal_pair_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="semantic/order mismatch"):
        BandSpec.resolve(contract(), {
            "s1_t1": ["VV_pre_db", "VH_pre_db"],
            "s1_t2": ["VV_event_db", "angle_event_deg"],
            "s1_change": ["VV_delta_db"],
            "terrain": ["elevation_m_DSM"],
        }, s1_groups())
