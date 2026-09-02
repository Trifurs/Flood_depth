from pathlib import Path
import pytest

from datasets.band_selection import BandSpec
from datasets.contract import DatasetContract
from utils.config import load_config


def contract() -> DatasetContract:
    config = load_config(Path("configs/pa_hydrokan/subset150_main.xml"))
    return DatasetContract.load(config["dataset"]["contract"])


def test_exact_selection_preserves_arbitrary_order_and_resolves_indexes() -> None:
    spec = BandSpec.resolve(contract(), {
        "s1_t1": ["VH_pre_db", "VV_pre_db"], "s1_t2": ["VH_event_db", "VV_event_db"],
        "s1_change": ["anomaly_raw"], "s1_conditioning": ["angle_event_deg", "angle_pre_deg"],
        "s2_t1": ["B11_pre_reflectance"], "s2_t2": ["B11_event_reflectance"],
        "s2_change": ["MNDWI_delta"], "terrain": ["slope_deg", "elevation_m_DSM"],
    })
    assert spec.names("s1_t1") == ("VH_pre_db", "VV_pre_db")
    assert spec.indexes("s1_t1") == (1, 0)
    assert spec.conditioning_sources(contract()) == (("s1_t2", 2), ("s1_t1", 2))


def test_unknown_band_fails_and_legacy_uses_every_channel() -> None:
    with pytest.raises(ValueError, match="Unknown band"):
        BandSpec.resolve(contract(), {"s1_t1": ["not-a-band"]})
    legacy = BandSpec.resolve(contract(), None)
    assert legacy.legacy_full_bands
    assert legacy.channels("s2_t1") == 6


def test_temporal_pair_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="semantic/order mismatch"):
        BandSpec.resolve(contract(), {
            "s1_t1": ["VV_pre_db", "VH_pre_db"],
            "s1_t2": ["VV_event_db", "angle_event_deg"],
            "s1_change": ["VV_delta_db"],
            "s2_t1": ["B3_pre_reflectance"],
            "s2_t2": ["B3_event_reflectance"],
            "s2_change": ["NDWI_delta"],
            "terrain": ["elevation_m_DSM"],
        })
