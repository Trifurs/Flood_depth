from __future__ import annotations

import pytest

from datasets.model_input_spec import ModelInputSpec
from datasets.reliability_spec import RELIABILITY_NAMES, ReliabilitySpec


def test_production_input_contract_has_only_sar_and_terrain_groups() -> None:
    spec = ModelInputSpec.from_mode()
    assert spec.is_s1_only
    assert spec.continuous_groups == ("s1_t1", "s1_t2", "s1_change", "terrain")
    assert spec.qa_groups == ("s1_qa",)
    assert ReliabilitySpec.from_mode(spec.mode).names == RELIABILITY_NAMES


def test_nonproduction_input_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelInputSpec.from_mode("multisensor")
