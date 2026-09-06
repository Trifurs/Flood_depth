import torch
from datasets.preprocessing import RELIABILITY_NAMES
from datasets.transforms import SynchronousAugment


def test_forced_modality_dropout_keeps_validity_and_reliability_consistent(monkeypatch) -> None:
    values = iter([torch.tensor(1.), torch.tensor(1.), torch.tensor(1.), torch.tensor(0.), torch.tensor(0.)])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: next(values))
    sample = {key: torch.ones(2, 4, 4) for key in ("s1_t1", "s1_t2", "s1_change", "s1_conditioning", "s2_t1", "s2_t2", "s2_change")}
    sample.update({"reliability": torch.ones(len(RELIABILITY_NAMES), 4, 4),
                   "validity": {"s1_valid": torch.ones(1, 4, 4), "s2_valid": torch.ones(1, 4, 4),
                                "dem_valid": torch.ones(1, 4, 4), "output_valid": torch.ones(1, 4, 4)},
                   "metadata": {}})
    result = SynchronousAugment(0, 0, 0, 1)(sample)
    assert result["metadata"]["modality_dropout"] == "s1"
    assert result["s1_conditioning"].sum() == 0 and result["validity"]["s1_valid"].sum() == 0
    assert result["reliability"][RELIABILITY_NAMES.index("s1_valid")].sum() == 0
    assert torch.all(result["validity"]["output_valid"] == 1)
