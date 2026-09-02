import pytest
import torch
from utils.amp import resolve_amp


def test_cpu_disables_amp_and_invalid_dtype_fails() -> None:
    enabled, dtype, scaler = resolve_amp(torch.device("cpu"), True, "auto")
    assert not enabled and dtype == torch.float16 and not scaler
    with pytest.raises(ValueError): resolve_amp(torch.device("cpu"), True, "tf32")

