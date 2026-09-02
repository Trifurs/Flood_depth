import torch
from tools.train import accumulation_window_sizes, normalize_accumulated_gradients


def test_remainder_windows_and_sample_normalization() -> None:
    assert accumulation_window_sizes(7, 4) == [4, 3]
    assert accumulation_window_sizes(3, 4) == [3]
    assert accumulation_window_sizes(4, 4) == [4]
    model = torch.nn.Linear(1, 1, bias=False); model.weight.grad = torch.tensor([[12.]])
    normalize_accumulated_gradients(model, 3)
    torch.testing.assert_close(model.weight.grad, torch.tensor([[4.]]))

