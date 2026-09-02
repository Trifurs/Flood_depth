import torch
from models.terrain_features_v13 import valid_central_gradients


def test_gradient_uses_metric_pixel_size_and_rejects_invalid_neighbours() -> None:
    x = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(1, 1, 5, 5) * 20
    valid = torch.ones_like(x)
    gx, gy, gx_valid, _ = valid_central_gradients(x, valid, 20., 20.)
    torch.testing.assert_close(gx[..., 2, 2], torch.tensor([[1.]]))
    torch.testing.assert_close(gy[..., 2, 2], torch.tensor([[0.]]))
    valid[..., 2, 1] = 0
    gx, _, gx_valid, _ = valid_central_gradients(x, valid, 20., 20.)
    assert gx[..., 2, 2].item() == 0 and gx_valid[..., 2, 2].item() == 0
