import torch
from losses.multiscale_losses import auxiliary_depth_loss, masked_average_target, masked_gradient_consistency_loss


def test_masked_pool_excludes_tensor_safety_zeros() -> None:
    target = torch.tensor([[[[2., 0.], [4., 0.]]]])
    mask = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.bool)
    pooled, fraction = masked_average_target(target, mask, (1, 1))
    torch.testing.assert_close(pooled, torch.tensor([[[[3.]]]])); assert fraction.item() == .5


def test_empty_auxiliary_and_gradient_losses_are_differentiable_zero() -> None:
    prediction = torch.randn(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros_like(prediction); mask = torch.zeros_like(prediction, dtype=torch.bool)
    auxiliary, _ = auxiliary_depth_loss([prediction], target, mask, [1.], .25)
    gradient = masked_gradient_consistency_loss(prediction, target, mask, .1)
    (auxiliary + gradient).backward(); assert prediction.grad is not None

