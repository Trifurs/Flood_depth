import torch

from losses.task_adaptive_depth_loss import (
    soft_depth_balance_weights,
    tail_underprediction_factor,
    task_adaptive_positive_depth_loss,
)


def test_soft_weights_bounded_and_positive_mean():
    target = torch.tensor([[[[.1, .2, .5, 1.0, 3.0]]]])
    positive = torch.ones_like(target)
    weights = soft_depth_balance_weights(target, positive, [.1, .23, .48, .83, 1.22, 2.14, 24.82])
    assert torch.isfinite(weights).all()
    assert float(weights.min()) >= .5 and float(weights.max()) <= 3.0
    assert abs(float(weights.mean()) - 1.0) < 1e-5


def test_tail_factor_only_deep_underprediction():
    target = torch.tensor([[[[.2, 1.0, 1.0, 1.0]]]])
    pred = torch.tensor([[[[.1, .5, 1.5, .5]]]])
    factor = tail_underprediction_factor(pred, target, torch.ones_like(target), [.1, .23, .48, .83, 1.22, 2.14, 24.82], .2)
    assert float(factor[0, 0, 0, 0]) == 1.0
    assert float(factor[0, 0, 0, 1]) > 1.0 and float(factor[0, 0, 0, 2]) == 1.0


def test_task_loss_finite_zero_positive():
    prediction = torch.ones(1, 1, 2, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    positive = torch.zeros_like(prediction)
    result = task_adaptive_positive_depth_loss(prediction, target, positive, [.1, .23, .48], .5)
    assert torch.isfinite(result["depth"])
    result["depth"].backward()
