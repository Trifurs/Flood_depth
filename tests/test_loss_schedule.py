from losses.composite_loss import CompositeFloodDepthLoss


def test_linear_warmup_schedule() -> None:
    loss = CompositeFloodDepthLoss({"lambda_pu": .2, "pu_start_epoch": 5, "pu_warmup_epochs": 10,
                                    "lambda_wse": .02, "wse_start_epoch": 5, "wse_warmup_epochs": 15}, .1)
    assert loss.scheduled_weight("pu", 4) == 0
    assert abs(loss.scheduled_weight("pu", 5) - .02) < 1e-9
    assert loss.scheduled_weight("pu", 14) == .2

