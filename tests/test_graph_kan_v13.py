import pytest
import torch

from models.terrain_graph_kan_v13 import MultiHeadTerrainGraphKAN


def inputs(dtype=torch.float32):
    features = torch.randn(1, 16, 8, 8, dtype=dtype, requires_grad=True)
    physical = {key: torch.ones(1, 1, 64, 64, dtype=dtype) for key in ("dem_valid", "z_hyd", "slope", "z_barrier")}
    return features, physical, torch.zeros(1, 1, 64, 64, dtype=dtype), torch.full((1, 2, 8, 8), .5, dtype=dtype), torch.ones(1, 1, 64, 64, dtype=dtype)


def test_zero_gamma_is_identity_and_no_edges_remain_finite() -> None:
    graph = MultiHeadTerrainGraphKAN(16, 4, 4, 3, 12, "explicit_fixed_scaling", "gate_sum")
    args = inputs(); output, diagnostics = graph(*args)
    torch.testing.assert_close(output, args[0])
    output.sum().backward()
    assert all(torch.isfinite(value).all().item() for value in diagnostics.values() if torch.is_tensor(value))
    args = list(inputs()); args[-1].zero_(); output, _ = graph(*args)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_cpu_autocast_forward_backward_finite(dtype) -> None:
    graph = MultiHeadTerrainGraphKAN(16, 4, 4, 3, 12, "explicit_fixed_scaling", "gate_sum")
    args = inputs()
    with torch.autocast("cpu", dtype=dtype):
        output, _ = graph(*args); loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_forward_backward_finite(dtype) -> None:
    graph = MultiHeadTerrainGraphKAN(16, 4, 4, 3, 12, "explicit_fixed_scaling", "gate_sum").cuda()
    features, physical, reliability, directional_weights, valid = inputs()
    args = (
        features.cuda(),
        {key: value.cuda() for key, value in physical.items()},
        reliability.cuda(),
        directional_weights.cuda(),
        valid.cuda(),
    )
    with torch.autocast("cuda", dtype=dtype):
        output, _ = graph(*args); loss = output.square().mean()
    loss.backward(); assert torch.isfinite(output.float()).all()
