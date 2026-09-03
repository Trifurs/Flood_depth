import torch

from models.hydro_edge_kan import HydroEdgeKAN
from models.kan_layers import KANLinear


def _inputs(size=8):
    features = torch.randn(2, 8, size, size, requires_grad=True)
    physical = {k: torch.ones(2, 1, size, size) for k in ("dem_valid", "z_hyd", "z_barrier", "local_relief")}
    physical["z_hyd"] = torch.arange(size, dtype=torch.float32).view(1, 1, 1, size).expand(2, 1, size, size)
    reliability = torch.zeros(2, 1, size, size)
    modality = torch.full((2, 2, size // 2, size // 2), .5)
    sensor = torch.ones(2, 1, size, size)
    return features, physical, reliability, modality, sensor


def test_featurewise_kan_sum_and_multihead_shape():
    layer = KANLinear(5, 3, grid_size=4, spline_order=3, normalization="explicit_fixed_scaling", input_bounding="prebounded", base_scale_init=.5, spline_scale_init=1., learnable_base_scale=True, learnable_spline_scale=True)
    x = torch.randn(2, 4, 5)
    total, base, spline = layer.forward_with_contributions(x)
    b, s = layer.featurewise_contributions(x)
    assert torch.allclose(total, b.sum(-2) + s.sum(-2) + layer.base_bias, atol=1e-6)
    assert torch.allclose(total, base + spline, atol=1e-6)


def test_hydro_edge_gamma_gradient_and_identity():
    model = HydroEdgeKAN(8, heads=2, grid_size=4, spline_order=3)
    args = _inputs()
    assert torch.allclose(model.gamma, torch.full_like(model.gamma, 0.02), atol=1e-6)
    out, diag = model(*args)
    assert torch.all(model.gamma >= 0) and torch.all(model.gamma <= model.gamma_max)
    out.mean().backward()
    assert model.edge_kan.spline_coefficients.grad is not None
    assert model.edge_kan.spline_coefficients.grad.abs().sum() > 0
    assert torch.isfinite(model.edge_kan.spline_coefficients.grad).all()
    invalid = list(args)
    invalid[4] = torch.zeros_like(invalid[4]); invalid[2] = torch.ones_like(invalid[2])
    identity, _ = model(*invalid)
    assert torch.equal(identity, invalid[0])


def test_all_small_gates_suppress_graph_update():
    model = HydroEdgeKAN(8, heads=2, grid_size=4)
    with torch.no_grad():
        model.edge_kan.base_weight.zero_()
        model.edge_kan.base_bias.fill_(-100.0)
        model.edge_kan.spline_coefficients.zero_()
    features, physical, reliability, modality, sensor = _inputs()
    output, diag = model(features, physical, reliability, modality, sensor)
    assert float(diag["gate_mean"].detach()) < 1e-6
    assert torch.allclose(output, features, atol=1e-5)


def test_constant_latent_difference_has_no_message_and_direction_order():
    model = HydroEdgeKAN(8, heads=2, grid_size=4)
    features, physical, reliability, modality, sensor = _inputs()
    features = torch.ones_like(features)
    output, _ = model(features, physical, reliability, modality, sensor)
    assert torch.allclose(output, features, atol=1e-6)
    import models.hydro_edge_kan as module
    original = module.DIRECTIONS
    try:
        reversed_model = HydroEdgeKAN(8, heads=2, grid_size=4)
        reversed_model.load_state_dict(model.state_dict())
        module.DIRECTIONS = tuple(reversed(original))
        reverse_output, _ = reversed_model(features, physical, reliability, modality, sensor)
    finally:
        module.DIRECTIONS = original
    assert torch.allclose(output, reverse_output, atol=1e-6)


def test_prebounded_kan_is_finite_for_out_of_range_inputs():
    layer = KANLinear(5, 2, grid_size=4, input_bounding="prebounded")
    values = torch.full((3, 5), 50.0, dtype=torch.float32, requires_grad=True)
    output = layer(values)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_kan_forward_backward_fp32_fp16_bf16():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        layer = KANLinear(5, 2, grid_size=4, input_bounding="prebounded").to(dtype=dtype)
        values = torch.randn(3, 5, dtype=dtype, requires_grad=True)
        output = layer(values)
        output.square().mean().backward()
        assert torch.isfinite(output).all() and torch.isfinite(values.grad).all()


def test_diagonal_edge_uses_euclidean_distance():
    model = HydroEdgeKAN(8, heads=2, grid_size=4, graph_scale=8, terrain_pixel_size_m=20.0)
    features, physical, reliability, modality, sensor = _inputs(size=8)
    descriptor, _, _ = model._descriptor(
        physical["z_hyd"], physical["z_barrier"], physical["local_relief"],
        torch.ones_like(physical["dem_valid"]), (1, 1),
    )
    # Default distance center=160 m and scale=80 m; diagonal is 160*sqrt(2).
    expected = (160.0 * (2.0 ** 0.5) - 160.0) / 80.0
    assert abs(float(descriptor[..., 4].mean()) - expected) < 1e-5
