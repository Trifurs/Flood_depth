from __future__ import annotations

import torch

from models.decoder_v14 import IndependentGatedFPNDecoderV14


def test_independent_decoder_gates_do_not_compete_and_respect_validity() -> None:
    decoder = IndependentGatedFPNDecoderV14(
        [4, 8, 16, 24], 0.0, 4, "efficient", False, widths=[8, 6, 4]
    )
    for layer in decoder.gates:
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.constant_(layer.bias, 5.0)
    skips = [torch.randn(1, 4, 16, 16), torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)]
    terrain = [torch.randn(1, 4, 16, 16), torch.randn(1, 8, 8, 8), torch.randn(1, 16, 4, 4)]
    fractions = [torch.ones(1, 1, 16, 16), torch.ones(1, 1, 8, 8), torch.ones(1, 1, 4, 4)]
    decoded, auxiliaries, gates = decoder(
        torch.randn(1, 24, 2, 2), skips, terrain, fractions, torch.ones(1, 1, 16, 16)
    )
    assert decoded.shape == (1, 4, 16, 16) and not auxiliaries
    assert all(item["sensor_p50"].item() > 0.9 and item["terrain_p50"].item() > 0.9 for item in gates)

    _, _, missing_sensor_gates = decoder(
        torch.randn(1, 24, 2, 2), skips, terrain, fractions, torch.zeros(1, 1, 16, 16)
    )
    assert all(torch.count_nonzero(item["sensor"]).item() == 0 for item in missing_sensor_gates)
    assert all(item["terrain_p50"].item() > 0.9 for item in missing_sensor_gates)

