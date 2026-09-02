from __future__ import annotations

import random

import numpy as np
import torch

from utils.checkpoint import capture_rng_state, restore_rng_state


def test_restore_rng_accepts_list_like_cpu_state() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng_state()
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    random.random()
    np.random.random()
    torch.rand(8)
    portable_state = dict(state)
    portable_state["torch_cpu"] = state["torch_cpu"].tolist()
    portable_state["torch_cuda"] = None
    restore_rng_state(portable_state)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
