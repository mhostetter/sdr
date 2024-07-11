import numpy as np
import pytest

import sdr


def test_bsc():
    rng = np.random.default_rng()
    p = rng.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = rng.integers(0, 2, N)
    P = [[1 - p, p], [p, 1 - p]]
    dmc = sdr.DiscreteMemorylessChannel(P)
    y = dmc(x)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)


def test_bec():
    rng = np.random.default_rng()
    p = rng.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = rng.integers(0, 2, N)
    P = [[1 - p, 0, p], [0, 1 - p, p]]
    dmc = sdr.DiscreteMemorylessChannel(P, Y=[0, 1, -1])
    y = dmc(x)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)
