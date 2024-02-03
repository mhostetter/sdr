import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # P must be 2D
        sdr.DiscreteMemorylessChannel([0.2, 0.8])
    with pytest.raises(ValueError):
        # P must be non-negative
        sdr.DiscreteMemorylessChannel([[0.2, 0.8], [-0.1, 1.1]])
    with pytest.raises(ValueError):
        # P must sum to 1
        sdr.DiscreteMemorylessChannel([[0.2, 0.8], [0.3, 0.8]])
    with pytest.raises(ValueError):
        # X must be 1D
        sdr.DiscreteMemorylessChannel([[0.2, 0.8], [0.3, 0.7]], X=[[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        # Y must be 1D
        sdr.DiscreteMemorylessChannel([[0.2, 0.8], [0.3, 0.7]], Y=[[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        # P must have X rows
        sdr.DiscreteMemorylessChannel([[0.2, 0.8], [0.3, 0.7]], X=[0, 1, 2])
    with pytest.raises(ValueError):
        # P must have Y columns
        sdr.DiscreteMemorylessChannel([[0.2, 0.8], [0.3, 0.7]], Y=[0, 1, 2])


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
