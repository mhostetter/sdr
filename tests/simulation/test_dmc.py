import numpy as np
import pytest

import sdr


def test_exceptions():
    x = [0, 1]
    with pytest.raises(ValueError):
        # P must be 2D
        sdr.dmc(x, [0.2, 0.8])
    with pytest.raises(ValueError):
        # P must be non-negative
        sdr.dmc(x, [[0.2, 0.8], [-0.1, 1.1]])
    with pytest.raises(ValueError):
        # P must sum to 1
        sdr.dmc(x, [[0.2, 0.8], [0.3, 0.8]])
    with pytest.raises(ValueError):
        # X must be 1D
        sdr.dmc(x, [[0.2, 0.8], [0.3, 0.7]], X=[[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        # Y must be 1D
        sdr.dmc(x, [[0.2, 0.8], [0.3, 0.7]], Y=[[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        # P must have X rows
        sdr.dmc(x, [[0.2, 0.8], [0.3, 0.7]], X=[0, 1, 2])
    with pytest.raises(ValueError):
        # P must have Y columns
        sdr.dmc(x, [[0.2, 0.8], [0.3, 0.7]], Y=[0, 1, 2])


def test_bsc():
    p = np.random.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = np.random.randint(0, 2, N)
    P = [[1 - p, p], [p, 1 - p]]
    y = sdr.dmc(x, P)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)


def test_bec():
    p = np.random.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = np.random.randint(0, 2, N)
    P = [[1 - p, 0, p], [0, 1 - p, p]]
    y = sdr.dmc(x, P, Y=[0, 1, -1])
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)
