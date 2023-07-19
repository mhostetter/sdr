import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # p must be between 0 and 1
        sdr.bsc([0, 1], -0.1)
    with pytest.raises(ValueError):
        # p must be between 0 and 1
        sdr.bsc([0, 1], 1.1)


def test_types():
    y = sdr.bsc(0, 0.5)
    assert isinstance(y, int)

    y = sdr.bsc([0, 1], 0.5)
    assert isinstance(y, np.ndarray)


def test_bit_flips():
    p = np.random.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = np.random.randint(0, 2, N)
    y = sdr.bsc(x, p)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)
