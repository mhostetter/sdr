import numpy as np
import pytest

import sdr


def test_types():
    y = sdr.bsc(0, 0.5)
    assert isinstance(y, int)

    y = sdr.bsc([0, 1], 0.5)
    assert isinstance(y, np.ndarray)


def test_bit_flips():
    rng = np.random.default_rng()
    p = rng.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = rng.integers(0, 2, N)
    y = sdr.bsc(x, p)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)
