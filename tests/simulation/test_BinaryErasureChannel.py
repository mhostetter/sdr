import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # p must be between 0 and 1
        sdr.BinaryErasureChannel(-0.1)
    with pytest.raises(ValueError):
        # p must be between 0 and 1
        sdr.BinaryErasureChannel(1.1)


def test_types():
    bsc = sdr.BinaryErasureChannel(0.5)

    y = bsc(0)
    assert isinstance(y, int)

    y = bsc([0, 1])
    assert isinstance(y, np.ndarray)


def test_bit_erasures():
    rng = np.random.default_rng()
    p = rng.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = rng.integers(0, 2, N)
    bsc = sdr.BinaryErasureChannel(p)
    y = bsc(x)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)


def test_capacities_exceptions():
    with pytest.raises(ValueError):
        # Erasure probability must be between 0 and 1
        sdr.BinaryErasureChannel.capacities(-0.1)
    with pytest.raises(ValueError):
        # Erasure probability must be between 0 and 1
        sdr.BinaryErasureChannel.capacities(1.1)


def test_capacities_limit_cases():
    assert sdr.BinaryErasureChannel.capacities(0) == 1
    assert sdr.BinaryErasureChannel.capacities(1) == 0


def test_capacities_outputs():
    p = [0, 0.5, 1]
    C = sdr.BinaryErasureChannel.capacities(p)
    assert isinstance(C, np.ndarray)
    assert np.array_equal(C, [1, 0.5, 0])

    C = sdr.BinaryErasureChannel.capacities(0.5)
    assert isinstance(C, float)
    assert C == 0.5
