import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # p must be between 0 and 1
        sdr.BinarySymmetricChannel(-0.1)
    with pytest.raises(ValueError):
        # p must be between 0 and 1
        sdr.BinarySymmetricChannel(1.1)


def test_types():
    bsc = sdr.BinarySymmetricChannel(0.5)

    y = bsc(0)
    assert isinstance(y, int)

    y = bsc([0, 1])
    assert isinstance(y, np.ndarray)


def test_bit_flips():
    p = np.random.uniform(0.2, 0.8)
    N = int(1000 / p)
    x = np.random.randint(0, 2, N)
    bsc = sdr.BinarySymmetricChannel(p)
    y = bsc(x)
    assert np.count_nonzero(x != y) / N == pytest.approx(p, rel=1e-1)


def test_capacities_exceptions():
    with pytest.raises(ValueError):
        # Transition probability must be between 0 and 1
        sdr.BinarySymmetricChannel.capacities(-0.1)
    with pytest.raises(ValueError):
        # Transition probability must be between 0 and 1
        sdr.BinarySymmetricChannel.capacities(1.1)


def test_capacities_limit_cases():
    assert sdr.BinarySymmetricChannel.capacities(0) == 1
    assert sdr.BinarySymmetricChannel.capacities(1) == 1


def test_capacities_outputs():
    p = [0, 0.5, 1]
    C = sdr.BinarySymmetricChannel.capacities(p)
    assert isinstance(C, np.ndarray)
    assert np.array_equal(C, [1, 0, 1])

    C = sdr.BinarySymmetricChannel.capacities(0.5)
    assert isinstance(C, float)
    assert C == 0
