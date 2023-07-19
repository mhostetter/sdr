import numpy as np

import sdr


def test_constant_increment():
    K0 = 1
    increment = 0.1
    offset = 0
    nco = sdr.NCO(K0, increment, offset)

    assert nco.K0 == K0
    assert nco.increment == increment
    assert nco.offset == offset

    N = 100
    y = nco.step(N)
    y_truth = np.arange(1, N + 1) * increment + offset
    assert np.allclose(y, y_truth)


def test_constant_increment_constant_offset():
    K0 = 1
    increment = 0.1
    offset = 0.5
    nco = sdr.NCO(K0, increment, offset)

    assert nco.K0 == K0
    assert nco.increment == increment
    assert nco.offset == offset

    N = 100
    y = nco.step(N)
    y_truth = np.arange(1, N + 1) * increment + offset
    assert np.allclose(y, y_truth)
