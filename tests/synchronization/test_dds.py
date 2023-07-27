import numpy as np

import sdr


def test_constant_increment():
    K0 = 1
    increment = 0.1
    offset = 0
    dds = sdr.DDS(K0, increment, offset)

    assert dds.nco.K0 == K0
    assert dds.nco.increment == increment
    assert dds.nco.offset == offset

    N = 100
    y = dds.step(N)
    y_truth = np.exp(1j * (np.arange(1, N + 1) * increment + offset))
    assert np.allclose(y, y_truth)


def test_constant_increment_constant_offset():
    K0 = 1
    increment = 0.1
    offset = 0.5
    dds = sdr.DDS(K0, increment, offset)

    assert dds.nco.K0 == K0
    assert dds.nco.increment == increment
    assert dds.nco.offset == offset

    N = 100
    y = dds.step(N)
    y_truth = np.exp(1j * (np.arange(1, N + 1) * increment + offset))
    assert np.allclose(y, y_truth)
