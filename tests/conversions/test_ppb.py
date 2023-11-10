import numpy as np

import sdr


def test_0_00005():
    assert sdr.ppb(0.000005) == 5000


def test_0_00000025():
    assert sdr.ppb(0.000000025) == 25


def test_array():
    assert np.array_equal(sdr.ppb(np.array([0.000005, 0.000000025])), np.array([5000, 25]))
