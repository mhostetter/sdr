import numpy as np

import sdr


def test_0_005():
    assert sdr.ppm(0.005) == 5000


def test_0_000025():
    assert sdr.ppm(0.000025) == 25


def test_array():
    assert np.array_equal(sdr.ppm(np.array([0.005, 0.000025])), np.array([5000, 25]))
