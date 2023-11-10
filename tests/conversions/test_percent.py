import numpy as np

import sdr


def test_0_5():
    assert sdr.percent(0.5) == 50


def test_0_25():
    assert sdr.percent(0.25) == 25


def test_array():
    assert np.array_equal(sdr.percent(np.array([0.5, 0.25])), np.array([50, 25]))
