import numpy as np
import pytest

import sdr


def test_scalar():
    assert sdr.max_integration_time(3, 235) == pytest.approx(0.0018818867640235891)


def test_cgl_vector():
    t = sdr.max_integration_time([1, 2, 3, 4, 5], 100)
    t_truth = np.array([0.002615, 0.00365466, 0.00442243, 0.00504438, 0.00556994])
    assert isinstance(t, np.ndarray)
    assert np.allclose(t, t_truth)


def test_freq_vector():
    t = sdr.max_integration_time(3, [100, 200, 300, 400, 500])
    t_truth = np.array([0.00442243, 0.00221122, 0.00147414, 0.00110561, 0.00088449])
    assert isinstance(t, np.ndarray)
    assert np.allclose(t, t_truth)
