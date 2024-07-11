import numpy as np
import pytest

import sdr


def test_scalar():
    assert sdr.coherent_gain_loss(1e-3, 235) == pytest.approx(0.8038919141626675)


def test_time_vector():
    cgl = sdr.coherent_gain_loss([1e-3, 2e-3, 3e-3, 4e-4, 5e-3], 100)
    cgl_truth = np.array([0.14335017, 0.57922366, 1.32626966, 0.02287239, 3.92239754])
    assert isinstance(cgl, np.ndarray)
    assert np.allclose(cgl, cgl_truth)


def test_freq_vector():
    cgl = sdr.coherent_gain_loss(1e-3, [0, 100, 200, 300, 400, 500])
    cgl_truth = np.array([0.0, 0.14335017, 0.57922366, 1.32626966, 2.42007077, 3.92239754])
    assert isinstance(cgl, np.ndarray)
    assert np.allclose(cgl, cgl_truth)
