import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Integration must be non-negative
        sdr.coherent_gain_loss(-1e3, 100)
    with pytest.raises(ValueError):
        # Integration must be non-negative
        sdr.coherent_gain_loss([-1e3, 0, 1e3], 100)


def test_scalar():
    assert sdr.coherent_gain_loss(1e-3, 235) == pytest.approx(0.8038919141626675)


def test_vector():
    cgl = sdr.coherent_gain_loss(1e-3, [0, 100, 200, 300, 400, 500])
    cgl_truth = np.array([0.0, 0.14335017, 0.57922366, 1.32626966, 2.42007077, 3.92239754])
    assert isinstance(cgl, np.ndarray)
    assert np.allclose(cgl, cgl_truth)
