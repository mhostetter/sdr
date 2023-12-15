import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        sdr.Integrator("invalid")


def test_backward():
    iir = sdr.Integrator("backward")
    b_truth = [0, 1]
    a_truth = [1, -1]
    assert np.allclose(iir.b_taps, b_truth)
    assert np.allclose(iir.a_taps, a_truth)


def test_trapezoidal():
    iir = sdr.Integrator("trapezoidal")
    b_truth = [0.5, 0.5]
    a_truth = [1, -1]
    assert np.allclose(iir.b_taps, b_truth)
    assert np.allclose(iir.a_taps, a_truth)


def test_forward():
    iir = sdr.Integrator("forward")
    b_truth = [1]
    a_truth = [1, -1]
    assert np.allclose(iir.b_taps, b_truth)
    assert np.allclose(iir.a_taps, a_truth)


def test_gaussian():
    iir = sdr.Integrator("forward")
    x = sdr.gaussian(0.3, 5, 10)
    y = iir(x)
    y_truth = np.cumsum(x)
    assert np.allclose(y, y_truth)


def test_raised_cosine():
    iir = sdr.Integrator("forward")
    x = sdr.root_raised_cosine(0.1, 8, 10)
    y = iir(x)
    y_truth = np.cumsum(x)
    assert np.allclose(y, y_truth)
