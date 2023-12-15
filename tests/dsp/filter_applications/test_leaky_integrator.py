import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        sdr.LeakyIntegrator("invalid")
    with pytest.raises(ValueError):
        sdr.LeakyIntegrator(-0.1)
    with pytest.raises(ValueError):
        sdr.LeakyIntegrator(1.1)


def test_impulse_response():
    iir = sdr.LeakyIntegrator(0.9)
    b_truth = [0.1]
    a_truth = [1, -0.9]
    assert np.allclose(iir.b_taps, b_truth)
    assert np.allclose(iir.a_taps, a_truth)
