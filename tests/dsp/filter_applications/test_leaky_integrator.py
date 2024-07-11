import numpy as np

import sdr


def test_impulse_response():
    iir = sdr.LeakyIntegrator(0.9)
    b_truth = [0.1]
    a_truth = [1, -0.9]
    assert np.allclose(iir.b_taps, b_truth)
    assert np.allclose(iir.a_taps, a_truth)
