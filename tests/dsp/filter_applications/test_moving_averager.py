import numpy as np

import sdr


def test_impulse_response():
    fir = sdr.MovingAverager(10)
    h_truth = np.ones(10) / 10
    assert np.allclose(fir.taps, h_truth)
