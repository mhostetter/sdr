import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        sdr.MovingAverage(10.0)
    with pytest.raises(ValueError):
        sdr.MovingAverage(1)
    with pytest.raises(ValueError):
        sdr.MovingAverage(-1)


def test_impulse_response():
    fir = sdr.MovingAverage(10)
    h_truth = np.ones(10) / 10
    assert np.allclose(fir.taps, h_truth)
