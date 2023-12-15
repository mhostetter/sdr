import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        sdr.MovingAverager(10.0)
    with pytest.raises(ValueError):
        sdr.MovingAverager(1)
    with pytest.raises(ValueError):
        sdr.MovingAverager(-1)


def test_impulse_response():
    fir = sdr.MovingAverager(10)
    h_truth = np.ones(10) / 10
    assert np.allclose(fir.taps, h_truth)
