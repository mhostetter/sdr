import numpy as np

import sdr


def test_zero_error():
    x = np.exp(1j * 2 * np.pi * 5 * np.arange(100) / 100)
    y = sdr.sample_rate_offset(x, 0)

    assert x.shape == y.shape
    assert np.allclose(x, y)
