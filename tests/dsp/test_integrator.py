import numpy as np

import sdr


def test_gaussian():
    fir = sdr.Integrator()
    x = sdr.gaussian(0.3, 5, 10)
    y = fir(x)

    y_truth = np.cumsum(x)

    assert np.allclose(y, y_truth)


def test_raised_cosine():
    fir = sdr.Integrator()
    x = sdr.root_raised_cosine(0.1, 8, 10)
    y = fir(x)

    y_truth = np.cumsum(x)

    assert np.allclose(y, y_truth)
