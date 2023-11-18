import numpy as np

import sdr


def test_gaussian():
    fir = sdr.Differentiator()
    x = sdr.gaussian(0.3, 5, 10)
    y = fir(x)

    xx = np.concatenate(([0], x, [0]))
    y_truth = np.diff(xx)

    assert np.allclose(y, y_truth)


def test_raised_cosine():
    fir = sdr.Differentiator()
    x = sdr.root_raised_cosine(0.1, 8, 10)
    y = fir(x)

    xx = np.concatenate(([0], x, [0]))
    y_truth = np.diff(xx)

    assert np.allclose(y, y_truth)
