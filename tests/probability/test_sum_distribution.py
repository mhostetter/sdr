import numpy as np
import scipy.stats

import sdr


def test_normal():
    X = scipy.stats.norm(loc=-1, scale=0.5)
    _verify(X, 2)
    _verify(X, 3)
    _verify(X, 4)


def test_rayleigh():
    X = scipy.stats.rayleigh(scale=1)
    _verify(X, 2)
    _verify(X, 3)
    _verify(X, 4)


def test_rician():
    X = scipy.stats.rice(2)
    _verify(X, 2)
    _verify(X, 3)
    _verify(X, 4)


def _verify(X, n):
    # Numerically compute the distribution
    Y = sdr.sum_distribution(X, n)

    # Empirically compute the distribution
    y = X.rvs((100_000, n)).sum(axis=1)
    hist, bins = np.histogram(y, bins=101, density=True)
    x = bins[1:] - np.diff(bins) / 2

    if False:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, X.pdf(x), label="X")
        plt.plot(x, Y.pdf(x), label="X + ... + X")
        plt.hist(y, bins=101, cumulative=False, density=True, histtype="step", label="X + .. + X empirical")
        plt.legend()
        plt.xlabel("Random variable")
        plt.ylabel("Probability density")
        plt.title("Sum of distribution")
        plt.show()

    assert np.allclose(Y.pdf(x), hist, atol=np.max(hist) * 0.1)
