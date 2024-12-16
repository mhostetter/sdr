import numpy as np
import scipy.stats

import sdr


def test_normal_normal():
    X = scipy.stats.norm(loc=-1, scale=0.5)
    Y = scipy.stats.norm(loc=2, scale=1.5)
    _verify(X, Y)


def test_rayleigh_rayleigh():
    X = scipy.stats.rayleigh(scale=1)
    Y = scipy.stats.rayleigh(loc=1, scale=2)
    _verify(X, Y)


def test_rician_rician():
    X = scipy.stats.rice(2)
    Y = scipy.stats.rice(3)
    _verify(X, Y)


def test_normal_rayleigh():
    X = scipy.stats.norm(loc=-1, scale=0.5)
    Y = scipy.stats.rayleigh(loc=2, scale=1.5)
    _verify(X, Y)


def test_rayleigh_rician():
    X = scipy.stats.rayleigh(scale=1)
    Y = scipy.stats.rice(3)
    _verify(X, Y)


def _verify(X, Y):
    # Numerically compute the distribution
    Z = sdr.add_distributions(X, Y)

    # Empirically compute the distribution
    z = X.rvs(100_000) + Y.rvs(100_000)
    hist, bins = np.histogram(z, bins=101, density=True)
    x = bins[1:] - np.diff(bins) / 2

    if False:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, X.pdf(x), label="X")
        plt.plot(x, Y.pdf(x), label="Y")
        plt.plot(x, Z.pdf(x), label="X + Y")
        plt.hist(z, bins=101, cumulative=False, density=True, histtype="step", label="X + Y empirical")
        plt.legend()
        plt.xlabel("Random variable")
        plt.ylabel("Probability density")
        plt.title("Sum of two distributions")
        plt.show()

    assert np.allclose(Z.pdf(x), hist, atol=np.max(hist) * 0.1)
