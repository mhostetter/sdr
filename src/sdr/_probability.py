"""
A module containing various probability functions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.special

from ._helper import convert_output, export, verify_arraylike


@export
def Q(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Computes the CCDF of the standard normal distribution $\mathcal{N}(0, 1)$..

    The complementary cumulative distribution function (CCDF) $Q(x)$ is the probability that a random variable
    exceeds a given value.

    Arguments:
        x: The real-valued input $x$.

    Returns:
        The probability $p$ that $x$ is exceeded.

    See Also:
        sdr.Qinv

    Examples:
        .. ipython:: python

            sdr.Q(1)
            sdr.Qinv(0.158655)

    Group:
        probability
    """
    x = verify_arraylike(x, float=True)

    p = scipy.special.erfc(x / np.sqrt(2)) / 2

    return convert_output(p)


@export
def Qinv(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Computes the inverse CCDF of the standard normal distribution $\mathcal{N}(0, 1)$.

    The inverse complementary cumulative distribution function (CCDF) $Q^{-1}(p)$ is the value that is exceeded
    with a given probability.

    Arguments:
        p: The probability $p$ of exceeding the returned value $x$.

    Returns:
        The real-valued $x$ that is exceeded with probability $p$.

    See Also:
        sdr.Q

    Examples:
        .. ipython:: python

            sdr.Qinv(0.158655)
            sdr.Q(1)

    Group:
        probability
    """
    p = verify_arraylike(p, float=True)

    x = np.sqrt(2) * scipy.special.erfcinv(2 * p)

    return convert_output(x)


@export
def sum_distributions(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    Y: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the sum of two independent random variables.

    Arguments:
        X: The distribution of the first random variable $X$.
        Y: The distribution of the second random variable $Y$.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the sum $Z = X + Y$.

    Notes:
        The PDF of the sum of two independent random variables is the convolution of the PDF of the two distributions.

        $$f_{X+Y}(t) = (f_X * f_Y)(t)$$

    Examples:
        Compute the distribution of the sum of two normal distributions.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            Y = scipy.stats.norm(loc=2, scale=1.5)
            x = np.linspace(-5, 10, 1000)

            @savefig sdr_sum_distributions_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.sum_distributions(X, Y).pdf(x), label="X + Y"); \
            plt.hist(X.rvs(100_000) + Y.rvs(100_000), bins=101, density=True, histtype="step", label="X + Y empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of two Normal distributions");

        Compute the distribution of the sum of two Rayleigh distributions.

        .. ipython:: python

            X = scipy.stats.rayleigh(scale=1)
            Y = scipy.stats.rayleigh(loc=1, scale=2)
            x = np.linspace(0, 12, 1000)

            @savefig sdr_sum_distributions_2.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.sum_distributions(X, Y).pdf(x), label="X + Y"); \
            plt.hist(X.rvs(100_000) + Y.rvs(100_000), bins=101, density=True, histtype="step", label="X + Y empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of two Rayleigh distributions");

        Compute the distribution of the sum of two Rician distributions.

        .. ipython:: python

            X = scipy.stats.rice(2)
            Y = scipy.stats.rice(3)
            x = np.linspace(0, 12, 1000)

            @savefig sdr_sum_distributions_3.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.sum_distributions(X, Y).pdf(x), label="X + Y"); \
            plt.hist(X.rvs(100_000) + Y.rvs(100_000), bins=101, density=True, histtype="step", label="X + Y empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of two Rician distributions");

    Group:
        probability
    """
    # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
    # is p.
    x1_min, x1_max = _x_range(X, p)
    x2_min, x2_max = _x_range(Y, p)
    dx1 = (x1_max - x1_min) / 1_000
    dx2 = (x2_max - x2_min) / 1_000
    dx = np.min([dx1, dx2])  # Use the smaller delta x -- must use the same dx for both distributions
    x1 = np.arange(x1_min, x1_max, dx)
    x2 = np.arange(x2_min, x2_max, dx)

    # Compute the PDF of each distribution
    f_X = X.pdf(x1)
    f_Y = Y.pdf(x2)

    # The PDF of the sum of two independent random variables is the convolution of the PDF of the two distributions
    f_Z = np.convolve(f_X, f_Y, mode="full") * dx

    # Determine the x axis for the output convolution
    x = np.arange(f_Z.size) * dx + x1[0] + x2[0]

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    x = np.append(x, x[-1] + dx)
    x -= dx / 2

    return scipy.stats.rv_histogram((f_Z, x))


def _x_range(X: scipy.stats.rv_continuous, p: float) -> tuple[float, float]:
    r"""
    Determines the range of x values for a given distribution such that the probability of exceeding the x axis, on
    either side, is p.
    """
    # Need to have these loops because for very small p, sometimes SciPy will return NaN instead of a valid value.
    # The while loops will increase the p value until a valid value is returned.

    pp = p
    while True:
        x_min = X.ppf(pp)
        if not np.isnan(x_min):
            break
        pp *= 10

    pp = p
    while True:
        x_max = X.isf(pp)
        if not np.isnan(x_max):
            break
        pp *= 10

    return x_min, x_max
