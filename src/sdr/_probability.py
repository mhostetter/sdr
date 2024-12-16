"""
A module containing various probability functions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.special

from ._helper import convert_output, export, verify_arraylike, verify_scalar


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
def sum_distribution(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    n_terms: int,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the sum of $n$ i.i.d. random variables $X_i$.

    Arguments:
        X: The distribution of the i.i.d. random variables $X_i$.
        n_terms: The number $n$ of random variables to sum.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the sum $Z = X_1 + X_2 + \dots + X_n$.

    Notes:
        The PDF of the sum of $n$ independent random variables is the convolution of the PDF of the base distribution.

        $$f_{X_1 + X_2 + \dots + X_n}(t) = (f_X * f_X * \dots * f_X)(t)$$

    Examples:
        Compute the distribution of the sum of two normal distributions.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            n_terms = 2
            x = np.linspace(-6, 2, 1_001)

            @savefig sdr_sum_distribution_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.sum_distribution(X, n_terms).pdf(x), label="X + X"); \
            plt.hist(X.rvs((100_000, n_terms)).sum(axis=1), bins=101, density=True, histtype="step", label="X + X empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of two Normal distributions");

        Compute the distribution of the sum of three Rayleigh distributions.

        .. ipython:: python

            X = scipy.stats.rayleigh(scale=1)
            n_terms = 3
            x = np.linspace(0, 10, 1_001)

            @savefig sdr_sum_distribution_2.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.sum_distribution(X, n_terms).pdf(x), label="X + X + X"); \
            plt.hist(X.rvs((100_000, n_terms)).sum(axis=1), bins=101, density=True, histtype="step", label="X + X + X empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of three Rayleigh distributions");

        Compute the distribution of the sum of four Rician distributions.

        .. ipython:: python

            X = scipy.stats.rice(2)
            n_terms = 4
            x = np.linspace(0, 18, 1_001)

            @savefig sdr_sum_distribution_3.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.sum_distribution(X, n_terms).pdf(x), label="X + X + X + X"); \
            plt.hist(X.rvs((100_000, n_terms)).sum(axis=1), bins=101, density=True, histtype="step", label="X + X + X + X empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of four Rician distributions");

    Group:
        probability
    """
    verify_scalar(n_terms, int=True, positive=True)
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    if n_terms == 1:
        return X

    # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
    # is p.
    x1_min, x1_max = _x_range(X, p)
    x = np.linspace(x1_min, x1_max, 1_001)
    dx = np.mean(np.diff(x))

    # Compute the PDF of the base distribution
    f_X = X.pdf(x)

    # The PDF of the sum of n_terms independent random variables is the convolution of the PDF of the base distribution.
    # This is efficiently computed in the frequency domain by exponentiating the FFT. The FFT must be zero-padded
    # enough that the circular convolutions do not wrap around.
    n_fft = scipy.fft.next_fast_len(f_X.size * n_terms)
    f_X_fft = np.fft.fft(f_X, n_fft)
    f_X_fft = f_X_fft**n_terms
    f_Y = np.fft.ifft(f_X_fft).real
    f_Y /= f_Y.sum() * dx
    x = np.arange(f_Y.size) * dx + x[0] * (n_terms)

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    x = np.append(x, x[-1] + dx)
    x -= dx / 2

    return scipy.stats.rv_histogram((f_Y, x))


@export
def sum_distributions(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    Y: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the sum of two independent random variables $X$ and $Y$.

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
            x = np.linspace(-5, 10, 1_001)

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
            x = np.linspace(0, 12, 1_001)

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
            x = np.linspace(0, 12, 1_001)

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
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

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


@export
def multiply_distributions(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    Y: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    x: npt.ArrayLike | None = None,
    p: float = 1e-10,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the product of two independent random variables $X$ and $Y$.

    Arguments:
        X: The distribution of the first random variable $X$.
        Y: The distribution of the second random variable $Y$.
        x: The x values at which to evaluate the PDF of the product. If None, the x values are determined based on `p`.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the product $Z = X \cdot Y$.

    Notes:
        The PDF of the product of two independent random variables is calculated as follows.

        $$
        f_{X \cdot Y}(t) =
        \int_{0}^{\infty} f_X(k) f_Y(t/k) \frac{1}{k} dk -
        \int_{-\infty}^{0} f_X(k) f_Y(t/k) \frac{1}{k} dk
        $$

    Examples:
        Compute the distribution of the product of two normal distributions.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            Y = scipy.stats.norm(loc=2, scale=1.5)
            x = np.linspace(-15, 10, 1_001)

            @savefig sdr_multiply_distributions_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.multiply_distributions(X, Y).pdf(x), label="X * Y"); \
            plt.hist(X.rvs(100_000) * Y.rvs(100_000), bins=101, density=True, histtype="step", label="X * Y empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Product of two Normal distributions");

    Group:
        probability
    """
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    if x is None:
        # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
        # is p.
        x1_min, x1_max = _x_range(X, np.sqrt(p))
        x2_min, x2_max = _x_range(Y, np.sqrt(p))
        bounds = np.array([x1_min * x2_min, x1_min * x2_max, x1_max * x2_min, x1_max * x2_max])
        x_min = np.min(bounds)
        x_max = np.max(bounds)
        x = np.linspace(x_min, x_max, 1_001)
    else:
        x = verify_arraylike(x, float=True, atleast_1d=True, ndim=1)
        x = np.sort(x)
    dx = np.mean(np.diff(x))

    def integrand(k: float, xi: float) -> float:
        return X.pdf(k) * Y.pdf(xi / k) * 1 / k

    f_Z = np.zeros_like(x)
    for i, xi in enumerate(x):
        f_Z[i] = scipy.integrate.quad(integrand, 0, np.inf, args=(xi,))[0]
        f_Z[i] -= scipy.integrate.quad(integrand, -np.inf, 0, args=(xi,))[0]

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    x = np.append(x, x[-1] + dx)
    x -= dx / 2

    return scipy.stats.rv_histogram((f_Z, x))


@export
def max_distribution(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    n_samples: int,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the maximum of $n$ i.i.d. random variables $X_i$.

    Arguments:
        X: The distribution of the i.i.d. random variables $X_i$.
        n_samples: The number $n$ of random variables to compute the maximum.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the sum $Z = \text{max}(X_1, X_2, \dots, X_n)$.

    Notes:
        Given a random variable $X$ with PDF $f_X(x)$ and CDF $F_X(x)$, we compute the PDF of
        $Z = \max(X_1, X_2, \dots, X_n)$, where $X_1, X_2, \dots, X_n$ are independent and identically distributed
        (i.i.d.), as follows.

        The CDF of $Z$, denoted $F_Z(z)$, is $F_Z(z) = P(Z \leq z)$. Since $Z = \max(X_1, X_2, \dots, X_n)$, the
        event $Z \leq z$ occurs if and only if all $X_i \leq z$. Using independence,
        $F_Z(z) = \prod_{i=1}^n P(X_i \leq z) = [F_X(z)]^n$.

        The PDF of $Z$, denoted $f_Z(z)$, is the derivative of $F_Z(z)$, so $f_Z(z) = \frac{d}{dz} F_Z(z)$.
        Substitute $F_Z(z) = [F_X(z)]^n$, which yields $f_Z(z) = n \cdot [F_X(z)]^{n-1} \cdot f_X(z)$.

        Therefore, the PDF of $Z = \max(X_1, X_2, \dots, X_n)$ is

        $$f_Z(z) = n \cdot [F_X(z)]^{n-1} \cdot f_X(z)$$

        where $F_X(z)$ is the CDF of the original random variable $X$, $f_X(z)$ is the PDF of $X$, and $n$ is the
        number of samples.

    Examples:
        Compute the distribution of the maximum of two samples from a normal distribution.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            n_samples = 2
            x = np.linspace(-3, 1, 1_001)

            @savefig sdr_max_distribution_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.max_distribution(X, n_samples).pdf(x), label=r"$\text{max}(X_1, X_2)$"); \
            plt.hist(X.rvs((100_000, n_samples)).max(axis=1), bins=101, density=True, histtype="step", label=r"$\text{max}(X_1, X_2)$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Maximum of two samples from a Normal distribution");

        Compute the distribution of the maximum of ten samples from a Rayleigh distribution.

        .. ipython:: python

            X = scipy.stats.rayleigh(scale=1)
            n_samples = 10
            x = np.linspace(0, 6, 1_001)

            @savefig sdr_max_distribution_2.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.max_distribution(X, n_samples).pdf(x), label="$\\text{max}(X_1, \dots, X_3)$"); \
            plt.hist(X.rvs((100_000, n_samples)).max(axis=1), bins=101, density=True, histtype="step", label="$\\text{max}(X_1, \dots, X_3)$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Maximum of ten samples from a Rayleigh distribution");

        Compute the distribution of the maximum of 100 samples from a Rician distribution.

        .. ipython:: python

            X = scipy.stats.rice(2)
            n_samples = 100
            x = np.linspace(0, 8, 1_001)

            @savefig sdr_max_distribution_3.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.max_distribution(X, n_samples).pdf(x), label=r"$\text{max}(X_1, \dots, X_{100})$"); \
            plt.hist(X.rvs((100_000, n_samples)).max(axis=1), bins=101, density=True, histtype="step", label=r"$\text{max}(X_1, \dots, X_{100})$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Maximum of 100 samples from a Rician distribution");

    Group:
        probability
    """
    verify_scalar(n_samples, int=True, positive=True)
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    if n_samples == 1:
        return X

    # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
    # is p.
    x1_min, x1_max = _x_range(X, p)
    x = np.linspace(x1_min, x1_max, 1_001)
    dx = np.mean(np.diff(x))

    # Compute the PDF and CDF of the base distribution
    f_X = X.pdf(x)
    F_X = X.cdf(x)

    f_Z = n_samples * F_X ** (n_samples - 1) * f_X

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
