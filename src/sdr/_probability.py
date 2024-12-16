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
    n_vars: int,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the sum of $n$ i.i.d. random variables $X_i$.

    Arguments:
        X: The distribution of the i.i.d. random variables $X_i$.
        n_vars: The number $n$ of random variables.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the sum $Z = X_1 + X_2 + \dots + X_n$.

    Notes:
        Given a random variable $X$ with PDF $f_X(x)$, we compute the PDF of
        $Z = X_1 + X_2 + \dots + X_n$, where $X_1, X_2, \dots, X_n$ are independent and identically distributed
        (i.i.d.), as follows.

        The PDF of $Z$, denoted $f_Z(z)$, can be obtained by using the convolution formula for independent
        random variables. Specifically, the PDF of the sum of $n$ i.i.d. random variables is given by the $n$-fold
        convolution of the PDF of $X$ with itself.

        For $n = 2$, $Z = X_1 + X_2$. The PDF of $Z$ is

        $$f_Z(z) = \int_{-\infty}^\infty f_X(x) f_X(z - x) \, dx$$

        For $n > 2$, the PDF of $Z = X_1 + X_2 + \dots + X_n$ is computed recursively

        $$f_Z(z) = \int_{-\infty}^\infty f_X(x) f_{Z_{n-1}}(z - x) \, dx$$

        where $f_{Z_{n-1}}(z)$ is the PDF of the sum of $n-1$ random variables.

        For large $n$, the Central Limit Theorem may be used as an approximation. If $X_i$ have mean $\mu$ and
        variance $\sigma^2$, then $Z$ approximately follows $Z \sim \mathcal{N}(n\mu, n\sigma^2)$
        for sufficiently large $n$.

    Examples:
        Compute the distribution of the sum of two normal distributions.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            n_vars = 2
            x = np.linspace(-6, 2, 1_001)

            @savefig sdr_sum_distribution_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.sum_distribution(X, n_vars).pdf(x), label="$X_1 + X_2$"); \
            plt.hist(X.rvs((100_000, n_vars)).sum(axis=1), bins=101, density=True, histtype="step", label="$X_1 + X_2$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of two Normal distributions");

        Compute the distribution of the sum of three Rayleigh distributions.

        .. ipython:: python

            X = scipy.stats.rayleigh(scale=1)
            n_vars = 3
            x = np.linspace(0, 10, 1_001)

            @savefig sdr_sum_distribution_2.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.sum_distribution(X, n_vars).pdf(x), label="$X_1 + X_2 + X_3$"); \
            plt.hist(X.rvs((100_000, n_vars)).sum(axis=1), bins=101, density=True, histtype="step", label="$X_1 + X_2 + X_3$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of three Rayleigh distributions");

        Compute the distribution of the sum of four Rician distributions.

        .. ipython:: python

            X = scipy.stats.rice(2)
            n_vars = 4
            x = np.linspace(0, 18, 1_001)

            @savefig sdr_sum_distribution_3.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.sum_distribution(X, n_vars).pdf(x), label="$X_1 + X_2 + X_3 + X_4$"); \
            plt.hist(X.rvs((100_000, n_vars)).sum(axis=1), bins=101, density=True, histtype="step", label="$X_1 + X_2 + X_3 + X_4$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of four Rician distributions");

    Group:
        probability
    """
    verify_scalar(n_vars, int=True, positive=True)
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    if n_vars == 1:
        return X

    # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
    # is p.
    z1_min, z1_max = _x_range(X, p)
    z = np.linspace(z1_min, z1_max, 1_001)
    dz = np.mean(np.diff(z))

    # Compute the PDF of the base distribution
    f_X = X.pdf(z)

    # The PDF of the sum of n_vars independent random variables is the convolution of the PDF of the base distribution.
    # This is efficiently computed in the frequency domain by exponentiating the FFT. The FFT must be zero-padded
    # enough that the circular convolutions do not wrap around.
    n_fft = scipy.fft.next_fast_len(f_X.size * n_vars)
    f_X_fft = np.fft.fft(f_X, n_fft)
    f_X_fft = f_X_fft**n_vars
    f_Z = np.fft.ifft(f_X_fft).real
    f_Z /= f_Z.sum() * dz
    z = np.arange(f_Z.size) * dz + z[0] * (n_vars)

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    z = np.append(z, z[-1] + dz)
    z -= dz / 2

    return scipy.stats.rv_histogram((f_Z, z))


@export
def sum_distributions(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    Y: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the sum of two independent random variables $X$ and $Y$.

    Arguments:
        X: The distribution of the random variable $X$.
        Y: The distribution of the random variable $Y$.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the sum $Z = X + Y$.

    Notes:
        Given two independent random variables $X$ and $Y$ with PDFs $f_X(x)$ and $f_Y(y)$, we compute the PDF of
        $Z = X + Y$ as follows.

        The PDF of $Z$, denoted $f_Z(z)$, can be obtained using the convolution formula for independent random
        variables.

        $$f_Z(z) = \int_{-\infty}^\infty f_X(x) f_Y(z - x) \, dx$$

    Examples:
        Compute the distribution of the sum of Normal and Rayleigh random variables.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            Y = scipy.stats.rayleigh(loc=1, scale=2)
            x = np.linspace(-4, 10, 1_001)

            @savefig sdr_sum_distributions_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.sum_distributions(X, Y).pdf(x), label="$X + Y$"); \
            plt.hist(X.rvs(100_000) + Y.rvs(100_000), bins=101, density=True, histtype="step", label="$X + Y$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of Normal and Rayleigh random variables");

        Compute the distribution of the sum of Rayleigh and Rician random variables.

        .. ipython:: python

            X = scipy.stats.rayleigh(scale=1)
            Y = scipy.stats.rice(3)
            x = np.linspace(0, 12, 1_001)

            @savefig sdr_sum_distributions_2.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.sum_distributions(X, Y).pdf(x), label="$X + Y$"); \
            plt.hist(X.rvs(100_000) + Y.rvs(100_000), bins=101, density=True, histtype="step", label="$X + Y$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of Rayleigh and Rician random variables");

        Compute the distribution of the sum of Rician and Chi-squared random variables.

        .. ipython:: python

            X = scipy.stats.rice(3)
            Y = scipy.stats.chi2(3)
            x = np.linspace(0, 20, 1_001)

            @savefig sdr_sum_distributions_3.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.sum_distributions(X, Y).pdf(x), label="$X + Y$"); \
            plt.hist(X.rvs(100_000) + Y.rvs(100_000), bins=101, density=True, histtype="step", label="$X + Y$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Sum of Rician and Chi-squared random variables");

    Group:
        probability
    """
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
    # is p.
    x_min, x_max = _x_range(X, p)
    y_min, y_max = _x_range(Y, p)
    dx = (x_max - x_min) / 1_000
    dy = (y_max - y_min) / 1_000
    dz = np.min([dx, dy])  # Use the smaller delta x -- must use the same dx for both distributions
    x = np.arange(x_min, x_max, dz)
    y = np.arange(y_min, y_max, dz)

    # Compute the PDF of each distribution
    f_X = X.pdf(x)
    f_Y = Y.pdf(y)

    # The PDF of the sum of two independent random variables is the convolution of the PDF of the two distributions
    f_Z = np.convolve(f_X, f_Y, mode="full") * dz

    # Determine the x axis for the output convolution
    z = np.arange(f_Z.size) * dz + x[0] + y[0]

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    z = np.append(z, z[-1] + dz)
    z -= dz / 2

    return scipy.stats.rv_histogram((f_Z, z))


@export
def multiply_distributions(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    Y: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    z: npt.ArrayLike | None = None,
    p: float = 1e-10,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the product of two independent random variables $X$ and $Y$.

    Arguments:
        X: The distribution of the random variable $X$.
        Y: The distribution of the random variable $Y$.
        z: The $z$ values at which to evaluate the PDF of $Z$. If None, the $z$ values are determined based on `p`.
        p: The probability of exceeding the x axis, on either side, for each distribution. This is used to determine
            the bounds on the x axis for the numerical convolution. Smaller values of $p$ will result in more accurate
            analysis, but will require more computation.

    Returns:
        The distribution of the product $Z = X \cdot Y$.

    Notes:
        Given two independent random variables $X$ and $Y$ with PDFs $f_X(x)$ and $f_Y(y)$, and CDFs $F_X(x)$ and
        $F_Y(y)$, we compute the PDF of $Z = X \cdot Y$ as follows.

        The PDF of $Z$, denoted $f_Z(z)$, can be derived using the joint distribution of $X$ and $Y$. Since
        $Z = X \cdot Y$, we express the relationship between $x$, $y$, and $z$ and use a transformation approach.

        Let $Z = X \cdot Y$. The PDF $f_Z(z)$ is given by

        $$
        f_Z(z) = \int_{-\infty}^\infty \frac{1}{|y|} f_X\left(\frac{z}{y}\right) f_Y(y) \, dy
        $$

        The Jacobian adjustment for this transformation contributes the factor $\frac{1}{|y|}$.

    Examples:
        Compute the distribution of the product of two Normal random variables.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            Y = scipy.stats.norm(loc=2, scale=1.5)
            x = np.linspace(-15, 10, 1_001)

            @savefig sdr_multiply_distributions_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, Y.pdf(x), label="Y"); \
            plt.plot(x, sdr.multiply_distributions(X, Y).pdf(x), label=r"$X \cdot Y$"); \
            plt.hist(X.rvs(100_000) * Y.rvs(100_000), bins=101, density=True, histtype="step", label=r"$X \cdot Y$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Product of two Normal random variables");

    Group:
        probability
    """
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    if z is None:
        # Determine the z axis of each distribution such that the probability of exceeding the z axis, on either side,
        # is p.
        x_min, x_max = _x_range(X, np.sqrt(p))
        y_min, y_max = _x_range(Y, np.sqrt(p))
        bounds = np.array([x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max])
        z_min = np.min(bounds)
        z_max = np.max(bounds)
        z = np.linspace(z_min, z_max, 1_001)
    else:
        z = verify_arraylike(z, float=True, atleast_1d=True, ndim=1)
        z = np.sort(z)
    dz = np.mean(np.diff(z))

    def integrand(y: float, z: float) -> float:
        return 1 / np.abs(y) * X.pdf(z / y) * Y.pdf(y)

    f_Z = np.zeros_like(z)
    for i, zi in enumerate(z):
        f_Z[i] = scipy.integrate.quad(integrand, -np.inf, np.inf, args=(zi,))[0]

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    z = np.append(z, z[-1] + dz)
    z -= dz / 2

    return scipy.stats.rv_histogram((f_Z, z))


@export
def max_distribution(
    X: scipy.stats.rv_continuous | scipy.stats.rv_histogram,
    n_vars: int,
    p: float = 1e-16,
) -> scipy.stats.rv_histogram:
    r"""
    Numerically calculates the distribution of the maximum of $n$ i.i.d. random variables $X_i$.

    Arguments:
        X: The distribution of the i.i.d. random variables $X_i$.
        n_vars: The number $n$ of random variables.
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
        $P(Z \leq z) = \prod_{i=1}^n P(X_i \leq z)$. So, $F_Z(z) = [F_X(z)]^n$.

        The PDF of $Z$, denoted $f_Z(z)$, is the derivative of $F_Z(z)$. Therefore, $f_Z(z) = \frac{d}{dz} F_Z(z)$.
        Substituting $F_Z(z) = [F_X(z)]^n$ yields $f_Z(z) = n \cdot [F_X(z)]^{n-1} \cdot f_X(z)$.

        Therefore, the PDF of $Z = \max(X_1, X_2, \dots, X_n)$ is

        $$f_Z(z) = n \cdot [F_X(z)]^{n-1} \cdot f_X(z)$$

        where $F_X(z)$ is the CDF of the original random variable $X$, $f_X(z)$ is the PDF of $X$, and $n$ is the
        number of samples.

    Examples:
        Compute the distribution of the maximum of two Normal random variables.

        .. ipython:: python

            X = scipy.stats.norm(loc=-1, scale=0.5)
            n_vars = 2
            x = np.linspace(-3, 1, 1_001)

            @savefig sdr_max_distribution_1.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.max_distribution(X, n_vars).pdf(x), label=r"$\text{max}(X_1, X_2)$"); \
            plt.hist(X.rvs((100_000, n_vars)).max(axis=1), bins=101, density=True, histtype="step", label=r"$\text{max}(X_1, X_2)$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Maximum of two Normal random variables");

        Compute the distribution of the maximum of ten Rayleigh random variables.

        .. ipython:: python

            X = scipy.stats.rayleigh(scale=1)
            n_vars = 10
            x = np.linspace(0, 6, 1_001)

            @savefig sdr_max_distribution_2.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.max_distribution(X, n_vars).pdf(x), label="$\\text{max}(X_1, \dots, X_3)$"); \
            plt.hist(X.rvs((100_000, n_vars)).max(axis=1), bins=101, density=True, histtype="step", label="$\\text{max}(X_1, \dots, X_3)$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Maximum of ten Rayleigh random variables");

        Compute the distribution of the maximum of 100 Rician random variables.

        .. ipython:: python

            X = scipy.stats.rice(2)
            n_vars = 100
            x = np.linspace(0, 8, 1_001)

            @savefig sdr_max_distribution_3.png
            plt.figure(); \
            plt.plot(x, X.pdf(x), label="X"); \
            plt.plot(x, sdr.max_distribution(X, n_vars).pdf(x), label=r"$\text{max}(X_1, \dots, X_{100})$"); \
            plt.hist(X.rvs((100_000, n_vars)).max(axis=1), bins=101, density=True, histtype="step", label=r"$\text{max}(X_1, \dots, X_{100})$ empirical"); \
            plt.legend(); \
            plt.xlabel("Random variable"); \
            plt.ylabel("Probability density"); \
            plt.title("Maximum of 100 Rician random variables");

    Group:
        probability
    """
    verify_scalar(n_vars, int=True, positive=True)
    verify_scalar(p, float=True, exclusive_min=0, inclusive_max=0.1)

    if n_vars == 1:
        return X

    # Determine the x axis of each distribution such that the probability of exceeding the x axis, on either side,
    # is p.
    z1_min, z1_max = _x_range(X, p)
    z = np.linspace(z1_min, z1_max, 1_001)
    dz = np.mean(np.diff(z))

    # Compute the PDF and CDF of the base distribution
    f_X = X.pdf(z)
    F_X = X.cdf(z)

    f_Z = n_vars * F_X ** (n_vars - 1) * f_X

    # Adjust the histograms bins to be on either side of each point. So there is one extra point added.
    z = np.append(z, z[-1] + dz)
    z -= dz / 2

    return scipy.stats.rv_histogram((f_Z, z))


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
