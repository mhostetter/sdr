"""
A module containing functions for calculating the capacity of a channel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.stats

from .._conversion import linear
from .._helper import export


def Hb(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Computes the binary entropy function $H_b(x)$.
    """
    x = np.asarray(x)

    return scipy.stats.entropy([x, 1 - x], base=2)  # type: ignore


@export
def bsc_capacity(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Calculates the capacity of a binary symmetric channel (BSC).

    Arguments:
        p: The transition probability $p$ of the BSC channel.

    Returns:
        The capacity $C$ of the channel in bits/channel use.

    Notes:
        The inputs to the BSC are $x_i \in \{0, 1\}$ and the outputs are $y_i \in \{0, 1\}$.
        The capacity of the BSC is

        $$C = 1 - H_b(p) \ \ \text{bits/channel use} .$$

    Examples:
        When the probability $p$ of bit error is 0, the capacity of the channel is 1 bit/channel use.
        However, as the probability of bit error approaches 0.5, the capacity of the channel approaches
        0.

        .. ipython:: python

            p = np.linspace(0, 1, 100); \
            C = sdr.bsc_capacity(p)

            @savefig sdr_bsc_capacity_1.png
            plt.figure(); \
            plt.plot(p, C); \
            plt.xlabel("Transition probability, $p$"); \
            plt.ylabel("Capacity (bits/channel use), $C$"); \
            plt.title("Capacity of the Binary Symmetric Channel");

    Group:
        link-budget-channel-capacity
    """
    p = np.asarray(p)
    if not (np.all(0 <= p) and np.all(p <= 1)):
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    return 1 - Hb(p)


@export
def bec_capacity(p: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Calculates the capacity of a binary erasure channel (BEC).

    Arguments:
        p: The erasure probability $p$ of the BEC channel.

    Returns:
        The capacity $C$ of the channel in bits/channel use.

    Notes:
        The inputs to the BEC are $x_i \in \{0, 1\}$ and the outputs are $y_i \in \{0, 1, e\}$.
        Erasures $e$ are represented by -1. The capacity of the BEC is

        $$C = 1 - p \ \ \text{bits/channel use} .$$

    Examples:
        When the probability $p$ of erasure is 0, the capacity of the channel is 1 bit/channel use.
        However, as the probability of erasure approaches 1, the capacity of the channel approaches
        0.

        .. ipython:: python

            p = np.linspace(0, 1, 100); \
            C = sdr.bec_capacity(p)

            @savefig sdr_bec_capacity_1.png
            plt.figure(); \
            plt.plot(p, C); \
            plt.xlabel("Erasure probability, $p$"); \
            plt.ylabel("Capacity (bits/channel use), $C$"); \
            plt.title("Capacity of the Binary Erasure Channel");

    Group:
        link-budget-channel-capacity
    """
    p = np.asarray(p)
    if not (np.all(0 <= p) and np.all(p <= 1)):
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    return 1 - p  # type: ignore


@export
def awgn_capacity(snr: npt.ArrayLike, bandwidth: float | None = None) -> npt.NDArray[np.float64]:
    r"""
    Calculates the capacity of an additive white Gaussian noise (AWGN) channel.

    Arguments:
        snr: The signal-to-noise ratio $S / N$ in dB of the channel.
        bandwidth: The bandwidth $B$ of the channel in Hz. If specified, the capacity is calculated in bits/s.
            If `None`, the capacity is calculated in bits/2D.

    Returns:
        The capacity $C$ of the channel in bits/2D, or bits/s if bandwidth was specified.

    Notes:
        The inputs to the AWGN channel are $x_i \in \mathbb{C}$ and the outputs are $y_i \in \mathbb{C}$.
        The capacity of the AWGN channel is

        $$C = \log_2\left(1 + \frac{S}{N}\right) \ \ \text{bits/2D} ,$$

        where $S = \frac{1}{N} \sum_{i=0}^{N-1} \left| x_i \right|^2$ is the average signal power
        and $N = \sigma^2$ is the complex noise power. The units are bits/2D, which is equivalent to
        bits per complex channel use.

        If the channel bandwidth $B$ is specified, the channel capacity is

        $$C = B\log_2\left(1 + \frac{S}{N}\right) \ \ \text{bits/s} .$$

    Examples:
        The capacity monotonically decreases as the SNR decreases. In the limit as the SNR approaches 0
        ($-\infty$ dB), the capacity approaches 0.

        .. ipython:: python

            snr = np.linspace(-20, 20, 100); \
            C = sdr.awgn_capacity(snr)

            @savefig sdr_awgn_capacity_1.png
            plt.figure(); \
            plt.plot(snr, C); \
            plt.xlabel("Signal-to-noise ratio (dB), $S/N$"); \
            plt.ylabel("Capacity (bits/2D), $C$"); \
            plt.title("Capacity of the AWGN Channel");

        At capacity, which occurs when $R = C$, $E_b/N_0$ is related to $E_s/N_0$ by

        $$\frac{E_b}{N_0} = \frac{1}{R} \frac{E_s}{N_0} = \frac{1}{C} \frac{E_s}{N_0} .$$

        When viewing the capacity as a function of $E_b/N_0$, the capacity approaches 0 as $E_b/N_0$ approaches
        -1.59 dB. This is the *absolute Shannon limit*.

        .. ipython:: python

            ebn0 = sdr.snr_to_ebn0(snr, C)

            @savefig sdr_awgn_capacity_2.png
            plt.figure(); \
            plt.semilogy(ebn0, C); \
            plt.xlabel("Bit energy-to-noise PSD ratio (dB), $E_b/N_0$"); \
            plt.ylabel("Capacity (bits/2D), $C$"); \
            plt.title("Capacity of the AWGN Channel");

    Group:
        link-budget-channel-capacity
    """
    snr = np.asarray(snr)
    snr_linear = linear(snr)

    if bandwidth:
        return bandwidth * np.log2(1 + snr_linear)  # bits/s

    return np.log2(1 + snr_linear)  # bits/2D


@export
def biawgn_capacity(snr: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Calculates the capacity of a binary-input additive white Gaussian noise (BI-AWGN) channel.

    Arguments:
        snr: The signal-to-noise ratio $S / N = A^2 / \sigma^2$ at the output of the channel in dB.

            .. note::
                This SNR is for a real signal. In the case of soft-decision BPSK, the SNR after coherent detection is
                $S / N = 2 E_s / N_0$, where $E_s$ is the energy per symbol and $N_0$ is the noise power spectral
                density.

    Returns:
        The capacity $C$ of the channel in bits/1D.

    Notes:
        The BI-AWGN channel is defined as

        $$Y = A \cdot X + W ,$$

        where $X$ is the input with $x_i \in \{-1, 1\}$, $A$ is the signal amplitude, $W \sim \mathcal{N}(0, \sigma^2)$
        is the AWGN noise, the SNR is $A^2 / \sigma^2$, and $Y$ is the output with $y_i \in \mathbb{R}$.

        The capacity of the BI-AWGN channel is

        $$C = \max_{f_X} I(X; Y)$$

        $$I(X; Y) = H(Y) - H(Y | X) ,$$

        where $I(X; Y)$ is the mutual information between the input $X$ and output $Y$, $H(Y)$ is the differential
        entropy of the output $Y$, and $H(Y | X)$ is the conditional entropy of the output $Y$ given the input $X$.
        The maximum mutual information is achieved when $X$ is equally likely to be $-1$ or $1$.

        The conditional probability density function (PDF) of $Y$ given $X = x$ is

        $$f_{Y|X}(y|x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(y - A x)^2}{2\sigma^2} \right) .$$

        The marginal PDF of $Y$ is

        $$f_Y(y) = \frac{1}{2} f_{Y|X}(y|1) + \frac{1}{2} f_{Y|X}(y|-1) .$$

        The differential entropy of the output $Y$ is

        $$H(Y) = - \int_{-\infty}^{\infty} f_Y(y) \log f_Y(y) \, dy .$$

        The conditional entropy of the output $Y$ given the input $X$ is

        $$H(Y | X) = - \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{Y|X}(y|x) \log f_{Y|X}(y|x) \, dy \, dx .$$

        In this case, since $Y = X + W$, the conditional entropy simplifies to

        $$H(Y | X) = H(W) = \frac{1}{2} \log(2\pi e \sigma^2) .$$

        The capacity of the BI-AWGN channel is

        $$
        \begin{align*}
        C
        &= H(Y) - H(Y | X) \\
        &= - \int_{-\infty}^{\infty} f_Y(y) \log f_Y(y) \, dy - \frac{1}{2} \log(2\pi e \sigma^2) .
        \end{align*}
        $$

        However, the integral must be solved using numerical methods. A convenient closed-form upper bound,
        provided by Yang, is

        $$C \leq C_{ub} = \log(2) - \log(1 + e^{-\text{SNR}}) ,$$

        where $\text{SNR} = A^2 / \sigma^2$ in linear units.

    References:
        - `Giuseppe Durisi, Chapter 3: The binary-input AWGN channel.
          <https://gdurisi.github.io/fbl-notes/bi-awgn.html>`_
        - `Pei Yang, A simple upper bound on the capacity of BI-AWGN channel.
          <https://www.jstage.jst.go.jp/article/comex/6/8/6_2017XBL0074/_pdf/-char/en>`_
        - `Tomáš Filler, Binary Additive White-Gaussian-Noise Channel.
          <https://dde.binghamton.edu/filler/mct/lectures/25/mct-lect25-bawgnc.pdf>`_

    Examples:
        .. ipython:: python

            snr = np.linspace(-30, 20, 51)

            C_ub = np.log2(2) - np.log2(1 + np.exp(-sdr.linear(snr)))
            C = sdr.biawgn_capacity(snr)

            @savefig sdr_biawgn_capacity_1.png
            plt.figure(); \
            plt.plot(snr, C_ub, linestyle="--", label="$C_{ub}$"); \
            plt.plot(snr, C, label="$C$"); \
            plt.legend(); \
            plt.xlabel("Signal-to-noise ratio (dB), $A^2/\sigma^2$"); \
            plt.ylabel("Capacity (bits/1D), $C$"); \
            plt.title("Capacity of the BI-AWGN Channel");

        .. ipython:: python

            snr = np.linspace(-30, 20, 51)
            p = sdr.Q(np.sqrt(sdr.linear(snr)))
            C_hard = sdr.bsc_capacity(p)
            C_soft = sdr.biawgn_capacity(snr)

            @savefig sdr_biawgn_capacity_2.png
            plt.figure(); \
            plt.plot(snr, C_hard, label="BSC (hard)"); \
            plt.plot(snr, C_soft, label="BI-AWGN (soft)"); \
            plt.legend(); \
            plt.xlabel("Signal-to-noise ratio (dB), $A^2/\sigma^2$"); \
            plt.ylabel("Capacity (bits/1D), $C$"); \
            plt.title("Capacity of the BSC (hard-decision) and BI-AWGN (soft-decision) Channels");

    Group:
        link-budget-channel-capacity
    """

    @np.vectorize
    def _calculate(snr: float) -> float:
        sigma2 = 1  # Noise power (variance), sigma^2
        A2 = linear(snr) * sigma2  # Signal power, A^2

        # f_Y|X(y | 1) is the PDF of Y given X = 1 was sent
        f_y_p1 = scipy.stats.norm(np.sqrt(A2), np.sqrt(sigma2)).pdf

        # f_Y|X(y | -1) is the PDF of Y given X = -1 was sent
        f_y_n1 = scipy.stats.norm(-np.sqrt(A2), np.sqrt(sigma2)).pdf

        # f_Y(y) is the marginal PDF of Y. This assumes equal probability of X = 0 and X = 1, which is required to
        # maximize the mutual information I(X; Y) and achieve capacity.
        def f_y(y):
            return 0.5 * f_y_p1(y) + 0.5 * f_y_n1(y)

        # H(Y | X) is the conditional entropy of the output Y given the input X. Since Y = X + W, the conditional
        # entropy is the differential entropy of the noise W.
        H_y_x = 0.5 * np.log2(2 * np.pi * np.e * sigma2)

        # H(Y) is the differential entropy of the output Y
        H_y = scipy.integrate.quad(lambda y: np.nan_to_num(-f_y(y) * np.log2(f_y(y))), -np.inf, np.inf)[0]

        # I(X; Y) is the mutual information between the input X and the output Y. This is the maximum achievable
        # mutual information and the channel capacity.
        I_x_y = H_y - H_y_x

        return I_x_y  # bits/1D

    rho = _calculate(snr)
    if rho.ndim == 0:
        rho = float(rho)

    return rho
