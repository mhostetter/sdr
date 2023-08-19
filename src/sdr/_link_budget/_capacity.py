"""
A module containing functions for calculating the capacity of a channel.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats

from .._conversion import linear
from .._helper import export


def Hb(x: npt.ArrayLike) -> np.ndarray:
    """
    Computes the binary entropy function $H_b(x)$.
    """
    x = np.asarray(x)

    return scipy.stats.entropy([x, 1 - x], base=2)


@export
def bsc_capacity(p: npt.ArrayLike) -> np.ndarray:
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
            plt.figure(figsize=(8, 4)); \
            plt.plot(p, C); \
            plt.xlabel("Transition probability, $p$"); \
            plt.ylabel("Capacity (bits/channel use), $C$"); \
            plt.title("Capacity of the Binary Symmetric Channel"); \
            plt.grid(True); \
            plt.tight_layout()

    Group:
        link-budget-channel-capacity
    """
    p = np.asarray(p)
    if not (np.all(0 <= p) and np.all(p <= 1)):
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    return 1 - Hb(p)


@export
def bec_capacity(p: npt.ArrayLike) -> np.ndarray:
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
            plt.figure(figsize=(8, 4)); \
            plt.plot(p, C); \
            plt.xlabel("Erasure probability, $p$"); \
            plt.ylabel("Capacity (bits/channel use), $C$"); \
            plt.title("Capacity of the Binary Erasure Channel"); \
            plt.grid(True); \
            plt.tight_layout()

    Group:
        link-budget-channel-capacity
    """
    p = np.asarray(p)
    if not (np.all(0 <= p) and np.all(p <= 1)):
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    return 1 - p


@export
def awgn_capacity(snr: npt.ArrayLike, bandwidth: float | None = None) -> np.ndarray:
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

        $$C = \log_2(1 + \frac{S}{N}) \ \ \text{bits/2D} ,$$

        where $S = \frac{1}{N} \sum_{i=0}^{N-1} \left| x_i \right|^2$ is the average signal power
        and $N = \sigma^2$ is the complex noise power. The units are bits/2D, which is equivalent to
        bits per complex channel use.

        If the channel bandwidth $B$ is specified, the channel capacity is

        $$C = B\log_2(1 + \frac{S}{N}) \ \ \text{bits/s} .$$

    Examples:
        The capacity monotonically decreases as the SNR decreases. In the limit as the SNR approaches 0
        ($-\infty$ dB), the capacity approaches 0.

        .. ipython:: python

            snr = np.linspace(-20, 20, 100); \
            C = sdr.awgn_capacity(snr)

            @savefig sdr_awgn_capacity_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(snr, C); \
            plt.xlabel("Signal-to-noise ratio (dB), $S/N$"); \
            plt.ylabel("Capacity (bits/2D), $C$"); \
            plt.title("Capacity of the AWGN Channel"); \
            plt.grid(True); \
            plt.tight_layout()

        At capacity, which occurs when $R = C$, $E_b/N_0$ is related to $E_s/N_0$ by

        $$\frac{E_b}{N_0} = \frac{1}{R} \frac{E_s}{N_0} = \frac{1}{C} \frac{E_s}{N_0} .$$

        When viewing the capacity as a function of $E_b/N_0$, the capacity approaches 0 as $E_b/N_0$ approaches
        -1.59 dB. This is the *absolute Shannon limit*.

        .. ipython:: python

            ebn0 = sdr.snr_to_ebn0(snr, C)

            @savefig sdr_awgn_capacity_2.png
            plt.figure(figsize=(8, 4)); \
            plt.semilogy(ebn0, C); \
            plt.xlabel("Bit energy to noise PSD ratio (dB), $E_b/N_0$"); \
            plt.ylabel("Capacity (bits/2D), $C$"); \
            plt.title("Capacity of the AWGN Channel"); \
            plt.grid(True, which="both"); \
            plt.tight_layout()

    Group:
        link-budget-channel-capacity
    """
    snr = np.asarray(snr)
    snr_linear = linear(snr)

    if bandwidth:
        return bandwidth * np.log2(1 + snr_linear)  # bits/s

    return np.log2(1 + snr_linear)  # bits/2D
