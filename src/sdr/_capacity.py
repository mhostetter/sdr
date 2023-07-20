"""
A module containing functions for calculating the capacity of a channel.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def awgn_capacity(snr: npt.ArrayLike, bandwidth: float | None = None) -> np.ndarray:
    r"""
    Calculates the capacity of an additive white Gaussian noise (AWGN) channel.

    Arguments:
        snr: The signal-to-noise ratio $S / N$ in dB of the channel.
        bandwidth: The bandwidth $B$ of the channel in Hz. If `None`, the capacity is calculated in bits/2D.

    Returns:
        The capacity $C$ of the channel in bits/2D, or bits/s if bandwidth was specified.

    Notes:
        The inputs to the AWGN channel are $x_i \in \mathbb{C}$ and the outputs are $y_i \in \mathbb{C}$.
        The capacity of the AWGN channel is

        $$C = \log_2(1 + \frac{S}{N}) \ \ \text{bits/2D} ,$$

        where $S$ is the signal power and $N = \sigma^2$ is the complex noise power.
        The units are bits/2D, which is equivalent to bits per complex channel use.

        If the bandwidth $B$ of waveform is specified, the capacity is

        $$C = B\log_2(1 + \frac{S}{N}) \ \ \text{bits/s} .$$

    Examples:
        The capacity monotonically decreases as the SNR decreases. In the limit as the SNR approaches 0
        ($-\infty$ dB), the capacity approaches 0.

        .. ipython:: python

            esn0 = np.linspace(-20, 10, 100); \
            C = sdr.awgn_capacity(esn0)

            @savefig sdr_awgn_capacity_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(esn0, C); \
            plt.xlabel("Symbol energy to noise PSD ratio (dB), $E_s/N_0$"); \
            plt.ylabel("Capacity (bits/2D), $C$"); \
            plt.title("Capacity of the AWGN Channel"); \
            plt.grid(True); \
            plt.tight_layout()

        At capacity, $E_b/N_0$ is related to $E_s/N_0$ by

        $$\frac{E_b}{N_0} = \frac{1}{C} \frac{E_s}{N_0} .$$

        When viewing the capacity as a function of $E_b/N_0$, the capacity approaches 0 as $E_b/N_0$ approaches
        -1.59 dB. This is the absolute Shannon limit.

        .. ipython:: python

            ebn0 = esn0 - 10 * np.log10(C)

            @savefig sdr_awgn_capacity_2.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(ebn0, C); \
            plt.xlabel("Bit energy to noise PSD ratio (dB), $E_b/N_0$"); \
            plt.ylabel("Capacity (bits/2D), $C$"); \
            plt.title("Capacity of the AWGN Channel"); \
            plt.grid(True); \
            plt.tight_layout()

    Group:
        link-budget
    """
    snr = np.asarray(snr)
    snr_linear = 10 ** (snr / 10)

    if bandwidth:
        C = bandwidth * np.log2(1 + snr_linear)  # bits/s
    else:
        C = np.log2(1 + snr_linear)  # bits/2D

    return C if C.ndim > 0 else C.item()
