"""
A module that converts between various types of signal-to-noise ratios (SNRs).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export

##############################################################################
# From Eb/N0
##############################################################################


@export
def ebn0_to_esn0(ebn0: npt.ArrayLike, bps: int, rate: int = 1) -> np.ndarray:
    r"""
    Converts from $E_b/N_0$ to $E_s/N_0$.

    $$
    \frac{E_s}{N_0} = \frac{E_b}{N_0} \frac{k}{n} \log_2 M
    $$

    Arguments:
        ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.
        bps: Bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.

    Returns:
        The symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.

    Examples:
        Convert from $E_b/N_0 = 5$ dB to $E_s/N_0$ for a 4-QAM signal with $r = 2/3$.

        .. ipython:: python

            sdr.ebn0_to_esn0(5, 2, rate=2/3)

        Convert from $E_b/N_0 = 10$ dB to $E_s/N_0$ for a 16-QAM signal with $r = 1$.

        .. ipython:: python

            sdr.ebn0_to_esn0(10, 4, rate=1)

    Group:
        conversions-from-ebn0
    """
    ebn0 = np.asarray(ebn0)  # Energy per information bit
    ecn0 = ebn0 + 10 * np.log10(rate)  # Energy per coded bit
    esn0 = ecn0 + 10 * np.log10(bps)  # Energy per symbol
    return esn0


@export
def ebn0_to_snr(ebn0: npt.ArrayLike, bps: int, rate: int = 1, sps: int = 1) -> np.ndarray:
    r"""
    Converts from $E_b/N_0$ to $S/N$.

    $$
    \frac{S}{N} = \frac{E_s}{N_0} \frac{f_{sym}}{f_s} = \frac{E_b}{N_0} \frac{k}{n} \log_2 M \frac{f_{sym}}{f_s}
    $$

    Arguments:
        ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.
        bps: Bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.
        sps: Samples per symbol $f_s / f_{sym}$.

    Returns:
        The signal-to-noise ratio $S/N$ in dB.

    Examples:
        Convert from $E_b/N_0 = 5$ dB to $S/N$ for a 4-QAM signal with $r = 2/3$ and 1 sample per symbol.

        .. ipython:: python

            sdr.ebn0_to_snr(5, 2, rate=2/3, sps=1)

        Convert from $E_b/N_0 = 10$ dB to $S/N$ for a 16-QAM signal with $r = 1$ and 4 samples per symbol.

        .. ipython:: python

            sdr.ebn0_to_snr(10, 4, rate=1, sps=4)

    Group:
        conversions-from-ebn0
    """
    esn0 = ebn0_to_esn0(ebn0, bps, rate=rate)  # SNR per symbol
    snr = esn0 - 10 * np.log10(sps)  # SNR per sample
    return snr


##############################################################################
# From Es/N0
##############################################################################


@export
def esn0_to_ebn0(esn0: npt.ArrayLike, bps: int, rate: int = 1) -> np.ndarray:
    r"""
    Converts from $E_s/N_0$ to $E_b/N_0$.

    $$
    \frac{E_b}{N_0} = \frac{E_s}{N_0} \frac{n}{k} \frac{1}{\log_2 M}
    $$

    Arguments:
        esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.
        bps: Bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.

    Returns:
        The bit energy $E_b$ to noise PSD $N_0$ ratio in dB.

    Examples:
        Convert from $E_s/N_0 = 5$ dB to $E_b/N_0$ for a 4-QAM signal with $r = 2/3$.

        .. ipython:: python

            sdr.esn0_to_ebn0(5, 2, rate=2/3)

        Convert from $E_s/N_0 = 10$ dB to $E_b/N_0$ for a 16-QAM signal with $r = 1$.

        .. ipython:: python

            sdr.esn0_to_ebn0(10, 4, rate=1)

    Group:
        conversions-from-esn0
    """
    esn0 = np.asarray(esn0)
    ecn0 = esn0 - 10 * np.log10(bps)  # Energy per coded bit
    ebn0 = ecn0 - 10 * np.log10(rate)  # Energy per information bit
    return ebn0
