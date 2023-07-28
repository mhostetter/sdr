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
        Convert from $E_b/N_0 = 10$ dB to $E_s/N_0$ for a 4-QAM signal with $r = 2/3$.

        .. ipython:: python

            sdr.ebn0_to_esn0(10, 2, rate=2/3)

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
