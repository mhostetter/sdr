"""
A module containing various power measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def peak_power(x: npt.ArrayLike) -> float:
    r"""
    Measures the peak power of a time-domain signal $x[n]$.

    $$P_{\text{peak}} = \max \left( \left| x[n] \right|^2 \right)$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The peak power of $x[n]$ in units^2.

    Group:
        measurement-power
    """
    x = np.asarray(x)
    return np.max(np.abs(x) ** 2)


@export
def average_power(x: npt.ArrayLike) -> float:
    r"""
    Measures the average power of a time-domain signal $x[n]$.

    $$P_{\text{avg}} = \frac{E}{N} = \frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The average power of $x[n]$ in units^2.

    Group:
        measurement-power
    """
    x = np.asarray(x)
    return np.mean(np.abs(x) ** 2)


@export
def papr(x: npt.ArrayLike) -> float:
    r"""
    Measures the peak-to-average power ratio (PAPR) of a time-domain signal $x[n]$.

    $$\text{PAPR} = 10 \log_{10} \frac{P_{\text{peak}}}{P_{\text{avg}}}$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The PAPR of $x[n]$ in dB.

    See Also:
        sdr.peak_power, sdr.average_power

    References:
        - https://en.wikipedia.org/wiki/Crest_factor

    Group:
        measurement-power
    """
    x = np.asarray(x)
    return 10 * np.log10(peak_power(x) / average_power(x))
