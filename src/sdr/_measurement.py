"""
A module containing various measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def papr(x: npt.ArrayLike) -> float:
    r"""
    Measures the peak-to-average power ratio (PAPR) of a signal.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The PAPR of the signal $x[n]$ in dB.

    Notes:
        $$\text{PAPR} = 10 \log_{10} \frac{P_{\text{peak}}}{P_{\text{avg}}}$$

        $$P_{\text{peak}} = \max \left( \left| x[n] \right|^2 \right)$$

        $$P_{\text{avg}} = \frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2$$

    References:
        - https://en.wikipedia.org/wiki/Crest_factor

    Group:
        measurement
    """
    x = np.asarray(x)

    mag2 = np.abs(x) ** 2
    peak_power = np.max(mag2)
    average_power = np.mean(mag2)

    return 10 * np.log10(peak_power / average_power)


@export
def crest_factor(x: npt.ArrayLike) -> float:
    r"""
    Measures the crest factor of a signal.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The crest factor of the signal $x[n]$.

    Notes:
        $$\text{CF} = \frac{V_{\text{peak}}}{V_{\text{rms}}}$$

        $$V_{\text{peak}} = \max \left( \left| x[n] \right| \right)$$

        $$V_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2}$$

    References:
        - https://en.wikipedia.org/wiki/Crest_factor

    Group:
        measurement
    """
    x = np.asarray(x)

    peak_voltage = np.max(np.abs(x))
    rms_voltage = np.sqrt(np.mean(np.abs(x) ** 2))

    return peak_voltage / rms_voltage
