"""
A module containing various voltage measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def peak_voltage(x: npt.ArrayLike) -> float:
    r"""
    Measures the peak voltage of a time-domain signal $x[n]$.

    $$V_{\text{peak}} = \max \left( \left| x[n] \right| \right)$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The peak voltage of $x[n]$ in units.

    Group:
        measurement-voltage
    """
    x = np.asarray(x)
    return np.max(np.abs(x))


@export
def rms_voltage(x: npt.ArrayLike) -> float:
    r"""
    Measures the root-mean-square (RMS) voltage of a time-domain signal $x[n]$.

    $$V_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2}$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The RMS voltage of $x[n]$ in units.

    Group:
        measurement-voltage
    """
    x = np.asarray(x)
    return np.sqrt(np.mean(np.abs(x) ** 2))


@export
def crest_factor(x: npt.ArrayLike) -> float:
    r"""
    Measures the crest factor of a time-domain signal $x[n]$.

    $$\text{CF} = \frac{V_{\text{peak}}}{V_{\text{rms}}}$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The crest factor of $x[n]$.

    See Also:
        sdr.peak_voltage, sdr.rms_voltage

    References:
        - https://en.wikipedia.org/wiki/Crest_factor

    Group:
        measurement-voltage
    """
    x = np.asarray(x)
    return peak_voltage(x) / rms_voltage(x)
