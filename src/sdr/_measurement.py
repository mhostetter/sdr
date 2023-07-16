"""
A module containing various measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def papr(x: npt.ArrayLike) -> float:
    """
    Measures the peak-to-average power ratio (PAPR) of a signal.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The PAPR of the signal $x[n]$ in dB.

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
    """
    Measures the crest factor of a signal.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The crest factor of the signal $x[n]$.

    Group:
        measurement
    """
    x = np.asarray(x)

    peak_voltage = np.max(np.abs(x))
    rms_voltage = np.sqrt(np.mean(np.abs(x) ** 2))

    return peak_voltage / rms_voltage
