"""
A module containing various measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def papr(x: npt.ArrayLike):
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
    peak = np.max(mag2)
    average = np.mean(mag2)

    return 10 * np.log10(peak / average)
