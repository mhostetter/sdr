"""
A module to determine axis units.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def time_units(time: npt.ArrayLike) -> tuple[str, float]:
    """
    Determines the appropriate time units to use for a given time array.
    """
    time = np.asarray(time)
    max_time = np.nanmax(np.abs(time))

    if max_time > 1:
        scalar = 1
        units = "s"
    elif max_time > 1e-3:
        scalar = 1e3
        units = "ms"
    elif max_time > 1e-6:
        scalar = 1e6
        units = "μs"
    elif max_time > 1e-9:
        scalar = 1e9
        units = "ns"
    elif max_time > 1e-12:
        scalar = 1e12
        units = "ps"
    else:
        scalar = 1e15
        units = "fs"

    return units, scalar


def freq_units(freq: npt.ArrayLike) -> tuple[str, float]:
    """
    Determines the appropriate frequency units to use for a given frequency array.
    """
    freq = np.asarray(freq)
    max_freq = np.nanmax(np.abs(freq))

    if max_freq > 10e12:
        scalar = 1e-12
        units = "THz"
    elif max_freq > 10e9:
        scalar = 1e-9
        units = "GHz"
    elif max_freq > 10e6:
        scalar = 1e-6
        units = "MHz"
    elif max_freq > 10e3:
        scalar = 1e-3
        units = "kHz"
    elif max_freq > 10:
        scalar = 1
        units = "Hz"
    elif max_freq > 10e-3:
        scalar = 1e3
        units = "mHz"
    else:
        scalar = 1e6
        units = "μHz"

    return units, scalar
