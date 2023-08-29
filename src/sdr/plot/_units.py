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
    max_time = np.max(np.abs(time))

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
    elif max_time > 1e-15:
        scalar = 1e15
        units = "fs"

    return units, scalar
