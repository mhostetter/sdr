"""
A module containing various energy measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def energy(x: npt.ArrayLike) -> float:
    r"""
    Measures the energy of a time-domain signal $x[n]$.

    $$E = \sum_{n=0}^{N-1} \left| x[n] \right|^2$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The energy of $x[n]$ in units^2.

    Group:
        measurement-energy
    """
    x = np.asarray(x)
    return np.sum(np.abs(x) ** 2)
