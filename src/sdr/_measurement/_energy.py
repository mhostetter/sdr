"""
A module containing various energy measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export
from .._conversion import db as to_db


@export
def energy(x: npt.ArrayLike, db: bool = False) -> float:
    r"""
    Measures the energy of a time-domain signal $x[n]$.

    $$E = \sum_{n=0}^{N-1} \left| x[n] \right|^2$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        db: Indicates whether to return the result in decibels (dB).

    Returns:
        The energy of $x[n]$ in units^2.

    Group:
        measurement-energy
    """
    x = np.asarray(x)
    E = np.sum(np.abs(x) ** 2)
    if db:
        E = to_db(E)
    return E
