"""
A module containing various energy measurement functions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import db as to_db
from .._helper import export


@export
def energy(
    x: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    db: bool = False,
) -> npt.NDArray[np.float64]:
    r"""
    Measures the energy of a time-domain signal $x[n]$.

    $$E = \sum_{n=0}^{N-1} \left| x[n] \right|^2$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        axis: Axis or axes along which to compute the energy. The default is `None`, which computes the energy of
            the entire array.
        db: Indicates whether to return the result in decibels (dB).

    Returns:
        The signal energy. If `db=False`, $E$ is returned. If `db=True`, $10 \log_{10} E$ is returned.

    Group:
        measurement-energy
    """
    x = np.asarray(x)
    E = np.sum(np.abs(x) ** 2, axis=axis)
    if db:
        E = to_db(E)
    return E
