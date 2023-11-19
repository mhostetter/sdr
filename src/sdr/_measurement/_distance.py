"""
A module containing various distance measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def euclidean(
    x: npt.NDArray,
    y: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> npt.NDArray:
    r"""
    Measures the Euclidean distance between two signals $x[n]$ and $y[n]$.

    $$d = \sqrt{\sum_{n=0}^{N-1} \left| x[n] - y[n] \right|^2}$$

    Arguments:
        x: The time-domain signal $x[n]$.
        y: The time-domain signal $y[n]$.
        axis: Axis or axes along which to compute the distance. The default is `None`, which computes the distance
            across the entire array.

    Returns:
        The Euclidean distance between $x[n]$ and $y[n]$.

    Group:
        measurement-distance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    d = np.sqrt(np.sum(np.abs(x - y) ** 2, axis=axis))
    return d
