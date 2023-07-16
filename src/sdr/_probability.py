"""
A module containing various probability functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.special

from ._helper import export


@export
def Q(x: npt.ArrayLike) -> np.ndarray:
    """
    Computes the complementary cumulative distribution function $Q(x)$ of the standard normal distribution.

    Arguments:
        x: The real-valued input $x$.

    Returns:
        The probability $p$ that $x$ is exceeded by samples drawn from \mathcal{N}(0, 1)$.

    Group:
        probability
    """
    x = np.asarray(x)
    p = scipy.special.erfc(x / np.sqrt(2)) / 2  # pylint: disable=no-member
    return p
