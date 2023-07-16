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
    r"""
    Computes the complementary cumulative distribution function $Q(x)$
    of the standard normal distribution $\mathcal{N}(0, 1)$.

    Arguments:
        x: The real-valued input $x$.

    Returns:
        The probability $p$ that $x$ is exceeded.

    See Also:
        sdr.Qinv

    Examples:
        .. ipython:: python

            sdr.Q(1)
            sdr.Qinv(0.158655)

    Group:
        probability
    """
    x = np.asarray(x)
    p = scipy.special.erfc(x / np.sqrt(2)) / 2  # pylint: disable=no-member
    return p


@export
def Qinv(p: npt.ArrayLike) -> np.ndarray:
    r"""
    Computes the inverse complementary cumulative distribution function $Q^{-1}(p)$
    of the standard normal distribution $\mathcal{N}(0, 1)$.

    Arguments:
        p: The probability $p$ of exceeding the returned value $x$.

    Returns:
        The real-valued $x$ that is exceeded with probability $p$.

    See Also:
        sdr.Q

    Examples:
        .. ipython:: python

            sdr.Qinv(0.158655)
            sdr.Q(1)

    Group:
        probability
    """
    p = np.asarray(p)
    x = np.sqrt(2) * scipy.special.erfcinv(2 * p)  # pylint: disable=no-member
    return x
