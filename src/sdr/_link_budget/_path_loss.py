"""
A module containing functions for calculating path losses.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._constants import SPEED_OF_LIGHT
from .._helper import export


@export
def fspl(d: npt.ArrayLike, f: npt.ArrayLike) -> np.ndarray:
    r"""
    Calculates the free-space path loss (FSPL) in dB.

    $$\text{FSPL} = 10 \log_{10} \left( \frac{4 \pi d f}{c} \right)^2$$

    Arguments:
        d: The distance $d$ in meters between the transmitter and receiver.
        f: The frequency $f$ in Hz of the signal.

    Returns:
        The free-space path loss (FSPL) in dB.

    Examples:
        Compute the free-space path loss for a 1 km link at 1 GHz.

        .. ipython:: python

            sdr.fspl(1e3, 1e9)

        The free-space path loss is proportional to the square of the distance. So, doubling the distance
        results in a 6 dB increase in the free-space path loss.

        .. ipython:: python

            sdr.fspl(2e3, 1e9)

        The free-space path loss is also proportional to the square of the frequency. So, doubling the frequency
        results in a 6 dB increase in the free-space path loss.

        .. ipython:: python

            sdr.fspl(1e3, 2e9)

    Group:
        link-budget-path-losses
    """
    d = np.asarray(d)
    f = np.asarray(f)
    loss = 20 * np.log10(4 * np.pi * d * f / SPEED_OF_LIGHT)
    return loss
