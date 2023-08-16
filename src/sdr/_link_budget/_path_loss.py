"""
A module containing functions for calculating path losses.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants

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

    Note:
        The free-space path loss equation is only valid in the far field. For $d < \lambda / 4 \pi$, the FSPL
        equation has a negative result, implying a path gain. This is not possible. So these path losses
        are set to 0 dB.

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

        Plot the free-space path loss at 1 GHz for distances up to 1 km.

        .. ipython:: python

            d = np.linspace(0, 1_000, 1000)  # m
            fspl = sdr.fspl(d, 1e9)  # dB

            @savefig sdr_fspl_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(d, fspl); \
            plt.xlabel('Distance (m)'); \
            plt.ylabel('Free-space path loss (dB)'); \
            plt.grid(True); \
            plt.tight_layout()

    Group:
        link-budget-path-losses
    """
    d = np.asarray(d)
    f = np.asarray(f)

    # The free-space path loss equation is only valid in the far field. For very small distances, the FSPL
    # equation could have a negative result, implying a path gain. This is not possible. So for distances less
    # than lambda / (4 * pi), the path loss is set to 0 dB.
    lambda_ = scipy.constants.speed_of_light / f
    d = np.maximum(d, lambda_ / (4 * np.pi))

    # The free-space path loss equation
    loss = 20 * np.log10(4 * np.pi * d * f / scipy.constants.speed_of_light)

    return loss
