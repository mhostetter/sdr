"""
A module containing functions for calculating antenna gains.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants

from .._conversion import db
from .._helper import convert_output, export, verify_arraylike


@export
def wavelength(freq: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Calculates the wavelength $\lambda$ of an electromagnetic wave with frequency $f$.

    $$\lambda = \frac{c}{f}$$

    Arguments:
        freq: The frequency $f$ in Hz of the signal.

    Returns:
        The wavelength $\lambda$ in meters.

    Examples:
        The wavelength of a 1 GHz signal is 0.3 meters.

        .. ipython:: python

            sdr.wavelength(1e9)

    Group:
        link-budget-antennas
    """
    freq = verify_arraylike(freq, float=True, positive=True)

    lambda_ = scipy.constants.speed_of_light / freq

    return convert_output(lambda_)


@export
def parabolic_antenna(
    freq: npt.ArrayLike,
    diameter: npt.ArrayLike,
    efficiency: npt.ArrayLike = 0.55,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""
    Calculates the gain $G$ and beamwidth $\theta$ of a parabolic reflector.

    Arguments:
        freq: The frequency $f$ in Hz of the signal.
        diameter: The diameter $d$ in meters of the parabolic reflector.
        efficiency: The efficiency $0 \le \eta \le 1$ of the parabolic reflector.

    Returns:
        - The gain $G$ in dBi.
        - The half-power beamwidth $\theta$ in degrees.

    Notes:
        $$G = \left( \frac{\pi d f}{c} \right)^2 \eta$$

        $$\theta = \arcsin \left( \frac{3.83 c}{\pi d f} \right)$$

    Examples:
        A 1 meter dish at 1 GHz with 55% efficiency has a gain of 17.8 dBi and a 3-dB beamwidth of 21.4 degrees.

        .. ipython:: python

            sdr.parabolic_antenna(1e9, 1)

        A 2 meter dish at 1 GHz with 55% efficiency has a gain of 23.8 dBi and a 3-dB beamwidth of 10.5 degrees.
        Since the antenna gain is proportional to the square of the reflector diameter, doubling the diameter
        results in a 6 dB increase in the gain, which we observe.

        .. ipython:: python

            sdr.parabolic_antenna(1e9, 2)

    Group:
        link-budget-antennas
    """
    freq = verify_arraylike(freq, float=True, positive=True)
    diameter = verify_arraylike(diameter, float=True, positive=True)
    efficiency = verify_arraylike(efficiency, float=True, inclusive_min=0, inclusive_max=1)

    lambda_ = wavelength(freq)  # Wavelength in meters
    G = (np.pi * diameter / lambda_) ** 2 * efficiency  # Gain in linear units
    G = db(G)  # Gain in dBi

    theta = np.arcsin(3.83 * lambda_ / (np.pi * diameter))  # Beamwidth in radians
    theta = np.rad2deg(theta)  # Beamwidth in degrees

    return convert_output(G), convert_output(theta)
