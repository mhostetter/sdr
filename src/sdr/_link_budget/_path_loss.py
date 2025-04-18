"""
A module containing functions for calculating path losses.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.constants

from .._conversion import db
from .._helper import convert_output, export, verify_arraylike


@export
def free_space_path_loss(
    distance: npt.ArrayLike,
    freq: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the free-space path loss (FSPL) in dB.

    $$\text{FSPL} = 10 \log_{10} \left( \frac{4 \pi d f}{c} \right)^2$$

    Arguments:
        distance: The distance $d$ in meters between the transmitter and receiver.
        freq: The frequency $f$ in Hz of the signal.

    Returns:
        The free-space path loss (FSPL) in dB.

    Note:
        The free-space path loss equation is only valid in the far field. For $d < \lambda / 4 \pi$, the FSPL
        equation has a negative result, implying a path gain. This is not possible. So these path losses
        are set to 0 dB.

    Examples:
        Compute the free-space path loss for a 1 km link at 1 GHz.

        .. ipython:: python

            sdr.free_space_path_loss(1e3, 1e9)

        The free-space path loss is proportional to the square of the distance. So, doubling the distance
        results in a 6 dB increase in the free-space path loss.

        .. ipython:: python

            sdr.free_space_path_loss(2e3, 1e9)

        The free-space path loss is also proportional to the square of the frequency. So, doubling the frequency
        results in a 6 dB increase in the free-space path loss.

        .. ipython:: python

            sdr.free_space_path_loss(1e3, 2e9)

        Plot the free-space path loss at 1 GHz for distances up to 1 km.

        .. ipython:: python

            d = np.linspace(0, 1_000, 1000)  # m
            fspl = sdr.free_space_path_loss(d, 1e9)  # dB

            @savefig sdr_fspl_1.svg
            plt.figure(); \
            plt.plot(d, fspl); \
            plt.xlabel('Distance (m)'); \
            plt.ylabel('Free-space path loss (dB)');

        It is confusing why free-space path loss is proportional to the square of frequency. Is RF energy attenuated
        more at higher frequencies? The answer is no. The reason the FSPL equation has a frequency
        dependence is that it assumes omnidirectional, isotropic antennas are used at both the transmitter and
        receiver. The isotropic antenna has a gain of 0 dBi. The physical size of a roughly isotropic antenna
        (think dipole) is a function of frequency as well. So, as the frequency increases, the physical size of the
        isotropic antenna decreases. But what if the size of the antenna was fixed across frequency, as is the case
        with a parabolic dish antenna? You'll note that the gain of a parabolic dish antenna is also proportional to
        the square of the frequency, see :func:`sdr.parabolic_antenna`.

        It turns out that if omnidirectional antennas are used at both the transmitter and receiver, the total path
        loss increases with frequency. But if a parabolic reflector is used at one end, the total path loss is
        constant across frequency. Furthermore, if a parabolic reflector is used at both ends, as is the case in
        VSAT systems, the total path loss decreases with frequency.

        .. ipython:: python

            freq = np.linspace(1e6, 40e9, 1_001)

            # Free-space path loss at 1 km
            fspl = sdr.free_space_path_loss(1e3, freq)

            # Isotropic antenna gain in dBi
            iso = 0

            # 1-meter diameter parabolic dish antenna gain in dBi
            par = sdr.parabolic_antenna(freq, 1)[0]

            @savefig sdr_fspl_2.svg
            plt.figure(); \
            plt.plot(freq / 1e9, fspl - iso - iso, label="Isotropic -> Isotropic"); \
            plt.plot(freq / 1e9, fspl - iso - par, label="Isotropic -> Parabolic"); \
            plt.plot(freq / 1e9, fspl - par - par, label="Parabolic -> Parabolic"); \
            plt.legend(title="Antennas", loc="center right"); \
            plt.xlabel("Frequency (GHz), $f$"); \
            plt.ylabel("Path loss (dB)"); \
            plt.title("Path loss across center frequency");

    Group:
        link-budget-path-losses
    """
    distance = verify_arraylike(distance, float=True, non_negative=True)
    freq = verify_arraylike(freq, float=True, positive=True)

    # The free-space path loss equation is only valid in the far field. For very small distances, the FSPL
    # equation could have a negative result, implying a path gain. This is not possible. So for distances less
    # than lambda / (4 * pi), the path loss is set to 0 dB.
    lambda_ = scipy.constants.speed_of_light / freq
    distance = np.maximum(distance, lambda_ / (4 * np.pi))

    # The free-space path loss equation
    loss = db((4 * np.pi * distance * freq / scipy.constants.speed_of_light) ** 2)

    return convert_output(loss)
