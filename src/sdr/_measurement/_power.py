"""
A module containing various power measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import db as to_db
from .._helper import export


@export
def peak_power(
    x: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
    db: bool = False,
) -> float:
    r"""
    Measures the peak power of a time-domain signal $x[n]$.

    $$P_{\text{peak}} = \max \left| x[n] \right|^2$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        axis: Axis or axes along which to compute the peak power. The default is `None`, which computes the peak power
            of the entire array.
        db: Indicates whether to return the result in decibels (dB).

    Returns:
        The peak power. If `db=False`, $P_{\text{peak}}$ is returned.
        If `db=True`, $10 \log_{10} P_{\text{peak}}$ is returned.

    Group:
        measurement-power
    """
    P_peak = np.max(np.abs(x) ** 2, axis=axis)
    if db:
        P_peak = to_db(P_peak, type="power")
    return P_peak


@export
def average_power(
    x: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
    db: bool = False,
) -> float:
    r"""
    Measures the average power of a time-domain signal $x[n]$.

    $$P_{\text{avg}} = \frac{E}{N} = \frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        axis: Axis or axes along which to compute the average power. The default is `None`, which computes the average
            power of the entire array.
        db: Indicates whether to return the result in decibels (dB).

    Returns:
        The average power. If `db=False`, $P_{\text{avg}}$ is returned.
        If `db=True`, $10 \log_{10} P_{\text{avg}}$ is returned.

    Group:
        measurement-power
    """
    P_avg = np.mean(np.abs(x) ** 2, axis=axis)
    if db:
        P_avg = to_db(P_avg, type="power")
    return P_avg


@export
def papr(
    x: npt.NDArray,
    axis: int | tuple[int, ...] | None = None,
) -> float:
    r"""
    Measures the peak-to-average power ratio (PAPR) of a time-domain signal $x[n]$.

    $$\text{PAPR} = 10 \log_{10} \frac{P_{\text{peak}}}{P_{\text{avg}}}$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        axis: Axis or axes along which to compute the PAPR. The default is `None`, which computes the PAPR of the
            entire array.

    Returns:
        The PAPR of $x[n]$ in dB.

    See Also:
        sdr.peak_power, sdr.average_power

    References:
        - https://en.wikipedia.org/wiki/Crest_factor

    Group:
        measurement-power
    """
    return to_db(peak_power(x, axis=axis) / average_power(x, axis=axis))
