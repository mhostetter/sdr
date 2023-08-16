"""
A module containing various voltage measurement functions.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import db as to_db
from .._helper import export


@export
def peak_voltage(x: npt.ArrayLike, db: bool = False) -> float:
    r"""
    Measures the peak voltage of a time-domain signal $x[n]$.

    $$V_{\text{peak}} = \max \left| x[n] \right|$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        db: Indicates whether to return the result in dB.

    Returns:
        The peak voltage. If `db=False`, $V_{\text{peak}}$ is returned.
        If `db=True`, $20 \log_{10} V_{\text{peak}}$ is returned.

    Group:
        measurement-voltage
    """
    x = np.asarray(x)
    V_peak = np.max(np.abs(x))
    if db:
        V_peak = to_db(V_peak, type="voltage")
    return V_peak


@export
def rms_voltage(x: npt.ArrayLike, db: bool = False) -> float:
    r"""
    Measures the root-mean-square (RMS) voltage of a time-domain signal $x[n]$.

    $$V_{\text{rms}} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2}$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.
        db: Indicates whether to return the result in dB.

    Returns:
        The root-mean-square voltage. If `db=False`, $V_{\text{rms}}$ is returned.
        If `db=True`, $20 \log_{10} V_{\text{rms}}$ is returned.

    Group:
        measurement-voltage
    """
    x = np.asarray(x)
    V_rms = np.sqrt(np.mean(np.abs(x) ** 2))
    if db:
        V_rms = to_db(V_rms, type="voltage")
    return V_rms


@export
def crest_factor(x: npt.ArrayLike) -> float:
    r"""
    Measures the crest factor of a time-domain signal $x[n]$.

    $$\text{CF} = \frac{V_{\text{peak}}}{V_{\text{rms}}}$$

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

    Returns:
        The crest factor of $x[n]$.

    See Also:
        sdr.peak_voltage, sdr.rms_voltage

    References:
        - https://en.wikipedia.org/wiki/Crest_factor

    Group:
        measurement-voltage
    """
    x = np.asarray(x)
    return peak_voltage(x) / rms_voltage(x)
