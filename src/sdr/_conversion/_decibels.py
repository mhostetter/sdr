"""
Decibel conversion functions for transforming between linear and logarithmic scales.

Provides conversions for value, power, and voltage quantities between linear units
and decibels (dB).
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import normalize_output_type, export, verify_and_validate_arraylike, verify_literal


@export
def db(
    x: npt.ArrayLike,
    type: Literal["value", "power", "voltage"] = "value",
) -> npt.NDArray[np.float64]:
    r"""
    Converts from linear units to decibels.

    Arguments:
        x: The input value or signal.
        type: The type of input value or signal.

            - `"value"`: The input value/signal is any value.

            $$x_{\text{dB}} = 10 \log_{10} x_{\text{linear}}$$

            - `"power"`: The input value/signal is a power measurement.

            $$P_{\text{dB}} = 10 \log_{10} P_{\text{linear}}$$

            - `"voltage"`: The input value/signal is a voltage measurement.

            $$V_{\text{dB}} = 20 \log_{10} V_{\text{linear}}$$

    Returns:
        The value or signal in dB.

    Examples:
        Convert 50 MHz to 77 dB-Hz.

        .. ipython:: python

            sdr.db(50e6)

        Convert 100 mW to 20 dBm.

        .. ipython:: python

            sdr.db(100, type="power")

        Convert 2 V to 6 dBV.

        .. ipython:: python

            sdr.db(2, type="voltage")

    Group:
        conversions-decibels
    """
    x = verify_and_validate_arraylike(x, non_negative=True)
    verify_literal(type, ["value", "power", "voltage"])

    with np.errstate(divide="ignore"):
        # Ignore divide by zero warning -- we're okay with -inf
        if type == "voltage":
            x_db = 20 * np.log10(x)
        else:
            x_db = 10 * np.log10(x)

    return normalize_output_type(x_db)


@export
def linear(
    x: npt.ArrayLike,
    type: Literal["value", "power", "voltage"] = "value",
) -> npt.NDArray[np.float64]:
    r"""
    Converts from decibels to linear units.

    Arguments:
        x: The input value or signal in dB.
        type: The type of output value or signal.

            - `"value"`: The output value/signal is any value.

            $$x_{\text{linear}} = 10^{\frac{x_{\text{dB}}}{10}}$$

            - `"power"`: The output value/signal is a power measurement.

            $$P_{\text{linear}} = 10^{\frac{P_{\text{dB}}}{10}}$$

            - `"voltage"`: The output value/signal is a voltage measurement.

            $$V_{\text{linear}} = 10^{\frac{V_{\text{dB}}}{20}}$$

    Returns:
        The value or signal in linear units.

    Examples:
        Convert 77 dB-Hz to 50 MHz.

        .. ipython:: python

            sdr.linear(77)

        Convert 20 dBm to 100 mW.

        .. ipython:: python

            sdr.linear(20, type="power")

        Convert 6 dBV to 2 V.

        .. ipython:: python

            sdr.linear(6, type="voltage")

    Group:
        conversions-decibels
    """
    x = verify_and_validate_arraylike(x)
    verify_literal(type, ["value", "power", "voltage"])

    with warnings.catch_warnings():
        # Ignore scalar power overflow warning -- we're okay with inf
        warnings.simplefilter("ignore", RuntimeWarning)
        if type == "voltage":
            x_linear = 10 ** (x / 20)
        else:
            x_linear = 10 ** (x / 10)

    return normalize_output_type(x_linear)
