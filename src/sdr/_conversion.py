"""
A module that contains various conversion functions.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from ._helper import convert_output, export, verify_arraylike, verify_literal

##############################################################################
# Decibels
##############################################################################


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
    x = verify_arraylike(x, non_negative=True)
    verify_literal(type, ["value", "power", "voltage"])

    with np.errstate(divide="ignore"):
        # Ignore divide by zero warning -- we're okay with -inf
        if type == "voltage":
            x_db = 20 * np.log10(x)
        else:
            x_db = 10 * np.log10(x)

    return convert_output(x_db)


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
    x = verify_arraylike(x)
    verify_literal(type, ["value", "power", "voltage"])

    with warnings.catch_warnings():
        # Ignore scalar power overflow warning -- we're okay with inf
        warnings.simplefilter("ignore", RuntimeWarning)
        if type == "voltage":
            x_linear = 10 ** (x / 20)
        else:
            x_linear = 10 ** (x / 10)

    return convert_output(x_linear)


##############################################################################
# From Eb/N0
##############################################################################


@export
def ebn0_to_esn0(
    ebn0: npt.ArrayLike,
    bps: npt.ArrayLike,
    rate: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    r"""
    Converts from $E_b/N_0$ to $E_s/N_0$.

    $$
    \frac{E_s}{N_0} = \frac{E_b}{N_0} \frac{k}{n} \log_2 M
    $$

    Arguments:
        ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.
        bps: Coded bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.

    Returns:
        The symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.

    Examples:
        Convert from $E_b/N_0 = 5$ dB to $E_s/N_0$ for a 4-QAM signal with $r = 2/3$.

        .. ipython:: python

            sdr.ebn0_to_esn0(5, 2, rate=2/3)

        Convert from $E_b/N_0 = 10$ dB to $E_s/N_0$ for a 16-QAM signal with $r = 1$.

        .. ipython:: python

            sdr.ebn0_to_esn0(10, 4, rate=1)

    Group:
        conversions-snrs
    """
    ebn0 = verify_arraylike(ebn0)  # Energy per information bit
    bps = verify_arraylike(bps, int=True, positive=True)
    rate = verify_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)

    ecn0 = ebn0 + db(rate)  # Energy per coded bit
    esn0 = ecn0 + db(bps)  # Energy per symbol

    return convert_output(esn0)


@export
def ebn0_to_snr(
    ebn0: npt.ArrayLike,
    bps: npt.ArrayLike,
    rate: npt.ArrayLike = 1.0,
    sps: npt.ArrayLike = 1,
) -> npt.NDArray[np.float64]:
    r"""
    Converts from $E_b/N_0$ to $S/N$.

    $$
    \frac{S}{N} = \frac{E_b}{N_0} \frac{k}{n} \log_2 M \frac{f_{sym}}{f_s}
    $$

    Arguments:
        ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.
        bps: Coded bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.
        sps: Samples per symbol $f_s / f_{sym}$.

    Returns:
        The signal-to-noise ratio $S/N$ in dB.

    Examples:
        Convert from $E_b/N_0 = 5$ dB to $S/N$ for a 4-QAM signal with $r = 2/3$ and 1 sample per symbol.

        .. ipython:: python

            sdr.ebn0_to_snr(5, 2, rate=2/3, sps=1)

        Convert from $E_b/N_0 = 10$ dB to $S/N$ for a 16-QAM signal with $r = 1$ and 4 samples per symbol.

        .. ipython:: python

            sdr.ebn0_to_snr(10, 4, rate=1, sps=4)

    Group:
        conversions-snrs
    """
    ebn0 = verify_arraylike(ebn0)  # Energy per information bit
    bps = verify_arraylike(bps, int=True, positive=True)
    rate = verify_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)
    sps = verify_arraylike(sps, int=True, positive=True)

    esn0 = ebn0_to_esn0(ebn0, bps, rate=rate)  # SNR per symbol
    snr = esn0 - db(sps)  # SNR per sample

    return convert_output(snr)


##############################################################################
# From Es/N0
##############################################################################


@export
def esn0_to_ebn0(
    esn0: npt.ArrayLike,
    bps: npt.ArrayLike,
    rate: npt.ArrayLike = 1.0,
) -> npt.NDArray[np.float64]:
    r"""
    Converts from $E_s/N_0$ to $E_b/N_0$.

    $$
    \frac{E_b}{N_0} = \frac{E_s}{N_0} \frac{n}{k} \frac{1}{\log_2 M}
    $$

    Arguments:
        esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.
        bps: Coded bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.

    Returns:
        The bit energy $E_b$ to noise PSD $N_0$ ratio in dB.

    Examples:
        Convert from $E_s/N_0 = 5$ dB to $E_b/N_0$ for a 4-QAM signal with $r = 2/3$.

        .. ipython:: python

            sdr.esn0_to_ebn0(5, 2, rate=2/3)

        Convert from $E_s/N_0 = 10$ dB to $E_b/N_0$ for a 16-QAM signal with $r = 1$.

        .. ipython:: python

            sdr.esn0_to_ebn0(10, 4, rate=1)

    Group:
        conversions-snrs
    """
    esn0 = verify_arraylike(esn0)  # Energy per symbol
    bps = verify_arraylike(bps, int=True, positive=True)
    rate = verify_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)

    ecn0 = esn0 - db(bps)  # Energy per coded bit
    ebn0 = ecn0 - db(rate)  # Energy per information bit

    return convert_output(ebn0)


@export
def esn0_to_snr(
    esn0: npt.ArrayLike,
    sps: npt.ArrayLike = 1,
) -> npt.NDArray[np.float64]:
    r"""
    Converts from $E_s/N_0$ to $S/N$.

    $$
    \frac{S}{N} = \frac{E_s}{N_0} \frac{f_{sym}}{f_s}
    $$

    Arguments:
        esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.
        sps: Samples per symbol $f_s / f_{sym}$.

    Returns:
        The signal-to-noise ratio $S/N$ in dB.

    Examples:
        Convert from $E_s/N_0 = 5$ dB to $S/N$ with 1 sample per symbol. In discrete-time systems,
        when there is 1 sample per symbol, $S/N$ is equivalent to $E_s/N_0$.

        .. ipython:: python

            sdr.esn0_to_snr(5, sps=1)

        Convert from $E_s/N_0 = 10$ dB to $S/N$ with 4 samples per symbol.

        .. ipython:: python

            sdr.esn0_to_snr(10, sps=4)

    Group:
        conversions-snrs
    """
    esn0 = verify_arraylike(esn0)  # Energy per symbol
    sps = verify_arraylike(sps, int=True, positive=True)

    snr = esn0 - db(sps)  # SNR per sample

    return convert_output(snr)


##############################################################################
# From SNR
##############################################################################


@export
def snr_to_ebn0(
    snr: npt.ArrayLike,
    bps: npt.ArrayLike,
    rate: npt.ArrayLike = 1.0,
    sps: npt.ArrayLike = 1,
) -> npt.NDArray[np.float64]:
    r"""
    Converts from $S/N$ to $E_b/N_0$.

    $$
    \frac{E_b}{N_0} = \frac{S}{N} \frac{f_s}{f_{sym}} \frac{n}{k} \frac{1}{\log_2 M}
    $$

    Arguments:
        snr: Signal-to-noise ratio $S/N$ in dB.
        bps: Coded bits per symbol $\log_2 M$, where $M$ is the modulation order.
        rate: Code rate $r = k/n$, where $k$ is the number of information bits and $n$ is the
            number of coded bits.
        sps: Samples per symbol $f_s / f_{sym}$.

    Returns:
        The bit energy $E_b$ to noise PSD $N_0$ ratio in dB.

    Examples:
        Convert from $S/N = 5$ dB to $E_b/N_0$ for a 4-QAM signal with $r = 2/3$ and 1 sample per symbol.

        .. ipython:: python

            sdr.snr_to_ebn0(5, 2, rate=2/3, sps=1)

        Convert from $S/N = 10$ dB to $E_b/N_0$ for a 16-QAM signal with $r = 1$ and 4 samples per symbol.

        .. ipython:: python

            sdr.snr_to_ebn0(10, 4, rate=1, sps=4)

    Group:
        conversions-snrs
    """
    snr = verify_arraylike(snr)  # SNR per sample
    bps = verify_arraylike(bps, int=True, positive=True)
    rate = verify_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)
    sps = verify_arraylike(sps, int=True, positive=True)

    esn0 = snr_to_esn0(snr, sps=sps)  # Energy per symbol
    ebn0 = esn0_to_ebn0(esn0, bps, rate=rate)  # Energy per information bit

    return convert_output(ebn0)


@export
def snr_to_esn0(
    snr: npt.ArrayLike,
    sps: npt.ArrayLike = 1,
) -> npt.NDArray[np.float64]:
    r"""
    Converts from $S/N$ to $E_s/N_0$.

    $$
    \frac{E_s}{N_0} = \frac{S}{N} \frac{f_s}{f_{sym}}
    $$

    Arguments:
        snr: Signal-to-noise ratio $S/N$ in dB.
        sps: Samples per symbol $f_s / f_{sym}$.

    Returns:
        The symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.

    Examples:
        Convert from $S/N = 5$ dB to $E_s/N_0$ with 1 sample per symbol. In discrete-time systems,
        when there is 1 sample per symbol, $S/N$ is equivalent to $E_s/N_0$.

        .. ipython:: python

            sdr.snr_to_esn0(5, sps=1)

        Convert from $S/N = 10$ dB to $E_s/N_0$ with 4 samples per symbol.

        .. ipython:: python

            sdr.snr_to_esn0(10, sps=4)

    Group:
        conversions-snrs
    """
    snr = verify_arraylike(snr)  # SNR per sample
    sps = verify_arraylike(sps, int=True, positive=True)

    esn0 = snr + db(sps)  # Energy per symbol

    return convert_output(esn0)
