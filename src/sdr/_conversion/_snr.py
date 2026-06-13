"""
Signal-to-noise ratio (SNR) conversion functions.

Provides conversions between Eb/N0, Es/N0, and S/N representations,
accounting for modulation order, code rate, and samples per symbol.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import normalize_output_type, export, verify_and_validate_arraylike
from ._decibels import db


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
    ebn0 = verify_and_validate_arraylike(ebn0)  # Energy per information bit
    bps = verify_and_validate_arraylike(bps, int=True, positive=True)
    rate = verify_and_validate_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)

    ecn0 = ebn0 + db(rate)  # Energy per coded bit
    esn0 = ecn0 + db(bps)  # Energy per symbol

    return normalize_output_type(esn0)


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
    ebn0 = verify_and_validate_arraylike(ebn0)  # Energy per information bit
    bps = verify_and_validate_arraylike(bps, int=True, positive=True)
    rate = verify_and_validate_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)
    sps = verify_and_validate_arraylike(sps, int=True, positive=True)

    esn0 = ebn0_to_esn0(ebn0, bps, rate=rate)  # SNR per symbol
    snr = esn0 - db(sps)  # SNR per sample

    return normalize_output_type(snr)


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
    esn0 = verify_and_validate_arraylike(esn0)  # Energy per symbol
    bps = verify_and_validate_arraylike(bps, int=True, positive=True)
    rate = verify_and_validate_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)

    ecn0 = esn0 - db(bps)  # Energy per coded bit
    ebn0 = ecn0 - db(rate)  # Energy per information bit

    return normalize_output_type(ebn0)


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
    esn0 = verify_and_validate_arraylike(esn0)  # Energy per symbol
    sps = verify_and_validate_arraylike(sps, int=True, positive=True)

    snr = esn0 - db(sps)  # SNR per sample

    return normalize_output_type(snr)


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
    snr = verify_and_validate_arraylike(snr)  # SNR per sample
    bps = verify_and_validate_arraylike(bps, int=True, positive=True)
    rate = verify_and_validate_arraylike(rate, float=True, inclusive_min=0, inclusive_max=1)
    sps = verify_and_validate_arraylike(sps, int=True, positive=True)

    esn0 = snr_to_esn0(snr, sps=sps)  # Energy per symbol
    ebn0 = esn0_to_ebn0(esn0, bps, rate=rate)  # Energy per information bit

    return normalize_output_type(ebn0)


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
    snr = verify_and_validate_arraylike(snr)  # SNR per sample
    sps = verify_and_validate_arraylike(sps, int=True, positive=True)

    esn0 = snr + db(sps)  # Energy per symbol

    return normalize_output_type(esn0)
