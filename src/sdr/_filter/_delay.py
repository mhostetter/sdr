"""
A module for fractional delay filters.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal

from .._helper import export
from ._design_fir import _normalize_passband


def _ideal_frac_delay(length: int, delay: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal fractional delay filter impulse response.
    """
    n = np.arange(-length // 2 + 1, length // 2 + 1)  # Sample indices
    h_ideal = np.sinc(n - delay)  # Ideal filter impulse response
    return h_ideal


@export
def design_frac_delay_fir(
    length: int,
    delay: float,
) -> npt.NDArray[np.float_]:
    r"""
    Designs a fractional delay FIR filter impulse response $h[n]$ using the Kaiser window method.

    Arguments:
        length: The filter length $L$. Filters with even length have best performance.
            Filters with odd length are equivalent to an even-length filter with an appended zero.
        delay: The fractional delay $0 \le \Delta n \le 1$.

    Returns:
        The filter impulse response $h[n]$ with length $L$. The center of the passband has 0 dB gain.

    Notes:
        The filter group delay is $\tau = L_{even}/2 - 1 + \Delta n$ at DC.

    References:
        - https://www.mathworks.com/help/dsp/ref/designfracdelayfir.html

    Examples:
        Design a $\Delta n = 0.25$ delay filter with length 8. Observe the width and flatness of the frequency
        response passband. Also observe the group delay of 3.25 at DC.

        .. ipython:: python

            h_8 = sdr.design_frac_delay_fir(8, 0.25)

            @savefig sdr_design_frac_delay_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_8);

            @savefig sdr_design_frac_delay_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_8); \
            plt.ylim(-4, 1);

            @savefig sdr_design_frac_delay_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.group_delay(h_8);

        Compare the magnitude response and group delay of filters with different lengths.

        .. ipython:: python

            h_16 = sdr.design_frac_delay_fir(16, 0.25); \
            h_32 = sdr.design_frac_delay_fir(32, 0.25); \
            h_64 = sdr.design_frac_delay_fir(64, 0.25)

            @savefig sdr_design_frac_delay_fir_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_8, label="Length 8"); \
            sdr.plot.magnitude_response(h_16, label="Length 16"); \
            sdr.plot.magnitude_response(h_32, label="Length 32"); \
            sdr.plot.magnitude_response(h_64, label="Length 64"); \
            plt.legend(); \
            plt.ylim(-4, 1);

            @savefig sdr_design_frac_delay_fir_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.group_delay(h_8, label="Length 8"); \
            sdr.plot.group_delay(h_16, label="Length 16"); \
            sdr.plot.group_delay(h_32, label="Length 32"); \
            sdr.plot.group_delay(h_64, label="Length 64"); \
            plt.legend();

    Group:
        dsp-arbitrary-resampling
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be an integer, not {type(length).__name__}.")
    if not length >= 2:
        raise ValueError(f"Argument 'length' must be at least 2, not {length}.")

    if not isinstance(delay, (int, float)):
        raise TypeError(f"Argument 'delay' must be a number, not {type(delay).__name__}.")
    if not 0 <= delay <= 1:
        raise ValueError(f"Argument 'delay' must be between 0 and 1, not {delay}.")

    N = length - (length % 2)  # The length guaranteed to be even
    h_ideal = _ideal_frac_delay(N, delay)

    if N == 2:
        beta = 0
    elif N == 4:
        beta = 2.21
    else:
        beta = (11.01299 * N**2 + 2395.00455 * N - 6226.46055) / (1.00000 * N**2 + 326.73886 * N + 1094.40241)
    h_window = scipy.signal.windows.kaiser(N, beta=beta)

    h = h_ideal * h_window

    if N < length:
        h = np.pad(h, (0, length - N), mode="constant")

    h = _normalize_passband(h, 0)

    return h
