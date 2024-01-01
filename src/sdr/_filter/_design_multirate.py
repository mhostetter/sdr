"""
A module for designing finite impulse response (FIR) multirate filters.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.signal

from .._helper import export


@export
def design_multirate_fir(
    up: int,
    down: int = 1,
    half_length: int = 12,
    A_stop: float = 80,
) -> npt.NDArray[np.float_]:
    r"""
    Designs a multirate FIR filter impulse response $h[n]$ using the Kaiser window method.

    Arguments:
        up: The interpolation rate $P$.
        down: The decimation rate $Q$.
        half_length: The half-length of the polyphase filters.
        A_stop: The stopband attenuation $A_{\text{stop}}$ in dB.

    Returns:
        The multirate filter impulse response $h[n]$.

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters
        - https://www.mathworks.com/help/dsp/ref/designmultiratefir.html

    Examples:
        Design a multirate FIR filter for rational resampling by 11/3.

        .. ipython:: python

            h = sdr.design_multirate_fir(11, 3)

            @savefig sdr_design_multirate_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h);

            @savefig sdr_design_multirate_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h);

    Group:
        dsp-multirate-filtering
    """
    if not isinstance(up, int):
        raise TypeError(f"Argument 'up' must be an integer, not {up}.")
    if not up >= 1:
        raise ValueError(f"Argument 'up' must be at least 1, not {up}.")
    P = up

    if not isinstance(down, int):
        raise TypeError(f"Argument 'down' must be an integer, not {down}.")
    if not down >= 1:
        raise ValueError(f"Argument 'down' must be at least 1, not {down}.")
    Q = down

    B = P if P > 1 else Q  # The number of polyphase branches
    R = max(P, Q)  # Inverse of the filter's fractional bandwidth

    # Compute the filter order, which is length - 1
    N = 2 * half_length * B

    # Compute ideal lowpass filter
    n = np.arange(N + 1)
    h = P / R * np.sinc((n - N // 2) / R)

    # Compute Kaiser window
    # beta = scipy.signal.windows.kaiser_beta(A_stop)
    if A_stop >= 50:
        beta = 0.1102 * (A_stop - 8.71)  # TODO: MATLAB uses 8.71 and SciPy uses 8.7
    elif A_stop > 21:
        beta = 0.5842 * (A_stop - 21) ** 0.4 + 0.07886 * (A_stop - 21)
    else:
        beta = 0
    w = scipy.signal.windows.kaiser(N + 1, beta)

    # Compute windowed filter
    h = h * w

    if not (Q > P > 1 and (half_length * P) % Q != 0):
        # The first and last elements are zero. Remove the last zero so that the filter is evenly
        # partitioned across the polyphase branches. The filter now has length 2 * half_length * B and
        # each polyphase branch has length 2 * half_length.
        h = h[:-1]

    # If the above condition is not true, the first and last elements are non-zero. The filter length
    # is 2 * half_length * B + 1 and each polyphase branch has length 2 * half_length + 1. The final
    # column in the polyphase matrix will be padded with zeros.

    return h


def design_multirate_fir_linear(rate: int) -> npt.NDArray[np.float_]:
    r"""
    The multirate filter is designed to linearly interpolate between samples. The filter coefficients are a
    length-$2r$ linear ramp $\frac{1}{r} [0, ..., r-1, r, r-1, ..., 1]$. The first output sample aligns with the
    first input sample.
    """
    h = np.zeros(2 * rate, dtype=float)
    h[:rate] = np.arange(0, rate) / rate
    h[rate:] = np.arange(rate, 0, -1) / rate
    return h


def design_multirate_fir_linear_matlab(rate: int) -> npt.NDArray[np.float_]:
    r"""
    The multirate filter is designed to linearly interpolate between samples. The filter coefficients are a
    length-$2r$ linear ramp $\frac{1}{r} [1, ..., r-1, r, r-1, ..., 0]$. This is method MATLAB uses. The first
    output sample is advanced from the first input sample.
    """
    h = np.zeros(2 * rate, dtype=float)
    h[:rate] = np.arange(1, rate + 1) / rate
    h[rate:] = np.arange(rate - 1, -1, -1) / rate
    return h


def design_multirate_fir_zoh(rate: int) -> npt.NDArray[np.float_]:
    """
    The multirate filter is designed to be a zero-order hold. The filter coefficients are a length-$r$ array of ones.
    """
    h = np.ones(rate, dtype=float)
    return h


@export
def polyphase_decompose(taps: npt.ArrayLike, phases: int) -> npt.NDArray:
    r"""
    Decomposes the prototype filter taps $h[n]$ into the polyphase matrix $h_i[n]$ with $B$ phases.

    Arguments:
        taps: The prototype filter feedforward coefficients $h[n]$.
        phases: The number of phases $B$.

    Returns:
        The polyphase matrix $h_i[n]$.

    Notes:
        The multirate FIR filter taps $h[n]$ are arranged down the columns of the polyphase matrix
        $h_i[n]$ as follows:

        .. code-block:: text
            :caption: Polyphase Matrix with 3 Phases

            +------+------+------+------+
            | h[0] | h[3] | h[6] | h[9] |
            +------+------+------+------+
            | h[1] | h[4] | h[7] | 0    |
            +------+------+------+------+
            | h[2] | h[5] | h[8] | 0    |
            +------+------+------+------+

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters

    Examples:
        Decompose the multirate FIR filter (notional taps for demonstration) into polyphase matrices
        with 3 and 6 phases.

        .. ipython:: python

            h = np.arange(0, 20)
            sdr.polyphase_decompose(h, 3)
            sdr.polyphase_decompose(h, 6)

    Group:
        dsp-multirate-filtering
    """
    taps = np.asarray(taps)

    if not isinstance(phases, int):
        raise TypeError(f"Argument 'phases' must be an integer, not {phases}.")
    if not phases >= 1:
        raise ValueError(f"Argument 'phases' must be at least 1, not {phases}.")
    B = phases

    N = math.ceil(taps.size / B) * B  # Filter length

    # Append zeros to the end of the taps so that the filter length is a multiple of B
    taps_pad = np.pad(taps, (0, N - taps.size), mode="constant")

    # Reshape the taps down the columns of H
    H = taps_pad.reshape(-1, B).T

    return H
