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
    interpolation: int,
    decimation: int = 1,
    polyphase_order: int = 23,
    atten: float = 80,
) -> npt.NDArray[np.float64]:
    r"""
    Designs a multirate FIR filter impulse response $h[n]$ using the Kaiser window method.

    Arguments:
        interpolation: The interpolation rate $P$.
        decimation: The decimation rate $Q$.
        polyphase_order: The order of each polyphase filter. Must be odd, such that the filter lengths are even.
        atten: The stopband attenuation $A_{\text{stop}}$ in dB.

    Returns:
        The multirate filter impulse response $h[n]$.

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters.
        - https://www.mathworks.com/help/dsp/ref/designmultiratefir.html

    Examples:
        Design a multirate FIR filter for rational resampling by 11/3.

        .. ipython:: python

            h = sdr.design_multirate_fir(11, 3)

            @savefig sdr_design_multirate_fir_1.png
            plt.figure(); \
            sdr.plot.impulse_response(h);

            @savefig sdr_design_multirate_fir_2.png
            plt.figure(); \
            sdr.plot.magnitude_response(h);

    Group:
        dsp-polyphase-filtering
    """
    if not isinstance(interpolation, int):
        raise TypeError(f"Argument 'interpolation' must be an integer, not {interpolation}.")
    if not interpolation >= 1:
        raise ValueError(f"Argument 'interpolation' must be at least 1, not {interpolation}.")
    P = interpolation

    if not isinstance(decimation, int):
        raise TypeError(f"Argument 'decimation' must be an integer, not {decimation}.")
    if not decimation >= 1:
        raise ValueError(f"Argument 'decimation' must be at least 1, not {decimation}.")
    Q = decimation

    if not isinstance(polyphase_order, int):
        raise TypeError(f"Argument 'polyphase_order' must be an integer, not {polyphase_order}.")
    if not polyphase_order >= 1:
        raise ValueError(f"Argument 'polyphase_order' must be at least 1, not {polyphase_order}.")
    if not (polyphase_order + 1) % 2 == 0:
        raise ValueError(f"Argument 'polyphase_order' must be odd, not {polyphase_order}.")
    half_length = (polyphase_order + 1) // 2

    B = P if P > 1 else Q  # The number of polyphase branches
    R = max(P, Q)  # Inverse of the filter's fractional bandwidth

    # Compute the filter order, which is length - 1
    N = 2 * half_length * B

    # Compute ideal lowpass filter
    n = np.arange(N + 1)
    h = P / R * np.sinc((n - N // 2) / R)

    # Compute Kaiser window
    # beta = scipy.signal.windows.kaiser_beta(atten)
    if atten >= 50:
        beta = 0.1102 * (atten - 8.71)  # TODO: MATLAB uses 8.71 and SciPy uses 8.7
    elif atten > 21:
        beta = 0.5842 * (atten - 21) ** 0.4 + 0.07886 * (atten - 21)
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


def design_multirate_fir_linear(rate: int) -> npt.NDArray[np.float64]:
    r"""
    The multirate filter is designed to linearly interpolate between samples.

    The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [0, ..., r-1, r, r-1, ..., 1]$.
    The first output sample aligns with the first input sample.
    """
    h = np.zeros(2 * rate, dtype=float)
    h[:rate] = np.arange(0, rate) / rate
    h[rate:] = np.arange(rate, 0, -1) / rate
    return h


def design_multirate_fir_linear_matlab(rate: int) -> npt.NDArray[np.float64]:
    r"""
    The multirate filter is designed to linearly interpolate between samples.

    The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [1, ..., r-1, r, r-1, ..., 0]$.
    This is method MATLAB uses. The first output sample is advanced from the first input sample.
    """
    h = np.zeros(2 * rate, dtype=float)
    h[:rate] = np.arange(1, rate + 1) / rate
    h[rate:] = np.arange(rate - 1, -1, -1) / rate
    return h


def design_multirate_fir_zoh(rate: int) -> npt.NDArray[np.float64]:
    """
    The multirate filter is designed to be a zero-order hold.

    The filter coefficients are a length-$r$ array of ones.
    """
    h = np.ones(rate, dtype=float)
    return h


@export
def polyphase_decompose(branches: int, taps: npt.ArrayLike) -> npt.NDArray:
    r"""
    Decomposes the prototype filter taps $h[n]$ into the polyphase matrix $h_i[n]$ with $B$ phases.

    Arguments:
        branches: The number of polyphase branches $B$.
        taps: The prototype filter feedforward coefficients $h[n]$.

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
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters.

    Examples:
        Decompose the multirate FIR filter (notional taps for demonstration) into polyphase matrices
        with 3 and 6 phases.

        .. ipython:: python

            h = np.arange(0, 20)
            sdr.polyphase_decompose(3, h)
            sdr.polyphase_decompose(6, h)

    Group:
        dsp-polyphase-filtering
    """
    if not isinstance(branches, int):
        raise TypeError(f"Argument 'branches' must be an integer, not {branches}.")
    if not branches >= 1:
        raise ValueError(f"Argument 'branches' must be at least 1, not {branches}.")
    B = branches

    taps = np.asarray(taps)
    if not taps.ndim == 1:
        raise ValueError(f"Argument 'taps' must be a 1-D array, not {taps.ndim}-D.")

    N = math.ceil(taps.size / B) * B  # Filter length

    # Append zeros to the end of the taps so that the filter length is a multiple of B
    taps_pad = np.pad(taps, (0, N - taps.size), mode="constant")

    # Reshape the taps down the columns of H
    H = taps_pad.reshape(-1, B).T

    return H
