"""
A module for designing finite impulse response (FIR) filters.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export


def _normalize(h: npt.NDArray[np.float_], norm: Literal["power", "energy", "passband"]) -> npt.NDArray[np.float_]:
    if norm == "power":
        h /= np.sqrt(np.max(np.abs(h) ** 2))
    elif norm == "energy":
        h /= np.sqrt(np.sum(np.abs(h) ** 2))
    elif norm == "passband":
        h /= np.sum(h)
    else:
        raise ValueError(f"Argument 'norm' must be 'power', 'energy', or 'passband', not {norm}.")

    return h


def _ideal_lowpass(order: int, cutoff_freq: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal lowpass filter impulse response.
    """
    wc = np.pi * cutoff_freq  # Cutoff frequency in radians/s
    n = np.arange(-order // 2, order // 2 + 1)  # Sample indices
    h_ideal = wc / np.pi * np.sinc(wc * n / np.pi)  # Ideal filter impulse response
    return h_ideal


def _window(
    order: int,
    window: Literal["hamming", "hann", "blackman", "blackman-harris", "chebyshev", "kaiser"]
    | npt.ArrayLike
    | None = None,
) -> npt.NDArray[np.float_]:
    if window is None:
        h_window = np.ones(order + 1)
    elif isinstance(window, str):
        if window == "hamming":
            h_window = scipy.signal.windows.hamming(order + 1)
        elif window == "hann":
            h_window = scipy.signal.windows.hann(order + 1)
        elif window == "blackman":
            h_window = scipy.signal.windows.blackman(order + 1)
        elif window == "blackman-harris":
            h_window = scipy.signal.windows.blackmanharris(order + 1)
        elif window == "chebyshev":
            h_window = scipy.signal.windows.chebwin(order + 1, at=60)
        elif window == "kaiser":
            beta = 0.5  # Beta was reverse-engineered from MATLAB's outputs
            h_window = scipy.signal.windows.kaiser(order + 1, beta=beta)
        else:
            raise ValueError(
                f"Argument 'window' must be in ['hamming', 'hann', 'blackman', 'blackman-harris', 'chebyshev', 'kaiser'], not {window!r}."
            )
    else:
        h_window = np.asarray(window)
        if not h_window.shape == (order + 1,):
            raise ValueError(f"Argument 'window' must be a length-{order + 1} vector, not {h_window.shape}.")

    return h_window


@export
def design_lowpass_fir(
    order: int,
    cutoff_freq: float,
    window: None
    | Literal["hamming", "hann", "blackman", "blackman-harris", "chebyshev", "kaiser"]
    | npt.ArrayLike = "hamming",
) -> npt.NDArray[np.float_]:
    r"""
    Designs a lowpass FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        cutoff_freq: The cutoff frequency $f_c$, normalized to the Nyquist frequency $f_s / 2$.
        window: The time-domain window to use.

            - `None`: No windowing. Equivalently, a length-$N + 1$ vector of ones.
            - `"hamming"`: Hamming window, see :func:`scipy.signal.windows.hamming`.
            - `"hann"`: Hann window, see :func:`scipy.signal.windows.hann`.
            - `"blackman"`: Blackman window, see :func:`scipy.signal.windows.blackman`.
            - `"blackman-harris"`: Blackman-Harris window, see :func:`scipy.signal.windows.blackmanharris`.
            - `"chebyshev"`: Chebyshev window, see :func:`scipy.signal.windows.chebwin`. The sidelobe attenuation
              is 60 dB.
            - `"kaiser"`: Kaiser window, see :func:`scipy.signal.windows.kaiser`. The beta parameter is 0.5.
            - `npt.ArrayLike`: A custom window. Must be a length-$N + 1$ vector.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$.

    References:
        - https://www.mathworks.com/help/dsp/ref/designlowpassfir.html

    Examples:
        Design a length-33 lowpass FIR filter with cutoff frequency $f_c = 0.2 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.design_lowpass_fir(32, 0.2, window="hamming")

            @savefig sdr_design_lowpass_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_design_lowpass_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming);

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.design_lowpass_fir(32, 0.2, window="hann"); \
            h_blackman = sdr.design_lowpass_fir(32, 0.2, window="blackman"); \
            h_blackman_harris = sdr.design_lowpass_fir(32, 0.2, window="blackman-harris"); \
            h_chebyshev = sdr.design_lowpass_fir(32, 0.2, window="chebyshev"); \
            h_kaiser = sdr.design_lowpass_fir(32, 0.2, window="kaiser")

            @savefig sdr_design_lowpass_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, label="Kaiser"); \
            plt.legend(); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be an integer, not {type(order).__name__}.")
    if not order % 2 == 0:
        raise ValueError(f"Argument 'order' must be even, not {order}.")

    if not isinstance(cutoff_freq, (int, float)):
        raise TypeError(f"Argument 'cutoff_freq' must be a number, not {type(cutoff_freq).__name__}.")
    if not 0 <= cutoff_freq <= 1:
        raise ValueError(f"Argument 'cutoff_freq' must be between 0 and 0.5, not {cutoff_freq}.")

    h_ideal = _ideal_lowpass(order, cutoff_freq)
    h_window = _window(order, window)
    h = h_ideal * h_window
    h = _normalize(h, "passband")

    return h
