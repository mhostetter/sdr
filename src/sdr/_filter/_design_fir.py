"""
A module for designing finite impulse response (FIR) filters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export, verify_scalar


def _normalize(h: npt.NDArray[np.float64], norm: Literal["power", "energy", "passband"]) -> npt.NDArray[np.float64]:
    if norm == "power":
        h /= np.sqrt(np.max(np.abs(h) ** 2))
    elif norm == "energy":
        h /= np.sqrt(np.sum(np.abs(h) ** 2))
    elif norm == "passband":
        h /= np.sum(h)
    else:
        raise ValueError(f"Argument 'norm' must be 'power', 'energy', or 'passband', not {norm}.")

    return h


def _normalize_passband(h: npt.NDArray[np.float64], center_freq: float) -> npt.NDArray[np.float64]:
    order = h.size - 1
    lo = np.exp(-1j * np.pi * center_freq * np.arange(-order // 2, order // 2 + 1))
    h = h / np.abs(np.sum(h * lo))
    return h


def _ideal_lowpass(order: int, cutoff_freq: float) -> npt.NDArray[np.float64]:
    """
    Returns the ideal lowpass filter impulse response.
    """
    wc = np.pi * cutoff_freq  # Cutoff frequency in radians/s
    n = np.arange(-order // 2, order // 2 + 1)  # Sample indices
    h_ideal = wc / np.pi * np.sinc(wc * n / np.pi)  # Ideal filter impulse response
    return h_ideal


def _ideal_highpass(order: int, cutoff_freq: float) -> npt.NDArray[np.float64]:
    """
    Returns the ideal highpass filter impulse response.
    """
    lo = np.cos(np.pi * np.arange(-order // 2, order // 2 + 1))
    h_ideal = _ideal_lowpass(order, 1 - cutoff_freq) * lo
    return h_ideal


def _ideal_bandpass(order: int, center_freq: float, bandwidth: float) -> npt.NDArray[np.float64]:
    """
    Returns the ideal bandpass filter impulse response.
    """
    h_lp1 = _ideal_lowpass(order, center_freq + bandwidth / 2)
    h_lp2 = _ideal_lowpass(order, center_freq - bandwidth / 2)
    h_ideal = h_lp1 - h_lp2
    return h_ideal


def _ideal_bandstop(order: int, center_freq: float, bandwidth: float) -> npt.NDArray[np.float64]:
    """
    Returns the ideal bandstop filter impulse response.
    """
    h_lp = _ideal_lowpass(order, center_freq - bandwidth / 2)
    h_hp = _ideal_highpass(order, center_freq + bandwidth / 2)
    h_ideal = h_lp + h_hp
    return h_ideal


@export
def lowpass_fir(
    order: int,
    cutoff_freq: float,
    window: str | float | tuple | None = "hamming",
) -> npt.NDArray[np.float64]:
    r"""
    Designs a lowpass FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        cutoff_freq: The cutoff frequency $f_c$, normalized to the Nyquist frequency $f_s / 2$.
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designlowpassfir.html

    Examples:
        Design a length-101 lowpass FIR filter with cutoff frequency $f_c = 0.2 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.lowpass_fir(100, 0.2, window="hamming")

            @savefig sdr_lowpass_fir_1.png
            plt.figure(); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_lowpass_fir_2.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming);

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.lowpass_fir(100, 0.2, window="hann"); \
            h_blackman = sdr.lowpass_fir(100, 0.2, window="blackman"); \
            h_blackman_harris = sdr.lowpass_fir(100, 0.2, window="blackmanharris"); \
            h_chebyshev = sdr.lowpass_fir(100, 0.2, window=("chebwin", 60)); \
            h_kaiser = sdr.lowpass_fir(100, 0.2, window=("kaiser", 0.5))

            @savefig sdr_lowpass_fir_3.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming, label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, label="Kaiser"); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    verify_scalar(order, int=True, even=True)
    verify_scalar(cutoff_freq, float=True, inclusive_min=0, inclusive_max=1)

    h = _ideal_lowpass(order, cutoff_freq)
    if window is not None:
        h *= scipy.signal.windows.get_window(window, h.size, fftbins=False)
    h = _normalize_passband(h, 0)

    return h


@export
def highpass_fir(
    order: int,
    cutoff_freq: float,
    window: str | float | tuple | None = "hamming",
) -> npt.NDArray[np.float64]:
    r"""
    Designs a highpass FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        cutoff_freq: The cutoff frequency $f_c$, normalized to the Nyquist frequency $f_s / 2$.
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designhighpassfir.html

    Examples:
        Design a length-101 highpass FIR filter with cutoff frequency $f_c = 0.7 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.highpass_fir(100, 0.7, window="hamming")

            @savefig sdr_highpass_fir_1.png
            plt.figure(); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_highpass_fir_2.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming);

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.highpass_fir(100, 0.7, window="hann"); \
            h_blackman = sdr.highpass_fir(100, 0.7, window="blackman"); \
            h_blackman_harris = sdr.highpass_fir(100, 0.7, window="blackmanharris"); \
            h_chebyshev = sdr.highpass_fir(100, 0.7, window=("chebwin", 60)); \
            h_kaiser = sdr.highpass_fir(100, 0.7, window=("kaiser", 0.5))

            @savefig sdr_highpass_fir_3.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming, label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, label="Kaiser"); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    verify_scalar(order, int=True, even=True)
    verify_scalar(cutoff_freq, float=True, inclusive_min=0, inclusive_max=1)

    h = _ideal_highpass(order, cutoff_freq)
    if window is not None:
        h *= scipy.signal.windows.get_window(window, h.size, fftbins=False)
    h = _normalize_passband(h, 1)

    return h


@export
def bandpass_fir(
    order: int,
    center_freq: float,
    bandwidth: float,
    window: str | float | tuple | None = "hamming",
) -> npt.NDArray[np.float64]:
    r"""
    Designs a bandpass FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        center_freq: The center frequency $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        bandwidth: The two-sided bandwidth about $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designbandpassfir.html

    Examples:
        Design a length-101 bandpass FIR filter with center frequency $f_{center} = 0.4 \cdot f_s / 2$
        and bandwidth $0.1 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.bandpass_fir(100, 0.4, 0.1, window="hamming")

            @savefig sdr_bandpass_fir_1.png
            plt.figure(); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_bandpass_fir_2.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming);

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.bandpass_fir(100, 0.4, 0.1, window="hann"); \
            h_blackman = sdr.bandpass_fir(100, 0.4, 0.1, window="blackman"); \
            h_blackman_harris = sdr.bandpass_fir(100, 0.4, 0.1, window="blackmanharris"); \
            h_chebyshev = sdr.bandpass_fir(100, 0.4, 0.1, window=("chebwin", 60)); \
            h_kaiser = sdr.bandpass_fir(100, 0.4, 0.1, window=("kaiser", 0.5))

            @savefig sdr_bandpass_fir_3.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming, label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, label="Kaiser"); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    verify_scalar(order, int=True, even=True)
    verify_scalar(center_freq, float=True, inclusive_min=0, inclusive_max=1)
    verify_scalar(bandwidth, float=True, inclusive_min=0, inclusive_max=2 * min(center_freq, 1 - center_freq))

    h = _ideal_bandpass(order, center_freq, bandwidth)
    if window is not None:
        h *= scipy.signal.windows.get_window(window, h.size, fftbins=False)
    h = _normalize_passband(h, center_freq)

    return h


@export
def bandstop_fir(
    order: int,
    center_freq: float,
    bandwidth: float,
    window: str | float | tuple | None = "hamming",
) -> npt.NDArray[np.float64]:
    r"""
    Designs a bandstop FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        center_freq: The center frequency $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        bandwidth: The two-sided bandwidth about $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the larger passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designbandstopfir.html

    Examples:
        Design a length-101 bandstop FIR filter with center frequency $f_{center} = 0.4 \cdot f_s / 2$
        and bandwidth $0.75 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.bandstop_fir(100, 0.4, 0.75, window="hamming")

            @savefig sdr_bandstop_fir_1.png
            plt.figure(); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_bandstop_fir_2.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming);

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.bandstop_fir(100, 0.4, 0.75, window="hann"); \
            h_blackman = sdr.bandstop_fir(100, 0.4, 0.75, window="blackman"); \
            h_blackman_harris = sdr.bandstop_fir(100, 0.4, 0.75, window="blackmanharris"); \
            h_chebyshev = sdr.bandstop_fir(100, 0.4, 0.75, window=("chebwin", 60)); \
            h_kaiser = sdr.bandstop_fir(100, 0.4, 0.75, window=("kaiser", 0.5))

            @savefig sdr_bandstop_fir_3.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_hamming, label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, label="Kaiser"); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    verify_scalar(order, int=True, even=True)
    verify_scalar(center_freq, float=True, inclusive_min=0, inclusive_max=1)
    verify_scalar(bandwidth, float=True, inclusive_min=0, inclusive_max=2 * min(center_freq, 1 - center_freq))

    h = _ideal_bandstop(order, center_freq, bandwidth)
    if window is not None:
        h *= scipy.signal.windows.get_window(window, h.size, fftbins=False)
    if center_freq > 0.5:
        h = _normalize_passband(h, 0)
    else:
        h = _normalize_passband(h, 1)

    return h
