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


def _normalize_passband(h: npt.NDArray[np.float_], center_freq: float) -> npt.NDArray[np.float_]:
    order = h.size - 1
    lo = np.exp(-1j * np.pi * center_freq * np.arange(-order // 2, order // 2 + 1))
    h = h / np.abs(np.sum(h * lo))
    return h


def _ideal_lowpass(order: int, cutoff_freq: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal lowpass filter impulse response.
    """
    wc = np.pi * cutoff_freq  # Cutoff frequency in radians/s
    n = np.arange(-order // 2, order // 2 + 1)  # Sample indices
    h_ideal = wc / np.pi * np.sinc(wc * n / np.pi)  # Ideal filter impulse response
    return h_ideal


def _ideal_highpass(order: int, cutoff_freq: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal highpass filter impulse response.
    """
    lo = np.cos(np.pi * np.arange(-order // 2, order // 2 + 1))
    h_ideal = _ideal_lowpass(order, 1 - cutoff_freq) * lo
    return h_ideal


def _ideal_bandpass(order: int, center_freq: float, bandwidth: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal bandpass filter impulse response.
    """
    h_lp1 = _ideal_lowpass(order, center_freq + bandwidth / 2)
    h_lp2 = _ideal_lowpass(order, center_freq - bandwidth / 2)
    h_ideal = h_lp1 - h_lp2
    return h_ideal


def _ideal_bandstop(order: int, center_freq: float, bandwidth: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal bandstop filter impulse response.
    """
    h_lp = _ideal_lowpass(order, center_freq - bandwidth / 2)
    h_hp = _ideal_highpass(order, center_freq + bandwidth / 2)
    h_ideal = h_lp + h_hp
    return h_ideal


def _ideal_frac_delay(length: int, delay: float) -> npt.NDArray[np.float_]:
    """
    Returns the ideal fractional delay filter impulse response.
    """
    n = np.arange(-length // 2 + 1, length // 2 + 1)  # Sample indices
    h_ideal = np.sinc(n - delay)  # Ideal filter impulse response
    return h_ideal


def _window(
    order: int,
    window: Literal["hamming", "hann", "blackman", "blackman-harris", "chebyshev", "kaiser"]
    | npt.ArrayLike
    | None = None,
    atten: float = 60,
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
            h_window = scipy.signal.windows.chebwin(order + 1, at=atten)
        elif window == "kaiser":
            beta = scipy.signal.kaiser_beta(atten)
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
    atten: float = 60,
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
            - `"chebyshev"`: Chebyshev window, see :func:`scipy.signal.windows.chebwin`.
            - `"kaiser"`: Kaiser window, see :func:`scipy.signal.windows.kaiser`.
            - `npt.ArrayLike`: A custom window. Must be a length-$N + 1$ vector.

        atten: The sidelobe attenuation in dB. Only used if `window` is `"chebyshev"` or `"kaiser"`.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designlowpassfir.html

    Examples:
        Design a length-101 lowpass FIR filter with cutoff frequency $f_c = 0.2 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.design_lowpass_fir(100, 0.2, window="hamming")

            @savefig sdr_design_lowpass_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_design_lowpass_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided");

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.design_lowpass_fir(100, 0.2, window="hann"); \
            h_blackman = sdr.design_lowpass_fir(100, 0.2, window="blackman"); \
            h_blackman_harris = sdr.design_lowpass_fir(100, 0.2, window="blackman-harris"); \
            h_chebyshev = sdr.design_lowpass_fir(100, 0.2, window="chebyshev"); \
            h_kaiser = sdr.design_lowpass_fir(100, 0.2, window="kaiser")

            @savefig sdr_design_lowpass_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided", label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, x_axis="one-sided", label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, x_axis="one-sided", label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, x_axis="one-sided", label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, x_axis="one-sided", label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, x_axis="one-sided", label="Kaiser"); \
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
        raise ValueError(f"Argument 'cutoff_freq' must be between 0 and 1, not {cutoff_freq}.")

    if not isinstance(atten, (int, float)):
        raise TypeError(f"Argument 'atten' must be a number, not {type(atten).__name__}.")
    if not 0 <= atten:
        raise ValueError(f"Argument 'atten' must be non-negative, not {atten}.")

    h_ideal = _ideal_lowpass(order, cutoff_freq)
    h_window = _window(order, window, atten=atten)
    h = h_ideal * h_window
    h = _normalize_passband(h, 0)

    return h


@export
def design_highpass_fir(
    order: int,
    cutoff_freq: float,
    window: None
    | Literal["hamming", "hann", "blackman", "blackman-harris", "chebyshev", "kaiser"]
    | npt.ArrayLike = "hamming",
    atten: float = 60,
) -> npt.NDArray[np.float_]:
    r"""
    Designs a highpass FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        cutoff_freq: The cutoff frequency $f_c$, normalized to the Nyquist frequency $f_s / 2$.
        window: The time-domain window to use.

            - `None`: No windowing. Equivalently, a length-$N + 1$ vector of ones.
            - `"hamming"`: Hamming window, see :func:`scipy.signal.windows.hamming`.
            - `"hann"`: Hann window, see :func:`scipy.signal.windows.hann`.
            - `"blackman"`: Blackman window, see :func:`scipy.signal.windows.blackman`.
            - `"blackman-harris"`: Blackman-Harris window, see :func:`scipy.signal.windows.blackmanharris`.
            - `"chebyshev"`: Chebyshev window, see :func:`scipy.signal.windows.chebwin`.
            - `"kaiser"`: Kaiser window, see :func:`scipy.signal.windows.kaiser`.
            - `npt.ArrayLike`: A custom window. Must be a length-$N + 1$ vector.

        atten: The sidelobe attenuation in dB. Only used if `window` is `"chebyshev"` or `"kaiser"`.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designhighpassfir.html

    Examples:
        Design a length-101 highpass FIR filter with cutoff frequency $f_c = 0.7 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.design_highpass_fir(100, 0.7, window="hamming")

            @savefig sdr_design_highpass_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_design_highpass_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided");

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.design_highpass_fir(100, 0.7, window="hann"); \
            h_blackman = sdr.design_highpass_fir(100, 0.7, window="blackman"); \
            h_blackman_harris = sdr.design_highpass_fir(100, 0.7, window="blackman-harris"); \
            h_chebyshev = sdr.design_highpass_fir(100, 0.7, window="chebyshev"); \
            h_kaiser = sdr.design_highpass_fir(100, 0.7, window="kaiser")

            @savefig sdr_design_highpass_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided", label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, x_axis="one-sided", label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, x_axis="one-sided", label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, x_axis="one-sided", label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, x_axis="one-sided", label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, x_axis="one-sided", label="Kaiser"); \
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
        raise ValueError(f"Argument 'cutoff_freq' must be between 0 and 1, not {cutoff_freq}.")

    if not isinstance(atten, (int, float)):
        raise TypeError(f"Argument 'atten' must be a number, not {type(atten).__name__}.")
    if not 0 <= atten:
        raise ValueError(f"Argument 'atten' must be non-negative, not {atten}.")

    h_ideal = _ideal_highpass(order, cutoff_freq)
    h_window = _window(order, window, atten=atten)
    h = h_ideal * h_window
    h = _normalize_passband(h, 1)

    return h


@export
def design_bandpass_fir(
    order: int,
    center_freq: float,
    bandwidth: float,
    window: None
    | Literal["hamming", "hann", "blackman", "blackman-harris", "chebyshev", "kaiser"]
    | npt.ArrayLike = "hamming",
    atten: float = 60,
) -> npt.NDArray[np.float_]:
    r"""
    Designs a bandpass FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        center_freq: The center frequency $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        bandwidth: The two-sided bandwidth about $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        window: The time-domain window to use.

            - `None`: No windowing. Equivalently, a length-$N + 1$ vector of ones.
            - `"hamming"`: Hamming window, see :func:`scipy.signal.windows.hamming`.
            - `"hann"`: Hann window, see :func:`scipy.signal.windows.hann`.
            - `"blackman"`: Blackman window, see :func:`scipy.signal.windows.blackman`.
            - `"blackman-harris"`: Blackman-Harris window, see :func:`scipy.signal.windows.blackmanharris`.
            - `"chebyshev"`: Chebyshev window, see :func:`scipy.signal.windows.chebwin`.
            - `"kaiser"`: Kaiser window, see :func:`scipy.signal.windows.kaiser`.
            - `npt.ArrayLike`: A custom window. Must be a length-$N + 1$ vector.

        atten: The sidelobe attenuation in dB. Only used if `window` is `"chebyshev"` or `"kaiser"`.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designbandpassfir.html

    Examples:
        Design a length-101 bandpass FIR filter with center frequency $f_{center} = 0.4 \cdot f_s / 2$
        and bandwidth $0.1 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.design_bandpass_fir(100, 0.4, 0.1, window="hamming")

            @savefig sdr_design_bandpass_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_design_bandpass_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided");

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.design_bandpass_fir(100, 0.4, 0.1, window="hann"); \
            h_blackman = sdr.design_bandpass_fir(100, 0.4, 0.1, window="blackman"); \
            h_blackman_harris = sdr.design_bandpass_fir(100, 0.4, 0.1, window="blackman-harris"); \
            h_chebyshev = sdr.design_bandpass_fir(100, 0.4, 0.1, window="chebyshev"); \
            h_kaiser = sdr.design_bandpass_fir(100, 0.4, 0.1, window="kaiser")

            @savefig sdr_design_bandpass_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided", label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, x_axis="one-sided", label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, x_axis="one-sided", label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, x_axis="one-sided", label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, x_axis="one-sided", label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, x_axis="one-sided", label="Kaiser"); \
            plt.legend(); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be an integer, not {type(order).__name__}.")
    if not order % 2 == 0:
        raise ValueError(f"Argument 'order' must be even, not {order}.")

    if not isinstance(center_freq, (int, float)):
        raise TypeError(f"Argument 'center_freq' must be a number, not {type(center_freq).__name__}.")
    if not 0 <= center_freq <= 1:
        raise ValueError(f"Argument 'center_freq' must be between 0 and 1, not {center_freq}.")

    if not isinstance(bandwidth, (int, float)):
        raise TypeError(f"Argument 'bandwidth' must be a number, not {type(bandwidth).__name__}.")
    if not 0 <= center_freq + bandwidth / 2 <= 1:
        raise ValueError(f"Argument 'bandwidth' must be between 0 and 0.5 - 'center_freq', not {bandwidth}.")
    if not 0 <= center_freq - bandwidth / 2 <= 1:
        raise ValueError(f"Argument 'bandwidth' must be between 0 and 'center_freq', not {bandwidth}.")

    if not isinstance(atten, (int, float)):
        raise TypeError(f"Argument 'atten' must be a number, not {type(atten).__name__}.")
    if not 0 <= atten:
        raise ValueError(f"Argument 'atten' must be non-negative, not {atten}.")

    h_ideal = _ideal_bandpass(order, center_freq, bandwidth)
    h_window = _window(order, window, atten=atten)
    h = h_ideal * h_window
    h = _normalize_passband(h, center_freq)

    return h


@export
def design_bandstop_fir(
    order: int,
    center_freq: float,
    bandwidth: float,
    window: None
    | Literal["hamming", "hann", "blackman", "blackman-harris", "chebyshev", "kaiser"]
    | npt.ArrayLike = "hamming",
    atten: float = 60,
) -> npt.NDArray[np.float_]:
    r"""
    Designs a bandstop FIR filter impulse response $h[n]$ using the window method.

    Arguments:
        order: The filter order $N$. Must be even.
        center_freq: The center frequency $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        bandwidth: The two-sided bandwidth about $f_{center}$, normalized to the Nyquist frequency $f_s / 2$.
        window: The time-domain window to use.

            - `None`: No windowing. Equivalently, a length-$N + 1$ vector of ones.
            - `"hamming"`: Hamming window, see :func:`scipy.signal.windows.hamming`.
            - `"hann"`: Hann window, see :func:`scipy.signal.windows.hann`.
            - `"blackman"`: Blackman window, see :func:`scipy.signal.windows.blackman`.
            - `"blackman-harris"`: Blackman-Harris window, see :func:`scipy.signal.windows.blackmanharris`.
            - `"chebyshev"`: Chebyshev window, see :func:`scipy.signal.windows.chebwin`.
            - `"kaiser"`: Kaiser window, see :func:`scipy.signal.windows.kaiser`.
            - `npt.ArrayLike`: A custom window. Must be a length-$N + 1$ vector.

        atten: The sidelobe attenuation in dB. Only used if `window` is `"chebyshev"` or `"kaiser"`.

    Returns:
        The filter impulse response $h[n]$ with length $N + 1$. The center of the larger passband has 0 dB gain.

    References:
        - https://www.mathworks.com/help/dsp/ref/designbandstopfir.html

    Examples:
        Design a length-101 bandstop FIR filter with center frequency $f_{center} = 0.4 \cdot f_s / 2$
        and bandwidth $0.75 \cdot f_s / 2$, using a Hamming window.

        .. ipython:: python

            h_hamming = sdr.design_bandstop_fir(100, 0.4, 0.75, window="hamming")

            @savefig sdr_design_bandstop_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_hamming);

            @savefig sdr_design_bandstop_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided");

        Compare filter designs using different windows.

        .. ipython:: python

            h_hann = sdr.design_bandstop_fir(100, 0.4, 0.75, window="hann"); \
            h_blackman = sdr.design_bandstop_fir(100, 0.4, 0.75, window="blackman"); \
            h_blackman_harris = sdr.design_bandstop_fir(100, 0.4, 0.75, window="blackman-harris"); \
            h_chebyshev = sdr.design_bandstop_fir(100, 0.4, 0.75, window="chebyshev"); \
            h_kaiser = sdr.design_bandstop_fir(100, 0.4, 0.75, window="kaiser")

            @savefig sdr_design_bandstop_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_hamming, x_axis="one-sided", label="Hamming"); \
            sdr.plot.magnitude_response(h_hann, x_axis="one-sided", label="Hann"); \
            sdr.plot.magnitude_response(h_blackman, x_axis="one-sided", label="Blackman"); \
            sdr.plot.magnitude_response(h_blackman_harris, x_axis="one-sided", label="Blackman-Harris"); \
            sdr.plot.magnitude_response(h_chebyshev, x_axis="one-sided", label="Chebyshev"); \
            sdr.plot.magnitude_response(h_kaiser, x_axis="one-sided", label="Kaiser"); \
            plt.legend(); \
            plt.ylim(-100, 10);

    Group:
        dsp-fir-filtering
    """
    if not isinstance(order, int):
        raise TypeError(f"Argument 'order' must be an integer, not {type(order).__name__}.")
    if not order % 2 == 0:
        raise ValueError(f"Argument 'order' must be even, not {order}.")

    if not isinstance(center_freq, (int, float)):
        raise TypeError(f"Argument 'center_freq' must be a number, not {type(center_freq).__name__}.")
    if not 0 <= center_freq <= 1:
        raise ValueError(f"Argument 'center_freq' must be between 0 and 1, not {center_freq}.")

    if not isinstance(bandwidth, (int, float)):
        raise TypeError(f"Argument 'bandwidth' must be a number, not {type(bandwidth).__name__}.")
    if not 0 <= center_freq + bandwidth / 2 <= 1:
        raise ValueError(f"Argument 'bandwidth' must be between 0 and 0.5 - 'center_freq', not {bandwidth}.")
    if not 0 <= center_freq - bandwidth / 2 <= 1:
        raise ValueError(f"Argument 'bandwidth' must be between 0 and 'center_freq', not {bandwidth}.")

    if not isinstance(atten, (int, float)):
        raise TypeError(f"Argument 'atten' must be a number, not {type(atten).__name__}.")
    if not 0 <= atten:
        raise ValueError(f"Argument 'atten' must be non-negative, not {atten}.")

    h_ideal = _ideal_bandstop(order, center_freq, bandwidth)
    h_window = _window(order, window, atten=atten)
    h = h_ideal * h_window
    if center_freq > 0.5:
        h = _normalize_passband(h, 0)
    else:
        h = _normalize_passband(h, 1)

    return h


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
            sdr.plot.magnitude_response(h_8, x_axis="one-sided"); \
            plt.ylim(-4, 1);

            @savefig sdr_design_frac_delay_fir_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.group_delay(h_8, x_axis="one-sided");

        Compare the magnitude response and group delay of filters with different lengths.

        .. ipython:: python

            h_16 = sdr.design_frac_delay_fir(16, 0.25); \
            h_32 = sdr.design_frac_delay_fir(32, 0.25); \
            h_64 = sdr.design_frac_delay_fir(64, 0.25)

            @savefig sdr_design_frac_delay_fir_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_8, x_axis="one-sided", label="Length 8"); \
            sdr.plot.magnitude_response(h_16, x_axis="one-sided", label="Length 16"); \
            sdr.plot.magnitude_response(h_32, x_axis="one-sided", label="Length 32"); \
            sdr.plot.magnitude_response(h_64, x_axis="one-sided", label="Length 64"); \
            plt.legend(); \
            plt.ylim(-4, 1);

            @savefig sdr_design_frac_delay_fir_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.group_delay(h_8, x_axis="one-sided", label="Length 8"); \
            sdr.plot.group_delay(h_16, x_axis="one-sided", label="Length 16"); \
            sdr.plot.group_delay(h_32, x_axis="one-sided", label="Length 32"); \
            sdr.plot.group_delay(h_64, x_axis="one-sided", label="Length 64"); \
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
