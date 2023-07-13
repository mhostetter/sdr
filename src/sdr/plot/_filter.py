"""
A module containing filter-related plotting functions.
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def impulse_response(b: np.ndarray, a: np.ndarray = 1, N: Optional[int] = None, **kwargs):
    r"""
    Plots the impulse response $h[n]$ of a filter.

    The impulse response $h[n]$ is the filter output when the input is an impulse $\delta[n]$.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        N: The number of samples to plot. If `None`, the length of `b` is used for FIR filters and
            100 for IIR filters.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plot-filter
    """
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    if N is None:
        if a.size == 1 and a[0] == 1:
            N = b.size  # FIR filter
        else:
            N = 100  # IIR filter

    # Delta impulse function
    d = np.zeros(N, dtype=np.float32)
    d[0] = 1

    # Filter the impulse
    zi = scipy.signal.lfiltic(b, a, y=[], x=[])
    h, zi = scipy.signal.lfilter(b, a, d, zi=zi)

    with plt.rc_context(RC_PARAMS):
        label = kwargs.pop("label", None)
        if np.iscomplexobj(h):
            if label is None:
                label = "real"
                label2 = "imag"
            else:
                label = label + " (real)"
                label2 = label + " (imag)"
            plt.plot(np.arange(h.size), h.real, label=label, **kwargs)
            plt.plot(np.arange(h.size), h.imag, label=label2, **kwargs)
        else:
            plt.plot(np.arange(h.size), h, label=label, **kwargs)

        if label:
            plt.legend()

        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Impulse Response, $h[n]$")


@export
def step_response(b: np.ndarray, a: np.ndarray = 1, N: Optional[int] = None, **kwargs):
    r"""
    Plots the step response $s[n]$ of a filter.

    The step response $s[n]$ is the filter output when the input is a unit step $u[n]$.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        N: The number of samples to plot. If `None`, the length of `b` is used for FIR filters and
            100 for IIR filters.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plot-filter
    """
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    if N is None:
        if a.size == 1 and a[0] == 1:
            N = b.size  # FIR filter
        else:
            N = 100  # IIR filter

    # Unit step function
    u = np.ones(N, dtype=np.float32)

    # Filter the impulse
    zi = scipy.signal.lfiltic(b, a, y=[], x=[])
    s, zi = scipy.signal.lfilter(b, a, u, zi=zi)

    with plt.rc_context(RC_PARAMS):
        label = kwargs.pop("label", None)
        if np.iscomplexobj(s):
            if label is None:
                label = "real"
                label2 = "imag"
            else:
                label = label + " (real)"
                label2 = label + " (imag)"
            plt.plot(np.arange(s.size), s.real, label=label, **kwargs)
            plt.plot(np.arange(s.size), s.imag, label=label2, **kwargs)
        else:
            plt.plot(np.arange(s.size), s, label=label, **kwargs)

        if label:
            plt.legend()

        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Step Response, $s[n]$")


@export
def zeros_poles(b: np.ndarray, a: np.ndarray = 1, **kwargs):
    r"""
    Plots the zeros and poles of the filter.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plot-filter
    """
    z, p, _ = scipy.signal.tf2zpk(b, a)
    unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 100))

    label = kwargs.pop("label", None)
    if label is None:
        z_label = "Zeros"
        p_label = "Poles"
    else:
        z_label = label + " (zeros)"
        p_label = label + " (poles)"

    with plt.rc_context(RC_PARAMS):
        plt.plot(unit_circle.real, unit_circle.imag, color="k", linestyle="--", label="Unit circle")
        plt.scatter(z.real, z.imag, marker="o", label=z_label)
        plt.scatter(p.real, p.imag, marker="x", label=p_label)
        plt.axis("equal")
        plt.legend()
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.title("Zeros and Poles of $H(z)$")


@export
def frequency_response(
    b: np.ndarray,
    a: np.ndarray = 1,
    sample_rate: float = 1.0,
    N: int = 1024,
    x_axis: Literal["one-sided", "two-sided", "log"] = "two-sided",
    decades: int = 4,
    **kwargs,
):
    r"""
    Plots the frequency response $H(e^{j\omega})$ of the filter.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        sample_rate: The sample rate of the filter in samples/s.
        N: The number of samples in the frequency response.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis.
        decades: The number of decades to plot when `x_axis="log"`.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plot-filter
    """
    if x_axis == "log":
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, H = scipy.signal.freqz(b, a, worN=w, whole=False, fs=sample_rate)
    else:
        w, H = scipy.signal.freqz(b, a, worN=N, whole=x_axis == "two-sided", fs=sample_rate)

    if x_axis == "two-sided":
        w[w >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        w = np.fft.fftshift(w)
        H = np.fft.fftshift(H)

    with plt.rc_context(RC_PARAMS):
        if x_axis == "log":
            plt.semilogx(w, 10 * np.log10(np.abs(H) ** 2), **kwargs)
        else:
            plt.plot(w, 10 * np.log10(np.abs(H) ** 2), **kwargs)

        plt.grid(True, which="both")

        if sample_rate == 1.0:
            plt.xlabel("Normalized Frequency, $f /f_s$")
        else:
            plt.xlabel("Frequency (Hz), $f$")
        plt.ylabel(r"Power (dB), $|H(\omega)|^2$")
        plt.title(r"Frequency Response, $H(\omega)$")


@export
def group_delay(
    b: np.ndarray,
    a: np.ndarray = 1,
    sample_rate: float = 1.0,
    N: int = 1024,
    x_axis: Literal["one-sided", "two-sided", "log"] = "two-sided",
    decades: int = 4,
    **kwargs,
):
    r"""
    Plots the group delay $\tau_g(\omega)$ of the IIR filter.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        sample_rate: The sample rate of the filter in samples/s.
        N: The number of samples in the frequency response.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis.
        decades: The number of decades to plot when `x_axis="log"`.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plot-filter
    """
    if x_axis == "log":
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, tau_g = scipy.signal.group_delay((b, a), w=w, whole=False, fs=sample_rate)
    else:
        w, tau_g = scipy.signal.group_delay((b, a), w=N, whole=x_axis == "two-sided", fs=sample_rate)

    if x_axis == "two-sided":
        w[w >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        w = np.fft.fftshift(w)
        tau_g = np.fft.fftshift(tau_g)

    with plt.rc_context(RC_PARAMS):
        if x_axis == "log":
            plt.semilogx(w, tau_g, **kwargs)
        else:
            plt.plot(w, tau_g, **kwargs)

        plt.grid(True, which="both")

        if sample_rate == 1.0:
            plt.xlabel("Normalized Frequency, $f /f_s$")
        else:
            plt.xlabel("Frequency (Hz), $f$")
        plt.ylabel(r"Group Delay (samples), $\tau_g(\omega)$")
        plt.title(r"Group Delay, $\tau_g(\omega)$")


@export
def filter(  # pylint: disable=redefined-builtin
    b: np.ndarray,
    a: np.ndarray = 1,
    sample_rate: float = 1.0,
    N_time: Optional[int] = None,
    N_freq: int = 1024,
    x_axis: Literal["one-sided", "two-sided", "log"] = "two-sided",
    decades: int = 4,
):
    r"""
    Plots the frequency response $H(e^{j\omega})$, impulse response $h[n]$, step response $s[n]$,
    and zeros and poles of the filter.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        sample_rate: The sample rate of the filter in samples/s.
        N_time: The number of samples in the time domain. If `None`, the length of `b` is used
            for FIR filters and 100 for IIR filters.
        N_freq: The number of samples in the frequency response.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis.
        decades: The number of decades to plot when `x_axis="log"`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plot-filter
    """
    with plt.rc_context(RC_PARAMS):
        plt.subplot2grid((4, 3), (0, 0), 2, 3)
        frequency_response(b, a, sample_rate=sample_rate, N=N_freq, x_axis=x_axis, decades=decades)

        plt.subplot2grid((4, 3), (2, 0), 2, 1)
        zeros_poles(b, a)

        plt.subplot2grid((4, 3), (2, 1), 1, 2)
        impulse_response(b, a, N=N_time)

        plt.subplot2grid((4, 3), (3, 1), 1, 2)
        step_response(b, a, N=N_time)

        plt.tight_layout()
