"""
A module containing frequency-domain plotting functions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.signal
from typing_extensions import Literal

from .._helper import export, verify_arraylike, verify_bool, verify_isinstance, verify_literal, verify_scalar
from ._helper import integer_x_axis, standard_plot, verify_sample_rate
from ._rc_params import RC_PARAMS


@export
def dft(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | float | tuple | None = None,
    size: int | None = None,
    oversample: int | None = None,
    fast: bool = False,
    centered: bool = True,
    ax: plt.Axes | None = None,
    type: Literal["plot", "stem"] = "plot",
    x_axis: Literal["freq", "bin"] = "freq",
    y_axis: Literal["complex", "mag", "mag^2", "db"] = "mag",
    diff: Literal["color", "line"] = "color",
    **kwargs,
):
    r"""
    Plots the discrete Fourier transform (DFT) of the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.
        size: The number of points to use for the DFT. If `None`, the length of the signal is used.
        oversample: The factor to oversample the DFT. If `None`, the DFT is not oversampled. This is only considered
            if `size` is `None`.
        fast: Indicates whether to use the fast Fourier transform (FFT) algorithm. If `True`, the DFT size is set
            to the next power of 2.
        centered: Indicates whether to center the DFT about 0.
        ax: The axis to plot on. If `None`, the current axis is used.
        type: The type of plot to use.
        x_axis: The x-axis scaling.
        y_axis: The y-axis scaling.
        diff: Indicates how to differentiate the real and imaginary parts of a complex signal. If `"color"`, the
            real and imaginary parts will have different colors based on the current Matplotlib color cycle.
            If `"line"`, the real part will have a solid line and the imaginary part will have a dashed line,
            and both lines will share the same color.
        kwargs: Additional keyword arguments to pass to the plotting function.

    See Also:
        sdr.plot.dtft

    Notes:
        The discrete Fourier transform (DFT) is defined as

        $$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2 \pi k n / N},$$

        where $x[n]$ is the time-domain signal, $X[k]$ is the DFT, $k$ is the DFT frequency bin, and $N$ is the number
        of samples in the DFT. The frequency corresponding to the k-th bin is $\frac{k}{N} f_s$.

        The DFT is a sampled version of the discrete-time Fourier transform (DTFT).

    Examples:
        Create a tone whose frequency will straddle two DFT bins.

        .. ipython:: python

            n = 10  # samples
            f = 1.5 / n  # cycles/sample
            x = np.exp(1j * 2 * np.pi * f * np.arange(n))

            @savefig sdr_plot_dft_1.png
            plt.figure(); \
            sdr.plot.dft(x, x_axis="bin", type="stem");

        Plot the DFT against normalized frequency. Compare the DTFT, an oversampled DFT, and a critically sampled DFT.
        Notice that the critically sampled DFT has scalloping loss (difference between the true peak and the DFT peak)
        when the signal frequency does not align with a bin. Furthermore, the sidelobes are non-zero and large.
        This is known as spectral leakage -- the spreading of a tone's power from a single bins to all bins.

        .. ipython:: python

            @savefig sdr_plot_dft_2.png
            plt.figure(); \
            sdr.plot.dtft(x, color="k", label="DTFT"); \
            sdr.plot.dft(x, oversample=4, type="stem", label="4x oversampled DFT"); \
            sdr.plot.dft(x, type="stem", label="DFT");

        If a window is applied to the signal before the DFT is computed, the main lobe is widened and the sidelobes are
        reduced. Given the wider main lobe, the scalloping loss of the critically sampled DFT is also reduced. These
        benefits come at the cost of reduced frequency resolution.

        .. ipython:: python

            @savefig sdr_plot_dft_3.png
            plt.figure(); \
            sdr.plot.dtft(x, window="hamming", color="k", label="DTFT"); \
            sdr.plot.dft(x, oversample=4, window="hamming", type="stem", label="4x oversampled DFT"); \
            sdr.plot.dft(x, window="hamming", type="stem", label="DFT");

    Group:
        plot-frequency-domain
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    # verify_isinstance(window, str, optional=True)
    verify_scalar(size, optional=True, int=True, positive=True)
    verify_scalar(oversample, optional=True, int=True, positive=True)
    verify_bool(fast)
    verify_bool(centered)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(type, ["plot", "stem"])
    verify_literal(x_axis, ["freq", "bin"])
    verify_literal(y_axis, ["complex", "mag", "mag^2", "db"])
    verify_literal(diff, ["color", "line"])

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if window is not None:
            x *= scipy.signal.windows.get_window(window, x.size)

        if size is None:
            if oversample is None:
                size = x.size
            else:
                size = x.size * oversample

        if fast:
            size = scipy.fft.next_fast_len(size)

        X = np.fft.fft(x, size)
        if x_axis == "freq":
            f = np.fft.fftfreq(size, 1 / sample_rate)
        elif x_axis == "bin":
            f = np.fft.fftfreq(size, 1 / size)

        if centered:
            X = np.fft.fftshift(X)
            f = np.fft.fftshift(f)

        standard_plot(f, X, ax=ax, y_axis=y_axis, diff=diff, type=type, **kwargs)

        if x_axis == "bin":
            integer_x_axis(ax)
            ax.set_xlabel("DFT bin index, $k$")
        elif sample_rate_provided:
            ax.set_xlabel("Frequency (Hz), $f$")
        else:
            ax.set_xlabel("Normalized frequency, $f/f_s$")

        if y_axis == "complex":
            ax.set_ylabel("Amplitude, $X[k]$")
        elif y_axis == "mag":
            ax.set_ylabel(r"Magnitude, $\left| X[k] \right|$")
        elif y_axis == "mag^2":
            ax.set_ylabel(r"Power, $\left| X[k] \right|^2$")
        elif y_axis == "db":
            ax.set_ylabel(r"Power (dB), $\left| X[k] \right|^2$")

        ax.set_title("Discrete Fourier transform (DFT)")


@export
def dtft(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | float | tuple | None = None,
    size: int = int(2**20),  # ~1 million points
    centered: bool = True,
    ax: plt.Axes | None = None,
    y_axis: Literal["complex", "mag", "mag^2", "db"] = "mag",
    diff: Literal["color", "line"] = "color",
    **kwargs,
):
    r"""
    Plots the discrete-time Fourier transform (DTFT) of the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.
        size: The number of points to use for the DTFT. The actual size used will be the nearest power of 2.
        centered: Indicates whether to center the DTFT about 0.
        ax: The axis to plot on. If `None`, the current axis is used.
        y_axis: The y-axis scaling.
        diff: Indicates how to differentiate the real and imaginary parts of a complex signal. If `"color"`, the
            real and imaginary parts will have different colors based on the current Matplotlib color cycle.
            If `"line"`, the real part will have a solid line and the imaginary part will have a dashed line,
            and both lines will share the same color.
        kwargs: Additional keyword arguments to pass to the plotting function.

    See Also:
        sdr.plot.dft

    Notes:
        The discrete Fourier transform (DTFT) is defined as

        $$X(f) = \sum_{n=-\infty}^{\infty} x[n] e^{-j 2 \pi f n / f_s},$$

        where $x[n]$ is the time-domain signal, $X(f)$ is the DTFT, and $f$ is the frequency.

    Examples:
        Create a DC tone that is 10 samples long. Plot its DTFT. Notice that the width of the main lobe is $2 / T$,
        with nulls at $\pm 1 / T$.

        .. ipython:: python

            n = 10  # samples
            f = 0 / n  # cycles/sample
            x = np.exp(1j * 2 * np.pi * f * np.arange(n))

            @savefig sdr_plot_dtft_1.png
            plt.figure(); \
            sdr.plot.dtft(x);

        Plot a critically sampled DFT and an oversampled DFT of the signal. Notice that the DFT is a sampled version of
        the DTFT. The oversampled DFT has more samples and thus more closely resembles the DTFT.

        .. ipython:: python

            @savefig sdr_plot_dtft_2.png
            plt.figure(); \
            sdr.plot.dtft(x, label="DTFT"); \
            sdr.plot.dft(x, oversample=4, type="stem", label="4x oversampled DFT"); \
            sdr.plot.dft(x, type="stem", label="DFT");

    Group:
        plot-frequency-domain
    """
    with plt.rc_context(RC_PARAMS):
        dft(
            x,
            sample_rate=sample_rate,
            window=window,
            size=size,
            fast=True,
            centered=centered,
            ax=ax,
            type="plot",
            x_axis="freq",
            y_axis=y_axis,
            diff=diff,
            **kwargs,
        )

        ax = plt.gca()

        if y_axis == "complex":
            ax.set_ylabel("Amplitude, $X(f)$")
        elif y_axis == "mag":
            ax.set_ylabel(r"Magnitude, $\left| X(f) \right|$")
        elif y_axis == "mag^2":
            ax.set_ylabel(r"Power, $\left| X(f) \right|^2$")
        elif y_axis == "db":
            ax.set_ylabel(r"Power (dB), $\left| X(f) \right|^2$")

        ax.set_title("Discrete-time Fourier transform (DTFT)")
