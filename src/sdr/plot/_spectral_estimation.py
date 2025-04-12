"""
A module containing various spectral estimation techniques.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._conversion import db
from .._helper import export, verify_arraylike, verify_isinstance, verify_literal, verify_scalar
from ._helper import freq_x_axis, freq_y_axis, time_x_axis, verify_sample_rate
from ._rc_params import RC_PARAMS


@export
def periodogram(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | float | tuple | None = "hann",
    length: int | None = None,
    overlap: int | None = None,
    fft: int | None = None,
    detrend: Literal["constant", "linear", False] = False,
    average: Literal["mean", "median"] = "mean",
    ax: plt.Axes | None = None,
    x_axis: Literal["auto", "one-sided", "two-sided", "log"] = "auto",
    y_axis: Literal["linear", "log"] = "log",
    **kwargs,
):
    r"""
    Plots the estimated power spectral density $P_{xx}$ of a time-domain signal $x[n]$ using Welch's method.

    Note:
        This function uses :func:`scipy.signal.welch()` to estimate the power spectral density of the
        time-domain signal.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        detrend: The type of detrending to apply. Options are to remove the mean or a linear trend from each segment.
        average: The type of averaging to use. Options are to average the periodograms using the mean or median.
        ax: The axis to plot on. If `None`, the current axis is used.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis. The default is `"auto"` which selects `"one-sided"`
            for real-valued signals and `"two-sided"` for complex-valued signals.
        y_axis: The y-axis scaling. Options are to display a linear or logarithmic power spectral density.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Group:
        plot-spectral-estimation
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    # verify_isinstance(window, str, optional=True)
    verify_scalar(length, optional=True, int=True, positive=True)
    verify_scalar(overlap, optional=True, int=True, non_negative=True)
    verify_scalar(fft, optional=True, int=True, positive=True)
    verify_literal(detrend, ["constant", "linear", False])
    verify_literal(average, ["mean", "median"])
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(x_axis, ["auto", "one-sided", "two-sided", "log"])
    verify_literal(y_axis, ["linear", "log"])

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if x_axis == "auto":
            x_axis = "one-sided" if np.isrealobj(x) else "two-sided"

        f, Pxx = scipy.signal.welch(
            x,
            fs=sample_rate,
            window=window,
            nperseg=length,
            noverlap=overlap,
            nfft=fft,
            detrend=detrend,
            return_onesided=x_axis != "two-sided",
            average=average,
        )

        if y_axis == "log":
            Pxx = db(Pxx)

        if x_axis == "two-sided":
            f[f >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
            f = np.fft.fftshift(f)
            Pxx = np.fft.fftshift(Pxx)

        if x_axis == "log":
            ax.semilogx(f, Pxx, **kwargs)
        else:
            ax.plot(f, Pxx, **kwargs)

        ax.grid(True, which="both")
        if "label" in kwargs:
            ax.legend()
        freq_x_axis(ax, sample_rate_provided)
        if y_axis == "log":
            ax.set_ylabel("Power density (dB/Hz)")
        else:
            ax.set_ylabel("Power density (W/Hz)")
        ax.set_title("Power spectral density")


@export
def spectrogram(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | float | tuple | None = "hann",
    length: int | None = None,
    overlap: int | None = None,
    fft: int | None = None,
    detrend: Literal["constant", "linear", False] = False,
    ax: plt.Axes | None = None,
    y_axis: Literal["auto", "one-sided", "two-sided"] = "auto",
    # persistence: bool = False,
    # colorbar: bool = True,
    **kwargs,
):
    r"""
    Plots the spectrogram of a time-domain signal $x[n]$ using Welch's method.

    Note:
        This function uses :func:`scipy.signal.spectrogram` to estimate the spectrogram of the time-domain signal.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be label as "Samples" and the y-axis as "Normalized frequency".
        window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
            If `None`, no window is applied.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        detrend: The type of detrending to apply. Options are to remove the mean or a linear trend from each segment.
        ax: The axis to plot on. If `None`, the current axis is used.
        y_axis: The y-axis scaling. Options are to display a one-sided spectrum or a two-sided spectrum.
            The default is `"auto"` which selects `"one-sided"` for real-valued signals and `"two-sided"` for
            complex-valued signals.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.pcolormesh`.
            The following keyword arguments are set by default. The defaults may be overwritten.

            - `"vmin"`: 10th percentile
            - `"vmax"`: 100th percentile
            - `"shading"`: `"gouraud"`

    Group:
        plot-spectral-estimation
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    # verify_isinstance(window, str, optional=True)
    verify_scalar(length, optional=True, int=True, positive=True)
    verify_scalar(overlap, optional=True, int=True, non_negative=True)
    verify_scalar(fft, optional=True, int=True, positive=True)
    verify_literal(detrend, ["constant", "linear", False])
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(y_axis, ["auto", "one-sided", "two-sided"])

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if y_axis == "auto":
            y_axis = "one-sided" if np.isrealobj(x) else "two-sided"

        f, t, Sxx = scipy.signal.spectrogram(
            x,
            fs=sample_rate,
            window=window,
            nperseg=length,
            noverlap=overlap,
            nfft=fft,
            detrend=detrend,
            return_onesided=y_axis != "two-sided",
            mode="psd",
        )
        Sxx = db(Sxx)

        if y_axis == "one-sided" and np.iscomplexobj(x):
            # If complex data, the spectrogram always returns a two-sided spectrum. So we need to remove the second half.
            f = f[0 : f.size // 2]
            Sxx = Sxx[0 : Sxx.shape[0] // 2, :]
        if y_axis == "two-sided":
            f = np.fft.fftshift(f)
            Sxx = np.fft.fftshift(Sxx, axes=0)

        default_kwargs = {
            "vmin": np.percentile(Sxx, 10),
            "vmax": np.percentile(Sxx, 100),
            "shading": "gouraud",
        }
        kwargs = {**default_kwargs, **kwargs}

        ax.pcolormesh(t, f, Sxx, **kwargs)

        time_x_axis(ax, sample_rate_provided)
        freq_y_axis(ax, sample_rate_provided)
        ax.set_title("Spectrogram")
