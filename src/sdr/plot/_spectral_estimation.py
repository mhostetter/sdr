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
from .._helper import export
from ._helper import integer_x_axis, process_sample_rate
from ._rc_params import RC_PARAMS
from ._units import freq_units, time_units


@export
def periodogram(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | npt.ArrayLike = "hann",
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
        window: The windowing function to use. This can be a string or a vector of length `length`.
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
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plot-spectral-estimation
    """
    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        x = np.asarray(x)

        if not x_axis in ["auto", "one-sided", "two-sided", "log"]:
            raise ValueError(f"Argument 'x_axis' must be 'auto', 'one-sided', 'two-sided', or 'log', not {x_axis!r}.")
        if x_axis == "auto":
            x_axis = "one-sided" if np.isrealobj(x) else "two-sided"

        if not y_axis in ["linear", "log"]:
            raise ValueError(f"Argument 'y_axis' must be 'linear' or 'log', not {y_axis!r}.")

        sample_rate, sample_rate_provided = process_sample_rate(sample_rate)

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

        if sample_rate_provided:
            units, scalar = freq_units(f)
            f *= scalar

        if x_axis == "log":
            ax.semilogx(f, Pxx, **kwargs)
        else:
            ax.plot(f, Pxx, **kwargs)

        ax.grid(True, which="both")
        if "label" in kwargs:
            ax.legend()

        if sample_rate_provided:
            ax.set_xlabel(f"Frequency ({units}), $f$")
        else:
            ax.set_xlabel("Normalized frequency, $f / f_s$")

        if y_axis == "log":
            ax.set_ylabel("Power density (dB/Hz)")
        else:
            ax.set_ylabel("Power density (W/Hz)")

        ax.set_title("Power spectral density")


@export
def spectrogram(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | npt.ArrayLike = "hann",
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
        This function uses :func:`scipy.signal.spectrogram()` to estimate the spectrogram of the time-domain signal.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be label as "Samples" and the y-axis as "Normalized frequency".
        window: The windowing function to use. This can be a string or a vector of length `length`.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        detrend: The type of detrending to apply. Options are to remove the mean or a linear trend from each segment.
        ax: The axis to plot on. If `None`, the current axis is used.
        y_axis: The y-axis scaling. Options are to display a one-sided spectrum or a two-sided spectrum.
            The default is `"auto"` which selects `"one-sided"` for real-valued signals and `"two-sided"` for
            complex-valued signals.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.pcolormesh()`.
            The following keyword arguments are set by default. The defaults may be overwritten.

            - `"vmin"`: 10th percentile
            - `"vmax"`: 100th percentile
            - `"shading"`: `"gouraud"`

    Group:
        plot-spectral-estimation
    """
    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        x = np.asarray(x)

        if not y_axis in ["auto", "one-sided", "two-sided"]:
            raise ValueError(f"Argument 'y_axis' must be 'auto', 'one-sided', or 'two-sided', not {y_axis!r}.")
        if y_axis == "auto":
            y_axis = "one-sided" if np.isrealobj(x) else "two-sided"

        sample_rate, sample_rate_provided = process_sample_rate(sample_rate)

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
            # f[f >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
            f = np.fft.fftshift(f)
            Sxx = np.fft.fftshift(Sxx)

        if sample_rate_provided:
            f_units, scalar = freq_units(f)
            f *= scalar
            t_units, scalar = time_units(t)
            t *= scalar

        default_kwargs = {
            "vmin": np.percentile(Sxx, 10),
            "vmax": np.percentile(Sxx, 100),
            "shading": "gouraud",
        }
        kwargs = {**default_kwargs, **kwargs}

        ax.pcolormesh(t, f, Sxx, **kwargs)
        if sample_rate_provided:
            ax.set_xlabel(f"Time ({t_units}), $t$")
            ax.set_ylabel(f"Frequency ({f_units}), $f$")
        else:
            integer_x_axis(ax)
            ax.set_xlabel("Samples, $n$")
            ax.set_ylabel("Normalized frequency, $f / f_s$")
        ax.set_title("Spectrogram")
