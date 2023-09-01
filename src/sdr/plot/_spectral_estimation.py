"""
A module containing various spectral estimation techniques.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export
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
    x_axis: Literal["one-sided", "two-sided", "log"] = "two-sided",
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
            be labeled as "Normalized Frequency".
        window: The windowing function to use. This can be a string or a vector of length `length`.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        detrend: The type of detrending to apply. Options are to remove the mean or a linear trend from each segment.
        average: The type of averaging to use. Options are to average the periodograms using the mean or median.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Note:
        The default y-axis lower limit is set to the 10th percentile. This is to crop any deep nulls.

    Group:
        plot-spectral-estimation
    """
    if sample_rate is None:
        sample_rate_provided = False
        sample_rate = 1
    else:
        sample_rate_provided = True
        if not isinstance(sample_rate, (int, float)):
            raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")

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
    Pxx = 10 * np.log10(Pxx)

    if x_axis == "two-sided":
        f[f >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)

    if sample_rate_provided:
        units, scalar = freq_units(f)
        f *= scalar

    with plt.rc_context(RC_PARAMS):
        if x_axis == "log":
            plt.semilogx(f, Pxx, **kwargs)
        else:
            plt.plot(f, Pxx, **kwargs)

        plt.grid(True, which="both")
        if "label" in kwargs:
            plt.legend()

        # Avoid deep nulls
        y_max = plt.gca().get_ylim()[1]
        y_min = np.percentile(Pxx, 10)
        plt.ylim(y_min, y_max)

        if sample_rate_provided:
            plt.xlabel(f"Frequency ({units}), $f$")
        else:
            plt.xlabel("Normalized Frequency, $f /f_s$")
        plt.ylabel("Power density (dB/Hz)")
        plt.title("Power spectral density")
        plt.tight_layout()


@export
def spectrogram(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    window: str | npt.ArrayLike = "hann",
    length: int | None = None,
    overlap: int | None = None,
    fft: int | None = None,
    detrend: Literal["constant", "linear", False] = False,
    x_axis: Literal["one-sided", "two-sided"] = "two-sided",
    **kwargs,
):
    r"""
    Plots the spectrogram of a time-domain signal $x[n]$ using Welch's method.

    Note:
        This function uses :func:`scipy.signal.spectrogram()` to estimate the spectrogram of the time-domain signal.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be label as "Samples" and the y-axis as "Normalized Frequency".
        window: The windowing function to use. This can be a string or a vector of length `length`.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        detrend: The type of detrending to apply. Options are to remove the mean or a linear trend from each segment.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum or two-sided spectrum.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.pcolormesh()`.
            The following keyword arguments are set by default. The defaults may be overwritten.

            - `"vmin"`: 10th percentile
            - `"vmax"`: 100th percentile
            - `"shading"`: `"gouraud"`

    Group:
        plot-spectral-estimation
    """
    if sample_rate is None:
        sample_rate_provided = False
        sample_rate = 1
    else:
        sample_rate_provided = True
        if not isinstance(sample_rate, (int, float)):
            raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")

    f, t, Sxx = scipy.signal.spectrogram(
        x,
        fs=sample_rate,
        window=window,
        nperseg=length,
        noverlap=overlap,
        nfft=fft,
        detrend=detrend,
        return_onesided=x_axis != "two-sided",
        mode="psd",
    )
    Sxx = 10 * np.log10(Sxx)

    if x_axis == "one-sided" and np.iscomplexobj(x):
        # If complex data, the spectrogram always returns a two-sided spectrum. So we need to remove the second half.
        f = f[0 : f.size // 2]
        Sxx = Sxx[0 : Sxx.shape[0] // 2, :]
    if x_axis == "two-sided":
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

    plt.pcolormesh(t, f, Sxx, **kwargs)
    if sample_rate_provided:
        plt.xlabel(f"Time ({t_units}), $t$")
        plt.ylabel(f"Frequency ({f_units}), $f$")
    else:
        plt.xlabel("Samples, $n$")
        plt.ylabel("Normalized Frequency, $f /f_s$")
    plt.title("Spectrogram")
    plt.tight_layout()
