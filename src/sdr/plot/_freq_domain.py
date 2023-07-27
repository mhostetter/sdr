"""
A module containing frequency-domain plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def periodogram(
    x: npt.ArrayLike,
    sample_rate: float = 1.0,
    window: str | npt.ArrayLike = "hann",
    length: int | None = None,
    overlap: int | None = None,
    fft: int | None = None,
    average: Literal["mean", "median"] = "mean",
    x_axis: Literal["one-sided", "two-sided", "log"] = "two-sided",
    **kwargs,
):
    r"""
    Plots the estimated power spectral density $P_{xx}$ of a time-domain signal $x[n]$ using Welch's method.

    This function uses :func:`scipy.signal.welch()` to estimate the power spectral density of a time-domain signal.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s.
        window: The windowing function to use. This can be a string or a vector of length `length`.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        average: The type of averaging to use. Options are to average the periodograms using the mean or median.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plot-frequency-domain
    """
    f, Pxx = scipy.signal.welch(
        x,
        fs=sample_rate,
        window=window,
        nperseg=length,
        noverlap=overlap,
        nfft=fft,
        return_onesided=x_axis != "two-sided",
        average=average,
    )
    Pxx = 10 * np.log10(np.abs(Pxx) ** 2)

    if x_axis == "two-sided":
        f[f >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)

    with plt.rc_context(RC_PARAMS):
        if x_axis == "log":
            plt.semilogx(f, Pxx, **kwargs)
        else:
            plt.plot(f, Pxx, **kwargs)

        plt.grid(True, which="both")
        if "label" in kwargs:
            plt.legend()

        # y_max = plt.gca().get_ylim()[1]
        # y_min = np.percentile(Pxx, 0.2)
        # plt.ylim(y_min, y_max)

        if sample_rate == 1.0:
            plt.xlabel("Normalized Frequency, $f /f_s$")
        else:
            plt.xlabel("Frequency (Hz), $f$")
        plt.ylabel("Power density (dB/Hz)")
        plt.title("Power spectral density")
        plt.tight_layout()


@export
def spectrogram(
    x: npt.ArrayLike,
    sample_rate: float = 1.0,
    window: str | npt.ArrayLike = "hann",
    length: int | None = None,
    overlap: int | None = None,
    fft: int | None = None,
    x_axis: Literal["one-sided", "two-sided"] = "two-sided",
    **kwargs,
):
    r"""
    Plots the spectrogram of a time-domain signal $x[n]$ using Welch's method.

    This function uses :func:`scipy.signal.spectrogram()` to estimate the spectrogram of a time-domain signal.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If the sample rate is 1, the x-axis will
            be label as "Samples" and the y-axis as "Normalized Frequency".
        window: The windowing function to use. This can be a string or a vector of length `length`.
        length: The length of each segment in samples. If `None`, the length is set to 256.
        overlap: The number of samples to overlap between segments. If `None`, the overlap is set to `length // 2`.
        fft: The number of points to use in the FFT. If `None`, the FFT length is set to `length`.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum or two-sided spectrum.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.pcolormesh()`.

    Notes:
        By default, the color bar range is from the 10th to 100th percentile. This is to provide more dynamic range
        in the "active" values, and not the noise. This can be changed by passing `vmin` and `vmax` as keyword
        arguments.

    Group:
        plot-frequency-domain
    """
    f, t, Sxx = scipy.signal.spectrogram(
        x,
        fs=sample_rate,
        window=window,
        nperseg=length,
        noverlap=overlap,
        nfft=fft,
        return_onesided=x_axis != "two-sided",
        mode="psd",
    )
    Sxx = 10 * np.log10(np.abs(Sxx))

    if x_axis == "one-sided" and np.iscomplexobj(x):
        # If complex data, the spectrogram always returns a two-sided spectrum. So we need to remove the second half.
        f = f[0 : f.size // 2]
        Sxx = Sxx[0 : Sxx.shape[0] // 2, :]
    if x_axis == "two-sided":
        # f[f >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx)

    # Add the default kwargs to the kwargs passed to the function. If there are any conflicts, the kwargs passed
    # to the function take precedence.
    default_kwargs = {
        "vmin": np.percentile(Sxx, 10),
        "vmax": np.percentile(Sxx, 100),
        "shading": "gouraud",
    }
    kwargs = {**default_kwargs, **kwargs}

    plt.pcolormesh(t, f, Sxx, **kwargs)
    if sample_rate == 1.0:
        plt.xlabel("Samples, $n$")
        plt.ylabel("Normalized Frequency, $f /f_s$")
    else:
        plt.xlabel("Time (s), $t$")
        plt.ylabel("Frequency (Hz), $f$")
    plt.title("Spectrogram")
    plt.tight_layout()
