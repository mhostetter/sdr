"""
A module containing time-domain plotting functions.
"""

from __future__ import annotations

from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.collections import LineCollection
from typing_extensions import Literal

from .._conversion import db
from .._helper import export
from ._helper import real_or_complex_plot
from ._rc_params import RC_PARAMS
from ._units import time_units


@overload
def time_domain(
    x: npt.NDArray,
    *,
    sample_rate: float | None = None,
    centered: bool = False,
    offset: float = 0.0,
    diff: Literal["color", "line"] = "color",
    **kwargs,
): ...


@overload
def time_domain(
    t: npt.NDArray,
    x: npt.NDArray,
    *,
    sample_rate: float | None = None,
    diff: Literal["color", "line"] = "color",
    **kwargs,
): ...


@export
def time_domain(  # noqa: D417
    *args,
    sample_rate: float | None = None,
    centered: bool = False,
    offset: float = 0.0,
    diff: Literal["color", "line"] = "color",
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots a time-domain signal $x[n]$.

    Arguments:
        t: The time signal $t[n]$. The units are assumed to be $1/f_s$.
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Samples".
        centered: Indicates whether to center the x-axis about 0. This argument is mutually exclusive with
            `offset`.
        offset: The x-axis offset to apply to the first sample. The units of the offset are $1/f_s$.
            This argument is mutually exclusive with `centered`.
        diff: Indicates how to differentiate the real and imaginary parts of a complex signal. If `"color"`, the
            real and imaginary parts will have different colors based on the current Matplotlib color cycle.
            If `"line"`, the real part will have a solid line and the imaginary part will have a dashed line,
            and both lines will share the same color.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Examples:
        Plot a square-root raised cosine (SRRC) pulse shape centered about 0.

        .. ipython:: python

            qpsk = sdr.PSK(4, phase_offset=45, sps=10, pulse_shape="srrc"); \
            pulse_shape = qpsk.pulse_shape

            @savefig sdr_plot_time_domain_1.png
            plt.figure(); \
            sdr.plot.time_domain(pulse_shape, centered=True); \
            plt.title("SRRC pulse shape");

        Plot an imaginary QPSK signal at 10 kS/s.

        .. ipython:: python

            symbols = np.random.randint(0, 4, 50); \
            x = qpsk.modulate(symbols)

            @savefig sdr_plot_time_domain_2.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=10e3); \
            plt.title("SRRC pulse-shaped QPSK");

        Plot non-uniformly sampled data.

        .. ipython:: python

            t = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 55]); \
            x = np.random.randn(t.size)

            @savefig sdr_plot_time_domain_3.png
            plt.figure(); \
            sdr.plot.time_domain(t, x, marker="."); \
            plt.title("Non-uniformly sampled data");

    Group:
        plot-time-domain
    """
    with plt.rc_context(RC_PARAMS):
        if len(args) == 1:
            x = args[0]
            t = None
        elif len(args) == 2:
            t, x = args
        else:
            raise ValueError(f"Expected 1 or 2 positional arguments, got {len(args)}.")

        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be 1-D, not {x.ndim}-D.")

        if ax is None:
            ax = plt.gca()

        if sample_rate is None:
            sample_rate_provided = False
            sample_rate = 1
        else:
            sample_rate_provided = True
            if not isinstance(sample_rate, (int, float)):
                raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")

        if t is None:
            if centered:
                if x.size % 2 == 0:
                    t = np.arange(-x.size // 2, x.size // 2) / sample_rate
                else:
                    t = np.arange(-(x.size - 1) // 2, (x.size + 1) // 2) / sample_rate
            else:
                t = np.arange(x.size) / sample_rate + offset

        if sample_rate_provided:
            units, scalar = time_units(t)
            t *= scalar

        real_or_complex_plot(ax, t, x, diff=diff, **kwargs)
        if sample_rate_provided:
            ax.set_xlabel(f"Time ({units})")
        else:
            ax.set_xlabel("Sample, $n$")
        ax.set_ylabel("Amplitude")


@export
def raster(
    x: npt.NDArray,
    length: int | None = None,
    stride: int | None = None,
    sample_rate: float | None = None,
    color: Literal["index"] | str = "index",
    persistence: bool = False,
    colorbar: bool = True,
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots a raster of the time-domain signal $x[n]$.

    Arguments:
        x: The real time-domain signal $x[n]$. If `x` is 1D, the rastering is determined by `length` and `stride`.
            If `x` is 2D, the rows correspond to each raster.
        length: The length of each raster in samples. This must be provided if `x` is 1D.
        stride: The stride between each raster in samples. If `None`, the stride is set to `length`.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Samples".
        color: Indicates how to color the rasters. If `"index"`, the rasters are colored based on their index.
            If a valid Matplotlib color, the rasters are all colored with that color.
        persistence: Indicates whether to plot the raster as a persistence plot. A persistence plot is a
            2D histogram of the rasters.
        colorbar: Indicates whether to add a colorbar to the plot. This is only added if `color="index"` or
            `persistence=True`.
        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to Matplotlib functions.

            If `persistence=False`, the following keyword arguments are passed to
            :obj:`matplotlib.collections.LineCollection`. The defaults may be overwritten.

            - `"linewidths"`: 1
            - `"linestyles"`: `"solid"`
            - `"cmap"`: `"rainbow"`

            If `persistence=True`, the following keyword arguments are passed to :func:`matplotlib.pyplot.pcolormesh`.
            The defaults may be overwritten.

            - `"bins"`: `(800, 200)  # Passed to np.histogram2d()`
            - `"cmap"`: `"rainbow"`
            - `"norm"`: `"log"`
            - `"rasterized"`: `True`
            - `"show_zero"`: `False`

    Group:
        plot-time-domain
    """
    with plt.rc_context(RC_PARAMS):
        if not x.ndim in [1, 2]:
            raise ValueError(f"Argument 'x' must be 1-D or 2-D, not {x.ndim}-D.")
        if np.iscomplexobj(x):
            raise ValueError("Argument 'x' must be real, not complex.")

        if ax is None:
            ax = plt.gca()

        if x.ndim == 1:
            if not length is not None:
                raise ValueError("Argument 'length' must be specified if 'x' is 1-D.")
            if not isinstance(length, int):
                raise TypeError(f"Argument 'length' must be an integer, not {type(length)}.")
            if not 1 <= length <= x.size:
                raise ValueError(f"Argument 'length' must be at least 1 and less than the length of 'x', not {length}.")

            if stride is None:
                stride = length
            elif not isinstance(stride, int):
                raise TypeError(f"Argument 'stride' must be an integer, not {type(stride)}.")
            elif not 1 <= stride <= x.size:
                raise ValueError(f"Argument 'stride' must be at least 1 and less than the length of 'x', not {stride}.")

            # Compute the strided data and format into segments for LineCollection
            N_rasters = (x.size - length) // stride + 1
            x_strided = np.lib.stride_tricks.as_strided(
                x, shape=(N_rasters, length), strides=(x.strides[0] * stride, x.strides[0]), writeable=False
            )
        else:
            if not length is None:
                raise ValueError("Argument 'length' can not be specified if 'x' is 2-D.")
            if not stride is None:
                raise ValueError("Argument 'stride' can not be specified if 'x' is 2-D.")

            N_rasters, length = x.shape
            x_strided = x

        if sample_rate is None:
            sample_rate_provided = False
            t = np.arange(length)
        else:
            sample_rate_provided = True
            if not isinstance(sample_rate, (int, float)):
                raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")
            t = np.arange(length) / sample_rate
            units, scalar = time_units(t)
            t *= scalar

        if persistence:
            default_kwargs = {
                "bins": (800, 200),
                "cmap": "rainbow",
                "norm": "log",
                "rasterized": True,
                "show_zero": False,
            }
            kwargs = {**default_kwargs, **kwargs}

            bins = kwargs.pop("bins")
            t_fine = np.linspace(t.min(), t.max(), bins[0])
            x_fine = np.concatenate([np.interp(t_fine, t, x_row) for x_row in x_strided])
            t_fine = np.broadcast_to(t_fine, (x_strided.shape[0], bins[0])).ravel()
            h, t_edges, x_edges = np.histogram2d(t_fine, x_fine, bins=bins)

            cmap = kwargs.pop("cmap")  # Need to pop cmap to avoid passing it twice to pcolormesh
            cmap = plt.colormaps[cmap]
            show_zero = kwargs.pop("show_zero")
            if show_zero:
                cmap = cmap.with_extremes(bad=cmap(0))
            else:
                h[h == 0] = np.nan  # Set 0s to NaNs so they don't show up in the plot

            pcm = ax.pcolormesh(t_edges, x_edges, h.T, cmap=cmap, **kwargs)
            if colorbar:
                plt.colorbar(pcm, label="Points", pad=0.05)
        else:
            segments = [np.column_stack([t, x_raster]) for x_raster in x_strided]

            # Set the default keyword arguments and override with user-specified keyword arguments
            default_kwargs = {
                "linewidths": 1,
                "linestyles": "solid",
                "cmap": "rainbow",
            }
            if color == "index":
                default_kwargs["array"] = np.arange(N_rasters)
            else:
                default_kwargs["colors"] = color
            kwargs = {**default_kwargs, **kwargs}

            line_collection = LineCollection(
                segments,
                **kwargs,
            )

            ax.add_collection(line_collection)
            ax.set_xlim(t.min(), t.max())
            ax.set_ylim(x.min(), x.max())

            if colorbar and color == "index":
                plt.colorbar(line_collection, label="Raster Index", pad=0.05)

        if sample_rate_provided:
            ax.set_xlabel(f"Time ({units})")
        else:
            ax.set_xlabel("Sample, $n$")
        ax.set_ylabel("Amplitude")


@export
def correlation(
    x: npt.NDArray,
    y: npt.NDArray,
    sample_rate: float | None = None,
    mode: Literal["full", "valid", "same", "circular"] = "full",
    y_axis: Literal["complex", "mag", "mag^2", "db"] = "mag",
    diff: Literal["color", "line"] = "color",
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the correlation between two time-domain signals $x[n]$ and $y[n]$.

    Arguments:
        x: The first time-domain signal $x[n]$.
        y: The second time-domain signal $y[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Lag (samples)".
        mode: The :func:`numpy.correlate()` correlation mode. If `"circular"`, a circular correlation is computed
            using FFTs.
        y_axis: Indicates how to plot the y-axis. If `"complex"`, the real and imaginary parts are plotted separately.
        diff: Indicates how to differentiate the real and imaginary parts of a complex signal. If `"color"`, the
            real and imaginary parts will have different colors based on the current Matplotlib color cycle.
            If `"line"`, the real part will have a solid line and the imaginary part will have a dashed line,
            and both lines will share the same color.
        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Examples:
        Plot the auto-correlation of a length-63 $m$-sequence. Notice that the linear correlation produces sidelobes
        for non-zero lag.

        .. ipython:: python

            x = sdr.m_sequence(6, output="bipolar")

            @savefig sdr_plot_correlation_1.png
            plt.figure(); \
            sdr.plot.correlation(x, x, mode="full");

        However, the circular correlation only produces magnitudes of 1 for non-zero lag.

        .. ipython:: python

            @savefig sdr_plot_correlation_2.png
            plt.figure(); \
            sdr.plot.correlation(x, x, mode="circular");

    Group:
        plot-time-domain
    """
    with plt.rc_context(RC_PARAMS):
        if not x.ndim == y.ndim == 1:
            raise ValueError(f"Arguments 'x' and 'y' must be 1-D, not {x.ndim}-D and {y.ndim}-D.")

        if ax is None:
            ax = plt.gca()

        if sample_rate is None:
            sample_rate_provided = False
            sample_rate = 1
        else:
            sample_rate_provided = True
            if not isinstance(sample_rate, (int, float)):
                raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")

        if mode == "circular":
            n_fft = max(x.size, y.size)
            X = np.fft.fft(x, n_fft)
            Y = np.fft.fft(y, n_fft)
            corr = np.fft.ifft(X * Y.conj(), n_fft)
            corr = np.fft.fftshift(corr)
            if np.isrealobj(x) and np.isrealobj(y):
                # If both signals are real, the correlation is real
                corr = corr.real
        else:
            corr = np.correlate(x, y, mode=mode)

        if corr.size % 2 == 0:
            t = np.arange(-corr.size // 2, corr.size // 2)  # Lags
        else:
            t = np.arange(-corr.size // 2 + 1, corr.size // 2 + 1)  # Lags

        if sample_rate_provided:
            units, scalar = time_units(t)
            t *= scalar

        if x is y:
            equation = r"r_{xx}[n]"
            if mode == "circular":
                ax.set_title("Periodic auto-correlation function (PACF)")
            else:
                ax.set_title("Auto-correlation function (ACF)")
        else:
            equation = r"r_{xy}[n]"
            if mode == "circular":
                ax.set_title("Periodic cross-correlation function (PCCF)")
            else:
                ax.set_title("Cross-correlation function (CCF)")

        if y_axis == "complex":
            ax.set_ylabel(rf"Correlation, ${equation}$")
        elif y_axis == "mag":
            corr = np.abs(corr)
            ax.set_ylabel(rf"Correlation, $\left| {equation} \right|$")
        elif y_axis == "mag^2":
            corr = np.abs(corr) ** 2
            ax.set_ylabel(rf"Correlation, $\left| {equation} \right|^2$")
        elif y_axis == "db":
            corr = db(np.abs(corr) ** 2)
            ax.set_ylabel(rf"Correlation (dB), $\left| {equation} \right|^2$")
        else:
            raise ValueError(f"Argument 'y_axis' must be 'complex', 'mag', 'mag^2', or 'db', not {y_axis!r}.")

        real_or_complex_plot(ax, t, corr, diff=diff, **kwargs)
        if sample_rate_provided:
            ax.set_xlabel(rf"Lag ({units}), $\Delta t$")
        else:
            ax.set_xlabel(r"Lag (samples), $\Delta n$")
