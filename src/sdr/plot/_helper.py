from __future__ import annotations

from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@overload
def stem(
    x: npt.NDArray,
    *,
    ax: plt.Axes | None = None,
    **kwargs,
): ...


@overload
def stem(
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    ax: plt.Axes | None = None,
    **kwargs,
): ...


@export
def stem(  # noqa: D417
    *args,
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Wraps :func:`matplotlib.pyplot.stem` to style the plot more like MATLAB.

    Arguments:
        x: The x-coordinates of the stem plot.
        y: The y-coordinates of the stem plot.
        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.stem`.

    Notes:
        This function is a wrapper around :func:`matplotlib.pyplot.stem` that styles the plot more like MATLAB's
        `stem()`. It removes the base line and sets the stem line color and marker color to the next color in the
        current color cycle. It also sets the marker face color to none.

    Examples:
        Compare the :func:`matplotlib.pyplot.stem` plot to the styled :func:`sdr.plot.stem` plot.

        .. ipython:: python

            rrc = sdr.raised_cosine(0.1, 8, 10)

            @savefig sdr_plot_stem_1.png
            plt.figure(); \
            plt.stem(rrc);

            @savefig sdr_plot_stem_2.png
            plt.figure(); \
            sdr.plot.stem(rrc);

        The standard plot is even more unreadable when multiple stem plots are on the same axes.

        .. ipython:: python

            gaussian = sdr.gaussian(0.1, 8, 10)

            @savefig sdr_plot_stem_3.png
            plt.figure(); \
            plt.stem(rrc); \
            plt.stem(gaussian);

            @savefig sdr_plot_stem_4.png
            plt.figure(); \
            sdr.plot.stem(rrc); \
            sdr.plot.stem(gaussian);

    Group:
        plot-utility
    """
    with plt.rc_context(RC_PARAMS):
        if len(args) == 1:
            y = args[0]
            x = np.arange(y.size)
        elif len(args) == 2:
            x, y = args
        else:
            raise ValueError(f"Expected 1 or 2 positional arguments, got {len(args)}.")

        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be 1-D, not {x.ndim}-D.")
        if not y.ndim == 1:
            raise ValueError(f"Argument 'y' must be 1-D, not {y.ndim}-D.")

        if ax is None:
            ax = plt.gca()

        # Make a dummy plot to get the next color in the current color cycle
        (line,) = ax.plot([], [])
        color = line.get_color()

        # Plot matplotlib's standard stem plot
        markerline, stemlines, baseline = ax.stem(x, y, **kwargs)

        # Style the stem plot more like MATLAB
        plt.setp(markerline, "markerfacecolor", "none", "markeredgecolor", color)
        plt.setp(stemlines, "color", color, "linewidth", 1)
        plt.setp(baseline, "color", "none", "linewidth", 1)


def real_or_complex_plot(
    ax: plt.Axes,
    t: npt.NDArray,
    x: npt.NDArray,
    diff: Literal["color", "line"] = "color",
    **kwargs,
):
    label = kwargs.pop("label", None)
    if np.iscomplexobj(x):
        if label is None:
            label, label2 = "real", "imag"
        else:
            label, label2 = label + " (real)", label + " (imag)"

        if diff == "color":
            ax.plot(t, x.real, label=label, **kwargs)
            ax.plot(t, x.imag, label=label2, **kwargs)
        elif diff == "line":
            (real,) = ax.plot(t, x.real, "-", label=label, **kwargs)
            kwargs.pop("color", None)
            ax.plot(t, x.imag, "--", color=real.get_color(), label=label2, **kwargs)
        else:
            raise ValueError(f"Argument 'diff' must be 'color' or 'line', not {diff}.")
    else:
        ax.plot(t, x, label=label, **kwargs)

    if label:
        ax.legend()


def min_ylim(y: npt.NDArray, separation: float, sample_rate: float):
    ymin, ymax = plt.gca().get_ylim()
    if ymax - ymin < separation / 4:
        # Find the mean of the signal rounded to the nearest sample
        mean = int(round(np.nanmean(y) / sample_rate))
        ymin = mean - separation / 2
        ymax = mean + separation / 2
        plt.gca().set_ylim(ymin, ymax)
