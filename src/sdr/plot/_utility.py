"""
A module containing utility functions for plotting.
"""

from __future__ import annotations

from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .._helper import (
    export,
    verify_arraylike,
    verify_isinstance,
    verify_positional_args,
)
from ._rc_params import RC_PARAMS


@overload
def stem(
    x: npt.NDArray,  # TODO: Change to npt.ArrayLike once Sphinx has better overload support
    *,
    color: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
): ...


@overload
def stem(
    x: npt.NDArray,  # TODO: Change to npt.ArrayLike once Sphinx has better overload support
    y: npt.NDArray,  # TODO: Change to npt.ArrayLike once Sphinx has better overload support
    *,
    color: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
): ...


@export
def stem(  # noqa: D417
    *args,
    color: str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Wraps :func:`matplotlib.pyplot.stem` to style the plot more like MATLAB.

    Arguments:
        x: The x-coordinates of the stem plot.
        y: The y-coordinates of the stem plot.
        color: The color of the stem line and marker. If `None`, the next color in the current color cycle is used.
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
            plt.stem(rrc); \
            plt.title("matplotlib.pyplot.stem()");

            @savefig sdr_plot_stem_2.png
            plt.figure(); \
            sdr.plot.stem(rrc); \
            plt.title("sdr.plot.stem()");

        The standard plot is even more unreadable when multiple stem plots are on the same axes.

        .. ipython:: python

            gaussian = sdr.gaussian(0.1, 8, 10)

            @savefig sdr_plot_stem_3.png
            plt.figure(); \
            plt.stem(rrc); \
            plt.stem(gaussian); \
            plt.title("matplotlib.pyplot.stem()");

            @savefig sdr_plot_stem_4.png
            plt.figure(); \
            sdr.plot.stem(rrc); \
            sdr.plot.stem(gaussian); \
            plt.title("sdr.plot.stem()");

    Group:
        plot-utility
    """
    verify_positional_args(args, 2)
    if len(args) == 1:
        y = verify_arraylike(args[0], complex=True, ndim=1)
        x = np.arange(y.size)
    elif len(args) == 2:
        x = verify_arraylike(args[0], float=True, ndim=1)
        y = verify_arraylike(args[1], complex=True, ndim=1)
    verify_isinstance(color, str, optional=True)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        # Make a dummy plot to get the next color in the current color cycle
        if color is None:
            (line,) = ax.plot([], [])
            color = line.get_color()

        # Plot matplotlib's standard stem plot
        markerline, stemlines, baseline = ax.stem(x, y, **kwargs)

        # Style the stem plot more like MATLAB
        plt.setp(markerline, "markerfacecolor", "none", "markeredgecolor", color)
        plt.setp(stemlines, "color", color, "linewidth", 1)
        plt.setp(baseline, "color", "none", "linewidth", 1)
