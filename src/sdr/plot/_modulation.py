"""
A module containing various modulation-related plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def constellation(
    x_hat: npt.ArrayLike,
    heatmap: bool = False,
    limits: tuple[float, float] | None = None,
    **kwargs,
):
    r"""
    Plots the constellation of the complex symbols $\hat{x}[k]$.

    Arguments:
        x_hat: The complex symbols $\hat{x}[k]$.
        heatmap: If `True`, a heatmap is plotted instead of a scatter plot.
        limits: The axis limits, which apply to both the x- and y-axis. If `None`, the axis limits are
            set to 10% larger than the maximum value.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.scatter()` (`heatmap=False`)
            or :func:`matplotlib.pyplot.hist2d()` (`heatmap=True`).
    """
    x_hat = np.asarray(x_hat)

    # Set the axis limits to 10% larger than the maximum value
    if limits is None:
        lim = np.max(np.abs(x_hat)) * 1.1
        limits = (-lim, lim)

    with plt.rc_context(RC_PARAMS):
        if heatmap:
            default_kwargs = {
                "range": (limits, limits),
                "bins": 75,  # Number of bins per axis
            }
            kwargs = {**default_kwargs, **kwargs}
            plt.hist2d(x_hat.real, x_hat.imag, **kwargs)
        else:
            default_kwargs = {
                "marker": ".",
                "linestyle": "none",
            }
            kwargs = {**default_kwargs, **kwargs}
            plt.plot(x_hat.real, x_hat.imag, **kwargs)
        plt.xlabel("In-phase channel, $I$")
        plt.ylabel("Quadrature channel, $Q$")
        plt.axis("square")
        plt.xlim(limits)
        plt.ylim(limits)
        if not heatmap:
            plt.grid(True)
        plt.title("Constellation")
        plt.tight_layout()
