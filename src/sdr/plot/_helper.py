from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import MaxNLocator
from typing_extensions import Literal

from .._conversion import db
from .._helper import verify_scalar
from ._utility import stem


def real_or_complex_plot(
    x: npt.NDArray,
    y: npt.NDArray,
    ax: plt.Axes,
    diff: Literal["color", "line"] = "color",
    **kwargs,
):
    label = kwargs.pop("label", None)
    if np.iscomplexobj(y):
        if label is None:
            label, label2 = "real", "imag"
        else:
            label, label2 = label + " (real)", label + " (imag)"

        if diff == "color":
            ax.plot(x, y.real, label=label, **kwargs)
            ax.plot(x, y.imag, label=label2, **kwargs)
        elif diff == "line":
            (real,) = ax.plot(x, y.real, "-", label=label, **kwargs)
            kwargs.pop("color", None)
            ax.plot(x, y.imag, "--", color=real.get_color(), label=label2, **kwargs)
        else:
            raise ValueError(f"Argument 'diff' must be 'color' or 'line', not {diff}.")
    else:
        ax.plot(x, y, label=label, **kwargs)

    if label:
        ax.legend()


def standard_plot(
    x: npt.NDArray,
    y: npt.NDArray,
    ax: plt.Axes,
    type: Literal["plot", "stem"] = "plot",
    y_axis: Literal["complex", "mag", "mag^2", "db"] = "mag",
    diff: Literal["color", "line"] = "color",
    **kwargs,
):
    if y_axis == "complex":
        pass
    elif y_axis == "mag":
        y = np.abs(y)
    elif y_axis == "mag^2":
        y = np.abs(y) ** 2
    elif y_axis == "db":
        y = db(np.abs(y) ** 2)
    else:
        raise ValueError(f"Argument 'y_axis' must be 'complex', 'mag', 'mag^2', or 'db', not {y_axis!r}.")

    label = kwargs.pop("label", None)

    if type == "plot":
        if np.iscomplexobj(y):
            if label is None:
                label, label2 = "real", "imag"
            else:
                label, label2 = label + " (real)", label + " (imag)"

            if diff == "color":
                ax.plot(x, y.real, label=label, **kwargs)
                ax.plot(x, y.imag, label=label2, **kwargs)
            elif diff == "line":
                (real,) = ax.plot(x, y.real, "-", label=label, **kwargs)
                kwargs.pop("color", None)
                ax.plot(x, y.imag, "--", color=real.get_color(), label=label2, **kwargs)
            else:
                raise ValueError(f"Argument 'diff' must be 'color' or 'line', not {diff}.")
        else:
            ax.plot(x, y, label=label, **kwargs)

    elif type == "stem":
        stem(x, y, ax=ax, label=label, **kwargs)

    else:
        raise ValueError(f"Argument 'type' must be 'plot' or 'stem', not {type!r}.")

    if label:
        ax.legend()


def verify_sample_rate(sample_rate: float | None, default=1.0):
    if sample_rate is None:
        sample_rate_provided = False
        sample_rate = default
    else:
        sample_rate_provided = True
        verify_scalar(sample_rate, float=True, positive=True)

    return sample_rate, sample_rate_provided


def integer_x_axis(ax: plt.Axes):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(MaxNLocator(integer=True))


def min_ylim(y: npt.NDArray, separation: float, sample_rate: float):
    ymin, ymax = plt.gca().get_ylim()
    if ymax - ymin < separation / 4:
        # Find the mean of the signal rounded to the nearest sample
        mean = int(round(np.nanmean(y) / sample_rate))
        ymin = mean - separation / 2
        ymax = mean + separation / 2
        plt.gca().set_ylim(ymin, ymax)
