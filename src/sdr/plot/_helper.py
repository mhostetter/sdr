from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal


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
