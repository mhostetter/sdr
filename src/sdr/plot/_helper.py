from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import ticker
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


def sample_units(min: float, max: float) -> tuple[str, float]:
    """
    Determines the appropriate time units to use for a given sample array.
    """
    max_sample = np.max([np.abs(max), np.abs(min)])

    if max_sample > 1e12:
        scalar = 1e-12
        units = "Tsamples"
    elif max_sample > 1e9:
        scalar = 1e-9
        units = "Gsamples"
    elif max_sample > 1e6:
        scalar = 1e-6
        units = "Msamples"
    elif max_sample > 1e3:
        scalar = 1e-3
        units = "ksamples"
    else:
        scalar = 1
        units = "samples"

    return units, scalar


def time_units(min: float, max: float) -> tuple[str, float]:
    """
    Determines the appropriate time units to use for a given time array.
    """
    max_time = np.max([np.abs(max), np.abs(min)])

    if max_time > 1:
        scalar = 1
        units = "s"
    elif max_time > 1e-3:
        scalar = 1e3
        units = "ms"
    elif max_time > 1e-6:
        scalar = 1e6
        units = "μs"
    elif max_time > 1e-9:
        scalar = 1e9
        units = "ns"
    elif max_time > 1e-12:
        scalar = 1e12
        units = "ps"
    else:
        scalar = 1e15
        units = "fs"

    return units, scalar


def freq_units(min: float, max: float) -> tuple[str, float]:
    """
    Determines the appropriate frequency units to use for a given frequency array.
    """
    max_freq = np.max([np.abs(max), np.abs(min)])

    if max_freq > 1e12:
        scalar = 1e-12
        units = "THz"
    elif max_freq > 1e9:
        scalar = 1e-9
        units = "GHz"
    elif max_freq > 1e6:
        scalar = 1e-6
        units = "MHz"
    elif max_freq > 1e3:
        scalar = 1e-3
        units = "kHz"
    elif max_freq >= 0.5:
        scalar = 1
        units = "Hz"
    elif max_freq > 1e-3:
        scalar = 1e3
        units = "mHz"
    else:
        scalar = 1e6
        units = "μHz"

    return units, scalar


def smart_scaled_formatter(val, pos, scalar):
    """
    Format the tick label based on the scale factor and the value.

    The goal is to provide a label with the appropriate number of decimal places.
    """
    scaled = val * scalar

    if abs(scaled - round(scaled)) < 1e-6:
        label = f"{int(round(scaled))}"
    elif abs(1e1 * scaled - round(1e1 * scaled)) < 1e-6:
        label = f"{scaled:.1f}"
    elif abs(1e2 * scaled - round(1e2 * scaled)) < 1e-6:
        label = f"{scaled:.2f}"
    elif abs(1e3 * scaled - round(1e3 * scaled)) < 1e-6:
        label = f"{scaled:.3f}"
    elif abs(1e4 * scaled - round(1e4 * scaled)) < 1e-6:
        label = f"{scaled:.4f}"
    elif abs(1e5 * scaled - round(1e5 * scaled)) < 1e-6:
        label = f"{scaled:.5f}"
    else:
        label = f"{scaled:.6f}"

    return label


def time_x_axis(ax: plt.Axes, sample_rate_provided: bool) -> str:
    """
    Format the x-axis of a plot to display time in seconds.
    """
    if sample_rate_provided:
        units, scalar = time_units(*ax.get_xlim())
        ax.set_xlabel(f"Time ({units}), $t$")
    else:
        units, scalar = sample_units(*ax.get_xlim())
        if scalar == 1:
            ax.set_xlabel("Sample, $n$")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.xaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
        else:
            ax.set_xlabel(f"Sample ({units}), $n$")

    formatter = ticker.FuncFormatter(lambda pos, val: smart_scaled_formatter(pos, val, scalar))
    ax.xaxis.set_major_formatter(formatter)

    return units


def time_y_axis(ax: plt.Axes, sample_rate_provided: bool) -> str:
    """
    Format the y-axis of a plot to display time in seconds.
    """
    if sample_rate_provided:
        units, scalar = time_units(*ax.get_ylim())
        ax.set_ylabel(f"Time ({units}), $t$")
    else:
        units, scalar = sample_units(*ax.get_ylim())
        if scalar == 1:
            ax.set_ylabel("Sample, $n$")
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.yaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
        else:
            ax.set_ylabel(f"Sample ({units}), $n$")

    formatter = ticker.FuncFormatter(lambda pos, val: smart_scaled_formatter(pos, val, scalar))
    ax.yaxis.set_major_formatter(formatter)

    return units


def freq_x_axis(ax: plt.Axes, sample_rate_provided: bool, bins: bool = False) -> str:
    """
    Format the x-axis of a plot to display frequency in Hz.
    """
    if bins:
        units, scalar = "", 1
        ax.set_xlabel("DFT bin index, $k$")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
    elif sample_rate_provided:
        units, scalar = freq_units(*ax.get_xlim())
        ax.set_xlabel(f"Frequency ({units}), $f$")
    else:
        units, scalar = "", 1
        ax.set_xlabel("Normalized frequency, $f/f_s$")

    formatter = ticker.FuncFormatter(lambda pos, val: smart_scaled_formatter(pos, val, scalar))
    ax.xaxis.set_major_formatter(formatter)

    return units


def freq_y_axis(ax: plt.Axes, sample_rate_provided: bool, bins: bool = False) -> str:
    """
    Format the y-axis of a plot to display frequency in Hz.
    """
    if bins:
        units, scalar = "", 1
        ax.set_ylabel("DFT bin index, $k$")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
    elif sample_rate_provided:
        units, scalar = freq_units(*ax.get_ylim())
        ax.set_ylabel(f"Frequency ({units}), $f$")
    else:
        units, scalar = "", 1
        ax.set_ylabel("Normalized frequency, $f/f_s$")

    formatter = ticker.FuncFormatter(lambda pos, val: smart_scaled_formatter(pos, val, scalar))
    ax.yaxis.set_major_formatter(formatter)

    return units


def min_ylim(y: npt.NDArray, separation: float, sample_rate: float):
    ymin, ymax = plt.gca().get_ylim()
    if ymax - ymin < separation / 4:
        # Find the mean of the signal rounded to the nearest sample
        mean = int(round(np.nanmean(y) / sample_rate))
        ymin = mean - separation / 2
        ymax = mean + separation / 2
        plt.gca().set_ylim(ymin, ymax)
