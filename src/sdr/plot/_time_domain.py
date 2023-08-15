"""
A module containing time-domain plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def time_domain(
    x: npt.ArrayLike,
    sample_rate: float | None = None,
    centered: bool = False,
    offset: float = 0,
    diff: Literal["color", "line"] = "color",
    **kwargs,
):
    r"""
    Plots a time-domain signal $x[n]$.

    Arguments:
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
        .. ipython:: python

            # Create a BPSK impulse signal
            x = np.zeros(1000); \
            symbol_map = np.array([1, -1]); \
            x[::10] = symbol_map[np.random.randint(0, 2, 100)]

            # Pulse shape the signal with a square-root raised cosine filter
            h_srrc = sdr.root_raised_cosine(0.5, 7, 10); \
            y = np.convolve(x, h_srrc)

            @savefig sdr_plot_time_domain_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y, sample_rate=10e3); \
            plt.title("SRRC pulse-shaped BPSK"); \
            plt.tight_layout()

        .. ipython:: python

            # Create a QPSK impulse signal
            x = np.zeros(1000, dtype=np.complex64); \
            symbol_map = np.exp(1j * np.pi / 4) * np.array([1, 1j, -1, -1j]); \
            x[::10] = symbol_map[np.random.randint(0, 4, 100)]

            # Pulse shape the signal with a square-root raised cosine filter
            h_srrc = sdr.root_raised_cosine(0.5, 7, 10); \
            y = np.convolve(x, h_srrc)

            @savefig sdr_plot_time_domain_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y, sample_rate=10e3); \
            plt.title("SRRC pulse-shaped QPSK"); \
            plt.tight_layout()

    Group:
        plot-time-domain
    """
    x = np.asarray(x)
    if not x.ndim == 1:
        raise ValueError(f"Argument 'x' must be 1-D, not {x.ndim}-D.")

    if sample_rate is None:
        sample_rate_provided = False
        sample_rate = 1
    else:
        sample_rate_provided = True
        if not isinstance(sample_rate, (int, float)):
            raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")

    if centered:
        if x.size % 2 == 0:
            t = np.arange(-x.size // 2, x.size // 2) / sample_rate
        else:
            t = np.arange(-(x.size - 1) // 2, (x.size + 1) // 2) / sample_rate
    else:
        t = np.arange(x.size) / sample_rate + offset

    # with plt.style.context(Path(__file__).parent / ".." / "presentation.mplstyle"):
    with plt.rc_context(RC_PARAMS):
        label = kwargs.pop("label", None)
        if np.iscomplexobj(x):
            if label is None:
                label, label2 = "real", "imag"
            else:
                label, label2 = label + " (real)", label + " (imag)"

            if diff == "color":
                plt.plot(t, x.real, label=label, **kwargs)
                plt.plot(t, x.imag, label=label2, **kwargs)
            elif diff == "line":
                (real,) = plt.plot(t, x.real, "-", label=label, **kwargs)
                kwargs.pop("color", None)
                plt.plot(t, x.imag, "--", color=real.get_color(), label=label2, **kwargs)
            else:
                raise ValueError(f"Argument 'diff' must be 'color' or 'line', not {diff}.")
        else:
            plt.plot(t, x, label=label, **kwargs)

        if label:
            plt.legend()
        if sample_rate_provided:
            plt.xlabel("Time (s)")
        else:
            plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
