"""
A module containing time-domain plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def time_domain(x: npt.ArrayLike, sample_rate: float = 1.0, **kwargs):
    r"""
    Plots a time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If the sample rate is 1, the x-axis will
            be label as "Samples".
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

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
    t = np.arange(x.size) / sample_rate

    # with plt.style.context(Path(__file__).parent / ".." / "presentation.mplstyle"):
    with plt.rc_context(RC_PARAMS):
        label = kwargs.pop("label", None)
        if np.iscomplexobj(x):
            if label is None:
                label, label2 = "real", "imag"
            else:
                label, label2 = label + " (real)", label + " (imag)"
            plt.plot(t, x.real, label=label, **kwargs)
            plt.plot(t, x.imag, label=label2, **kwargs)
        else:
            plt.plot(t, x, label=label, **kwargs)

        if label:
            plt.legend()
        if sample_rate == 1:
            plt.xlabel("Samples")
        else:
            plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
