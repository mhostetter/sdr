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
    """
    Plots a time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If the sample rate is 1, the x-axis will
            be label as "Samples".
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plot-time
    """
    x = np.asarray(x)
    t = np.arange(x.size) / sample_rate

    # with plt.style.context(Path(__file__).parent / ".." / "presentation.mplstyle"):
    with plt.rc_context(RC_PARAMS):
        label = kwargs.pop("label", None)
        if np.iscomplexobj(x):
            if label is None:
                label = "real"
                label2 = "imag"
            else:
                label = label + " (real)"
                label2 = label + " (imag)"
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
