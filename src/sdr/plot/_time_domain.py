"""
A module containing time-domain plotting functions.
"""
import matplotlib.pyplot as plt
import numpy as np

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def time_domain(x: np.ndarray, sample_rate: float = 1.0, **kwargs):
    """
    Plots a time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to plot.
        sample_rate: The sample rate of the signal in samples/s. If the sample rate is 1, the x-axis will
            be label as "Samples".
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plotting
    """
    x = np.asarray(x)
    t = np.arange(x.size) / sample_rate

    label = kwargs.pop("label", None)

    # with plt.style.context(Path(__file__).parent / ".." / "presentation.mplstyle"):
    with plt.rc_context(RC_PARAMS):
        if np.iscomplexobj(x):
            x_label = y_label = label
            if x_label is not None:
                x_label += " (real)"
                y_label += " (imag)"
            plt.plot(t, x.real, label=x_label, **kwargs)
            plt.plot(t, x.imag, label=y_label, **kwargs)
        else:
            plt.plot(t, x, label=label, **kwargs)

        if sample_rate == 1:
            plt.xlabel("Samples")
        else:
            plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        if label:
            plt.legend()
