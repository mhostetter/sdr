"""
A module containing filter-related plotting functions.
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def impulse_response(b: np.ndarray, a: np.ndarray = 1, N: Optional[int] = None, **kwargs):
    """
    Plots the impulse response of a filter.

    Arguments:
        b: The feedforward coefficients, $b_i$.
        a: The feedback coefficients, $a_j$. For FIR filters, this is set to 1.
        N: The number of samples to plot. If `None`, the length of `b` is used for FIR filters and
            100 for IIR filters.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.IIR

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        plotting
    """
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)

    if N is None:
        if a.size == 1 and a[0] == 1:
            # FIR filter
            N = b.size
        else:
            N = 100

    # Delta impulse function
    d = np.zeros(N, dtype=np.float32)
    d[0] = 1

    # Filter the impulse
    zi = scipy.signal.lfiltic(b, a, y=[], x=[])
    h, zi = scipy.signal.lfilter(b, a, d, zi=zi)

    with plt.rc_context(RC_PARAMS):
        label = kwargs.pop("label", None)
        if np.iscomplexobj(h):
            if label is None:
                label = "real"
                label2 = "imag"
            else:
                label = label + " (real)"
                label2 = label + " (imag)"
            plt.plot(np.arange(h.size), h.real, label=label, **kwargs)
            plt.plot(np.arange(h.size), h.imag, label=label2, **kwargs)
        else:
            plt.plot(np.arange(h.size), h, label=label, **kwargs)

        if label:
            plt.legend()

        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Impulse Response, $h[n]$")
