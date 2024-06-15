"""
A module containing :obj:`sdr`'s default matplotlib rcParams.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .._helper import export

RC_PARAMS = {
    "lines.linewidth": 1,
    "axes.grid": True,
    "axes.grid.which": "both",
    # "axes.xmargin": 0,
    # "axes.ymargin": 0,
    # "axes.zmargin": 0,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.1,
    "figure.constrained_layout.w_pad": 0.1,
    "figure.figsize": (8, 4),
    "figure.max_open_warning": 0,
    "figure.titleweight": "bold",
    "grid.alpha": 0.8,
    "grid.linestyle": "--",
}


@export
def use_style():
    """
    Applies :obj:`sdr`'s default :obj:`matplotlib` rcParams.

    These style settings may be reverted with :func:`matplotlib.pyplot.rcdefaults()`.

    Examples:
        The following rcParams are applied.

        .. ipython:: python

            sdr.plot._rc_params.RC_PARAMS

    Group:
        plot-utility
    """
    plt.rcParams.update(RC_PARAMS)
