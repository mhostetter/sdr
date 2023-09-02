"""
A module containing detection-related plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def roc(
    p_fa: npt.ArrayLike,
    p_d: npt.ArrayLike,
    type: Literal["linear", "semilogx", "semilogy", "loglog"] = "semilogx",  # pylint: disable=redefined-builtin
    **kwargs,
):
    r"""
    Plots the receiver operating characteristic (ROC) curve as a function of $P_{FA}$.

    Arguments:
        p_fa: The probability of false alarm $P_{FA}$.
        p_d: The probability of detection $P_D$.
        type: The type of plot to generate.
        kwargs: Additional keyword arguments to pass to the plotting function defined by `type`.

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        if type == "linear":
            plt.plot(p_fa, p_d, **kwargs)
        elif type == "semilogx":
            plt.semilogx(p_fa, p_d, **kwargs)
        elif type == "semilogy":
            plt.semilogy(p_fa, p_d, **kwargs)
        elif type == "loglog":
            plt.loglog(p_fa, p_d, **kwargs)
        else:
            raise ValueError(
                f"Argument 'type' must be one of ['linear', 'semilogx', 'semilogy', 'loglog'], not {type!r}."
            )

        plt.ylim(0, 1)
        plt.grid(True, which="both")
        if "label" in kwargs:
            plt.legend()

        plt.xlabel("Probability of false alarm, $P_{FA}$")
        plt.ylabel("Probability of detection, $P_{D}$")
        plt.title("Receiver operating characteristic (ROC) curve")
        plt.tight_layout()
