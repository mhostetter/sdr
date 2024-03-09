"""
A module containing functions to compute theoretical detection performance.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats
from typing_extensions import Literal

from .._helper import export


@export
def threshold(
    p_fa: npt.ArrayLike,
    sigma2: npt.ArrayLike = 1,
    detector: Literal["square-law", "linear", "real"] = "square-law",
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical detection threshold.

    Arguments:
        p_fa: The desired probability of false alarm $P_{FA}$ in $(0, 1)$.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"square-law"`: The square-law detector.
            - `"linear"`: The linear detector.
            - `"real"`: The real detector.

    Returns:
        The detection threshold $\gamma$ in linear units.

    Examples:
        .. ipython:: python

            @savefig sdr_threshold_1.png
            plt.figure(); \
            p_fa = np.logspace(-15, 0, 101); \
            threshold = sdr.threshold(p_fa, 1); \
            plt.semilogx(p_fa, threshold); \
            plt.xlabel("Probability of false alarm, $P_{FA}$"); \
            plt.ylabel("Threshold"); \
            plt.title("Square-Law Detector: Thresholds");

    Group:
        detection-theory
    """
    p_fa = np.asarray(p_fa)
    sigma2 = np.asarray(sigma2)

    if detector == "square-law":
        threshold = _threshold_square_law(p_fa, sigma2)
    else:
        raise ValueError(f"Invalid detector type: {detector}")

    if threshold.ndim == 0:
        threshold = threshold.item()

    return threshold


def _threshold_square_law(
    p_fa: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    nu = 2  # Degrees of freedom
    h0_theory = scipy.stats.chi2(nu, scale=sigma2 / 2)
    threshold = h0_theory.isf(p_fa)

    return threshold
