"""
A module containing functions to approximate detection performance.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def albersheim(p_d: npt.ArrayLike, p_fa: npt.ArrayLike, n_nc: npt.ArrayLike = 1) -> npt.NDArray[np.float64]:
    r"""
    Estimates the minimum required single-sample SNR.

    Arguments:
        p_d: The desired probability of detection $P_D$ in $(0, 1)$.
        p_fa: The desired probability of false alarm $P_{FA}$ in $(0, 1)$.
        n_nc: The number of non-coherent combinations $N_{NC} \ge 1$.

    Returns:
        The minimum required single-sample SNR $\gamma$ in dB.

    Notes:
        This function implements Albersheim's equation, given by

        $$A = \ln \frac{0.62}{P_{FA}}$$

        $$B = \ln \frac{P_D}{1 - P_D}$$

        $$
        \text{SNR}_{\text{dB}} =
        -5 \log_{10} N_{NC} + \left(6.2 + \frac{4.54}{\sqrt{N_{NC} + 0.44}}\right)
        \log_{10} \left(A + 0.12AB + 1.7B\right) .
        $$

        The error in the estimated minimum SNR is claimed to be less than 0.2 dB for

        $$10^{-7} \leq P_{FA} \leq 10^{-3}$$
        $$0.1 \leq P_D \leq 0.9$$
        $$1 \le N_{NC} \le 8096 .$$

    References:
        - https://radarsp.weebly.com/uploads/2/1/4/7/21471216/albersheim_alternative_forms.pdf
        - https://bpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/5/462/files/2016/12/Noncoherent-Integration-Gain-Approximations.pdf
        - https://www.mathworks.com/help/phased/ref/albersheim.html

    Examples:
        .. ipython:: python

            p_d = 0.9; \
            p_fa = np.logspace(-7, -3, 100)

            @savefig sdr_albersheim_1.png
            plt.figure(); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=1), label="$N_{NC}$ = 1"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=2), label="$N_{NC}$ = 2"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=10), label="$N_{NC}$ = 10"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=20), label="$N_{NC}$ = 20"); \
            plt.legend(); \
            plt.xlabel("Probability of false alarm, $P_{FA}$"); \
            plt.ylabel("Minimum required SNR (dB)"); \
            plt.title(f"Estimated minimum required SNR across non-coherent combinations for $P_D = 0.9$\nusing Albersheim's approximation");

    Group:
        detection-approximation
    """
    p_d = np.asarray(p_d)
    if not np.all(np.logical_and(0 < p_d, p_d < 1)):
        raise ValueError("Argument 'p_d' must have values in (0, 1).")

    p_fa = np.asarray(p_fa)
    if not np.all(np.logical_and(0 < p_fa, p_fa < 1)):
        raise ValueError("Argument 'p_fa' must have values in (0, 1).")

    n_nc = np.asarray(n_nc)
    if not np.all(n_nc >= 1):
        raise ValueError("Argument 'n_nc' must be at least 1.")

    A = np.log(0.62 / p_fa)
    B = np.log(p_d / (1 - p_d))
    snr = -5 * np.log10(n_nc) + (6.2 + (4.54 / np.sqrt(n_nc + 0.44))) * np.log10(A + 0.12 * A * B + 1.7 * B)

    return snr
