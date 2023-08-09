"""
A module containing functions to compute theoretical detection performance.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def albersheim(p_d: npt.ArrayLike, p_fa: npt.ArrayLike, N_nc: npt.ArrayLike = 1) -> np.ndarray:
    r"""
    Estimates the minimum required single-sample SNR, given $N_{NC}$ non-coherent combinations, to achieve
    a probability of detection $P_D$ and probability of false alarm $P_{FA}$. This function implements
    Albersheim's equation.

    Arguments:
        p_d: The desired probability of detection $P_D$ in $(0, 1)$.
        p_fa: The desired probability of false alarm $P_{FA}$ in $(0, 1)$.
        N_nc: The number of non-coherent combinations $N_{NC} \ge 1$.

    Returns:
        The minimum required single-sample SNR $\gamma$ in dB.

    Notes:
        Albersheim's equation is given by:

        $$A = \ln \frac{0.62}{P_{FA}}$$

        $$B = \ln \frac{P_D}{1 - P_D}$$

        $$
        \text{SNR}_{\text{dB}} =
        -5 \log_{10} N_{NC} + \left(6.2 + \frac{4.54}{\sqrt{N_{NC} + 0.44}}\right)
        \log_{10} \left(A + 0.12AB + 1.7B\right)
        $$

        The error in the estimated minimum SNR is claimed to be less than 0.2 dB for
        $10^{-7} \leq P_{FA} \leq 10^{-3}$, $0.1 \leq P_D \leq 0.9$, and $1 \le N_{NC} \le 8096$.

    References:
        - https://radarsp.weebly.com/uploads/2/1/4/7/21471216/albersheim_alternative_forms.pdf
        - https://bpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/5/462/files/2016/12/Noncoherent-Integration-Gain-Approximations.pdf
        - https://www.mathworks.com/help/phased/ref/albersheim.html

    Examples:
        .. ipython:: python

            p_d = 0.9; \
            p_fa = np.logspace(-7, -3, 100)

            @savefig sdr_albersheim_1.png
            plt.figure(figsize=(8, 4)); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, N_nc=1), label="$N_{NC}$ = 1"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, N_nc=2), label="$N_{NC}$ = 2"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, N_nc=10), label="$N_{NC}$ = 10"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, N_nc=20), label="$N_{NC}$ = 20"); \
            plt.legend(); \
            plt.grid(True, which="both"); \
            plt.xlabel("Probability of false alarm, $P_{FA}$"); \
            plt.ylabel("Minimum required SNR (dB)"); \
            plt.title(f"Estimated minimum required SNR across non-coherent combinations for $P_D = 0.9$\nusing Albersheim's approximation"); \
            plt.tight_layout()

    Group:
        detection-theory
    """  # pylint: disable=line-too-long
    p_d = np.asarray(p_d)
    if not np.all(np.logical_and(0 < p_d, p_d < 1)):
        raise ValueError("Argument 'p_d' must have values in (0, 1).")

    p_fa = np.asarray(p_fa)
    if not np.all(np.logical_and(0 < p_fa, p_fa < 1)):
        raise ValueError("Argument 'p_fa' must have values in (0, 1).")

    N_nc = np.asarray(N_nc)
    if not np.all(N_nc >= 1):
        raise ValueError("Argument 'N_nc' must be at least 1.")

    A = np.log(0.62 / p_fa)
    B = np.log(p_d / (1 - p_d))
    snr = -5 * np.log10(N_nc) + (6.2 + (4.54 / np.sqrt(N_nc + 0.44))) * np.log10(A + 0.12 * A * B + 1.7 * B)

    return snr
