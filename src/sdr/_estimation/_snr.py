"""
A module containing estimation algorithms for signal-to-noise ratio (SNR) calculations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import db, linear
from .._helper import export


@export
def composite_snr(snr1: npt.ArrayLike, snr2: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Calculates the signal-to-noise ratio (SNR) of the product of two signals.

    Arguments:
        snr1: The signal-to-noise ratio (SNR) of the first signal $\gamma_1$ in dB.
        snr2: The signal-to-noise ratio (SNR) of the second signal $\gamma_2$ in dB.

    Returns:
        The signal-to-noise ratio (SNR) of the product of the two signals $\gamma$ in dB.

    Notes:
        The effective or composite SNR $\gamma$ of the product of two signals with SNRs $\gamma_1$ and $\gamma_2$
        is given by

        $$\frac{1}{\gamma} = \frac{1}{2} \left[ \frac{1}{\gamma_1} + \frac{1}{\gamma_2} + \frac{1}{\gamma_1 \gamma_2} \right] .$$

        When both $\gamma_1$ and $\gamma_2$ are greater than 0 dB, and if $\gamma_1 = \gamma_2$, then
        $\gamma = \gamma_1 = \gamma_2$. If one is much less than the other, then $\gamma$ is approximately 3 dB
        greater than the smaller one. When both are less than 0 dB, the composite SNR is approximately
        $\gamma = 2 \gamma_1 \gamma_2$.

    References:
        - `Seymour Stein, Algorithms for Ambiguity Function Processing <https://ieeexplore.ieee.org/document/1163621>`_

    Examples:
        Calculate the composite SNR of two signals with equal SNRs. Notice for positive input SNR, the composite SNR
        is linear with slope 1 and intercept 0 dB. For negative input SNR, the composite SNR is linear with slope 2 and
        offset 3 dB.

        .. ipython:: python

            snr1 = np.linspace(-60, 60, 101)

            @savefig sdr_composite_snr_1.png
            plt.figure(); \
            plt.plot(snr1, sdr.composite_snr(snr1, snr1)); \
            plt.xlim(-60, 60); \
            plt.ylim(-60, 60); \
            plt.xlabel("Input SNRs (dB), $\gamma_1$ and $\gamma_2$"); \
            plt.ylabel(r"Composite SNR (dB), $\gamma$"); \
            plt.title("Composite SNR of two signals with equal SNRs");

        Calculate the composite SNR of two signals with different SNRs. Notice the knee of the curve is located at
        `max(0, snr2)`. Left of the knee, the composite SNR is linear with slope 2 and intercept `snr2 + 3` dB.
        Right of the knee, the composite SNR is linear with slope 0 and intercept `snr2 + 3` dB.

        .. ipython:: python

            snr1 = np.linspace(-60, 60, 101)

            plt.figure();
            for snr2 in np.arange(-40, 40 + 10, 10):
                plt.plot(snr1, sdr.composite_snr(snr1, snr2), label=snr2)
            @savefig sdr_composite_snr_2.png
            plt.legend(title="SNR of signal 2 (dB), $\gamma_2$"); \
            plt.xlim(-60, 60); \
            plt.ylim(-60, 60); \
            plt.xlabel("SNR of signal 1 (dB), $\gamma_1$"); \
            plt.ylabel(r"Composite SNR (dB), $\gamma$"); \
            plt.title("Composite SNR of two signals with different SNRs");

    Group:
        estimation-snr
    """
    snr1 = np.asarray(snr1)
    snr2 = np.asarray(snr2)

    # Convert to linear
    snr1 = linear(snr1)
    snr2 = linear(snr2)

    inv_snr = 0.5 * (1 / snr1 + 1 / snr2 + 1 / (snr1 * snr2))
    snr = 1 / inv_snr

    # Convert to dB
    snr = db(snr)

    return snr
