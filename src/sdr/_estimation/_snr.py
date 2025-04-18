"""
A module containing estimation algorithms for signal-to-noise ratio (SNR) calculations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import db, linear
from .._helper import convert_output, export, verify_arraylike


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

        $$\frac{1}{\gamma} = \frac{1}{\gamma_1} + \frac{1}{\gamma_2} + \frac{1}{\gamma_1 \gamma_2} .$$

        When both $\gamma_1$ and $\gamma_2$ are greater than 0 dB, and if $\gamma_1 = \gamma_2$, then
        $\gamma = \gamma_1 / 2 = \gamma_2 / 2$. If one is much less than the other, then $\gamma$ is approximately
        equal to the smaller one. When both are less than 0 dB, the composite SNR is approximately
        $\gamma = \gamma_1 \gamma_2$.

        .. note::
            The constant terms from Stein's original equations were rearranged. The factor of 2 was removed and
            incorporated into the CRLB equations. This was done so that when $\gamma_2 = \infty$ then
            $\gamma = \gamma_1$.

    References:
        - `Seymour Stein, Algorithms for Ambiguity Function Processing <https://ieeexplore.ieee.org/document/1163621>`_

    Examples:
        Calculate the composite SNR of two signals with equal SNRs. Notice for positive input SNR, the composite SNR
        is linear with slope 1 and intercept -3 dB. For negative input SNR, the composite SNR is linear with slope 2
        and intercept 0 dB.

        .. ipython:: python

            snr1 = np.linspace(-60, 60, 101)

            @savefig sdr_composite_snr_1.svg
            plt.figure(); \
            plt.plot(snr1, sdr.composite_snr(snr1, snr1)); \
            plt.xlim(-60, 60); \
            plt.ylim(-60, 60); \
            plt.xlabel("Input SNRs (dB), $\gamma_1$ and $\gamma_2$"); \
            plt.ylabel(r"Composite SNR (dB), $\gamma$"); \
            plt.title("Composite SNR of two signals with equal SNRs");

        Calculate the composite SNR of two signals with different SNRs. Notice the knee of the curve is located at
        `max(0, snr2)`. Left of the knee, the composite SNR is linear with slope 1 and intercept `snr2`.
        Right of the knee, the composite SNR is linear with slope 0 and intercept `snr2`.

        .. ipython:: python

            snr1 = np.linspace(-60, 60, 101)

            plt.figure();
            for snr2 in np.arange(-40, 40 + 10, 10):
                plt.plot(snr1, sdr.composite_snr(snr1, snr2), label=snr2)
            @savefig sdr_composite_snr_2.svg
            plt.legend(title="SNR of signal 2 (dB), $\gamma_2$"); \
            plt.xlim(-60, 60); \
            plt.ylim(-60, 60); \
            plt.xlabel("SNR of signal 1 (dB), $\gamma_1$"); \
            plt.ylabel(r"Composite SNR (dB), $\gamma$"); \
            plt.title("Composite SNR of two signals with different SNRs");

    Group:
        estimation-snr
    """
    snr1 = verify_arraylike(snr1, float=True)
    snr2 = verify_arraylike(snr2, float=True)

    # Convert to linear
    snr1 = linear(snr1)
    snr2 = linear(snr2)

    inv_snr = 1 / snr1 + 1 / snr2 + 1 / (snr1 * snr2)
    snr = 1 / inv_snr

    # Convert back to dB
    snr = db(snr)

    return convert_output(snr)
