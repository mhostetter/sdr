"""
A module containing functions to estimate time-domain parameters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import linear
from .._helper import export
from ._snr import composite_snr


@export
def tdoa_crlb(
    snr1: npt.ArrayLike,
    snr2: npt.ArrayLike,
    time: npt.ArrayLike,
    bandwidth: npt.ArrayLike,
    rms_bandwidth: npt.ArrayLike | None = None,
    noise_bandwidth: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the Cramér-Rao lower bound (CRLB) on the time difference of arrival (TDOA) estimation.

    Arguments:
        snr1: The signal-to-noise ratio (SNR) of the first signal $\gamma_1 = S_1 / (N_0 B_n)$ in dB.
        snr2: The signal-to-noise ratio (SNR) of the second signal $\gamma_2 = S_2 / (N_0 B_n)$ in dB.
        time: The integration time $T$ in seconds.
        bandwidth: The signal bandwidth $B_s$ in Hz.
        rms_bandwidth: The root-mean-square (RMS) bandwidth $B_{s,\text{rms}}$ in Hz. If `None`, the RMS bandwidth
            is calculated assuming a rectangular spectrum, $B_{s,\text{rms}} = B_s/\sqrt{12}$.
        noise_bandwidth: The noise bandwidth $B_n$ in Hz. If `None`, the noise bandwidth is assumed to be the
            signal bandwidth $B_s$. The noise bandwidth must be the same for both signals.

    Returns:
        The Cramér-Rao lower bound (CRLB) on the time difference of arrival (TDOA) estimation standard deviation
        $\sigma_{\text{TDOA}}$ in seconds.

    Notes:
        The Cramér-Rao lower bound (CRLB) on the time difference of arrival (TDOA) estimation standard deviation
        $\sigma_{\text{TDOA}}$ is given by

        $$\sigma_{\text{TDOA}} = \frac{1}{2 \pi B_{s,\text{rms}}} \frac{1}{\sqrt{B_n T \gamma}} .$$

        The effective signal-to-noise ratio (SNR) $\gamma$ is improved by the coherent integration gain, which is the
        time-bandwidth product $B_n T$. The product $B_n T \gamma$ is the output SNR of the matched filter
        or correlator.

        The time measurement accuracy is inversely proportional to the bandwidth of the signal and the square root of
        the output SNR.

    Examples:
        .. ipython:: python

            snr = 10
            time = np.logspace(-6, 0, 101)

            @savefig sdr_tdoa_crlb_1.png
            plt.figure(); \
            plt.loglog(time, sdr.tdoa_crlb(snr, snr, time, 1e5), label="100 kHz"); \
            plt.loglog(time, sdr.tdoa_crlb(snr, snr, time, 1e6), label="1 MHz"); \
            plt.loglog(time, sdr.tdoa_crlb(snr, snr, time, 1e7), label="10 MHz"); \
            plt.loglog(time, sdr.tdoa_crlb(snr, snr, time, 1e8), label="100 MHz"); \
            plt.legend(title="Bandwidth"); \
            plt.xlim(1e-6, 1e0); \
            plt.ylim(1e-12, 1e-6); \
            plt.xlabel("Integration time (s), $T$"); \
            plt.ylabel(r"CRLB on TDOA (s), $\sigma_{\text{TDOA}}$"); \
            plt.title(f"Cramér-Rao lower bound (CRLB) on TDOA estimation error\nstandard deviation with {snr}-dB SNR");

    Group:
        estimation-time
    """
    snr = composite_snr(snr1, snr2)
    snr = np.asarray(snr)
    time = np.asarray(time)
    bandwidth = np.asarray(bandwidth)

    if rms_bandwidth is None:
        rms_bandwidth = 1 / np.sqrt(12) * bandwidth
    rms_bandwidth = np.asarray(rms_bandwidth)

    if noise_bandwidth is None:
        noise_bandwidth = bandwidth
    noise_bandwidth = np.asarray(noise_bandwidth)

    # The effective SNR is improved by the coherent integration gain, which is the time-bandwidth product
    snr = linear(snr)
    output_snr = time * noise_bandwidth * snr

    # Stein specifically mentions that the equations are only valid for output SNR greater than 10 dB
    output_snr = np.where(output_snr >= linear(10), output_snr, np.nan)

    return 1 / (2 * np.pi * rms_bandwidth * np.sqrt(output_snr))
