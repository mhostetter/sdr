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
def toa_crlb(
    snr: npt.ArrayLike,
    time: npt.ArrayLike,
    bandwidth: npt.ArrayLike,
    rms_bandwidth: npt.ArrayLike | None = None,
    noise_bandwidth: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the Cramér-Rao lower bound (CRLB) on time of arrival (TOA) estimation.

    Arguments:
        snr: The signal-to-noise ratio (SNR) of the signal $\gamma = S / (N_0 B_n)$ in dB.
        time: The integration time $T$ in seconds.
        bandwidth: The signal bandwidth $B_s$ in Hz.
        rms_bandwidth: The root-mean-square (RMS) bandwidth $B_{s,\text{rms}}$ in Hz. If `None`, the RMS bandwidth
            is calculated assuming a rectangular spectrum, $B_{s,\text{rms}} = B_s/\sqrt{12}$.
        noise_bandwidth: The noise bandwidth $B_n$ in Hz. If `None`, the noise bandwidth is assumed to be the
            signal bandwidth $B_s$. The noise bandwidth must be the same for both signals.

    Returns:
        The Cramér-Rao lower bound (CRLB) on the time of arrival (TOA) estimation error standard deviation
        $\sigma_{\text{toa}}$ in seconds.

    See Also:
        sdr.rms_bandwidth

    Notes:
        The Cramér-Rao lower bound (CRLB) on the time of arrival (TOA) estimation error standard deviation
        $\sigma_{\text{toa}}$ is given by

        $$\sigma_{\text{toa}} = \frac{1}{\pi \sqrt{8} B_{s,\text{rms}}} \frac{1}{\sqrt{B_n T \gamma}}$$

        $$
        B_{s,\text{rms}} = \sqrt{\frac
        {\int_{-\infty}^{\infty} (f - \mu_f)^2 \cdot S(f - \mu_f) \, df}
        {\int_{-\infty}^{\infty} S(f - \mu_f) \, df}
        }
        $$

        where $\gamma$ is the signal-to-noise ratio (SNR), $S(f)$ is the power spectral density (PSD) of the signal,
        and $\mu_f$ is the centroid of the PSD.

        .. note::
            The constant terms from Stein's original equations were rearranged. The factor of 2 was removed from
            $\gamma$ and the factor of $2\pi$ was removed from $B_{s,\text{rms}}$ and incorporated into the CRLB
            equation.

        The signal-to-noise ratio (SNR) $\gamma$ is improved by the coherent integration gain, which is the
        time-bandwidth product $B_n T$. The product $B_n T \gamma$ is the output SNR of the matched filter
        or correlator, which is equivalent to $E / N_0$.

        $$B_n T \gamma = B_n T \frac{S}{N_0 B_n} = \frac{S T}{N_0} = \frac{E}{N_0}$$

        .. warning::
            According to Stein, the CRLB equation only holds for output SNRs greater than 10 dB. This ensures there is
            sufficient SNR to correctly identify the time/frequency peak without high $P_{fa}$. Given the rearrangement
            of scaling factors, CRLB values with output SNRs less than 7 dB are set to NaN.

        The time measurement precision is inversely proportional to the bandwidth of the signal and the square root of
        the output SNR.

    Examples:
        .. ipython:: python

            snr = 10
            time = np.logspace(-6, 0, 101)

            @savefig sdr_toa_crlb_1.png
            plt.figure(); \
            plt.loglog(time, sdr.toa_crlb(snr, time, 1e5), label="100 kHz"); \
            plt.loglog(time, sdr.toa_crlb(snr, time, 1e6), label="1 MHz"); \
            plt.loglog(time, sdr.toa_crlb(snr, time, 1e7), label="10 MHz"); \
            plt.loglog(time, sdr.toa_crlb(snr, time, 1e8), label="100 MHz"); \
            plt.legend(title="Bandwidth"); \
            plt.xlim(1e-6, 1e0); \
            plt.ylim(1e-12, 1e-6); \
            plt.xlabel("Integration time (s), $T$"); \
            plt.ylabel(r"CRLB on TOA (s), $\sigma_{\text{toa}}$"); \
            plt.title(f"Cramér-Rao lower bound (CRLB) on TOA estimation error\nstandard deviation with {snr}-dB SNR");

    Group:
        estimation-time
    """
    # The second signal's SNR of 1 million dB is equivalent to a noiseless template
    return tdoa_crlb(snr, 1_000_000, time, bandwidth, rms_bandwidth, noise_bandwidth)


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
    Calculates the Cramér-Rao lower bound (CRLB) on time difference of arrival (TDOA) estimation.

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
        The Cramér-Rao lower bound (CRLB) on the time difference of arrival (TDOA) estimation error standard deviation
        $\sigma_{\text{tdoa}}$ in seconds.

    See Also:
        sdr.rms_bandwidth

    Notes:
        The Cramér-Rao lower bound (CRLB) on the time difference of arrival (TDOA) estimation error standard deviation
        $\sigma_{\text{tdoa}}$ is given by

        $$\sigma_{\text{tdoa}} = \frac{1}{\pi \sqrt{8} B_{s,\text{rms}}} \frac{1}{\sqrt{B_n T \gamma}}$$

        $$\frac{1}{\gamma} = \frac{1}{\gamma_1} + \frac{1}{\gamma_2} + \frac{1}{\gamma_1 \gamma_2}$$

        $$
        B_{s,\text{rms}} = \sqrt{\frac
        {\int_{-\infty}^{\infty} (f - \mu_f)^2 \cdot S(f - \mu_f) \, df}
        {\int_{-\infty}^{\infty} S(f - \mu_f) \, df}
        }
        $$

        where $\gamma$ is the effective signal-to-noise ratio (SNR), $S(f)$ is the power spectral density (PSD)
        of the signal, and $\mu_f$ is the centroid of the PSD.

        .. note::
            The constant terms from Stein's original equations were rearranged. The factor of 2 was removed from
            $\gamma$ and the factor of $2\pi$ was removed from $B_{s,\text{rms}}$ and incorporated into the CRLB
            equation.

        The effective signal-to-noise ratio (SNR) $\gamma$ is improved by the coherent integration gain, which is the
        time-bandwidth product $B_n T$. The product $B_n T \gamma$ is the output SNR of the matched filter
        or correlator, which is equivalent to $E / N_0$.

        $$B_n T \gamma = B_n T \frac{S}{N_0 B_n} = \frac{S T}{N_0} = \frac{E}{N_0}$$

        .. warning::
            According to Stein, the CRLB equation only holds for output SNRs greater than 10 dB. This ensures there is
            sufficient SNR to correctly identify the time/frequency peak without high $P_{fa}$. Given the rearrangement
            of scaling factors, CRLB values with output SNRs less than 7 dB are set to NaN.

        The time measurement precision is inversely proportional to the bandwidth of the signal and the square root of
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
            plt.ylabel(r"CRLB on TDOA (s), $\sigma_{\text{tdoa}}$"); \
            plt.title(f"Cramér-Rao lower bound (CRLB) on TDOA estimation error\nstandard deviation with {snr}-dB SNR");

    Group:
        estimation-time
    """
    snr = composite_snr(snr1, snr2)
    snr = np.asarray(snr)
    time = np.asarray(time)
    bandwidth = np.asarray(bandwidth)

    if rms_bandwidth is None:
        rms_bandwidth = bandwidth / np.sqrt(12)
    rms_bandwidth = np.asarray(rms_bandwidth)

    if noise_bandwidth is None:
        noise_bandwidth = bandwidth
    noise_bandwidth = np.asarray(noise_bandwidth)

    # The effective SNR is improved by the coherent integration gain, which is the time-bandwidth product
    snr = linear(snr)
    output_snr = time * noise_bandwidth * snr

    # Stein specifically mentions that the equations are only valid for output SNR greater than 10 dB.
    # Since we factored 2 out from the composite SNR, we need to compare against 7 dB.
    output_snr = np.where(output_snr >= linear(7), output_snr, np.nan)

    return 1 / (np.pi * np.sqrt(8) * rms_bandwidth * np.sqrt(output_snr))
