"""
A module containing functions to estimate frequency-domain parameters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import linear
from .._helper import export
from ._snr import composite_snr


@export
def foa_crlb(
    snr: npt.ArrayLike,
    time: npt.ArrayLike,
    bandwidth: npt.ArrayLike,
    rms_integration_time: npt.ArrayLike | None = None,
    noise_bandwidth: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the Cramér-Rao lower bound (CRLB) on frequency of arrival (FOA) estimation.

    Arguments:
        snr: The signal-to-noise ratio (SNR) of the signal $\gamma = S / (N_0 B_n)$ in dB.
        time: The integration time $T$ in seconds.
        bandwidth: The signal bandwidth $B_s$ in Hz.
        rms_integration_time: The root-mean-square (RMS) integration time $T_{\text{rms}}$ in Hz. If `None`, the RMS
            integration time is calculated assuming a rectangular power envelope, $T_{\text{rms}} = T/\sqrt{12}$.
        noise_bandwidth: The noise bandwidth $B_n$ in Hz. If `None`, the noise bandwidth is assumed to be the
            signal bandwidth $B_s$. The noise bandwidth must be the same for both signals.

    Returns:
        The Cramér-Rao lower bound (CRLB) on the frequency of arrival (FOA) estimation error standard
        deviation $\sigma_{\text{foa}}$ in Hz.

    See Also:
        sdr.rms_integration_time

    Notes:
        The Cramér-Rao lower bound (CRLB) on the frequency of arrival (FOA) estimation error standard
        deviation $\sigma_{\text{foa}}$ is given by

        $$\sigma_{\text{foa}} = \frac{1}{\pi \sqrt{8} T_{\text{rms}}} \frac{1}{\sqrt{B_n T \gamma}}$$

        $$
        T_{\text{rms}} = \sqrt{\frac
        {\int_{-\infty}^{\infty} (t - \mu_t)^2 \cdot \left| x(t - \mu_t) \right|^2 \, dt}
        {\int_{-\infty}^{\infty} \left| x(t - \mu_t) \right|^2 \, dt}
        }
        $$

        where $\gamma$ is the signal-to-noise ratio (SNR), $\left| x(t) \right|^2$ is the power envelope of the signal,
        and $\mu_t$ is the centroid of the power envelope.

        .. note::
            The constant terms from Stein's original equations were rearranged. The factor of 2 was removed from
            $\gamma$ and the factor of $2\pi$ was removed from $T_{\text{rms}}$ and incorporated into the CRLB
            equation.

        The signal-to-noise ratio (SNR) $\gamma$ is improved by the coherent integration gain, which is the
        time-bandwidth product $B_n T$. The product $B_n T \gamma$ is the output SNR of the matched filter
        or correlator, which is equivalent to $E / N_0$.

        $$B_n T \gamma = B_n T \frac{S}{N_0 B_n} = \frac{S T}{N_0} = \frac{E}{N_0}$$

        .. warning::
            According to Stein, the CRLB equation only holds for output SNRs greater than 10 dB. This ensures there is
            sufficient SNR to correctly identify the time/frequency peak without high $P_{fa}$. Given the rearrangement
            of scaling factors, CRLB values with output SNRs less than 7 dB are set to NaN.

        The frequency measurement precision is inversely proportional to the integration time of the signal and the
        square root of the output SNR.

    Examples:
        .. ipython:: python

            snr = 10
            bandwidth = np.logspace(5, 8, 101)

            @savefig sdr_foa_crlb_1.png
            plt.figure(); \
            plt.loglog(bandwidth, sdr.foa_crlb(snr, snr, 1e-6, bandwidth), label="1 μs"); \
            plt.loglog(bandwidth, sdr.foa_crlb(snr, snr, 1e-5, bandwidth), label="10 μs"); \
            plt.loglog(bandwidth, sdr.foa_crlb(snr, snr, 1e-4, bandwidth), label="100 μs"); \
            plt.loglog(bandwidth, sdr.foa_crlb(snr, snr, 1e-3, bandwidth), label="1 ms"); \
            plt.legend(title="Integration time"); \
            plt.xlim(1e5, 1e8); \
            plt.ylim(1e0, 1e6); \
            plt.xlabel("Bandwidth (Hz), $B$"); \
            plt.ylabel(r"CRLB on FOA (Hz), $\sigma_{\text{foa}}$"); \
            plt.title(f"Cramér-Rao lower bound (CRLB) on FOA estimation error\nstandard deviation with {snr}-dB SNR");

    Group:
        estimation-frequency
    """
    # The second signal's SNR of 1 million dB is equivalent to a noiseless template
    return fdoa_crlb(snr, 1_000_000, time, bandwidth, rms_integration_time, noise_bandwidth)


@export
def fdoa_crlb(
    snr1: npt.ArrayLike,
    snr2: npt.ArrayLike,
    time: npt.ArrayLike,
    bandwidth: npt.ArrayLike,
    rms_integration_time: npt.ArrayLike | None = None,
    noise_bandwidth: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the Cramér-Rao lower bound (CRLB) on frequency difference of arrival (FDOA) estimation.

    Arguments:
        snr1: The signal-to-noise ratio (SNR) of the first signal $\gamma_1 = S_1 / (N_0 B_n)$ in dB.
        snr2: The signal-to-noise ratio (SNR) of the second signal $\gamma_2 = S_2 / (N_0 B_n)$ in dB.
        time: The integration time $T$ in seconds.
        bandwidth: The signal bandwidth $B_s$ in Hz.
        rms_integration_time: The root-mean-square (RMS) integration time $T_{\text{rms}}$ in Hz. If `None`, the RMS
            integration time is calculated assuming a rectangular power envelope, $T_{\text{rms}} = T/\sqrt{12}$.
        noise_bandwidth: The noise bandwidth $B_n$ in Hz. If `None`, the noise bandwidth is assumed to be the
            signal bandwidth $B_s$. The noise bandwidth must be the same for both signals.

    Returns:
        The Cramér-Rao lower bound (CRLB) on the frequency difference of arrival (FDOA) estimation error standard
        deviation $\sigma_{\text{fdoa}}$ in Hz.

    See Also:
        sdr.rms_integration_time

    Notes:
        The Cramér-Rao lower bound (CRLB) on the frequency difference of arrival (FDOA) estimation error standard
        deviation $\sigma_{\text{fdoa}}$ is given by

        $$\sigma_{\text{fdoa}} = \frac{1}{\pi \sqrt{8} T_{\text{rms}}} \frac{1}{\sqrt{B_n T \gamma}}$$

        $$\frac{1}{\gamma} = \frac{1}{\gamma_1} + \frac{1}{\gamma_2} + \frac{1}{\gamma_1 \gamma_2}$$

        $$
        T_{\text{rms}} = \sqrt{\frac
        {\int_{-\infty}^{\infty} (t - \mu_t)^2 \cdot \left| x(t - \mu_t) \right|^2 \, dt}
        {\int_{-\infty}^{\infty} \left| x(t - \mu_t) \right|^2 \, dt}
        }
        $$

        where $\gamma$ is the effective signal-to-noise ratio (SNR), $\left| x(t) \right|^2$ is the power envelope of
        the signal, and $\mu_t$ is the centroid of the power envelope.

        .. note::
            The constant terms from Stein's original equations were rearranged. The factor of 2 was removed from
            $\gamma$ and the factor of $2\pi$ was removed from $T_{\text{rms}}$ and incorporated into the CRLB
            equation.

        The effective signal-to-noise ratio (SNR) $\gamma$ is improved by the coherent integration gain, which is the
        time-bandwidth product $B_n T$. The product $B_n T \gamma$ is the output SNR of the matched filter
        or correlator, which is equivalent to $E / N_0$.

        $$B_n T \gamma = B_n T \frac{S}{N_0 B_n} = \frac{S T}{N_0} = \frac{E}{N_0}$$

        .. warning::
            According to Stein, the CRLB equation only holds for output SNRs greater than 10 dB. This ensures there is
            sufficient SNR to correctly identify the time/frequency peak without high $P_{fa}$. Given the rearrangement
            of scaling factors, CRLB values with output SNRs less than 7 dB are set to NaN.

        The frequency measurement precision is inversely proportional to the integration time of the signal and the
        square root of the output SNR.

    Examples:
        .. ipython:: python

            snr = 10
            bandwidth = np.logspace(5, 8, 101)

            @savefig sdr_fdoa_crlb_1.png
            plt.figure(); \
            plt.loglog(bandwidth, sdr.fdoa_crlb(snr, snr, 1e-6, bandwidth), label="1 μs"); \
            plt.loglog(bandwidth, sdr.fdoa_crlb(snr, snr, 1e-5, bandwidth), label="10 μs"); \
            plt.loglog(bandwidth, sdr.fdoa_crlb(snr, snr, 1e-4, bandwidth), label="100 μs"); \
            plt.loglog(bandwidth, sdr.fdoa_crlb(snr, snr, 1e-3, bandwidth), label="1 ms"); \
            plt.legend(title="Integration time"); \
            plt.xlim(1e5, 1e8); \
            plt.ylim(1e0, 1e6); \
            plt.xlabel("Bandwidth (Hz), $B$"); \
            plt.ylabel(r"CRLB on FDOA (Hz), $\sigma_{\text{fdoa}}$"); \
            plt.title(f"Cramér-Rao lower bound (CRLB) on FDOA estimation error\nstandard deviation with {snr}-dB SNR");

    Group:
        estimation-frequency
    """
    snr = composite_snr(snr1, snr2)
    snr = np.asarray(snr)
    time = np.asarray(time)
    bandwidth = np.asarray(bandwidth)

    if rms_integration_time is None:
        rms_integration_time = time / np.sqrt(12)
    rms_integration_time = np.asarray(rms_integration_time)

    if noise_bandwidth is None:
        noise_bandwidth = bandwidth
    noise_bandwidth = np.asarray(noise_bandwidth)

    # The effective SNR is improved by the coherent integration gain, which is the time-bandwidth product
    snr = linear(snr)
    output_snr = time * noise_bandwidth * snr

    # Stein specifically mentions that the equations are only valid for output SNR greater than 10 dB.
    # Since we factored 2 out from the composite SNR, we need to compare against 7 dB.
    output_snr = np.where(output_snr >= linear(7), output_snr, np.nan)

    return 1 / (np.pi * np.sqrt(8) * rms_integration_time * np.sqrt(output_snr))
