"""
A module containing functions related to non-coherent integration.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.optimize

from .._conversion import db, linear
from .._helper import export


@export
def non_coherent_gain(
    n_nc: npt.ArrayLike,
    snr: npt.ArrayLike,
    p_fa: npt.ArrayLike = 1e-6,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the SNR improvement by non-coherently integrating $N_{nc}$ samples.

    Arguments:
        n_nc: The number of samples $N_{nc}$ to non-coherently integrate.
        snr: The SNR of the non-coherently integrated samples.
        p_fa: The desired probability of false alarm $P_{FA}$. This is used to compute the necessary thresholds before
            and after integration. The non-coherent gain is slightly affected by the $P_{FA}$.

    Returns:
        The non-coherent gain $G_{nc}$ in dB.

    Notes:
        $$y[m] = \sum_{n=0}^{N_{nc}-1} \left| x[m-n] \right|^2$$
        $$\text{SNR}_{y,\text{dB}} = \text{SNR}_{x,\text{dB}} + G_{nc}$$

    Examples:
        Compute the non-coherent gain for various integration lengths at 10-dB SNR.

        .. ipython:: python

            sdr.non_coherent_gain(1, 10)
            sdr.non_coherent_gain(2, 10)
            sdr.non_coherent_gain(10, 10)
            sdr.non_coherent_gain(20, 10)

        Plot the non-coherent gain parameterized by SNR. Notice that the gain is affected by the input SNR.
        For very large input SNRs, the non-coherent gain approaches the coherent gain
        $G_{NC} \approx 10 \log_{10} N_{NC}$. For very small SNRs, the non-coherent gain is approximated by
        $G_{NC} \approx 3.7 \log_{10} N_{NC}$.

        .. ipython:: python

            plt.figure(); \
            n = np.logspace(0, 3, 51); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for snr in np.arange(-20, 30, 10):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr), label=f"{snr} dB")
            @savefig sdr_non_coherent_gain_1.png
            plt.legend(title="SNR", loc="upper left"); \
            plt.xlabel("Number of samples, $N_{NC}$"); \
            plt.ylabel("Non-coherent gain, $G_{NC}$"); \
            plt.title("Non-coherent gain for various SNRs");

        Plot the non-coherent gain parameterized by the probability of false alarm for 5-dB SNR. Notice that the
        gain is only slightly affected by the $P_{FA}$.

        .. ipython:: python

            plt.figure(); \
            snr = 5; \
            n = np.logspace(0, 3, 51); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for exp in np.arange(-14, 0, 4):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr, p_fa=10.0**exp), label=f"$10^{{{exp}}}$")
            @savefig sdr_non_coherent_gain_2.png
            plt.legend(title="$P_{FA}$"); \
            plt.xlabel("Number of samples, $N_{NC}$"); \
            plt.ylabel("Non-coherent gain, $G_{NC}$"); \
            plt.title(f"Non-coherent gain at {snr}-dB SNR for various $P_{{FA}}$");

        However, when the input SNR is very low, for example -20 dB, the non-coherent gain is more affected by
        false alarm rate.

        .. ipython:: python

            plt.figure(); \
            snr = -20; \
            n = np.logspace(0, 3, 51); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for exp in np.arange(-14, 0, 4):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr, p_fa=10.0**exp), label=f"$10^{{{exp}}}$")
            @savefig sdr_non_coherent_gain_3.png
            plt.legend(title="$P_{FA}$"); \
            plt.xlabel("Number of samples, $N_{NC}$"); \
            plt.ylabel("Non-coherent gain, $G_{NC}$"); \
            plt.title(f"Non-coherent gain at {snr}-dB SNR for various $P_{{FA}}$");

    Group:
        detection-non-coherent-integration
    """
    n_nc = np.asarray(n_nc)
    snr = np.asarray(snr)
    p_fa = np.asarray(p_fa)

    if np.any(n_nc < 1):
        raise ValueError(f"Argument 'n_nc' must be at least 1, not {n_nc}.")
    if np.any(p_fa < 0) or np.any(p_fa > 1):
        raise ValueError(f"Argument 'p_fa' must be between 0 and 1, not {p_fa}.")

    g_nc = _non_coherent_gain(n_nc, snr, p_fa)
    if g_nc.ndim == 0:
        g_nc = float(g_nc)

    return g_nc


@np.vectorize
def _non_coherent_gain(n_nc: float, snr: float, p_fa: float) -> float:
    sigma2 = 1  # Noise variance (power), sigma^2
    A2 = linear(snr) * sigma2  # Signal power, A^2

    # Determine the threshold that yields the desired probability of false alarm. Then compute the probability
    # of detection for the specified SNR.
    df = 2 * 1  # Degrees of freedom
    threshold_in = scipy.stats.chi2.isf(p_fa, df, scale=sigma2 / 2)
    nc = 1 * A2 / (sigma2 / 2)  # Non-centrality parameter
    p_d_in = scipy.stats.ncx2.sf(threshold_in, df, nc, scale=sigma2 / 2)

    if p_d_in == 1:
        # The SNR is already large enough that the probability of detection is 1 before non-coherent integration.
        # After non-coherent integration, the probability of detection will still be 1.
        # We can't use numerical techniques to solve for the exact gain, since `scipy.stats.ncx2.sf()` is not providing
        # enough precision. So, we have to upper bound the non-coherent gain by the coherent gain.
        return db(n_nc)

    # Determine the threshold, after non-coherent integration, that yields the same probability of false alarm.
    df = 2 * n_nc  # Degrees of freedom
    threshold_out = scipy.stats.chi2.isf(p_fa, df, scale=sigma2 / 2)

    def root_eq(A2_db):
        nc = n_nc * linear(A2_db) / (sigma2 / 2)  # Non-centrality parameter
        p_d_out = scipy.stats.ncx2.sf(threshold_out, df, nc, scale=sigma2 / 2)
        return db(p_d_out) - db(p_d_in)  # Use logarithms for numerical stability

    # Determine the input signal power that, after non-coherent integration, yields the same probability of detection.
    # We use logarithms for power for numerical stability.
    A2_db = db(A2)
    A2_db_nc = scipy.optimize.brentq(root_eq, A2_db - db(n_nc), A2_db)
    g_nc = A2_db - A2_db_nc

    return g_nc
