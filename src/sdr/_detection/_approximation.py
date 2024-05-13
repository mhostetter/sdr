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
    Estimates the minimum signal-to-noise ratio (SNR) required to achieve the desired probability of detection $P_{D}$.

    Arguments:
        p_d: The desired probability of detection $P_D$ in $(0, 1)$.
        p_fa: The desired probability of false alarm $P_{FA}$ in $(0, 1)$.
        n_nc: The number of non-coherent combinations $N_{NC} \ge 1$.

    Returns:
        The minimum required single-sample SNR $\gamma$ in dB.

    See Also:
        sdr.min_snr

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

        Albersheim's equation approximates a linear detector. However, the difference between linear and square-law
        detectors in minimal, so Albersheim's equation finds wide use.

    References:
        - https://radarsp.weebly.com/uploads/2/1/4/7/21471216/albersheim_alternative_forms.pdf
        - https://bpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/5/462/files/2016/12/Noncoherent-Integration-Gain-Approximations.pdf
        - https://www.mathworks.com/help/phased/ref/albersheim.html

    Examples:
        Compare the theoretical minimum required SNR using a linear detector in :func:`sdr.min_snr` with the
        estimated minimum required SNR using Albersheim's approximation in :func:`sdr.albersheim`.

        .. ipython:: python

            p_d = 0.9; \
            p_fa = np.logspace(-12, -1, 21)

            @savefig sdr_albersheim_1.png
            plt.figure(); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=1), linestyle="--"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=2), linestyle="--"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=10), linestyle="--"); \
            plt.semilogx(p_fa, sdr.albersheim(p_d, p_fa, n_nc=20), linestyle="--"); \
            plt.gca().set_prop_cycle(None); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=1, detector="linear"), label="$N_{NC}$ = 1"); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=2, detector="linear"), label="$N_{NC}$ = 2"); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=10, detector="linear"), label="$N_{NC}$ = 10"); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=20, detector="linear"), label="$N_{NC}$ = 20"); \
            plt.legend(); \
            plt.xlabel("Probability of false alarm, $P_{FA}$"); \
            plt.ylabel("Minimum required SNR (dB)"); \
            plt.title("Minimum required SNR across non-coherent combinations for $P_D = 0.9$\nfrom theory (solid) and Albersheim's approximation (dashed)");

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


@export
def peebles(p_d: npt.ArrayLike, p_fa: npt.ArrayLike, n_nc: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Estimates the non-coherent integration gain for a given probability of detection $P_{D}$ and false alarm $P_{FA}$.

    Arguments:
        p_d: The desired probability of detection $P_D$ in $(0, 1)$.
        p_fa: The desired probability of false alarm $P_{FA}$ in $(0, 1)$.
        n_nc: The number of non-coherent combinations $N_{NC} \ge 1$.

    Returns:
        The non-coherent integration gain $G_{NC}$.

    See Also:
        sdr.non_coherent_gain

    Notes:
        This function implements Peebles' equation, given by

        $$
        G_{NC} = 6.79 \cdot \left(1 + 0.253 \cdot P_D\right) \cdot \left(1 + \frac{\log_{10}(1 / P_{FA})}{46.6}\right) \cdot \log_{10}(N_{NC}) \cdot \left(1 - 0.14 \cdot \log_{10}(N_{NC}) + 0.0183 \cdot (\log_{10}(n_{nc}))^2\right)
        $$

        The error in the estimated non-coherent integration gain is claimed to be less than 0.8 dB for

        $$10^{-10} \leq P_{FA} \leq 10^{-2}$$
        $$0.5 \leq P_D \leq 0.999$$
        $$1 \le N_{NC} \le 100 .$$

        Peebles' equation approximates the non-coherent integration gain using a square-law detector.

    Examples:
        Compare the theoretical non-coherent gain for a square-law detector against the approximation from Peebles's
        equation. This comparison plots curves for various post-integration probabilities of detection.

        .. ipython:: python

            fig, ax = plt.subplots(1, 2, sharey=True); \
            p_fa = 1e-8; \
            n = np.linspace(1, 100, 101).astype(int);
            for p_d in [0.5, 0.8, 0.95]:
                snr = sdr.min_snr(p_d, p_fa, detector="square-law")
                ax[0].plot(n, sdr.non_coherent_gain(n, snr, p_fa=p_fa, detector="square-law", snr_ref="output"), label=p_d)
            ax[0].plot(n, sdr.coherent_gain(n), color="k", label="Coherent"); \
            ax[0].legend(title="$P_D$"); \
            ax[0].set_xlabel("Number of samples, $N_{NC}$"); \
            ax[0].set_ylabel("Non-coherent gain, $G_{NC}$"); \
            ax[0].set_title("Theoretical");
            for p_d in [0.5, 0.8, 0.95]:
                ax[1].plot(n, sdr.peebles(p_d, p_fa, n), linestyle="--", label=p_d)
            @savefig sdr_peebles_1.png
            ax[1].plot(n, sdr.coherent_gain(n), color="k", label="Coherent"); \
            ax[1].legend(title="$P_D$"); \
            ax[1].set_xlabel("Number of samples, $N_{NC}$"); \
            ax[1].set_ylabel("Non-coherent gain, $G_{NC}$"); \
            ax[1].set_title("Peebles's approximation");

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

    g_nc = 6.79  # dB
    g_nc *= 1 + 0.253 * p_d
    g_nc *= 1 + np.log10(1 / p_fa) / 46.6
    g_nc *= np.log10(n_nc)
    g_nc *= 1 - 0.14 * np.log10(n_nc) + 0.0183 * np.log10(n_nc) ** 2

    return g_nc
