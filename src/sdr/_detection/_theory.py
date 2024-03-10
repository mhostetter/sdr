"""
A module containing functions to compute theoretical detection performance.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats
from typing_extensions import Literal

from .._conversion import linear
from .._helper import export


@export
def h0_theory(
    sigma2: float = 1.0,
    detector: Literal["real", "linear", "square-law"] = "square-law",
) -> scipy.stats.rv_continuous:
    r"""
    Computes the statistical distribution under the null hypothesis $\mathcal{H}_0$.

    Arguments:
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"real"`: The real detector.
            - `"linear"`: The linear detector.
            - `"square-law"`: The square-law detector.

    Returns:
        The probability density function under the null hypothesis $\mathcal{H}_0$.

    Examples:
        .. ipython:: python

            snr = 5  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        .. ipython:: python

            detector = "real"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector); \
            threshold = sdr.threshold(p_fa, sigma2, detector)

            @savefig sdr_h0_theory_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.title("Real Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector); \
            threshold = sdr.threshold(p_fa, sigma2, detector)

            @savefig sdr_h0_theory_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector); \
            threshold = sdr.threshold(p_fa, sigma2, detector)

            @savefig sdr_h0_theory_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    sigma2 = float(sigma2)
    if sigma2 <= 0:
        raise ValueError(f"Argument `sigma2` must be positive, not {sigma2}.")

    nu = 2  # Degrees of freedom

    if detector == "square-law":
        h0 = scipy.stats.chi2(nu, scale=sigma2 / 2)
    elif detector == "linear":
        h0 = scipy.stats.chi(nu, scale=sigma2 / 2)
    elif detector == "real":
        h0 = scipy.stats.norm(0, np.sqrt(sigma2 / 2))
    else:
        raise ValueError(f"Argument `detector` must be one of 'square-law', 'linear', or 'real', not {detector}.")

    return h0


@export
def p_d(
    snr: npt.ArrayLike,
    p_fa: npt.ArrayLike,
    detector: Literal["square-law", "linear", "real"] = "square-law",
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical probability of detection.

    Arguments:
        snr: The signal-to-noise ratio $S / \sigma^2$ in dB.
        p_fa: The probability of false alarm $P_{FA}$ in $(0, 1)$.
        detector: The detector type.

            - `"square-law"`: The square-law detector.
            - `"linear"`: The linear detector.
            - `"real"`: The real detector.

    Returns:
        The probability of detection $P_D$ in $(0, 1)$.

    Examples:
        .. ipython:: python

            plt.figure(); \
            snr = np.linspace(-10, 20, 101);
            for p_fa in [1e-12, 1e-9, 1e-6, 1e-3]:
                p_d = sdr.p_d(snr, p_fa)
                sdr.plot.p_d(snr, p_d, label=f"{p_fa:1.0e}")
            @savefig sdr_p_d_1.png
            plt.legend(title="$P_{FA}$", loc="upper left"); \
            plt.title("Square-Law Detector: Probability of detection");

        .. ipython:: python

            plt.figure(); \
            p_fa = np.logspace(-15, 0, 101);
            for snr in [-5, 0, 5, 10, 15, 20]:
                p_d = sdr.p_d(snr, p_fa)
                sdr.plot.roc(p_fa, p_d, label=f"{snr} dB")
            @savefig sdr_p_d_2.png
            plt.legend(title="SNR"); \
            plt.title("Square-Law Detector: Receiver operating characteristic");

    Group:
        detection-theory
    """
    snr = np.asarray(snr)
    p_fa = np.asarray(p_fa)

    snr = linear(snr)

    if detector == "square-law":
        p_d = _p_d_square_law(snr, p_fa)
    else:
        raise ValueError(f"Invalid detector type: {detector}")

    if p_d.ndim == 0:
        p_d = p_d.item()

    return p_d


def _p_d_square_law(
    snr: npt.NDArray[np.float64],
    p_fa: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    sigma2 = 1  # Noise variance
    A2 = snr * sigma2  # Signal power

    nu = 2  # Degrees of freedom
    lambda_ = A2 / (sigma2 / 2)  # Non-centrality parameter
    h1_theory = scipy.stats.ncx2(nu, lambda_, scale=sigma2 / 2)
    threshold = _threshold_square_law(p_fa, sigma2)
    p_d = h1_theory.sf(threshold)

    return p_d


@export
def p_fa(
    threshold: npt.ArrayLike,
    sigma2: npt.ArrayLike = 1,
    detector: Literal["square-law", "linear", "real"] = "square-law",
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical probability of false alarm.

    Arguments:
        threshold: The detection threshold $\gamma$ in linear units.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"square-law"`: The square-law detector.
            - `"linear"`: The linear detector.
            - `"real"`: The real detector.

    Returns:
        The probability of false alarm $P_{FA}$ in $(0, 1)$.

    Examples:
        .. ipython:: python

            @savefig sdr_p_fa_1.png
            plt.figure(); \
            threshold = np.linspace(0, 20, 101); \
            p_fa = sdr.p_fa(threshold, 1); \
            plt.semilogy(threshold, p_fa); \
            plt.xlabel("Threshold"); \
            plt.ylabel("Probability of false alarm, $P_{FA}$"); \
            plt.title("Square-Law Detector: False alarm performance");

    Group:
        detection-theory
    """
    threshold = np.asarray(threshold)
    sigma2 = np.asarray(sigma2)

    if detector == "square-law":
        p_fa = _p_fa_square_law(threshold, sigma2)
    else:
        raise ValueError(f"Invalid detector type: {detector}")

    if p_fa.ndim == 0:
        p_fa = p_fa.item()

    return p_fa


def _p_fa_square_law(
    threshold: npt.NDArray[np.float64],
    sigma2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    nu = 2  # Degrees of freedom
    h0_theory = scipy.stats.chi2(nu, scale=sigma2 / 2)
    p_fa = h0_theory.sf(threshold)

    return p_fa


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
