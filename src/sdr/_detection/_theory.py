"""
A module containing functions to compute theoretical detection performance.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
import scipy.stats
from typing_extensions import Literal

from .._conversion import db as to_db
from .._conversion import linear
from .._helper import (
    convert_output,
    export,
    verify_arraylike,
    verify_bool,
    verify_literal,
    verify_not_specified,
    verify_scalar,
)
from .._probability import add_distribution


@export
def h0(
    sigma2: float = 1.0,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> scipy.stats.rv_continuous:
    r"""
    Computes the statistical distribution under the null hypothesis $\mathcal{H}_0$.

    Arguments:
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.

    Returns:
        The distribution under the null hypothesis $\mathcal{H}_0$.

    See Also:
        sdr.plot.detector_pdfs

    Examples:
        .. ipython:: python

            snr = 5  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        Compare the detection statistics for complex signals.

        .. ipython:: python

            A2 = sdr.linear(snr) * sigma2; A2  # Signal power, A^2

            x_h0 = rng.normal(size=100_000, scale=np.sqrt(sigma2 / 2)) + 1j * rng.normal(size=100_000, scale=np.sqrt(sigma2 / 2)); \
            x_h1 = np.sqrt(A2) + x_h0

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h0_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h0_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h0_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Square-Law Detector: Probability density functions");

        Compare the detection statistics for real signals.

        .. ipython:: python

            A2 = sdr.linear(snr) * sigma2; A2  # Signal power, A^2

            x_h0 = rng.normal(size=100_000, scale=np.sqrt(sigma2)); \
            x_h1 = np.sqrt(A2) + x_h0

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector, complex=False); \
            h1 = sdr.h1(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h0_4.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector, complex=False); \
            h1 = sdr.h1(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h0_5.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector, complex=False); \
            h1 = sdr.h1(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h0_6.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold, x=np.linspace(0, 15, 1001)); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.xlim(0, 15); \
            plt.ylim(0, 0.5); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    verify_scalar(sigma2, float=True, non_negative=True)
    verify_literal(detector, ["coherent", "linear", "square-law"])
    verify_bool(complex)
    verify_scalar(n_c, int=True, positive=True)
    verify_scalar(n_nc, optional=True, int=True, positive=True)
    if detector == "coherent":
        verify_not_specified(n_nc)

    return _h0(sigma2, detector, complex, n_c, n_nc)


@lru_cache
def _h0(
    sigma2: float = 1.0,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> scipy.stats.rv_continuous:
    # sigma2 = float(sigma2)
    if n_nc is None:
        n_nc = 1

    if complex:
        nu = 2  # Degrees of freedom
        sigma2_per = sigma2 / 2  # Noise variance per dimension
    else:
        nu = 1
        sigma2_per = sigma2

    # Coherent integration scales the noise power by n_c
    sigma2_per *= n_c

    with np.errstate(invalid="ignore"):
        if detector == "coherent":
            h0 = scipy.stats.norm(0, np.sqrt(sigma2_per))
        elif detector == "linear":
            h0 = scipy.stats.chi(nu, scale=np.sqrt(sigma2_per))
            h0 = add_distribution(h0, n_nc)
        elif detector == "square-law":
            h0 = scipy.stats.chi2(nu * n_nc, scale=sigma2_per)

    return h0


@export
def h1(
    snr: float,
    sigma2: float = 1.0,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> scipy.stats.rv_continuous:
    r"""
    Computes the statistical distribution under the alternative hypothesis $\mathcal{H}_1$.

    Arguments:
        snr: The signal-to-noise ratio $S / \sigma^2$ in dB.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.

    Returns:
        The distribution under the alternative hypothesis $\mathcal{H}_1$.

    See Also:
        sdr.plot.detector_pdfs

    Examples:
        .. ipython:: python

            snr = 5  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        Compare the detection statistics for complex signals.

        .. ipython:: python

            A2 = sdr.linear(snr) * sigma2; A2  # Signal power, A^2

            x_h0 = rng.normal(size=100_000, scale=np.sqrt(sigma2 / 2)) + 1j * rng.normal(size=100_000, scale=np.sqrt(sigma2 / 2)); \
            x_h1 = np.sqrt(A2) + x_h0

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h1_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h1_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h1_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Square-Law Detector: Probability density functions");

        Compare the detection statistics for real signals.

        .. ipython:: python

            A2 = sdr.linear(snr) * sigma2; A2  # Signal power, A^2

            x_h0 = rng.normal(size=100_000, scale=np.sqrt(sigma2)); \
            x_h1 = np.sqrt(A2) + x_h0

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector, complex=False); \
            h1 = sdr.h1(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h1_4.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector, complex=False); \
            h1 = sdr.h1(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h1_5.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector, complex=False); \
            h1 = sdr.h1(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h1_6.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold, x=np.linspace(0, 15, 1001)); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.xlim(0, 15); \
            plt.ylim(0, 0.5); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    verify_scalar(snr, float=True)
    verify_scalar(sigma2, float=True, non_negative=True)
    verify_literal(detector, ["coherent", "linear", "square-law"])
    verify_bool(complex)
    verify_scalar(n_c, int=True, positive=True)
    verify_scalar(n_nc, optional=True, int=True, positive=True)
    if detector == "coherent":
        verify_not_specified(n_nc)

    return _h1(snr, sigma2, detector, complex, n_c, n_nc)


@lru_cache
def _h1(
    snr: float,
    sigma2: float = 1.0,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> scipy.stats.rv_continuous:
    # snr = float(snr)
    # sigma2 = float(sigma2)
    if n_nc is None:
        n_nc = 1

    A2 = linear(snr) * sigma2  # Signal power, A^2
    if complex:
        nu = 2  # Degrees of freedom
        sigma2_per = sigma2 / 2  # Noise variance per dimension
    else:
        nu = 1
        sigma2_per = sigma2

    # Coherent integration scales the signal power by n_c^2 and the noise power by n_c
    A2 *= n_c**2
    sigma2_per *= n_c

    lambda_ = A2 / (sigma2_per)  # Non-centrality parameter

    with np.errstate(invalid="ignore"):
        if detector == "coherent":
            h1 = scipy.stats.norm(np.sqrt(A2), np.sqrt(sigma2_per))
        elif detector == "linear":
            if complex:
                # Rice distribution has 2 degrees of freedom
                h1 = scipy.stats.rice(np.sqrt(lambda_), scale=np.sqrt(sigma2_per))

                # Sometimes this distribution has so much SNR that SciPy throws an error when computing the mean, etc.
                # We need to check for that condition. If true, we cant approximate the distribution by a Gaussian,
                # which doesn't suffer from the same error.
                if np.isnan(h1.mean()):
                    h1 = scipy.stats.norm(np.sqrt(lambda_ * sigma2_per), scale=np.sqrt(sigma2_per))
            else:
                # Folded normal distribution has 1 degree of freedom
                h1 = scipy.stats.foldnorm(np.sqrt(lambda_), scale=np.sqrt(sigma2_per))
            h1 = add_distribution(h1, n_nc)
        elif detector == "square-law":
            h1 = scipy.stats.ncx2(nu * n_nc, lambda_ * n_nc, scale=sigma2_per)

    return h1


@export
def p_d(
    snr: npt.ArrayLike,
    p_fa: npt.ArrayLike,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical probability of detection $P_d$.

    Arguments:
        snr: The signal-to-noise ratio $S / \sigma^2$ in dB.
        p_fa: The probability of false alarm $P_{fa}$ in $(0, 1)$.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.

    Returns:
        The probability of detection $P_d$ in $(0, 1)$.

    Notes:
        The probability of detection $P_d$ is defined as the probability that the detector output $T(x)$ exceeds the
        detection threshold $\gamma$ given that the signal is present.

        $$P_d = P\left(T(x) > \gamma \mid \mathcal{H}_1\right)$$

    Examples:
        .. ipython:: python

            snr = 3  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        Compute the probability of detection for the coherent detector. Plot the PDFs and observe the theoretical
        $P_d$ is achieved.

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            p_d = sdr.p_d(snr, p_fa, detector); p_d

            @savefig sdr_p_d_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, h1=h1, threshold=threshold); \
            plt.title("Coherent Detector: Probability density functions");

        Compute the probability of detection for the linear detector. Plot the PDFs and observe the theoretical
        $P_d$ is achieved.

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            p_d = sdr.p_d(snr, p_fa, detector); p_d

            @savefig sdr_p_d_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, h1=h1, threshold=threshold); \
            plt.title("Linear Detector: Probability density functions");

        Compute the probability of detection for the square-law detector. Plot the PDFs and observe the theoretical
        $P_d$ is achieved.

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            p_d = sdr.p_d(snr, p_fa, detector); p_d

            @savefig sdr_p_d_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, h1=h1, threshold=threshold); \
            plt.title("Square-Law Detector: Probability density functions");

        Compare the detection performance of each detector.

        .. ipython:: python

            p_fa = 1e-6  # Probability of false alarm

            @savefig sdr_p_d_4.png
            plt.figure(); \
            snr = np.linspace(0, 20, 101); \
            sdr.plot.p_d(snr, sdr.p_d(snr, p_fa, "coherent"), label="Coherent"); \
            sdr.plot.p_d(snr, sdr.p_d(snr, p_fa, "linear"), label="Linear"); \
            sdr.plot.p_d(snr, sdr.p_d(snr, p_fa, "square-law"), label="Square-Law"); \
            plt.legend(title="Detector"); \
            plt.title("Probability of detection");

        Compare the $P_d$ of the square-law detector for various $P_{fa}$.

        .. ipython:: python

            plt.figure(); \
            snr = np.linspace(-10, 20, 101);
            for p_fa in [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1]:
                p_d = sdr.p_d(snr, p_fa)
                sdr.plot.p_d(snr, p_d, label=f"{p_fa:1.0e}")
            @savefig sdr_p_d_5.png
            plt.legend(title="$P_{fa}$", loc="upper left"); \
            plt.title("Square-Law Detector: Probability of detection");

        Compare the receiver operating characteristics (ROCs) of the square-law detector for various SNRs.

        .. ipython:: python

            plt.figure(); \
            p_fa = np.logspace(-15, 0, 101);
            for snr in [-5, 0, 5, 10, 15, 20]:
                p_d = sdr.p_d(snr, p_fa)
                sdr.plot.roc(p_fa, p_d, label=f"{snr} dB")
            @savefig sdr_p_d_6.png
            plt.legend(title="SNR"); \
            plt.title("Square-Law Detector: Receiver operating characteristic");

    Group:
        detection-theory
    """
    snr = verify_arraylike(snr, float=True)
    p_fa = verify_arraylike(p_fa, float=True, inclusive_min=0, inclusive_max=1)
    verify_literal(detector, ["coherent", "linear", "square-law"])
    verify_bool(complex)
    verify_scalar(n_c, int=True, positive=True)
    verify_scalar(n_nc, optional=True, int=True, positive=True)
    if detector == "coherent":
        verify_not_specified(n_nc)

    sigma2 = 1

    @np.vectorize
    def _calculate(snr, p_fa):
        h1 = _h1(snr, sigma2, detector, complex, n_c, n_nc)
        gamma = threshold(p_fa, sigma2, detector, complex, n_c, n_nc)
        return h1.sf(gamma)

    p_d = _calculate(snr, p_fa)

    return convert_output(p_d)


@export
def p_fa(
    threshold: npt.ArrayLike,
    sigma2: npt.ArrayLike = 1,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical probability of false alarm $P_{fa}$.

    Arguments:
        threshold: The detection threshold $\gamma$ in linear units.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.

    Returns:
        The probability of false alarm $P_{fa}$ in $(0, 1)$.

    Notes:
        The probability of false alarm $P_{fa}$ is defined as the probability that the detector output $T(x)$ exceeds
        the detection threshold $\gamma$ given that the signal is absent.

        $$P_{fa} = P\left(T(x) > \gamma \mid \mathcal{H}_0\right)$$

    Examples:
        .. ipython:: python

            threshold = 1.0  # Detection threshold
            sigma2 = 1  # Noise variance

        Compute the probability of false alarm for the coherent detector. Plot the PDFs and observe the theoretical
        $P_{fa}$ is achieved.

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector)

            p_fa = sdr.p_fa(threshold, sigma2, detector); p_fa

            @savefig sdr_p_fa_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Coherent Detector: Probability density functions");

        Compute the probability of false alarm for the linear detector. Plot the PDFs and observe the theoretical
        $P_{fa}$ is achieved.

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector)

            p_fa = sdr.p_fa(threshold, sigma2, detector); p_fa

            @savefig sdr_p_fa_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Linear Detector: Probability density functions");

        Compute the probability of false alarm for the square-law detector. Plot the PDFs and observe the theoretical
        $P_{fa}$ is achieved.

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector)

            p_fa = sdr.p_fa(threshold, sigma2, detector); p_fa

            @savefig sdr_p_fa_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    threshold = verify_arraylike(threshold, float=True)
    sigma2 = verify_arraylike(sigma2, float=True, non_negative=True)
    verify_literal(detector, ["coherent", "linear", "square-law"])
    verify_bool(complex)
    verify_scalar(n_c, int=True, positive=True)
    verify_scalar(n_nc, optional=True, int=True, positive=True)
    if detector == "coherent":
        verify_not_specified(n_nc)

    @np.vectorize
    def _calculate(threshold, sigma2):
        h0 = _h0(sigma2, detector, complex, n_c, n_nc)
        return h0.sf(threshold)

    p_fa = _calculate(threshold, sigma2)

    return convert_output(p_fa)


@export
def threshold(
    p_fa: npt.ArrayLike,
    sigma2: npt.ArrayLike = 1,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical detection threshold $\gamma$.

    Arguments:
        p_fa: The desired probability of false alarm $P_{fa}$ in $(0, 1)$.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.

    Returns:
        The detection threshold $\gamma$ in linear units.

    Examples:
        .. ipython:: python

            p_fa = 1e-1  # Probability of false alarm
            sigma2 = 1  # Noise variance

        Compute the detection threshold for the coherent detector. Plot the PDFs and observe the desired $P_{fa}$
        is achieved.

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0(sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            @savefig sdr_threshold_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Coherent Detector: Probability density functions");

        Compute the detection threshold for the linear detector. Plot the PDFs and observe the desired $P_{fa}$
        is achieved.

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            @savefig sdr_threshold_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Linear Detector: Probability density functions");

        Compute the detection threshold for the square-law detector. Plot the PDFs and observe the desired $P_{fa}$
        is achieved.

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0(sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            @savefig sdr_threshold_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    p_fa = verify_arraylike(p_fa, float=True, inclusive_min=0, inclusive_max=1)
    sigma2 = verify_arraylike(sigma2, float=True, non_negative=True)
    verify_literal(detector, ["coherent", "linear", "square-law"])
    verify_bool(complex)
    verify_scalar(n_c, int=True, positive=True)
    verify_scalar(n_nc, optional=True, int=True, positive=True)
    if detector == "coherent":
        verify_not_specified(n_nc)

    @np.vectorize
    def _calculate(p_fa, sigma2):
        h0 = _h0(sigma2, detector, complex, n_c, n_nc)
        return h0.isf(p_fa)

    threshold = _calculate(p_fa, sigma2)

    return convert_output(threshold)


@export
def threshold_factor(
    p_fa: npt.ArrayLike,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
    db: bool = False,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical detection threshold factor $\alpha$.

    Arguments:
        p_fa: The desired probability of false alarm $P_{fa}$ in $(0, 1)$.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.
        db: Indicates whether to return the detection threshold $\alpha$ in dB.

    Returns:
        The detection threshold factor $\alpha$.

    Notes:
        The detection threshold factor $\alpha$ is defined as the ratio of the detection threshold $\gamma$ to the
        mean of the detector output under the null hypothesis. This is true for linear and square-law detectors.

        $$\alpha = \frac{\gamma}{\frac{1}{N} \sum_{i=1}^{N}\{T(x[i]) \mid \mathcal{H}_0\}}$$

        For coherent detectors, the detection threshold factor $\alpha$ is defined as the ratio of the detection
        threshold $\gamma$ to the mean of the square of the detector output under the null hypothesis.
        This is required because the mean of the coherent detector output is zero.

        $$\alpha = \frac{\gamma}{\frac{1}{N} \sum_{i=1}^{N}\{T(x[i])^2 \mid \mathcal{H}_0\}}$$

    Examples:
        .. ipython:: python

            p_fa = np.logspace(-16, -1, 101)  # Probability of false alarm

            @savefig sdr_threshold_factor_1.png
            plt.figure(); \
            plt.semilogx(p_fa, sdr.threshold_factor(p_fa, detector="coherent"), label="Coherent"); \
            plt.semilogx(p_fa, sdr.threshold_factor(p_fa, detector="linear"), label="Linear"); \
            plt.semilogx(p_fa, sdr.threshold_factor(p_fa, detector="square-law"), label="Square-Law"); \
            plt.xlabel("Probability of false alarm, $P_{fa}$"); \
            plt.ylabel(r"Detection threshold factor, $\alpha$"); \
            plt.legend(title="Detector"); \
            plt.title("Detection threshold factor across false alarm rate");

    Group:
        detection-theory
    """
    sigma2 = 1
    gamma = threshold(p_fa, sigma2, detector, complex, n_c, n_nc)
    null_hypothesis = h0(sigma2, detector, complex, n_c, n_nc)

    if detector == "coherent":
        # Since mean is zero, the variance is equivalent to the mean of the square
        alpha = gamma / null_hypothesis.var()
    else:
        alpha = gamma / null_hypothesis.mean()

    if db:
        alpha = to_db(alpha)

    return alpha


@export
def min_snr(
    p_d: npt.ArrayLike,
    p_fa: npt.ArrayLike,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
    n_c: int = 1,
    n_nc: int | None = None,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the minimum input signal-to-noise ratio (SNR) required to achieve the desired probability of detection
    $P_d$.

    Arguments:
        p_d: The desired probability of detection $P_d$ in $(0, 1)$.
        p_fa: The probability of false alarm $P_{fa}$ in $(0, 1)$.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        n_c: The number of samples to coherently integrate $N_c$.
        n_nc: The number of samples to non-coherently integrate $N_{nc}$. Non-coherent integration is only allowable
            for linear and square-law detectors.

    Returns:
        The minimum signal-to-noise ratio (SNR) required to achieve the desired $P_d$.

    See Also:
        sdr.albersheim

    Examples:
        Compute the minimum required SNR to achieve $P_d = 0.9$ and $P_{fa} = 10^{-6}$ with a square-law detector.

        .. ipython:: python

            sdr.min_snr(0.9, 1e-6, detector="square-law")

        Now suppose the signal is non-coherently integrated $N_{nc} = 10$ times. Notice the minimum required SNR
        decreases, but by less than 10 dB. This is because non-coherent integration is less efficient than coherent
        integration.

        .. ipython:: python

            sdr.min_snr(0.9, 1e-6, detector="square-law", n_nc=10)

        Now suppose the signal is coherently integrated for $N_c = 10$ samples before the square-law detector.
        Notice the SNR now decreases by exactly 10 dB.

        .. ipython:: python

            sdr.min_snr(0.9, 1e-6, detector="square-law", n_c=10, n_nc=10)

        Compare the theoretical minimum required SNR using linear and square-law detectors.

        .. ipython:: python

            p_d = 0.9; \
            p_fa = np.logspace(-12, -1, 21)

            @savefig sdr_min_snr_1.png
            plt.figure(); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=1, detector="square-law"), label=1); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=2, detector="square-law"), label=2); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=4, detector="square-law"), label=4); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=8, detector="square-law"), label=8); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=16, detector="square-law"), label=16); \
            plt.semilogx(p_fa, sdr.min_snr(p_d, p_fa, n_nc=32, detector="square-law"), label=32); \
            plt.legend(title="$N_{nc}$"); \
            plt.xlabel("Probability of false alarm, $P_{fa}$"); \
            plt.ylabel("Minimum required input SNR (dB)"); \
            plt.title("Minimum required input SNR across non-coherent combinations for $P_d = 0.9$");

    Group:
        detection-theory
    """
    p_d = verify_arraylike(p_d, float=True, inclusive_min=0, inclusive_max=1)
    p_fa = verify_arraylike(p_fa, float=True, inclusive_min=0, inclusive_max=1)
    verify_literal(detector, ["coherent", "linear", "square-law"])
    verify_bool(complex)
    verify_scalar(n_c, int=True, positive=True)
    verify_scalar(n_nc, optional=True, int=True, positive=True)
    if detector == "coherent":
        verify_not_specified(n_nc)

    calc_p_d = globals()["p_d"]

    @np.vectorize
    def _calculate(p_d, p_fa):
        def _objective(snr):
            return p_d - calc_p_d(snr, p_fa, detector, complex, n_c, n_nc)

        # The max SNR may return p_d = NaN. If so, we need to reduce the max value or optimize.brentq() will error.
        min_snr = -100  # dB
        max_snr = 30  # dB
        while True:
            if np.isnan(_objective(max_snr)):
                max_snr -= 10
            else:
                break

        snr = scipy.optimize.brentq(_objective, min_snr, max_snr)

        return snr

    snr = _calculate(p_d, p_fa)

    return convert_output(snr)
