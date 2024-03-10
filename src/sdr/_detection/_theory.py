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
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
) -> scipy.stats.rv_continuous:
    r"""
    Computes the statistical distribution under the null hypothesis $\mathcal{H}_0$.

    Arguments:
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $T(x) = \mathrm{Re}\{x[n]\}$.
            - `"linear"`: A linear detector, $T(x) = \left| x[n] \right|$.
            - `"square-law"`: A square-law detector, $T(x) = \left| x[n] \right|^2$.

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.

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
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h0_theory_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h0_theory_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h0_theory_3.png
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
            h0 = sdr.h0_theory(sigma2, detector, complex=False); \
            h1 = sdr.h1_theory(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h0_theory_4.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector, complex=False); \
            h1 = sdr.h1_theory(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h0_theory_5.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector, complex=False); \
            h1 = sdr.h1_theory(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h0_theory_6.png
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
    sigma2 = float(sigma2)
    if sigma2 <= 0:
        raise ValueError(f"Argument `sigma2` must be positive, not {sigma2}.")

    if complex:
        nu = 2  # Degrees of freedom
        sigma2_per = sigma2 / 2  # Noise variance per dimension
    else:
        nu = 1
        sigma2_per = sigma2

    if detector == "coherent":
        h0 = scipy.stats.norm(0, np.sqrt(sigma2_per))
    elif detector == "linear":
        h0 = scipy.stats.chi(nu, scale=np.sqrt(sigma2_per))
    elif detector == "square-law":
        h0 = scipy.stats.chi2(nu, scale=sigma2_per)
    else:
        raise ValueError(f"Argument `detector` must be one of 'coherent', 'linear', or 'square-law', not {detector!r}.")

    return h0


@export
def h1_theory(
    snr: float,
    sigma2: float = 1.0,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
) -> scipy.stats.rv_continuous:
    r"""
    Computes the statistical distribution under the alternative hypothesis $\mathcal{H}_1$.

    Arguments:
        snr: The signal-to-noise ratio $S / \sigma^2$ in dB.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $T(x) = \mathrm{Re}\{x[n]\}$.
            - `"linear"`: A linear detector, $T(x) = \left| x[n] \right|$.
            - `"square-law"`: A square-law detector, $T(x) = \left| x[n] \right|^2$.

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.

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
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h1_theory_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h1_theory_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold
            p_d = sdr.p_d(snr, p_fa, detector); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h1_theory_3.png
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
            h0 = sdr.h0_theory(sigma2, detector, complex=False); \
            h1 = sdr.h1_theory(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.real(x_h0); \
            z_h1 = np.real(x_h1)

            @savefig sdr_h1_theory_4.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Coherent Detector: Probability density functions");

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector, complex=False); \
            h1 = sdr.h1_theory(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0); \
            z_h1 = np.abs(x_h1)

            @savefig sdr_h1_theory_5.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.gca().set_prop_cycle(None); \
            plt.hist(z_h0, bins=101, histtype="step", density=True); \
            plt.hist(z_h1, bins=101, histtype="step", density=True); \
            plt.title("Linear Detector: Probability density functions");

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector, complex=False); \
            h1 = sdr.h1_theory(snr, sigma2, detector, complex=False)

            threshold = sdr.threshold(p_fa, sigma2, detector, complex=False); threshold
            p_d = sdr.p_d(snr, p_fa, detector, complex=False); p_d

            z_h0 = np.abs(x_h0) ** 2; \
            z_h1 = np.abs(x_h1) ** 2

            @savefig sdr_h1_theory_6.png
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
    snr = float(snr)
    sigma2 = float(sigma2)
    if sigma2 <= 0:
        raise ValueError(f"Argument `sigma2` must be positive, not {sigma2}.")

    A2 = linear(snr) * sigma2  # Signal power, A^2
    if complex:
        nu = 2  # Degrees of freedom
        sigma2_per = sigma2 / 2  # Noise variance per dimension
    else:
        nu = 1
        sigma2_per = sigma2
    lambda_ = A2 / (sigma2_per)  # Non-centrality parameter

    if detector == "coherent":
        h1 = scipy.stats.norm(np.sqrt(A2), np.sqrt(sigma2_per))
    elif detector == "linear":
        if complex:
            # Rice distribution has 2 degrees of freedom
            h1 = scipy.stats.rice(np.sqrt(lambda_), scale=np.sqrt(sigma2_per))
        else:
            # Folded normal distribution has 1 degree of freedom
            h1 = scipy.stats.foldnorm(np.sqrt(lambda_), scale=np.sqrt(sigma2_per))
    elif detector == "square-law":
        h1 = scipy.stats.ncx2(nu, lambda_, scale=sigma2_per)
    else:
        raise ValueError(f"Argument `detector` must be one of 'coherent', 'linear', or 'square-law', not {detector!r}.")

    return h1


@export
def p_d(
    snr: npt.ArrayLike,
    p_fa: npt.ArrayLike,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical probability of detection $P_{D}$.

    Arguments:
        snr: The signal-to-noise ratio $S / \sigma^2$ in dB.
        p_fa: The probability of false alarm $P_{FA}$ in $(0, 1)$.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $T(x) = \mathrm{Re}\{x[n]\}$.
            - `"linear"`: A linear detector, $T(x) = \left| x[n] \right|$.
            - `"square-law"`: A square-law detector, $T(x) = \left| x[n] \right|^2$.

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.

    Returns:
        The probability of detection $P_D$ in $(0, 1)$.

    Examples:
        .. ipython:: python

            snr = 3  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        Compute the probability of detection for the coherent detector. Plot the PDFs and observe the theoretical
        $P_{D}$ is achieved.

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            p_d = sdr.p_d(snr, p_fa, detector); p_d

            @savefig sdr_p_d_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, h1=h1, threshold=threshold); \
            plt.title("Coherent Detector: Probability density functions");

        Compute the probability of detection for the linear detector. Plot the PDFs and observe the theoretical
        $P_{D}$ is achieved.

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            p_d = sdr.p_d(snr, p_fa, detector); p_d

            @savefig sdr_p_d_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, h1=h1, threshold=threshold); \
            plt.title("Linear Detector: Probability density functions");

        Compute the probability of detection for the square-law detector. Plot the PDFs and observe the theoretical
        $P_{D}$ is achieved.

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector)

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

        Compare the $P_{D}$ of the square-law detector for various $P_{FA}$.

        .. ipython:: python

            plt.figure(); \
            snr = np.linspace(-10, 20, 101);
            for p_fa in [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1]:
                p_d = sdr.p_d(snr, p_fa)
                sdr.plot.p_d(snr, p_d, label=f"{p_fa:1.0e}")
            @savefig sdr_p_d_5.png
            plt.legend(title="$P_{FA}$", loc="upper left"); \
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
    snr = np.asarray(snr)
    p_fa = np.asarray(p_fa)

    sigma2 = 1

    @np.vectorize
    def _calculate(snr, p_fa):
        h1 = h1_theory(snr, sigma2, detector, complex)
        gamma = threshold(p_fa, sigma2, detector, complex)
        return h1.sf(gamma)

    p_d = _calculate(snr, p_fa)
    if p_d.ndim == 0:
        p_d = p_d.item()

    return p_d


@export
def p_fa(
    threshold: npt.ArrayLike,
    sigma2: npt.ArrayLike = 1,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical probability of false alarm $P_{FA}$.

    Arguments:
        threshold: The detection threshold $\gamma$ in linear units.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $T(x) = \mathrm{Re}\{x[n]\}$.
            - `"linear"`: A linear detector, $T(x) = \left| x[n] \right|$.
            - `"square-law"`: A square-law detector, $T(x) = \left| x[n] \right|^2$.

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.

    Returns:
        The probability of false alarm $P_{FA}$ in $(0, 1)$.

    Examples:
        .. ipython:: python

            threshold = 1.0  # Detection threshold
            sigma2 = 1  # Noise variance

        Compute the probability of false alarm for the coherent detector. Plot the PDFs and observe the theoretical
        $P_{FA}$ is achieved.

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0_theory(sigma2, detector)

            p_fa = sdr.p_fa(threshold, sigma2, detector); p_fa

            @savefig sdr_p_fa_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Coherent Detector: Probability density functions");

        Compute the probability of false alarm for the linear detector. Plot the PDFs and observe the theoretical
        $P_{FA}$ is achieved.

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector)

            p_fa = sdr.p_fa(threshold, sigma2, detector); p_fa

            @savefig sdr_p_fa_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Linear Detector: Probability density functions");

        Compute the probability of false alarm for the square-law detector. Plot the PDFs and observe the theoretical
        $P_{FA}$ is achieved.

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector)

            p_fa = sdr.p_fa(threshold, sigma2, detector); p_fa

            @savefig sdr_p_fa_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    threshold = np.asarray(threshold)
    sigma2 = np.asarray(sigma2)

    @np.vectorize
    def _calculate(threshold, sigma2):
        h0 = h0_theory(sigma2, detector, complex)
        return h0.sf(threshold)

    p_fa = _calculate(threshold, sigma2)
    if p_fa.ndim == 0:
        p_fa = p_fa.item()

    return p_fa


@export
def threshold(
    p_fa: npt.ArrayLike,
    sigma2: npt.ArrayLike = 1,
    detector: Literal["coherent", "linear", "square-law"] = "square-law",
    complex: bool = True,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the theoretical detection threshold $\gamma$.

    Arguments:
        p_fa: The desired probability of false alarm $P_{FA}$ in $(0, 1)$.
        sigma2: The noise variance $\sigma^2$ in linear units.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $T(x) = \mathrm{Re}\{x[n]\}$.
            - `"linear"`: A linear detector, $T(x) = \left| x[n] \right|$.
            - `"square-law"`: A square-law detector, $T(x) = \left| x[n] \right|^2$.

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.

    Returns:
        The detection threshold $\gamma$ in linear units.

    Examples:
        .. ipython:: python

            p_fa = 1e-1  # Probability of false alarm
            sigma2 = 1  # Noise variance

        Compute the detection threshold for the coherent detector. Plot the PDFs and observe the desired $P_{FA}$
        is achieved.

        .. ipython:: python

            detector = "coherent"; \
            h0 = sdr.h0_theory(sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            @savefig sdr_threshold_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Coherent Detector: Probability density functions");

        Compute the detection threshold for the linear detector. Plot the PDFs and observe the desired $P_{FA}$
        is achieved.

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            @savefig sdr_threshold_2.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Linear Detector: Probability density functions");

        Compute the detection threshold for the square-law detector. Plot the PDFs and observe the desired $P_{FA}$
        is achieved.

        .. ipython:: python

            detector = "square-law"; \
            h0 = sdr.h0_theory(sigma2, detector)

            threshold = sdr.threshold(p_fa, sigma2, detector); threshold

            @savefig sdr_threshold_3.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0=h0, threshold=threshold); \
            plt.title("Square-Law Detector: Probability density functions");

    Group:
        detection-theory
    """
    p_fa = np.asarray(p_fa)
    sigma2 = np.asarray(sigma2)

    @np.vectorize
    def _calculate(p_fa, sigma2):
        h0 = h0_theory(sigma2, detector, complex)
        return h0.isf(p_fa)

    threshold = _calculate(p_fa, sigma2)
    if threshold.ndim == 0:
        threshold = threshold.item()

    return threshold
