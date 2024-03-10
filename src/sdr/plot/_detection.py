"""
A module containing detection-related plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def p_d(
    x: npt.ArrayLike,
    p_d: npt.ArrayLike,
    x_label: Literal["snr", "enr"] = "snr",
    **kwargs,
):
    r"""
    Plots the probability of detection $P_D$ as a function of received SNR or ENR.

    Arguments:
        x: The SNR or ENR in dB.
        p_d: The probability of detection $P_D$.
        x_label: The x-axis label to use.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        plt.plot(x, p_d, **kwargs)

        plt.ylim(0, 1)
        if "label" in kwargs:
            plt.legend()

        if x_label == "snr":
            plt.xlabel(r"Signal-to-noise ratio (dB), $S/\sigma^2$")
        elif x_label == "enr":
            plt.xlabel(r"Energy-to-noise ratio (dB), $\mathcal{E}/\sigma^2$")
        else:
            raise ValueError(f"Argument 'x_label' must be one of ['snr', 'enr'], not {x_label!r}.")

        plt.ylabel("Probability of detection, $P_{D}$")
        plt.title("Detection performance")


@export
def roc(
    p_fa: npt.ArrayLike,
    p_d: npt.ArrayLike,
    type: Literal["linear", "semilogx", "semilogy", "loglog"] = "semilogx",
    **kwargs,
):
    r"""
    Plots the receiver operating characteristic (ROC) curve as a function of $P_{FA}$.

    Arguments:
        p_fa: The probability of false alarm $P_{FA}$.
        p_d: The probability of detection $P_D$.
        type: The type of plot to generate.
        kwargs: Additional keyword arguments to pass to the plotting function defined by `type`.

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        if type == "linear":
            plt.plot(p_fa, p_d, **kwargs)
        elif type == "semilogx":
            plt.semilogx(p_fa, p_d, **kwargs)
        elif type == "semilogy":
            plt.semilogy(p_fa, p_d, **kwargs)
        elif type == "loglog":
            plt.loglog(p_fa, p_d, **kwargs)
        else:
            raise ValueError(
                f"Argument 'type' must be one of ['linear', 'semilogx', 'semilogy', 'loglog'], not {type!r}."
            )

        if "label" in kwargs:
            plt.legend()

        plt.xlabel("Probability of false alarm, $P_{FA}$")
        plt.ylabel("Probability of detection, $P_{D}$")
        plt.title("Receiver operating characteristic (ROC) curve")


@export
def detector_pdfs(
    h0: scipy.stats.rv_continuous | None = None,
    h1: scipy.stats.rv_continuous | None = None,
    threshold: float | None = None,
    shade: bool = True,
    annotate: bool = True,
    x: npt.NDArray[np.float64] | None = None,
    points: int = 1001,
    p_h0: float = 1e-6,
    p_h1: float = 1e-3,
    **kwargs,
):
    r"""
    Plots the probability density functions (PDFs) of the detector under $\mathcal{H}_0$ and $\mathcal{H}_1$.

    Arguments:
        h0: The statistical distribution under $\mathcal{H}_0$.
        h1: The statistical distribution under $\mathcal{H}_1$.
        threshold: The detection threshold $\gamma$.
        shade: Indicates whether to shade the tails of the PDFs.
        annotate: Indicates whether to annotate the plot with the probabilities of false alarm and detection.
        x: The x-axis values to use for the plot. If not provided, it will be generated automatically.
        points: The number of points to use for the x-axis.
        p_h0: The probability of the $\mathcal{H}_0$ tails to plot. The smaller the value, the longer the x-axis.
        p_h1: The probability of the $\mathcal{H}_1$ tails to plot. The smaller the value, the longer the x-axis.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.h0_theory, sdr.h1_theory, sdr.threshold

    Example:
        .. ipython:: python

            snr = 5  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0_theory(sigma2, detector); \
            h1 = sdr.h1_theory(snr, sigma2, detector); \
            threshold = sdr.threshold(p_fa, sigma2, detector)

            @savefig sdr_plot_detector_pdfs_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.title("Linear Detector: Probability density functions");

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        if x is None:
            x_min = []
            if h0 is not None:
                x_min.append(h0.ppf(p_h0))
            if h1 is not None:
                x_min.append(h1.ppf(p_h1))
            if threshold is not None:
                x_min.append(threshold)
            x_min = np.nanmin(x_min)

            x_max = []
            if h0 is not None:
                x_max.append(h0.isf(p_h0))
            if h1 is not None:
                x_max.append(h1.isf(p_h1))
            if threshold is not None:
                x_max.append(threshold)
            x_max = np.nanmax(x_max)

            x = np.linspace(x_min, x_max, points)

        if h0 is not None:
            plt.plot(x, h0.pdf(x), label=r"$\mathcal{H}_0$: Noise", **kwargs)
            if shade and threshold:
                h0_color = plt.gca().lines[-1].get_color()
                plt.fill_between(x, 0, h0.pdf(x), where=(x >= threshold), interpolate=True, color=h0_color, alpha=0.1)
        if h1 is not None:
            plt.plot(x, h1.pdf(x), label=r"$\mathcal{H}_1$: Signal + Noise", **kwargs)
            if shade and threshold:
                h1_color = plt.gca().lines[-1].get_color()
                plt.fill_between(x, 0, h1.pdf(x), where=(x >= threshold), interpolate=True, color=h1_color, alpha=0.1)
        if threshold is not None:
            plt.axvline(threshold, color="k", linestyle="--", label="Threshold")

        if annotate:
            if h0 is not None and threshold is not None:
                p_fa = h0.sf(threshold)
                threshold_half = h0.isf(p_fa / 2)
                plt.text(
                    threshold_half,
                    h0.pdf(threshold_half) / 2,
                    rf"$P_{{FA}} = $ {p_fa:1.2e}",
                    color=h0_color,
                    ha="center",
                    va="center",
                )

            if h1 is not None and threshold is not None:
                p_d = h1.sf(threshold)
                threshold_half = h1.isf(p_d / 2)
                plt.text(
                    threshold_half,
                    h1.pdf(threshold_half) / 2,
                    rf"$P_{{D}} = $ {p_d:1.2e}",
                    color=h1_color,
                    ha="center",
                    va="center",
                )

        plt.legend()
        plt.xlabel("Test statistic, $T(x)$")
        plt.ylabel("Probability density")
        plt.title("Detector probability density functions (PDFs)")
