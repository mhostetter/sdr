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
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the probability of detection $P_d$ as a function of received SNR or ENR.

    Arguments:
        x: The SNR or ENR in dB.
        p_d: The probability of detection $P_d$.
        x_label: The x-axis label to use.
        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        ax.plot(x, p_d, **kwargs)

        if "label" in kwargs:
            ax.legend()

        if x_label == "snr":
            ax.set_xlabel(r"Signal-to-noise ratio (dB), $S/\sigma^2$")
        elif x_label == "enr":
            ax.set_xlabel(r"Energy-to-noise ratio (dB), $\mathcal{E}/\sigma^2$")
        else:
            raise ValueError(f"Argument 'x_label' must be one of ['snr', 'enr'], not {x_label!r}.")

        ax.set_ylabel("Probability of detection, $P_d$")
        ax.set_title("Detection performance")


@export
def roc(
    p_fa: npt.ArrayLike,
    p_d: npt.ArrayLike,
    type: Literal["linear", "semilogx", "semilogy", "loglog"] = "semilogx",
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the receiver operating characteristic (ROC) curve as a function of $P_{fa}$.

    Arguments:
        p_fa: The probability of false alarm $P_{fa}$.
        p_d: The probability of detection $P_d$.
        type: The type of plot to generate.
        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to the plotting function defined by `type`.

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        if type == "linear":
            ax.plot(p_fa, p_d, **kwargs)
        elif type == "semilogx":
            ax.semilogx(p_fa, p_d, **kwargs)
        elif type == "semilogy":
            ax.semilogy(p_fa, p_d, **kwargs)
        elif type == "loglog":
            ax.loglog(p_fa, p_d, **kwargs)
        else:
            raise ValueError(
                f"Argument 'type' must be one of ['linear', 'semilogx', 'semilogy', 'loglog'], not {type!r}."
            )

        if "label" in kwargs:
            ax.legend()

        ax.set_xlabel("Probability of false alarm, $P_{fa}$")
        ax.set_ylabel("Probability of detection, $P_d$")
        ax.set_title("Receiver operating characteristic (ROC) curve")


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
    ax: plt.Axes | None = None,
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
        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    See Also:
        sdr.h0, sdr.h1, sdr.threshold

    Example:
        .. ipython:: python

            snr = 5  # Signal-to-noise ratio in dB
            sigma2 = 1  # Noise variance
            p_fa = 1e-1  # Probability of false alarm

        .. ipython:: python

            detector = "linear"; \
            h0 = sdr.h0(sigma2, detector); \
            h1 = sdr.h1(snr, sigma2, detector); \
            threshold = sdr.threshold(p_fa, sigma2, detector)

            @savefig sdr_plot_detector_pdfs_1.png
            plt.figure(); \
            sdr.plot.detector_pdfs(h0, h1, threshold); \
            plt.title("Linear Detector: Probability density functions");

    Group:
        plot-detection
    """
    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

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
            ax.plot(x, h0.pdf(x), label=r"$\mathcal{H}_0$: Noise", **kwargs)
            if shade and threshold:
                h0_color = ax.lines[-1].get_color()
                ax.fill_between(x, 0, h0.pdf(x), where=(x >= threshold), interpolate=True, color=h0_color, alpha=0.1)
        if h1 is not None:
            ax.plot(x, h1.pdf(x), label=r"$\mathcal{H}_1$: Signal + Noise", **kwargs)
            if shade and threshold:
                h1_color = ax.lines[-1].get_color()
                ax.fill_between(x, 0, h1.pdf(x), where=(x >= threshold), interpolate=True, color=h1_color, alpha=0.1)
        if threshold is not None:
            ax.axvline(threshold, color="k", linestyle="--", label="Threshold")

        if annotate:
            if h0 is not None and threshold is not None:
                p_fa = h0.sf(threshold)
                threshold_half = h0.isf(p_fa / 2)
                ax.text(
                    threshold_half,
                    h0.pdf(threshold_half) / 2,
                    rf"$P_{{fa}} = $ {p_fa:1.2e}",
                    color=h0_color,
                    ha="center",
                    va="center",
                )

            if h1 is not None and threshold is not None:
                p_d = h1.sf(threshold)
                threshold_half = h1.isf(p_d / 2)
                ax.text(
                    threshold_half,
                    h1.pdf(threshold_half) / 2,
                    rf"$P_{{d}} = $ {p_d:1.2e}",
                    color=h1_color,
                    ha="center",
                    va="center",
                )

        ax.legend()
        ax.set_xlabel("Test statistic, $T(x)$")
        ax.set_ylabel("Probability density")
        ax.set_title("Detector probability density functions (PDFs)")
