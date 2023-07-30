"""
A module containing various modulation-related plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from ._rc_params import RC_PARAMS


@export
def constellation(
    x_hat: npt.ArrayLike,
    heatmap: bool = False,
    limits: tuple[float, float] | None = None,
    **kwargs,
):
    r"""
    Plots the constellation of the complex symbols $\hat{x}[k]$.

    Arguments:
        x_hat: The complex symbols $\hat{x}[k]$.
        heatmap: If `True`, a heatmap is plotted instead of a scatter plot.
        limits: The axis limits, which apply to both the x- and y-axis. If `None`, the axis limits are
            set to 10% larger than the maximum value.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.scatter()` (`heatmap=False`)
            or :func:`matplotlib.pyplot.hist2d()` (`heatmap=True`).

    Group:
        plot-modulation
    """
    x_hat = np.asarray(x_hat)

    # Set the axis limits to 10% larger than the maximum value
    if limits is None:
        lim = np.max(np.abs(x_hat)) * 1.1
        limits = (-lim, lim)

    with plt.rc_context(RC_PARAMS):
        if heatmap:
            default_kwargs = {
                "range": (limits, limits),
                "bins": 75,  # Number of bins per axis
            }
            kwargs = {**default_kwargs, **kwargs}
            plt.hist2d(x_hat.real, x_hat.imag, **kwargs)
        else:
            default_kwargs = {
                "marker": ".",
                "linestyle": "none",
            }
            kwargs = {**default_kwargs, **kwargs}
            plt.plot(x_hat.real, x_hat.imag, **kwargs)
        plt.axis("square")
        plt.xlim(limits)
        plt.ylim(limits)
        if not heatmap:
            plt.grid(True)
        if "label" in kwargs:
            plt.legend()
        plt.xlabel("In-phase channel, $I$")
        plt.ylabel("Quadrature channel, $Q$")
        plt.title("Constellation")
        plt.tight_layout()


@export
def symbol_map(
    symbol_map: npt.ArrayLike,  # pylint: disable=redefined-outer-name
    annotate: bool | Literal["bin"] = True,
    limits: tuple[float, float] | None = None,
    **kwargs,
):
    r"""
    Plots the symbol map of the complex symbols $\hat{x}[k]$.

    Arguments:
        symbol_map: The complex symbols $\hat{x}[k]$.
        annotate: If `True`, the symbols are annotated with their index.
            If `"bin"`, the symbols are annotated with their binary representation.
        limits: The axis limits, which apply to both the x- and y-axis.
            If `None`, the axis limits are set to 50% larger than the maximum value.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.

    Group:
        plot-modulation
    """
    symbol_map = np.asarray(symbol_map)
    k = int(np.log2(symbol_map.size))

    # Set the axis limits to 50% larger than the maximum value
    if limits is None:
        lim = np.max(np.abs(symbol_map)) * 1.5
        limits = (-lim, lim)

    with plt.rc_context(RC_PARAMS):
        default_kwargs = {
            "marker": "x",
            "markersize": 6,
            "linestyle": "none",
        }
        kwargs = {**default_kwargs, **kwargs}
        plt.plot(symbol_map.real, symbol_map.imag, **kwargs)

        if annotate:
            for i, symbol in enumerate(symbol_map):
                if annotate == "bin":
                    label = f"{i} = " + np.binary_repr(i, k)
                else:
                    label = i

                plt.annotate(
                    label,
                    (symbol.real, symbol.imag),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.axis("square")
        plt.xlim(limits)
        plt.ylim(limits)
        plt.grid(True)
        if "label" in kwargs:
            plt.legend()
        plt.xlabel("In-phase channel, $I$")
        plt.ylabel("Quadrature channel, $Q$")
        plt.title("Symbol Map")
        plt.tight_layout()


@export
def ber(
    ebn0: npt.ArrayLike,
    ber: npt.ArrayLike,  # pylint: disable=redefined-outer-name
    **kwargs,
):
    r"""
    Plots the bit error rate (BER) as a function of $E_b/N_0$.

    Arguments:
        ebn0: The bit energy $E_b$ to noise PSD $N_0$ ratio (dB).
        ber: The bit error rate $P_b$.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.semilogy()`.

    Examples:
        .. ipython:: python

            # bpsk = sdr.PSK(2); \
            # qpsk = sdr.PSK(4); \
            # psk8 = sdr.PSK(8); \
            # ebn0 = np.linspace(0, 10, 100)

            # @savefig sdr_plot_error_rate_1.png
            # plt.figure(figsize=(8, 4)); \
            # sdr.plot.error_rate(ebn0, bpsk.ber(ebn0), label="BPSK"); \
            # sdr.plot.error_rate(ebn0, bpsk.ber(ebn0), label="BPSK"); \
            # sdr.plot.error_rate(ebn0, bpsk.ber(ebn0), label="BPSK"); \
            # plt.title("Bit error rate curves for PSK in AWGN"); \
            # plt.tight_layout()

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        plt.semilogy(ebn0, ber, **kwargs)
        plt.grid(True, which="both")
        if "label" in kwargs:
            plt.legend()

        plt.xlabel("Bit energy to noise PSD ratio (dB), $E_b/N_0$")
        plt.ylabel("Probability of bit error, $P_b$")
        plt.title("Bit error rate curve")
        plt.tight_layout()


@export
def ser(
    esn0: npt.ArrayLike,
    ser: npt.ArrayLike,  # pylint: disable=redefined-outer-name
    **kwargs,
):
    r"""
    Plots the symbol error rate (SER) as a function of $E_s/N_0$.

    Arguments:
        esn0: The symbol energy $E_s$ to noise PSD $N_0$ ratio (dB).
        ser: The symbol error rate $P_e$.
        **kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.semilogy()`.

    Examples:
        .. ipython:: python

            # bpsk = sdr.PSK(2); \
            # qpsk = sdr.PSK(4); \
            # psk8 = sdr.PSK(8); \
            # esn0 = np.linspace(0, 10, 100)

            # @savefig sdr_plot_error_rate_2.png
            # plt.figure(figsize=(8, 4)); \
            # sdr.plot.error_rate(esn0, bpsk.ser(esn0), label="BPSK"); \
            # sdr.plot.error_rate(esn0, qpsk.ser(esn0), label="QPSK"); \
            # sdr.plot.error_rate(esn0, psk8.ser(esn0), label="8-PSK"); \
            # plt.title("Symbol error rate curves for PSK in AWGN"); \
            # plt.tight_layout()

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        plt.semilogy(esn0, ser, **kwargs)
        plt.grid(True, which="both")
        if "label" in kwargs:
            plt.legend()

        plt.xlabel("Symbol energy to noise PSD ratio (dB), $E_s/N_0$")
        plt.ylabel("Probability of symbol error, $P_e$")
        plt.title("Symbol error rate curve")
        plt.tight_layout()
