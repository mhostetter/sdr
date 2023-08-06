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
        kwargs: If `heatmap=False`, additional keyword arguments to pass to :func:`matplotlib.pyplot.scatter()`.
            The following keyword arguments are set by default. The defaults may be overwritten.

            - `"range"`: +/- 10% of the maximum value
            - `"bins"`: 75, which is the number of bins per axis

            If `heatmap=True`, additional keyword arguments to pass to :func:`matplotlib.pyplot.hist2d()`.
            The following keyword arguments are set by default. The defaults may be overwritten.

            - `"marker"`: `"."`
            - `"linestyle"`: `"none"`

    Example:
        Display the symbol constellation for Gray-coded QPSK at 6 dB $E_s/N_0$.

        .. ipython:: python

            qpsk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, qpsk.order, 10_000); \
            x = qpsk.modulate(s); \
            x_hat = sdr.awgn(x, 6);

            @savefig sdr_plot_constellation_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(x_hat);

        Display the symbol constellation using a heatmap.

        .. ipython:: python

            @savefig sdr_plot_constellation_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(x_hat, heatmap=True);

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
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot()`.
            The following keyword arguments are set by default. The defaults may be overwritten.

            - `"marker"`: `"x"`
            - `"markersize"`: 6
            - `"linestyle"`: `"none"`

    Example:
        Display the symbol mapping for Gray-coded QPSK.

        .. ipython:: python

            qpsk = sdr.PSK(4, phase_offset=45)

            @savefig sdr_plot_symbol_map_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.symbol_map(qpsk.symbol_map);

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
        ber: The bit error rate $P_{be}$.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.semilogy()`.

    Examples:
        Plot theoretical BER curves for BPSK, QPSK, 8-PSK, and 16-PSK in an AWGN channel.

        .. ipython:: python

            bpsk = sdr.PSK(2); \
            qpsk = sdr.PSK(4); \
            psk8 = sdr.PSK(8); \
            psk16 = sdr.PSK(16); \
            ebn0 = np.linspace(-2, 10, 100)

            @savefig sdr_plot_ber_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.ber(ebn0, bpsk.ber(ebn0), label="BPSK"); \
            sdr.plot.ber(ebn0, qpsk.ber(ebn0), label="QPSK"); \
            sdr.plot.ber(ebn0, psk8.ber(ebn0), label="8-PSK"); \
            sdr.plot.ber(ebn0, psk16.ber(ebn0), label="16-PSK"); \
            plt.title("BER curves for PSK modulation in an AWGN channel"); \
            plt.tight_layout();

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
        plt.ylabel("Probability of bit error, $P_{be}$")
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
        ser: The symbol error rate $P_{se}$.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.semilogy()`.

    Examples:
        Plot theoretical SER curves for BPSK, QPSK, 8-PSK, and 16-PSK in an AWGN channel.

        .. ipython:: python

            bpsk = sdr.PSK(2); \
            qpsk = sdr.PSK(4); \
            psk8 = sdr.PSK(8); \
            psk16 = sdr.PSK(16); \
            esn0 = np.linspace(-2, 10, 100)

            @savefig sdr_psk_ser_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.ser(esn0, bpsk.ser(esn0), label="BPSK"); \
            sdr.plot.ser(esn0, qpsk.ser(esn0), label="QPSK"); \
            sdr.plot.ser(esn0, psk8.ser(esn0), label="8-PSK"); \
            sdr.plot.ser(esn0, psk16.ser(esn0), label="16-PSK"); \
            plt.title("SER curves for PSK modulation in an AWGN channel"); \
            plt.tight_layout();

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
        plt.ylabel("Probability of symbol error, $P_{se}$")
        plt.title("Symbol error rate curve")
        plt.tight_layout()
