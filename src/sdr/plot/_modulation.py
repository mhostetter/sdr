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
                    label = f"{i} =" + np.binary_repr(i, k)
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
def error_rate(
    snr: npt.ArrayLike,
    error_rate: npt.ArrayLike,
    x_unit: Literal["ebn0", "esn0", "snr"] = "ebn0",
    y_unit: Literal["ber", "ser"] = "ber",
    **kwargs,
):
    r"""
    Plots the bit error rate (BER) or symbol error rate (SER) as a function of the signal-to-noise ratio (SNR).

    Arguments:
        snr: A form of signal-to-noise ratio in dB. Either $E_b/N_0$ if `x_unit="ebn0"`,
            $E_s/N_0$ if `x_unit="esn0"`, or $S/N$ if `x_unit="snr"`.
        error_rate: The bit error rate if `y_unit="ber"` or symbol error rate if `y_unit="ser"`.
        x_unit: The unit of the x-axis.
        y_unit: The unit of the y-axis.
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
        default_kwargs = {
            # "marker": "x",
            # "markersize": 6,
            # "linestyle": "none",
        }
        kwargs = {**default_kwargs, **kwargs}

        plt.semilogy(snr, error_rate, **kwargs)
        plt.grid(True, which="both")
        if "label" in kwargs:
            plt.legend()

        if x_unit == "ebn0":
            plt.xlabel("Bit energy to noise PSD ratio (dB), $E_b/N_0$")
        elif x_unit == "esn0":
            plt.xlabel("Symbol energy to noise PSD ratio (dB), $E_s/N_0$")
        elif x_unit == "snr":
            plt.xlabel("Bit energy to noise PSD ratio (dB), $S/N$")
        else:
            raise ValueError(f"Argument 'x_unit' can be either 'ebn0', 'esn0', or 'snr', not {x_unit}.")

        if y_unit == "ber":
            plt.ylabel("Probability of bit error, $P_b$")
        elif y_unit == "ser":
            plt.ylabel("Probability of symbol error, $P_e$")
        else:
            raise ValueError(f"Argument 'y_unit' can be either 'ber' or 'ser', not {y_unit}.")

        if y_unit == "ber":
            plt.title("Bit error rate curve")
        else:
            plt.title("Symbol error rate curve")

        plt.tight_layout()
