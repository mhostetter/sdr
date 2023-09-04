"""
A module containing various modulation-related plotting functions.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from .._modulation import LinearModulation, PiMPSK
from ._rc_params import RC_PARAMS
from ._time_domain import raster


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
            x = qpsk.map_symbols(s); \
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
    modulation: LinearModulation | npt.ArrayLike,
    annotate: bool | Literal["bin"] = True,
    limits: tuple[float, float] | None = None,
    **kwargs,
):
    r"""
    Plots the symbol map of the complex symbols $\hat{x}[k]$.

    Arguments:
        modulation: The linear modulation or symbol map $\{0, \dots, M-1\} \mapsto \mathbb{C}$.
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
    if isinstance(modulation, LinearModulation):
        symbol_map_ = modulation.symbol_map
    else:
        symbol_map_ = np.asarray(modulation)
    k = int(np.log2(symbol_map_.size))

    if isinstance(modulation, PiMPSK):
        label = kwargs.pop("label", None)
        if label:
            even_label = f"{label} (even)"
            odd_label = f"{label} (odd)"
        else:
            even_label = "even"
            odd_label = "odd"
        even_symbol_map = symbol_map_
        odd_symbol_map = symbol_map_ * np.exp(1j * np.pi / modulation.order)
        symbol_map(even_symbol_map, annotate=annotate, limits=limits, label=even_label, **kwargs)
        symbol_map(odd_symbol_map, annotate=annotate, limits=limits, label=odd_label, **kwargs)
        return

    # Set the axis limits to 50% larger than the maximum value
    if limits is None:
        lim = np.max(np.abs(symbol_map_)) * 1.5
        limits = (-lim, lim)

    with plt.rc_context(RC_PARAMS):
        default_kwargs = {
            "marker": "x",
            "markersize": 6,
            "linestyle": "none",
        }
        kwargs = {**default_kwargs, **kwargs}
        plt.plot(symbol_map_.real, symbol_map_.imag, **kwargs)

        if annotate:
            for i, symbol in enumerate(symbol_map_):
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
def eye(
    x: npt.ArrayLike,
    sps: int,
    span: int = 2,
    sample_rate: float | None = None,
    color: Literal["index"] | str = "index",
    **kwargs,
):
    r"""
    Plots the eye diagram of the baseband modulated signal $x[n]$.

    Arguments:
        x: The baseband modulated signal $x[n]$. If `x` is complex, the real and imaginary rasters are interleaved.
            Time order is preserved.
        sps: The number of samples per symbol.
        span: The number of symbols per raster.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Samples".
        color: Indicates how to color the rasters. If `"index"`, the rasters are colored based on their index.
            If a valid Matplotlib color, the rasters are all colored with that color.
        kwargs: Additional keyword arguments to pass to :func:`sdr.plot.raster()`.

    Example:
        Modulate 100 QPSK symbols.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 100); \
            a = psk.map_symbols(s)

        Apply a raised cosine pulse shape and examine the eye diagram. Since the raised cosine pulse shape
        is a Nyquist filter, there is no intersymbol interference (ISI) at the symbol decisions.

        .. ipython:: python

            sps = 25; \
            h = sdr.raised_cosine(0.5, 6, sps); \
            fir = sdr.Interpolator(sps, h); \
            x = fir(a)

            @savefig sdr_plot_eye_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.eye(x, sps)

        Apply a root raised cosine pulse shape and examine the eye diagram. The root raised cosine filter
        is not a Nyquist filter, and ISI can be observed. (It should be noted that two cascaded root raised
        cosine filters, one for transmit and one for receive, is a Nyquist filter. This is why SRRC pulse shaping
        is often used in practice.)

        .. ipython:: python

            sps = 25; \
            h = sdr.root_raised_cosine(0.5, 6, sps); \
            fir = sdr.Interpolator(sps, h); \
            x = fir(a)

            @savefig sdr_plot_eye_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.eye(x, sps)

    Group:
        plot-modulation
    """
    raster(x, span * sps + 1, stride=sps, sample_rate=sample_rate, color=color, **kwargs)

    # Make y-axis symmetric
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ylim = max(np.abs(ymin), np.abs(ymax))
    ax.set_ylim(-ylim, ylim)


@export
def phase_tree(
    x: npt.ArrayLike,
    sps: int,
    span: int = 4,
    sample_rate: float | None = None,
    color: Literal["index"] | str = "index",
    **kwargs,
):
    r"""
    Plots the phase tree of a continuous-phase modulated (CPM) signal signal $x[n]$.

    Arguments:
        x: The baseband CPM signal $x[n]$.
        sps: The number of samples per symbol.
        span: The number of symbols per raster.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Samples".
        color: Indicates how to color the rasters. If `"index"`, the rasters are colored based on their index.
            If a valid Matplotlib color, the rasters are all colored with that color.
        kwargs: Additional keyword arguments to pass to :func:`sdr.plot.raster()`.

    Example:
        Modulate 100 MSK symbols.

        .. ipython:: python

            msk = sdr.MSK(); \
            s = np.random.randint(0, msk.order, 100); \
            x = msk.modulate(s)

        .. ipython:: python

            @savefig sdr_plot_phase_tree_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.phase_tree(x, msk.sps)

    Group:
        plot-modulation
    """
    phase = np.angle(x)

    # Create a strided array of phase values
    length = sps * span
    stride = sps
    N_rasters = (phase.size - length) // stride + 1
    phase_strided = np.lib.stride_tricks.as_strided(
        phase, shape=(N_rasters, length), strides=(phase.strides[0] * stride, phase.strides[0]), writeable=False
    )

    # Unwrap the phase and convert to degrees
    phase_strided = np.unwrap(phase_strided, axis=1)
    phase_strided = np.rad2deg(phase_strided)

    raster(phase_strided, sample_rate=sample_rate, color=color, **kwargs)

    # Make y-axis symmetric and have ticks every 180 degrees
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ylim = max(np.abs(ymin), np.abs(ymax))
    ylim = np.ceil(ylim / 180) * 180
    ax.set_ylim(-ylim, ylim)
    ax.set_yticks(np.arange(-ylim, ylim + 1, 180))

    ax.set_ylabel("Phase (deg)")


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
