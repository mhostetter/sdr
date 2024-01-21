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
    x_hat: npt.NDArray[np.complex_],
    limits: tuple[float, float] | None = None,
    persistence: bool = False,
    colorbar: bool = True,
    **kwargs,
):
    r"""
    Plots the constellation of the complex symbols $\hat{x}[k]$.

    Arguments:
        x_hat: The complex symbols $\hat{x}[k]$.
        limits: The axis limits, which apply to both the x- and y-axis. If `None`, the axis limits are
            set to 10% larger than the maximum value.
        persistence: Indicates whether to plot the points as a persistence plot. A persistence plot is a
            2D histogram of the points.
        colorbar: Indicates whether to add a colorbar to the plot. This is only added if `persistence=True`.
        kwargs: Additional keyword arguments to pass to Matplotlib functions.

            If `persistence=False`, the following keyword arguments are passed to :func:`matplotlib.pyplot.scatter()`.
            The defaults may be overwritten.

            - `"marker"`: `"."`
            - `"linestyle"`: `"none"`

            If `persistence=True`, the following keyword arguments are passed to :func:`numpy.histogram2d()` and
            :func:`matplotlib.pyplot.pcolormesh`. The defaults may be overwritten.

            - `"range"`: +/- 10% of the maximum value
            - `"bins"`: `100  # Number of bins per axis`
            - `"cmap"`: `"rainbow"`
            - `"show_zero"`: `False`

    Example:
        Display the symbol constellation for Gray-coded QPSK at 6 dB $E_s/N_0$.

        .. ipython:: python

            qpsk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, qpsk.order, 100_000); \
            x = qpsk.map_symbols(s); \
            x_hat = sdr.awgn(x, 6);

            @savefig sdr_plot_constellation_1.png
            plt.figure(); \
            sdr.plot.constellation(x_hat[0:1_000])

        Display the symbol constellation using a persistence plot.

        .. ipython:: python

            @savefig sdr_plot_constellation_2.png
            plt.figure(); \
            sdr.plot.constellation(x_hat, persistence=True)

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
        # Set the axis limits to 10% larger than the maximum value
        if limits is None:
            lim = np.max(np.abs(x_hat)) * 1.1
            limits = (-lim, lim)

        if persistence:
            default_kwargs = {
                "range": (limits, limits),
                "bins": 100,  # Number of bins per axis
                "cmap": "rainbow",
                "show_zero": False,
            }
            kwargs = {**default_kwargs, **kwargs}

            bins = kwargs.pop("bins")
            range = kwargs.pop("range")
            h, t_edges, x_edges = np.histogram2d(x_hat.real, x_hat.imag, bins=bins, range=range)

            cmap = kwargs.pop("cmap")  # Need to pop cmap to avoid passing it twice to pcolormesh
            cmap = plt.colormaps[cmap]
            show_zero = kwargs.pop("show_zero")
            if show_zero:
                cmap = cmap.with_extremes(bad=cmap(0))
            else:
                h[h == 0] = np.nan  # Set 0s to NaNs so they don't show up in the plot

            pcm = plt.pcolormesh(t_edges, x_edges, h.T, cmap=cmap, **kwargs)
            if colorbar:
                plt.colorbar(pcm, label="Points", pad=0.05)
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
        if "label" in kwargs:
            plt.legend()
        plt.xlabel("In-phase channel, $I$")
        plt.ylabel("Quadrature channel, $Q$")
        plt.title("Constellation")


@export
def symbol_map(
    modulation: LinearModulation | npt.NDArray[np.complex_],
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
            plt.figure(); \
            sdr.plot.symbol_map(qpsk.symbol_map)

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
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
        if "label" in kwargs:
            plt.legend()
        plt.xlabel("In-phase channel, $I$")
        plt.ylabel("Quadrature channel, $Q$")
        plt.title("Symbol Map")


@export
def eye(
    x: npt.NDArray,
    sps: int,
    span: int = 2,
    sample_rate: float | None = None,
    color: Literal["index"] | str = "index",
    persistence: bool = False,
    colorbar: bool = True,
    **kwargs,
):
    r"""
    Plots the eye diagram of the baseband modulated signal $x[n]$.

    Arguments:
        x: The baseband modulated signal $x[n]$. If `x` is complex, in-phase and quadrature eye diagrams are plotted
            in separate subplots.
        sps: The number of samples per symbol.
        span: The number of symbols per raster.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Samples".
        color: Indicates how to color the rasters. If `"index"`, the rasters are colored based on their index.
            If a valid Matplotlib color, the rasters are all colored with that color.
        persistence: Indicates whether to plot the raster as a persistence plot. A persistence plot is a
            2D histogram of the rasters.
        colorbar: Indicates whether to add a colorbar to the plot. This is only added if `color="index"` or
            `persistence=True`.
        kwargs: Additional keyword arguments to pass to :func:`sdr.plot.raster()`.

    Example:
        Modulate 1,000 QPSK symbols using a square root raised cosine (SRRC) pulse shaping filter. The SRRC pulse shape
        is not a Nyquist filter, and intersymbol interference (ISI) can be observed in the eye diagram. Plot the eye
        diagram using index-colored rasters. Note, we are ignoring the transient response of the pulse shaping filter
        at the beginning and end of the signal.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45, pulse_shape="srrc"); \
            sps = psk.sps; \
            s = np.random.randint(0, psk.order, 1_000); \
            tx_samples = psk.modulate(s)

            @savefig sdr_plot_eye_1.png
            plt.figure(figsize=(8, 6)); \
            sdr.plot.eye(tx_samples[4*sps : -4*sps], sps); \
            plt.suptitle("Transmitted QPSK symbols with SRRC pulse shape");

        Plot the eye diagram using a persistence plot. This provides insight into the probability density
        function of the signal.

        .. ipython:: python

            @savefig sdr_plot_eye_2.png
            plt.figure(figsize=(8, 6)); \
            sdr.plot.eye(tx_samples[4*sps : -4*sps], sps, persistence=True); \
            plt.suptitle("Transmitted QPSK symbols with SRRC pulse shape");

        Apply a SRRC matched filter at the receiver. The cascaded transmit and receive SRRC filters are equivalent
        to a single raised cosine (RC) filter, which is a Nyquist filter. ISI is no longer observed, and the eye is
        open.

        .. ipython:: python

            mf = sdr.FIR(psk.pulse_shape); \
            rx_samples = mf(tx_samples, mode="same")

            @savefig sdr_plot_eye_3.png
            plt.figure(figsize=(8, 6)); \
            sdr.plot.eye(rx_samples[4*sps : -4*sps], sps, persistence=True); \
            plt.suptitle("Received and matched filtered QPSK symbols");

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):

        def _eye(xx):
            raster(
                xx,
                span * sps + 1,
                stride=sps,
                sample_rate=sample_rate,
                color=color,
                persistence=persistence,
                colorbar=colorbar,
                **kwargs,
            )

            # Make y-axis symmetric
            ax = plt.gca()
            ymin, ymax = ax.get_ylim()
            ylim = max(np.abs(ymin), np.abs(ymax))
            ax.set_ylim(-ylim, ylim)

        if np.iscomplexobj(x):
            plt.subplot(2, 1, 1)
            _eye(x.real)
            plt.title("In-phase eye diagram")
            plt.subplot(2, 1, 2)
            _eye(x.imag)
            plt.title("Quadrature eye diagram")
        else:
            _eye(x)
            plt.title("Eye diagram")


@export
def phase_tree(
    x: npt.NDArray,
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
            plt.figure(); \
            sdr.plot.phase_tree(x, msk.sps)

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
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
    ebn0: npt.NDArray[np.float_],
    ber: npt.NDArray[np.float_],
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
            plt.figure(); \
            sdr.plot.ber(ebn0, bpsk.ber(ebn0), label="BPSK"); \
            sdr.plot.ber(ebn0, qpsk.ber(ebn0), label="QPSK"); \
            sdr.plot.ber(ebn0, psk8.ber(ebn0), label="8-PSK"); \
            sdr.plot.ber(ebn0, psk16.ber(ebn0), label="16-PSK"); \
            plt.title("BER curves for PSK modulation in an AWGN channel");

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        plt.semilogy(ebn0, ber, **kwargs)
        if "label" in kwargs:
            plt.legend()

        plt.xlabel("Bit energy to noise PSD ratio (dB), $E_b/N_0$")
        plt.ylabel("Probability of bit error, $P_{be}$")
        plt.title("Bit error rate curve")


@export
def ser(
    esn0: npt.NDArray[np.float_],
    ser: npt.NDArray[np.float_],
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
            plt.figure(); \
            sdr.plot.ser(esn0, bpsk.ser(esn0), label="BPSK"); \
            sdr.plot.ser(esn0, qpsk.ser(esn0), label="QPSK"); \
            sdr.plot.ser(esn0, psk8.ser(esn0), label="8-PSK"); \
            sdr.plot.ser(esn0, psk16.ser(esn0), label="16-PSK"); \
            plt.title("SER curves for PSK modulation in an AWGN channel");

    Group:
        plot-modulation
    """
    with plt.rc_context(RC_PARAMS):
        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        plt.semilogy(esn0, ser, **kwargs)
        if "label" in kwargs:
            plt.legend()

        plt.xlabel("Symbol energy to noise PSD ratio (dB), $E_s/N_0$")
        plt.ylabel("Probability of symbol error, $P_{se}$")
        plt.title("Symbol error rate curve")
