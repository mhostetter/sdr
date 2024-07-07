"""
A module containing various modulation-related plotting functions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export, verify_arraylike, verify_bool, verify_isinstance, verify_scalar
from .._link_budget._capacity import shannon_limit_ebn0 as _shannon_limit_ebn0
from .._modulation import LinearModulation, PiMPSK
from ._helper import verify_sample_rate
from ._rc_params import RC_PARAMS
from ._time_domain import raster


@export
def constellation(
    x_hat: npt.ArrayLike,
    limits: tuple[float, float] | None = None,
    persistence: bool = False,
    colorbar: bool = True,
    ax: plt.Axes | None = None,
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
        ax: The axis to plot on. If `None`, the current axis is used.
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
    x_hat = verify_arraylike(x_hat, complex=True, ndim=1)
    verify_bool(persistence)
    verify_bool(colorbar)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        # Set the axis limits to 10% larger than the maximum value
        if limits is None:
            lim = np.max(np.abs(x_hat)) * 1.1
            lim = max(lim, ax.get_ylim()[1])  # Don't reduce the limits if they are already larger
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

            pcm = ax.pcolormesh(t_edges, x_edges, h.T, cmap=cmap, **kwargs)
            if colorbar:
                plt.colorbar(pcm, label="Points", pad=0.05)
        else:
            default_kwargs = {
                "marker": ".",
                "linestyle": "none",
            }
            kwargs = {**default_kwargs, **kwargs}
            ax.plot(x_hat.real, x_hat.imag, **kwargs)

        ax.axis("square")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        if "label" in kwargs:
            ax.legend()
        ax.set_xlabel("In-phase channel, $I$")
        ax.set_ylabel("Quadrature channel, $Q$")
        ax.set_title("Constellation")


@export
def symbol_map(
    modulation: LinearModulation | npt.ArrayLike,
    annotate: bool | Literal["bin"] = True,
    limits: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
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
        ax: The axis to plot on. If `None`, the current axis is used.
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
    verify_isinstance(modulation, (LinearModulation, np.ndarray))
    verify_isinstance(annotate, (bool, str))
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

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
        ax.plot(symbol_map_.real, symbol_map_.imag, **kwargs)

        if annotate:
            for i, symbol in enumerate(symbol_map_):
                if annotate == "bin":
                    label = f"{i} = " + np.binary_repr(i, k)
                else:
                    label = i

                ax.annotate(
                    label,
                    (symbol.real, symbol.imag),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.axis("square")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        if "label" in kwargs:
            ax.legend()
        ax.set_xlabel("In-phase channel, $I$")
        ax.set_ylabel("Quadrature channel, $Q$")
        ax.set_title("Symbol map")


@export
def eye(
    x: npt.ArrayLike,
    sps: int,
    span: int = 2,
    sample_rate: float | None = None,
    color: Literal["index"] | str = "index",
    persistence: bool = False,
    colorbar: bool = True,
    ax: plt.Axes | None = None,
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
            be labeled as "Symbol".
        color: Indicates how to color the rasters. If `"index"`, the rasters are colored based on their index.
            If a valid Matplotlib color, the rasters are all colored with that color.
        persistence: Indicates whether to plot the raster as a persistence plot. A persistence plot is a
            2D histogram of the rasters.
        colorbar: Indicates whether to add a colorbar to the plot. This is only added if `color="index"` or
            `persistence=True`.
        ax: The axis to plot on. If `None`, the current axis is used.
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
    x = verify_arraylike(x, complex=True, ndim=1)
    verify_scalar(sps, int=True, inclusive_min=2)
    verify_scalar(span, int=True, positive=True)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate, default=sps)
    verify_isinstance(color, str)
    verify_bool(persistence)
    verify_bool(colorbar)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):

        def _eye(ax, xx):
            raster(
                xx,
                length=span * sps + 1,
                stride=sps,
                sample_rate=sample_rate,
                color=color,
                persistence=persistence,
                colorbar=colorbar,
                ax=ax,
                **kwargs,
            )
            if not sample_rate_provided:
                ax.set_xlabel("Symbol, $k$")

            # Make y-axis symmetric
            ymin, ymax = ax.get_ylim()
            ylim = max(np.abs(ymin), np.abs(ymax))
            ax.set_ylim(-ylim, ylim)

        if np.iscomplexobj(x):
            ax = plt.subplot(2, 1, 1)
            _eye(ax, x.real)
            plt.title("In-phase eye diagram")
            ax = plt.subplot(2, 1, 2)
            _eye(ax, x.imag)
            plt.title("Quadrature eye diagram")
        else:
            if ax is None:
                ax = plt.gca()
            _eye(ax, x)
            plt.title("Eye diagram")


@export
def phase_tree(
    x: npt.ArrayLike,
    sps: int,
    span: int = 2,
    sample_rate: float | None = None,
    color: Literal["index"] | str = "index",
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the phase tree of a continuous-phase modulated (CPM) signal signal $x[n]$.

    Arguments:
        x: The baseband CPM signal $x[n]$.
        sps: The number of samples per symbol.
        span: The number of symbols per raster.
        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Symbol".
        color: Indicates how to color the rasters. If `"index"`, the rasters are colored based on their index.
            If a valid Matplotlib color, the rasters are all colored with that color.
        ax: The axis to plot on. If `None`, the current axis is used.
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
    x = verify_arraylike(x, complex=True, ndim=1)
    verify_scalar(sps, int=True, inclusive_min=2)
    verify_scalar(span, int=True, positive=True)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate, default=sps)
    verify_isinstance(color, str)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        phase = np.angle(x)

        # Create a strided array of phase values
        length = sps * span + 1
        stride = sps
        N_rasters = (phase.size - length) // stride + 1
        phase_strided = np.lib.stride_tricks.as_strided(
            phase, shape=(N_rasters, length), strides=(phase.strides[0] * stride, phase.strides[0]), writeable=False
        )

        # Unwrap the phase and convert to degrees
        phase_strided = np.unwrap(phase_strided, axis=1)
        phase_strided -= phase_strided[:, 0][:, np.newaxis]  # Normalize to 0 degrees at the first sample
        phase_strided = np.rad2deg(phase_strided)

        raster(
            phase_strided,
            sample_rate=sample_rate,
            color=color,
            ax=ax,
            **kwargs,
        )
        if not sample_rate_provided:
            ax.set_xlabel("Symbol, $k$")

        # Make y-axis symmetric and have ticks every 180 degrees
        ymin, ymax = ax.get_ylim()
        ylim = max(np.abs(ymin), np.abs(ymax))
        ylim = np.ceil(ylim / 180) * 180
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks(np.arange(-ylim, ylim + 1, 180))

        ax.set_ylabel(r"Phase (deg), $\phi$")


@export
def ber(
    ebn0: npt.ArrayLike,
    ber: npt.ArrayLike,
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the bit error rate (BER) as a function of $E_b/N_0$.

    Arguments:
        ebn0: The bit energy $E_b$ to noise PSD $N_0$ ratio (dB).
        ber: The bit error rate $P_{be}$.
        ax: The axis to plot on. If `None`, the current axis is used.
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
    ebn0 = verify_arraylike(ebn0, float=True, ndim=1)
    ber = verify_arraylike(ber, float=True, ndim=1)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        ax.semilogy(ebn0, ber, **kwargs)
        if "label" in kwargs:
            ax.legend()

        ax.set_xlabel("Bit energy-to-noise PSD ratio (dB), $E_b/N_0$")
        ax.set_ylabel("Probability of bit error, $P_{be}$")
        ax.set_title("Bit error rate curve")


@export
def ser(
    esn0: npt.ArrayLike,
    ser: npt.ArrayLike,
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the symbol error rate (SER) as a function of $E_s/N_0$.

    Arguments:
        esn0: The symbol energy $E_s$ to noise PSD $N_0$ ratio (dB).
        ser: The symbol error rate $P_{se}$.
        ax: The axis to plot on. If `None`, the current axis is used.
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
    esn0 = verify_arraylike(esn0, float=True, ndim=1)
    ser = verify_arraylike(ser, float=True, ndim=1)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        default_kwargs = {}
        kwargs = {**default_kwargs, **kwargs}

        ax.semilogy(esn0, ser, **kwargs)
        if "label" in kwargs:
            ax.legend()

        ax.set_xlabel("Symbol energy-to-noise PSD ratio (dB), $E_s/N_0$")
        ax.set_ylabel("Probability of symbol error, $P_{se}$")
        ax.set_title("Symbol error rate curve")


@export
def shannon_limit_ebn0(
    rho: float,
    ax: plt.Axes | None = None,
):
    r"""
    Plots the Shannon limit for the bit energy-to-noise PSD ratio $E_b/N_0$.

    Arguments:
        rho: The nominal spectral efficiency $\rho$ of the modulation in bits/2D.
        ax: The axis to plot on. If `None`, the current axis is used.

    Examples:
        Plot the absolute Shannon limit on $E_b/N_0$ and the Shannon limit for $\rho = 2$ bits/2D. Compare these to the
        theoretical BER curve for BPSK modulation, which has spectral efficiency $\rho = 2$ bits/2D.

        .. ipython:: python

            bpsk = sdr.PSK(2)
            ebn0 = np.linspace(-2, 10, 201)

            @savefig sdr_plot_shannon_limit_ebn0_1.png
            plt.figure(); \
            sdr.plot.ber(ebn0, bpsk.ber(ebn0), label="BPSK theoretical"); \
            plt.ylim(1e-6, 1e0); \
            sdr.plot.shannon_limit_ebn0(0); \
            sdr.plot.shannon_limit_ebn0(2); \
            plt.title("Bit error rate curve for PSK modulation in AWGN");

    Group:
        plot-modulation
    """
    verify_scalar(rho, float=True, non_negative=True)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        ebn0 = _shannon_limit_ebn0(rho)

        ax.axvline(x=ebn0, ymin=0, ymax=0.75, color="black", linestyle="--")
        ax.annotate(
            rf"Shannon limit for $\rho = {rho}$",
            xy=(ebn0, ax.get_ylim()[0]),
            xytext=(-1.25, 0.75),
            textcoords="offset fontsize",
            rotation=90,
        )
