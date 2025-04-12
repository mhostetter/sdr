"""
A module containing filter-related plotting functions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._conversion import db
from .._filter import FIR, IIR
from .._helper import export, verify_arraylike, verify_bool, verify_isinstance, verify_literal, verify_scalar
from ._helper import integer_x_axis, min_ylim, standard_plot, verify_sample_rate
from ._rc_params import RC_PARAMS
from ._units import freq_units, time_units


def _convert_to_taps(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
) -> tuple[npt.NDArray, npt.NDArray]:
    if isinstance(filter, FIR):
        b = filter.taps
        a = np.array([1])
    elif isinstance(filter, IIR):
        b = filter.b_taps
        a = filter.a_taps
    elif isinstance(filter, tuple):
        b = verify_arraylike(filter[0], complex=True, ndim=1)
        a = verify_arraylike(filter[1], complex=True, ndim=1)
    else:
        b = verify_arraylike(filter, complex=True, ndim=1)
        a = np.array([1])
    return b, a


@export
def impulse_response(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    n: int | None = None,
    offset: float = 0.0,
    ax: plt.Axes | None = None,
    type: Literal["plot", "stem"] = "stem",
    **kwargs,
):
    r"""
    Plots the impulse response $h[n]$ of a filter.

    The impulse response $h[n]$ is the filter output when the input is an impulse $\delta[n]$.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        n: The number of samples $N$ to plot. If `None`, the length of `b` is used for FIR filters and
            100 for IIR filters.
        offset: The x-axis offset to apply to the first sample. Can be useful for comparing the impulse
            response of filters with different lengths.
        ax: The axis to plot on. If `None`, the current axis is used.
        type: The type of plot to use.
        kwargs: Additional keyword arguments to pass to the plotting function.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_impulse_response_1.png
            plt.figure(); \
            sdr.plot.impulse_response(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_impulse_response_2.png
            plt.figure(); \
            sdr.plot.impulse_response(iir, n=30)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    verify_scalar(n, optional=True, int=True, positive=True)
    verify_scalar(offset, float=True)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(type, ["plot", "stem"])

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if n is None:
            if a.size == 1 and a[0] == 1:
                n = b.size  # FIR filter
            else:
                n = 100  # IIR filter

        # Delta impulse function
        d = np.zeros(n, dtype=float)
        d[0] = 1

        # Filter the impulse
        zi = scipy.signal.lfiltic(b, a, y=[], x=[])
        h, zi = scipy.signal.lfilter(b, a, d, zi=zi)
        t = np.arange(h.size) + offset

        integer_x_axis(ax)
        standard_plot(t, h, ax=ax, type=type, y_axis="complex", **kwargs)
        ax.set_xlabel("Sample, $n$")
        ax.set_ylabel("Amplitude")
        ax.set_title("Impulse response, $h[n]$")


@export
def step_response(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    n: int | None = None,
    ax: plt.Axes | None = None,
    type: Literal["plot", "stem"] = "stem",
    **kwargs,
):
    r"""
    Plots the step response $s[n]$ of a filter.

    The step response $s[n]$ is the filter output when the input is a unit step $u[n]$.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        n: The number of samples $N$ to plot. If `None`, the length of `b` is used for FIR filters and
            100 for IIR filters.
        ax: The axis to plot on. If `None`, the current axis is used.
        type: The type of plot to use.
        kwargs: Additional keyword arguments to pass to the plotting function.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_step_response_1.png
            plt.figure(); \
            sdr.plot.step_response(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_step_response_2.png
            plt.figure(); \
            sdr.plot.step_response(iir, n=30)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    verify_scalar(n, optional=True, int=True, positive=True)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(type, ["plot", "stem"])

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if n is None:
            if a.size == 1 and a[0] == 1:
                n = b.size  # FIR filter
            else:
                n = 100  # IIR filter

        # Unit step function
        u = np.ones(n, dtype=float)

        # Filter the impulse
        zi = scipy.signal.lfiltic(b, a, y=[], x=[])
        s, zi = scipy.signal.lfilter(b, a, u, zi=zi)
        t = np.arange(s.size)

        integer_x_axis(ax)
        standard_plot(t, s, ax=ax, type=type, y_axis="complex", **kwargs)
        ax.set_xlabel("Sample, $n$")
        ax.set_ylabel("Amplitude")
        ax.set_title("Step response, $s[n]$")


@export
def zeros_poles(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    ax: plt.Axes | None = None,
    **kwargs,
):
    r"""
    Plots the zeros and poles of the filter.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        ax: The axis to plot on. If `None`, the current axis is used.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_zeros_poles_1.png
            plt.figure(); \
            sdr.plot.zeros_poles(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_zeros_poles_2.png
            plt.figure(); \
            sdr.plot.zeros_poles(iir)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    verify_isinstance(ax, plt.Axes, optional=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        z, p, _ = scipy.signal.tf2zpk(b, a)
        unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 100))

        label = kwargs.pop("label", None)
        if label is None:
            z_label = "Zeros"
            p_label = "Poles"
        else:
            z_label = label + " (zeros)"
            p_label = label + " (poles)"

        ax.plot(unit_circle.real, unit_circle.imag, color="k", linestyle="--", label="Unit circle")
        ax.scatter(z.real, z.imag, marker="o", label=z_label)
        ax.scatter(p.real, p.imag, marker="x", label=p_label)
        ax.axis("equal")
        ax.legend()
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title("Zeros and poles of $H(z)$")


@export
def magnitude_response(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    sample_rate: float | None = None,
    n: int = 1024,
    ax: plt.Axes | None = None,
    x_axis: Literal["auto", "one-sided", "two-sided", "log"] = "auto",
    y_axis: Literal["linear", "log"] = "log",
    decades: int = 4,
    **kwargs,
):
    r"""
    Plots the magnitude response $|H(\omega)|^2$ of the filter.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        n: The number of samples $N$ in the frequency response.
        ax: The axis to plot on. If `None`, the current axis is used.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis. The default is `"auto"` which selects `"one-sided"`
            for real-valued filters and `"two-sided"` for complex-valued filters.
        y_axis: The y-axis scaling. Options are to display a linear or logarithmic magnitude response.
        decades: The number of decades to plot when `x_axis="log"`.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_magnitude_response_1.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_magnitude_response_2.png
            plt.figure(); \
            sdr.plot.magnitude_response(iir)

        .. ipython:: python

            @savefig sdr_plot_magnitude_response_3.png
            plt.figure(); \
            sdr.plot.magnitude_response(h_srrc, x_axis="two-sided")

        .. ipython:: python

            @savefig sdr_plot_magnitude_response_4.png
            plt.figure(); \
            sdr.plot.magnitude_response(iir, x_axis="log", decades=3)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    verify_scalar(n, optional=True, int=True, positive=True)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(x_axis, ["auto", "one-sided", "two-sided", "log"])
    verify_literal(y_axis, ["linear", "log"])
    verify_scalar(decades, int=True, positive=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if x_axis == "auto":
            x_axis = "one-sided" if np.isrealobj(b) and np.isrealobj(a) else "two-sided"

        with np.errstate(divide="ignore", invalid="ignore"):
            if x_axis == "log":
                f = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), n)
                f, H = scipy.signal.freqz(b, a, worN=f, whole=False, fs=sample_rate)
            else:
                f, H = scipy.signal.freqz(b, a, worN=n, whole=x_axis == "two-sided", fs=sample_rate)

        if x_axis == "two-sided":
            f -= sample_rate / 2
            H = np.fft.fftshift(H)

        if sample_rate_provided:
            units, scalar = freq_units(f)
            f *= scalar

        if y_axis == "log":
            H = db(np.abs(H) ** 2)
        else:
            H = np.abs(H) ** 2

        if x_axis == "log":
            ax.semilogx(f, H, **kwargs)
        else:
            ax.plot(f, H, **kwargs)

        ax.grid(True, which="both")
        if "label" in kwargs:
            ax.legend()

        if sample_rate_provided:
            ax.set_xlabel(f"Frequency ({units}), $f$")
        else:
            ax.set_xlabel("Normalized frequency, $f / f_s$")

        if y_axis == "log":
            ax.set_ylabel(r"Power (dB), $|H(\omega)|^2$")
        else:
            ax.set_ylabel(r"Power, $|H(\omega)|^2$")

        ax.set_title(r"Magnitude response, $|H(\omega)|^2$")


@export
def phase_response(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    sample_rate: float | None = None,
    n: int = 1024,
    unwrap: bool = True,
    ax: plt.Axes | None = None,
    x_axis: Literal["auto", "one-sided", "two-sided", "log"] = "auto",
    decades: int = 4,
    **kwargs,
):
    r"""
    Plots the phase response $\angle H(\omega)$ of the filter.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        n: The number of samples $N$ in the phase response.
        unwrap: Indicates whether to unwrap the phase response.
        ax: The axis to plot on. If `None`, the current axis is used.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis. The default is `"auto"` which selects `"one-sided"`
            for real-valued filters and `"two-sided"` for complex-valued filters.
        decades: The number of decades to plot when `x_axis="log"`.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_phase_response_1.png
            plt.figure(); \
            sdr.plot.phase_response(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_phase_response_2.png
            plt.figure(); \
            sdr.plot.phase_response(iir)

        .. ipython:: python

            @savefig sdr_plot_phase_response_3.png
            plt.figure(); \
            sdr.plot.phase_response(h_srrc, x_axis="two-sided")

        .. ipython:: python

            @savefig sdr_plot_phase_response_4.png
            plt.figure(); \
            sdr.plot.phase_response(iir, x_axis="log", decades=3)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    verify_scalar(n, optional=True, int=True, positive=True)
    verify_bool(unwrap)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(x_axis, ["auto", "one-sided", "two-sided", "log"])
    verify_scalar(decades, int=True, positive=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if x_axis == "auto":
            x_axis = "one-sided" if np.isrealobj(b) and np.isrealobj(a) else "two-sided"

        if x_axis == "log":
            f = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), n)
            f, H = scipy.signal.freqz(b, a, worN=f, whole=False, fs=sample_rate)
        else:
            f, H = scipy.signal.freqz(b, a, worN=n, whole=x_axis == "two-sided", fs=sample_rate)

        if x_axis == "two-sided":
            f -= sample_rate / 2
            H = np.fft.fftshift(H)

        if sample_rate_provided:
            units, scalar = freq_units(f)
            f *= scalar

        if unwrap:
            theta = np.rad2deg(np.unwrap(np.angle(H)))
        else:
            theta = np.rad2deg(np.angle(H))

        if x_axis == "two-sided":
            # Set omega=0 to have phase of 0
            theta -= theta[np.argmin(np.abs(f))]

        if x_axis == "log":
            ax.semilogx(f, theta, **kwargs)
        else:
            ax.plot(f, theta, **kwargs)

        ax.grid(True, which="both")
        if "label" in kwargs:
            ax.legend()
        if sample_rate_provided:
            ax.set_xlabel(f"Frequency ({units}), $f$")
        else:
            ax.set_xlabel("Normalized frequency, $f / f_s$")
        ax.set_ylabel(r"Phase (deg), $\angle H(\omega)$")
        ax.set_title(r"Phase response, $\angle H(\omega)$")


@export
def phase_delay(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    sample_rate: float | None = None,
    n: int = 1024,
    ax: plt.Axes | None = None,
    x_axis: Literal["auto", "one-sided", "two-sided", "log"] = "auto",
    decades: int = 4,
    **kwargs,
):
    r"""
    Plots the phase delay $\tau_{\phi}(\omega)$ of the filter.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        n: The number of samples $N$ in the phase delay.
        ax: The axis to plot on. If `None`, the current axis is used.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis. The default is `"auto"` which selects `"one-sided"`
            for real-valued filters and `"two-sided"` for complex-valued filters.
        decades: The number of decades to plot when `x_axis="log"`.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_phase_delay_1.png
            plt.figure(); \
            sdr.plot.phase_delay(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_phase_delay_2.png
            plt.figure(); \
            sdr.plot.phase_delay(iir)

        .. ipython:: python

            @savefig sdr_plot_phase_delay_3.png
            plt.figure(); \
            sdr.plot.phase_delay(h_srrc, x_axis="two-sided")

        .. ipython:: python

            @savefig sdr_plot_phase_delay_4.png
            plt.figure(); \
            sdr.plot.phase_delay(iir, x_axis="log", decades=3)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    verify_scalar(n, optional=True, int=True, positive=True)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(x_axis, ["auto", "one-sided", "two-sided", "log"])
    verify_scalar(decades, int=True, positive=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if x_axis == "auto":
            x_axis = "one-sided" if np.isrealobj(b) and np.isrealobj(a) else "two-sided"

        if x_axis == "log":
            f = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), n)
            f, H = scipy.signal.freqz(b, a, worN=f, whole=False, fs=sample_rate)
        else:
            f, H = scipy.signal.freqz(b, a, worN=n, whole=x_axis == "two-sided", fs=sample_rate)

        if x_axis == "two-sided":
            f -= sample_rate / 2
            H = np.fft.fftshift(H)

        theta = np.unwrap(np.angle(H))
        theta -= theta[np.argmin(np.abs(f))]  # Set omega=0 to have phase of 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_phi = -theta / (2 * np.pi * f)
        tau_phi[np.argmin(np.abs(f))] = np.nan  # Avoid crazy result when dividing by near zero

        if sample_rate_provided:
            f_units, scalar = freq_units(f)
            f *= scalar
            t_units, scalar = time_units(tau_phi)
            tau_phi *= scalar

        if x_axis == "log":
            ax.semilogx(f, tau_phi, **kwargs)
        else:
            ax.plot(f, tau_phi, **kwargs)

        min_ylim(tau_phi, 2 / sample_rate, sample_rate)

        ax.grid(True, which="both")
        if "label" in kwargs:
            ax.legend()
        if sample_rate_provided:
            ax.set_xlabel(f"Frequency ({f_units}), $f$")
            ax.set_ylabel(rf"Phase delay ({t_units}), $\tau_{{\phi}}{{\omega}}$")
        else:
            ax.set_xlabel("Normalized frequency, $f / f_s$")
            ax.set_ylabel(r"Phase delay (samples), $\tau_{\phi}(\omega)$")
        ax.set_title(r"Phase delay, $\tau_{\phi}(\omega)$")


@export
def group_delay(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    sample_rate: float | None = None,
    n: int = 1024,
    ax: plt.Axes | None = None,
    x_axis: Literal["auto", "one-sided", "two-sided", "log"] = "auto",
    decades: int = 4,
    **kwargs,
):
    r"""
    Plots the group delay $\tau_g(\omega)$ of the IIR filter.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        n: The number of samples $N$ in the frequency response.
        ax: The axis to plot on. If `None`, the current axis is used.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis. The default is `"auto"` which selects `"one-sided"`
            for real-valued filters and `"two-sided"` for complex-valued filters.
        decades: The number of decades to plot when `x_axis="log"`.
        kwargs: Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_group_delay_1.png
            plt.figure(); \
            sdr.plot.group_delay(h_srrc);

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_group_delay_2.png
            plt.figure(); \
            sdr.plot.group_delay(iir)

        .. ipython:: python

            @savefig sdr_plot_group_delay_3.png
            plt.figure(); \
            sdr.plot.group_delay(h_srrc, x_axis="two-sided");

        .. ipython:: python

            @savefig sdr_plot_group_delay_4.png
            plt.figure(); \
            sdr.plot.group_delay(iir, x_axis="log", decades=3)

    Group:
        plot-filter
    """
    b, a = _convert_to_taps(filter)
    sample_rate, sample_rate_provided = verify_sample_rate(sample_rate)
    verify_scalar(n, optional=True, int=True, positive=True)
    verify_isinstance(ax, plt.Axes, optional=True)
    verify_literal(x_axis, ["auto", "one-sided", "two-sided", "log"])
    verify_scalar(decades, int=True, positive=True)

    with plt.rc_context(RC_PARAMS):
        if ax is None:
            ax = plt.gca()

        if x_axis == "auto":
            x_axis = "one-sided" if np.isrealobj(b) and np.isrealobj(a) else "two-sided"

        if x_axis == "log":
            f = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), n)
            f, tau_g = scipy.signal.group_delay((b, a), w=f, whole=False, fs=sample_rate)
        else:
            f, tau_g = scipy.signal.group_delay((b, a), w=n, whole=x_axis == "two-sided", fs=sample_rate)

        if x_axis == "two-sided":
            f -= sample_rate / 2
            tau_g = np.fft.fftshift(tau_g)

        if sample_rate_provided:
            f_units, scalar = freq_units(f)
            f *= scalar

            tau_g /= sample_rate
            t_units, scalar = time_units(tau_g)
            tau_g *= scalar

        if x_axis == "log":
            ax.semilogx(f, tau_g, **kwargs)
        else:
            ax.plot(f, tau_g, **kwargs)

        min_ylim(tau_g, 2 / sample_rate, sample_rate)

        ax.grid(True, which="both")
        if "label" in kwargs:
            ax.legend()
        if sample_rate_provided:
            ax.set_xlabel(f"Frequency ({f_units}), $f$")
            ax.set_ylabel(rf"Group delay ({t_units}), $\tau_g(\omega)$")
        else:
            ax.set_xlabel("Normalized frequency, $f / f_s$")
            ax.set_ylabel(r"Group delay (samples), $\tau_g(\omega)$")
        ax.set_title(r"Group delay, $\tau_g(\omega)$")


@export
def filter(
    filter: FIR | IIR | npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike],
    sample_rate: float | None = None,
    n_time: int | None = None,
    n_freq: int = 1024,
    x_axis: Literal["one-sided", "two-sided", "log"] = "two-sided",
    decades: int = 4,
):
    r"""
    Plots the magnitude response $|H(\omega)|^2$, impulse response $h[n]$, and zeros and poles of the filter.

    Arguments:
        filter: The filter definition.

            - :class:`sdr.FIR`, :class:`sdr.IIR`: The filter object.
            - `npt.ArrayLike`: The feedforward coefficients $b_i$.
            - `tuple[npt.ArrayLike, npt.ArrayLike]`: The feedforward coefficients $b_i$ and
              feedback coefficients $a_j$.

        sample_rate: The sample rate $f_s$ of the signal in samples/s. If `None`, the x-axis will
            be labeled as "Normalized frequency".
        n_time: The number of samples $N_t$ in the time domain. If `None`, the length of `b` is used
            for FIR filters and 100 for IIR filters.
        n_freq: The number of samples $N_f$ in the frequency response.
        x_axis: The x-axis scaling. Options are to display a one-sided spectrum, a two-sided spectrum, or
            one-sided spectrum with a logarithmic frequency axis.
        decades: The number of decades to plot when `x_axis="log"`.

    Examples:
        See the :ref:`fir-filters` example.

        .. ipython:: python

            h_srrc = sdr.root_raised_cosine(0.5, 10, 10)

            @savefig sdr_plot_filter_1.png
            plt.figure(figsize=(8, 6)); \
            sdr.plot.filter(h_srrc)

        See the :ref:`iir-filters` example.

        .. ipython:: python

            zero = 0.6; \
            pole = 0.8 * np.exp(1j * np.pi / 8); \
            iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])

            @savefig sdr_plot_filter_2.png
            plt.figure(figsize=(8, 6)); \
            sdr.plot.filter(iir, n_time=30)

    Group:
        plot-filter
    """
    with plt.rc_context(RC_PARAMS):
        b, a = _convert_to_taps(filter)

        plt.subplot2grid((2, 3), (0, 0), 1, 3)
        magnitude_response((b, a), sample_rate=sample_rate, n=n_freq, x_axis=x_axis, decades=decades)

        plt.subplot2grid((2, 3), (1, 0), 1, 1)
        zeros_poles((b, a))

        plt.subplot2grid((2, 3), (1, 1), 1, 2)
        impulse_response((b, a), n=n_time)
