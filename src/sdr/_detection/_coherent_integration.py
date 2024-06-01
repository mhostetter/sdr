"""
A module containing functions related to coherent integration.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.optimize

from .._conversion import db
from .._helper import export


@export
def coherent_gain(time_bandwidth: npt.ArrayLike) -> npt.NDArray[np.float32]:
    r"""
    Computes the SNR improvement by coherent integration.

    Arguments:
        time_bandwidth: The time-bandwidth product $T_C B_C$ in seconds-Hz (unitless). If the signal bandwidth equals
            the sample rate, the argument equals the number of samples $N_c$ to coherently integrate.

    Returns:
        The coherent gain $G_c$ in dB.

    Notes:
        The signal $x[n]$ is coherently integrated over $N_c$ samples to produce the output $y[n]$.

        $$y[n] = \sum_{m=0}^{N_c-1} x[n-m]$$

        The coherent integration gain is the reduction in SNR of $x[n]$ compared to $y[n]$, such that both signals
        have the same detection performance.

        $$\text{SNR}_{y,\text{dB}} = \text{SNR}_{x,\text{dB}} + G_c$$

        The coherent integration gain is the time-bandwidth product

        $$G_c = 10 \log_{10} (T_C B_C) .$$

        If the signal bandwidth equals the sample rate, the coherent gain is simply

        $$G_c = 10 \log_{10} N_c .$$

    Examples:
        See the :ref:`coherent-integration` example.

        Compute the coherent gain for various integration lengths.

        .. ipython:: python

            sdr.coherent_gain(1)
            sdr.coherent_gain(2)
            sdr.coherent_gain(10)
            sdr.coherent_gain(20)

        Plot coherent gain as a function of the number of coherently integrated samples.

        .. ipython:: python

            n_c = np.logspace(0, 3, 1001)

            @savefig sdr_coherent_gain_1.png
            plt.figure(); \
            plt.semilogx(n_c, sdr.coherent_gain(n_c)); \
            plt.xlabel("Number of samples, $N_c$"); \
            plt.ylabel("Coherent gain (dB), $G_c$"); \
            plt.title("Coherent gain as a function of the number of integrated samples");

    Group:
        detection-coherent-integration
    """
    time_bandwidth = np.asarray(time_bandwidth)
    if np.any(time_bandwidth <= 0):
        raise ValueError(f"Argument 'time_bandwidth' must be greater than 0, not {time_bandwidth}.")

    return db(time_bandwidth)


@export
def coherent_gain_loss(
    time: npt.ArrayLike,
    freq: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the coherent gain loss (CGL) given a time or frequency offset.

    Arguments:
        time: The coherent integration time $T_C$ or time offset in $\Delta t$ in seconds.
        freq: The frequency offset $\Delta f$ or signal bandwidth $B_C$ in Hz.

    Returns:
        The coherent gain loss (CGL) in dB.

    Notes:
        Coherent gain loss is the reduction in SNR due to time or frequency offsets during coherent integration.
        These losses are similar to scalloping loss.

        The coherent gain loss of a signal integrated for $T_C$ seconds with a frequency offset of $\Delta f$ Hz is

        $$\text{CGL} = -10 \log_{10} \left( \text{sinc}^2 \left( T_c \Delta f \right) \right) ,$$

        where the sinc function is defined as

        $$\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x} .$$

        The coherent gain loss of a signal with bandwidth $B_C$ Hz with a detection time offset of $\Delta t$ seconds
        is

        $$\text{CGL} = -10 \log_{10} \left( \text{sinc}^2 \left( \Delta t B_C \right) \right) .$$

    Examples:
        Compute the coherent gain loss for an integration time of 1 ms and a frequency offset of 235 Hz.

        .. ipython:: python

            sdr.coherent_gain_loss(1e-3, 235)

        Compute the coherent gain loss for a signal with 1 MHz of bandwidth and a detection time offset of 0.25 μs.

        .. ipython:: python

            sdr.coherent_gain_loss(0.25e-6, 1e6)

        Compute the coherent gain loss of a signal detected between two DFT bins. This is commonly referred as the DFT
        scalloping loss. Suppose the DFT is 1 ms long, then the bin spacing is 1 kHz. The worst case scalloping loss
        occurs at 1/2 bin spacing, or 500 Hz in this example. Scalloping loss of 3.9 dB for an unwindowed DFT is a
        well-known figure.

        .. ipython:: python

            t_c = 1e-3  # s
            sdr.coherent_gain_loss(t_c, 1 / t_c / 2)

        If the DFT is zero-padded to twice the length, the scalloping loss is reduced to 0.9 dB.

        .. ipython:: python

            t_c = 1e-3  # s
            sdr.coherent_gain_loss(t_c, 1 / (2 * t_c) / 2)

        Compute the coherent gain loss for an integration time of 1 ms and an array of frequency offsets.

        .. ipython:: python

            sdr.coherent_gain_loss(1e-3, [0, 100, 200, 300, 400, 500])

        Plot coherent gain loss as a function of frequency offset.

        .. ipython:: python

            f = np.linspace(0, 2e3, 1001)

            @savefig sdr_coherent_gain_loss_1.png
            plt.figure(); \
            plt.plot(f, sdr.coherent_gain_loss(0.5e-3, f), label="0.5 ms"); \
            plt.plot(f, sdr.coherent_gain_loss(1e-3, f), label="1 ms"); \
            plt.plot(f, sdr.coherent_gain_loss(2e-3, f), label="2 ms"); \
            plt.legend(title="Integration time"); \
            plt.ylim(-5, 55); \
            plt.xlabel("Frequency offset (Hz)"); \
            plt.ylabel("Coherent gain loss (dB)"); \
            plt.title("Coherent gain loss for various integration times");

        Plot coherent gain loss as a function of integration time.

        .. ipython:: python

            t = np.linspace(0, 1e-2, 1001)

            @savefig sdr_coherent_gain_loss_2.png
            plt.figure(); \
            plt.plot(t * 1e3, sdr.coherent_gain_loss(t, 100), label="100 Hz"); \
            plt.plot(t * 1e3, sdr.coherent_gain_loss(t, 200), label="200 Hz"); \
            plt.plot(t * 1e3, sdr.coherent_gain_loss(t, 400), label="400 Hz"); \
            plt.legend(title="Frequency offset"); \
            plt.ylim(-5, 55); \
            plt.xlabel("Integration time (ms)"); \
            plt.ylabel("Coherent gain loss (dB)"); \
            plt.title("Coherent gain loss for various frequency offsets");

    Group:
        detection-coherent-integration
    """
    time = np.asarray(time)
    freq = np.asarray(freq)

    if np.any(time < 0):
        raise ValueError(f"Argument 'time' must be non-negative, not {time}.")
    if np.any(freq < 0):
        raise ValueError(f"Argument 'freq' must be non-negative, not {freq}.")

    cgl = np.sinc(time * freq) ** 2
    cgl_db = -1 * db(cgl)

    return cgl_db


@export
def max_integration_time(
    cgl: npt.ArrayLike,
    freq_offset: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    r"""
    Computes the maximum integration time that produces at most the provided coherent gain loss (CGL).

    Arguments:
        cgl: The coherent gain loss (CGL) in dB.
        freq_offset: The frequency offset $\Delta f$ in Hz.

    Returns:
        The maximum integration time $T_c$ in seconds.

    Notes:
        The inverse sinc function is calculated using numerical techniques.

    Examples:
        Compute the maximum integration time that produces at most 3 dB of coherent gain loss for a frequency offset
        of 235 Hz.

        .. ipython:: python

            sdr.max_integration_time(3, 235)

        Compute the maximum integration time that produces at most 3 dB of coherent gain loss for an array of
        frequency offsets.

        .. ipython:: python

            sdr.max_integration_time(3, [0, 100, 200, 300, 400, 500])

        Plot the maximum integration time as a function of frequency offset.

        .. ipython:: python

            f = np.linspace(0, 1e3, 1001)

            @savefig sdr_max_integration_time_1.png
            plt.figure(); \
            plt.plot(f, sdr.max_integration_time(0.1, f) * 1e3, label="0.1 dB"); \
            plt.plot(f, sdr.max_integration_time(1, f) * 1e3, label="1 dB"); \
            plt.plot(f, sdr.max_integration_time(3, f) * 1e3, label="3 dB"); \
            plt.legend(); \
            plt.ylim(0, 10); \
            plt.xlabel("Frequency offset (Hz)"); \
            plt.ylabel("Maximum integration time (ms)"); \
            plt.title("Maximum integration time for various coherent gain losses");

        Plot the maximum integration time as a function of coherent gain loss.

        .. ipython:: python

            cgl = np.linspace(0, 10, 1001)

            @savefig sdr_max_integration_time_2.png
            plt.figure(); \
            plt.plot(cgl, sdr.max_integration_time(cgl, 50) * 1e3, label="50 Hz"); \
            plt.plot(cgl, sdr.max_integration_time(cgl, 100) * 1e3, label="100 Hz"); \
            plt.plot(cgl, sdr.max_integration_time(cgl, 200) * 1e3, label="200 Hz"); \
            plt.legend(); \
            plt.ylim(0, 10); \
            plt.xlabel("Coherent gain loss (dB)"); \
            plt.ylabel("Maximum integration time (ms)"); \
            plt.title("Maximum integration time for various frequency offsets");

    Group:
        detection-coherent-integration
    """
    cgl = np.asarray(cgl)
    freq_offset = np.asarray(freq_offset)

    if np.any(cgl < 0):
        raise ValueError(f"Argument 'cgl' must be non-negative, not {cgl}.")

    t = _max_integration_time(cgl, freq_offset)
    if t.ndim == 0:
        t = float(t)

    return t


@np.vectorize
def _max_integration_time(cgl: float, freq_offset: float) -> float:
    if freq_offset == 0:
        return np.inf

    min_t = 0  # s
    max_t = 1 / freq_offset  # s
    t = scipy.optimize.brentq(lambda t: coherent_gain_loss(freq_offset, t) - cgl, min_t, max_t)

    return t


@export
def max_frequency_offset(
    cgl: npt.ArrayLike,
    integration_time: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    r"""
    Computes the maximum frequency offset that produces at most the provided coherent gain loss (CGL).

    Arguments:
        cgl: The coherent gain loss (CGL) in dB.
        integration_time: The coherent integration time $T_c$ in seconds.

    Returns:
        The maximum frequency offset $\Delta f$ in Hz.

    Notes:
        The inverse sinc function is calculated using numerical techniques.

    Examples:
        Compute the maximum frequency offset that produces at most 3 dB of coherent gain loss for an integration time
        of 1 ms.

        .. ipython:: python

            sdr.max_frequency_offset(3, 1e-3)

        Compute the maximum frequency offset that produces at most 3 dB of coherent gain loss for an array of
        integration times.

        .. ipython:: python

            sdr.max_frequency_offset(3, [1e-3, 2e-3, 3e-3])

        Plot the maximum frequency offset as a function of integration time.

        .. ipython:: python

            t = np.linspace(0, 10e-3, 1001)

            @savefig sdr_max_frequency_offset_1.png
            plt.figure(); \
            plt.plot(t * 1e3, sdr.max_frequency_offset(0.1, t), label="0.1 dB"); \
            plt.plot(t * 1e3, sdr.max_frequency_offset(1, t), label="1 dB"); \
            plt.plot(t * 1e3, sdr.max_frequency_offset(3, t), label="3 dB"); \
            plt.legend(); \
            plt.ylim(0, 1e3); \
            plt.xlabel("Integration time (ms)"); \
            plt.ylabel("Maximum frequency offset (Hz)"); \
            plt.title("Maximum frequency offset for various coherent gain losses");

        Plot the maximum frequency offset as a function of coherent gain loss.

        .. ipython:: python

            cgl = np.linspace(0, 10, 1001)

            @savefig sdr_max_frequency_offset_2.png
            plt.figure(); \
            plt.plot(cgl, sdr.max_frequency_offset(cgl, 0.5e-3), label="0.5 ms"); \
            plt.plot(cgl, sdr.max_frequency_offset(cgl, 1e-3), label="1 ms"); \
            plt.plot(cgl, sdr.max_frequency_offset(cgl, 2e-3), label="2 ms"); \
            plt.legend(); \
            plt.ylim(0, 1e3); \
            plt.xlabel("Coherent gain loss (dB)"); \
            plt.ylabel("Maximum frequency offset (Hz)"); \
            plt.title("Maximum frequency offset for various integration times");

    Group:
        detection-coherent-integration
    """
    cgl = np.asarray(cgl)
    integration_time = np.asarray(integration_time)

    if np.any(cgl < 0):
        raise ValueError(f"Argument 'cgl' must be non-negative, not {cgl}.")

    f = _max_frequency_offset(cgl, integration_time)
    if f.ndim == 0:
        f = float(f)

    return f


@np.vectorize
def _max_frequency_offset(cgl: float, integration_time: float) -> float:
    if integration_time == 0:
        return np.inf

    min_f = 0  # Hz
    max_f = 1 / integration_time  # Hz
    f = scipy.optimize.brentq(lambda f: coherent_gain_loss(integration_time, f) - cgl, min_f, max_f)

    return f
