"""
A module containing functions related to coherent gain loss.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.optimize

from .._conversion import db, linear
from .._helper import export


@export
def coherent_gain(
    n_c: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the SNR improvement by coherently integrating $N_c$ samples.

    Arguments:
        n_c: The number of samples $N_c$ to coherently integrate.

    Returns:
        The coherent gain $G_c$ in dB.

    Notes:
        $$y[m] = \sum_{n=0}^{N_c-1} x[m-n]$$
        $$\text{SNR}_{y,\text{dB}} = \text{SNR}_{x,\text{dB}} + G_c$$
        $$G_c = 10 \log_{10} N_c$$

    Examples:
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
    n_c = np.asarray(n_c)
    if np.any(n_c < 1):
        raise ValueError(f"Argument 'n_c' must be at least 1, not {n_c}.")

    return db(n_c)


@export
def coherent_gain_loss(
    integration_time: npt.ArrayLike,
    freq_offset: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the coherent gain loss (CGL) as a function of the given integration time and frequency offset.

    Arguments:
        integration_time: The coherent integration time $T_c$ in seconds.
        freq_offset: The frequency offset $\Delta f$ in Hz.

    Returns:
        The coherent gain loss (CGL) in dB.

    Notes:
        $$\text{CGL} = -10 \log_{10} \left( \text{sinc}^2 \left( T_c \Delta f \right) \right)$$
        $$\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$$

    Examples:
        Compute the coherent gain loss for an integration time of 1 ms and a frequency offset of 235 Hz.

        .. ipython:: python

            sdr.coherent_gain_loss(1e-3, 235)

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
    integration_time = np.asarray(integration_time)
    freq_offset = np.asarray(freq_offset)

    if np.any(integration_time < 0):
        raise ValueError(f"Argument 'integration_time' must be non-negative, not {integration_time}.")

    cgl = np.sinc(integration_time * freq_offset) ** 2
    cgl_db = -1 * db(cgl)

    return cgl_db


@export
def non_coherent_gain(
    n_nc: npt.ArrayLike,
    snr: npt.ArrayLike,
    p_fa: npt.ArrayLike = 1e-6,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the SNR improvement by non-coherently integrating $N_{nc}$ samples.

    Arguments:
        n_nc: The number of samples $N_{nc}$ to non-coherently integrate.
        snr: The SNR of the non-coherently integrated samples.
        p_fa: The desired probability of false alarm $P_{FA}$. This is used to compute the necessary thresholds before
            and after integration. The non-coherent gain is slightly affected by the $P_{FA}$.

    Returns:
        The non-coherent gain $G_{nc}$ in dB.

    Notes:
        $$y[m] = \sum_{n=0}^{N_{nc}-1} \left| x[m-n] \right|^2$$
        $$\text{SNR}_{y,\text{dB}} = \text{SNR}_{x,\text{dB}} + G_{nc}$$

    Examples:
        Compute the non-coherent gain for various integration lengths at 10-dB SNR.

        .. ipython:: python

            sdr.non_coherent_gain(1, 10)
            sdr.non_coherent_gain(2, 10)
            sdr.non_coherent_gain(10, 10)
            sdr.non_coherent_gain(20, 10)

        Plot the non-coherent gain parameterized by SNR. Notice that the gain is affected by the input SNR.
        For very large input SNRs, the non-coherent gain approaches the coherent gain
        $G_{NC} \approx 10 \log_{10} N_{NC}$. For very small SNRs, the non-coherent gain is approximated by
        $G_{NC} \approx 3.7 \log_{10} N_{NC}$.

        .. ipython:: python

            plt.figure(); \
            n = np.logspace(0, 3, 51); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for snr in np.arange(-20, 30, 10):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr), label=f"{snr} dB")
            @savefig sdr_non_coherent_gain_1.png
            plt.legend(title="SNR", loc="upper left"); \
            plt.xlabel("Number of samples, $N_{NC}$"); \
            plt.ylabel("Non-coherent gain, $G_{NC}$"); \
            plt.title("Non-coherent gain for various SNRs");

        Plot the non-coherent gain parameterized by the probability of false alarm for 5-dB SNR. Notice that the
        gain is only slightly affected by the $P_{FA}$.

        .. ipython:: python

            plt.figure(); \
            snr = 5; \
            n = np.logspace(0, 3, 51); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for exp in np.arange(-14, 0, 4):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr, p_fa=10.0**exp), label=f"$10^{{{exp}}}$")
            @savefig sdr_non_coherent_gain_2.png
            plt.legend(title="$P_{FA}$"); \
            plt.xlabel("Number of samples, $N_{NC}$"); \
            plt.ylabel("Non-coherent gain, $G_{NC}$"); \
            plt.title(f"Non-coherent gain at {snr}-dB SNR for various $P_{{FA}}$");

        However, when the input SNR is very low, for example -20 dB, the non-coherent gain is more affected by
        false alarm rate.

        .. ipython:: python

            plt.figure(); \
            snr = -20; \
            n = np.logspace(0, 3, 51); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for exp in np.arange(-14, 0, 4):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr, p_fa=10.0**exp), label=f"$10^{{{exp}}}$")
            @savefig sdr_non_coherent_gain_3.png
            plt.legend(title="$P_{FA}$"); \
            plt.xlabel("Number of samples, $N_{NC}$"); \
            plt.ylabel("Non-coherent gain, $G_{NC}$"); \
            plt.title(f"Non-coherent gain at {snr}-dB SNR for various $P_{{FA}}$");

    Group:
        detection-non-coherent-integration
    """
    n_nc = np.asarray(n_nc)
    snr = np.asarray(snr)
    p_fa = np.asarray(p_fa)

    if np.any(n_nc < 1):
        raise ValueError(f"Argument 'n_nc' must be at least 1, not {n_nc}.")
    if np.any(p_fa < 0) or np.any(p_fa > 1):
        raise ValueError(f"Argument 'p_fa' must be between 0 and 1, not {p_fa}.")

    g_nc = _non_coherent_gain(n_nc, snr, p_fa)
    if g_nc.ndim == 0:
        g_nc = float(g_nc)

    return g_nc


@np.vectorize
def _non_coherent_gain(n_nc: float, snr: float, p_fa: float) -> float:
    sigma2 = 1  # Noise variance (power), sigma^2
    A2 = linear(snr) * sigma2  # Signal power, A^2

    # Determine the threshold that yields the desired probability of false alarm. Then compute the probability
    # of detection for the specified SNR.
    df = 2 * 1  # Degrees of freedom
    threshold_in = scipy.stats.chi2.isf(p_fa, df, scale=sigma2 / 2)
    nc = 1 * A2 / (sigma2 / 2)  # Non-centrality parameter
    p_d_in = scipy.stats.ncx2.sf(threshold_in, df, nc, scale=sigma2 / 2)

    if p_d_in == 1:
        # The SNR is already large enough that the probability of detection is 1 before non-coherent integration.
        # After non-coherent integration, the probability of detection will still be 1.
        # We can't use numerical techniques to solve for the exact gain, since `scipy.stats.ncx2.sf()` is not providing
        # enough precision. So, we have to upper bound the non-coherent gain by the coherent gain.
        return db(n_nc)

    # Determine the threshold, after non-coherent integration, that yields the same probability of false alarm.
    df = 2 * n_nc  # Degrees of freedom
    threshold_out = scipy.stats.chi2.isf(p_fa, df, scale=sigma2 / 2)

    def root_eq(A2_db):
        nc = n_nc * linear(A2_db) / (sigma2 / 2)  # Non-centrality parameter
        p_d_out = scipy.stats.ncx2.sf(threshold_out, df, nc, scale=sigma2 / 2)
        return db(p_d_out) - db(p_d_in)  # Use logarithms for numerical stability

    # Determine the input signal power that, after non-coherent integration, yields the same probability of detection.
    # We use logarithms for power for numerical stability.
    A2_db = db(A2)
    A2_db_nc = scipy.optimize.brentq(root_eq, A2_db - db(n_nc), A2_db)
    g_nc = A2_db - A2_db_nc

    return g_nc


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
