"""
A module containing measurement functions related to modulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.integrate
from typing_extensions import Literal

from .._helper import export, verify_arraylike, verify_literal, verify_scalar
from ._power import average_power


@export
def evm(
    x_hat: npt.ArrayLike,
    ref: npt.ArrayLike,
    norm: Literal["average-power-ref", "average-power", "peak-power"] = "average-power-ref",
    output: Literal["rms", "all"] | float = "rms",
) -> float:
    r"""
    Measures the error-vector magnitude (EVM) of the complex symbols $\hat{x}[k]$.

    Arguments:
        x_hat: The complex symbols $\hat{x}[k]$ to be measured.
        ref: The complex reference symbols $x[k]$. This can be the noiseless transmitted symbols or the
            modulation's symbol map.
        norm: The normalization source used in the EVM calculation.

            - `"average-power-ref"`: The average power of the reference symbols $x[k]$.

            $$P_{\text{ref}} = \frac{1}{N} \sum_{k=0}^{N-1} \left| x[k] \right|^2$$

            - `"average-power"`: The average power of the received symbols $\hat{x}[k]$.

            $$P_{\text{ref}} = \frac{1}{N} \sum_{k=0}^{N-1} \left| \hat{x}[k] \right|^2$$

            - `"peak-power"`: The peak power of the received symbols $\hat{x}[k]$.

            $$P_{\text{ref}} = \text{max} \left| \hat{x}[k] \right|^2$$

        output: The output type of the EVM calculation.

            - `"rms"`: The root-mean-square (RMS) EVM.

            $$
            \text{EVM}_{\text{RMS}} =
            100 \sqrt{\frac{\frac{1}{N} \sum_{k=0}^{N-1} \left| \hat{x}[k] - x[k] \right|^2}{P_{\text{ref}}}}
            $$

            - `"all"`: The instantaneous EVM for each symbol.

            $$
            \text{EVM}_{k} =
            100 \sqrt{\frac{\left| \hat{x}[k] - x[k] \right|^2}{P_{\text{ref}}}}
            $$

            - `float`: The RMS EVM for the given percentile (0 - 100).

    Examples:
        Create QPSK symbols with $E_s/N_0$ of 20 dB.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1000); \
            x = psk.map_symbols(s); \
            x_hat = sdr.awgn(x, 20)

            @savefig sdr_evm_1.png
            plt.figure(); \
            sdr.plot.constellation(x_hat, label=r"$\hat{x}[k]$"); \
            sdr.plot.symbol_map(psk.symbol_map, label=r"Reference"); \
            plt.title("QPSK Constellation at 20 dB $E_s/N_0$");

        Measure the RMS EVM, normalizing with the average power of the reference symbols.
        Either the symbol map or noiseless transmitted symbols may be passed.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map)
            sdr.evm(x_hat, x)

        Measure the RMS EVM, normalizing with the average power of the received symbols.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map, norm="average-power")

        Measure the RMS EVM, normalizing with the peak power of the received symbols.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map, norm="peak-power")

        Measure the 95th percentile EVM.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map, output=95)

        Measure the instantaneous EVM for each symbol.

        .. ipython:: python

            inst_evm = sdr.evm(x_hat, psk.symbol_map, output="all")

            @savefig sdr_evm_2.png
            plt.figure(); \
            plt.hist(inst_evm, bins=20); \
            plt.xlabel("RMS EVM (%)"); \
            plt.ylabel("Count"); \
            plt.title("EVM Histogram");

    Group:
        measurement-modulation
    """
    x_hat = verify_arraylike(x_hat, complex=True)
    ref = verify_arraylike(ref, complex=True)
    norm = verify_literal(norm, ["average-power-ref", "average-power", "peak-power"])

    if norm == "average-power-ref":
        ref_power = average_power(ref)
    elif norm == "average-power":
        ref_power = average_power(x_hat)
    elif norm == "peak-power":
        ref_power = np.max(np.abs(x_hat) ** 2)

    if ref.shape == x_hat.shape:
        # Compute the error vectors to each reference symbol
        error_vectors = x_hat - ref
    else:
        # The reference symbols are the symbol map. We must first determine the most likely reference symbol for each
        # received symbol. Then we must compute the error vectors to those symbols.
        symbol_map = ref
        all_error_vectors = np.subtract.outer(x_hat, symbol_map)
        s_hat = np.argmin(np.abs(all_error_vectors), axis=-1)
        error_vectors = x_hat - symbol_map[s_hat]

    if output == "rms":
        rms_evm = 100 * np.sqrt(average_power(error_vectors) / ref_power)
        return rms_evm

    inst_evm = 100 * np.sqrt(np.abs(error_vectors) ** 2 / ref_power)
    if output == "all":
        return inst_evm

    verify_scalar(output, float=True, inclusive_min=0, inclusive_max=100)
    perc_evm = np.percentile(inst_evm, output)
    return perc_evm


@export
def rms_bandwidth(
    x: npt.ArrayLike,
    sample_rate: float = 1.0,
) -> float:
    r"""
    Measures the RMS bandwidth $B_{\text{rms}}$ of the signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

            .. note::
                For best measurement performance, the time-domain signal should be very long. This allows for
                more averaging of its power spectra.

        sample_rate: The sample rate $f_s$ in samples/s.

    Returns:
        The RMS signal bandwidth $B_{\text{rms}}$ in Hz.

    See Also:
        sdr.toa_crlb, sdr.tdoa_crlb

    Notes:
        The root-mean-square (RMS) bandwidth $B_{\text{rms}}$ is calculated by

        $$
        B_{\text{rms}} = \sqrt{\frac
        {\int_{-\infty}^{\infty} (f - \mu_f)^2 \cdot S(f - \mu_f) \, df}
        {\int_{-\infty}^{\infty} S(f - \mu_f) \, df}
        }
        $$

        where $S(f)$ is the power spectral density (PSD) of the signal $x[n]$. The RMS bandwidth is measured about the
        centroid of the spectrum

        $$
        \mu_f = \frac
        {\int_{-\infty}^{\infty} f \cdot S(f) \, df}
        {\int_{-\infty}^{\infty} S(f) \, df} .
        $$

        The RMS bandwidth is a measure of the energy spread of the spectrum about the centroid. For a rectangular
        spectrum, the RMS bandwidth is $B_{\text{rms}} = B_s / \sqrt{12}$.

    Examples:
        Calculate the RMS bandwidth of a signal with a rectangular spectrum with normalized bandwidth of 1.

        .. ipython:: python

            symbol_rate = 1  # symbols/s
            symbol_rate / np.sqrt(12)

        Create a BPSK signal with a rectangular pulse shape. Note, the time-domain pulse shape is rectangular, but the
        spectrum is sinc-shaped. Measure the RMS bandwidth of the signal and compare it to the ideal rectangular
        spectrum.

        .. ipython:: python

            psk = sdr.PSK(2, pulse_shape="rect")
            symbols = np.random.randint(0, psk.order, 10_000)
            x_rect = psk.modulate(symbols)
            sdr.rms_bandwidth(x_rect, sample_rate=symbol_rate * psk.sps)

        Make the same measurements with square-root raised cosine (SRRC) pulse shaping. The SRRC spectrum is narrower
        and, therefore, closer to the rectangular spectrum.

        .. ipython:: python

            psk = sdr.PSK(2, pulse_shape="srrc")
            symbols = np.random.randint(0, psk.order, 10_000)
            x_srrc = psk.modulate(symbols)
            sdr.rms_bandwidth(x_srrc, sample_rate=symbol_rate * psk.sps)

        Plot the power spectral density (PSD) of the rectangular and SRRC pulse-shaped signals.

        .. ipython:: python

            @savefig sdr_rms_bandwidth_1.png
            plt.figure(); \
            sdr.plot.periodogram(x_rect, sample_rate=symbol_rate * psk.sps, label="Rectangular"); \
            sdr.plot.periodogram(x_srrc, sample_rate=symbol_rate * psk.sps, label="SRRC");

    Group:
        measurement-modulation
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    verify_scalar(sample_rate, float=True, positive=True)

    f, psd = scipy.signal.welch(
        x, fs=sample_rate, detrend=False, return_onesided=False, scaling="density", average="mean"
    )

    # Shift so that the frequencies increase monotonically
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)

    # Calculate the centroid of the PSD
    f_mean = scipy.integrate.simpson(f * psd, x=f)
    f_mean /= scipy.integrate.simpson(psd, x=f)
    f -= f_mean

    # Calculate the RMS bandwidth
    ms_bandwidth = scipy.integrate.simpson(f**2 * psd, x=f)
    ms_bandwidth /= scipy.integrate.simpson(psd, x=f)
    rms_bandwidth = np.sqrt(float(ms_bandwidth))

    return rms_bandwidth


@export
def rms_integration_time(
    x: npt.ArrayLike,
    sample_rate: float = 1.0,
) -> float:
    r"""
    Measures the RMS integration time $T_{\text{rms}}$ of the signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

            .. note::
                For best measurement performance, the time-domain signal should be very oversampled. This allows
                for better time precision.

        sample_rate: The sample rate $f_s$ in samples/s.

    Returns:
        The RMS integration time $T_{\text{rms}}$ in seconds.

    See Also:
        sdr.foa_crlb, sdr.fdoa_crlb

    Notes:
        The root-mean-square (RMS) integration time $T_{\text{rms}}$ is calculated by

        $$
        T_{\text{rms}} = \sqrt{\frac
        {\int_{-\infty}^{\infty} (t - \mu_t)^2 \cdot \left| x(t - \mu_t) \right|^2 \, dt}
        {\int_{-\infty}^{\infty} \left| x(t - \mu_t) \right|^2 \, dt}
        }
        $$

        where $x(t)$ is the continuous-time signal that was discretely sampled as $x[n]$. The RMS integration time is
        measured about the centroid of the signal

        $$
        \mu_t = \frac
        {\int_{-\infty}^{\infty} t \cdot \left| x(t) \right|^2 \, dt}
        {\int_{-\infty}^{\infty} \left| x(t) \right|^2 \, dt} .
        $$

        The RMS integration time is a measure of the energy spread of the signal about the centroid. For a rectangular
        signal, the RMS integration time is $T_{\text{rms}} = T / \sqrt{12}$.

    Examples:
        Calculate the RMS integration time of a signal with a rectangular envelope and duration of 1 second.

        .. ipython:: python

            symbol_rate = 100  # symbols/s
            sps = 100  # samples/symbol
            sample_rate = symbol_rate * sps  # samples/s
            n_symbols = symbol_rate  # Make a 1-second long signal
            t_s = n_symbols / symbol_rate  # Integration time (s)
            t_s / np.sqrt(12)

        Create a BPSK signal with a rectangular pulse shape. Measure the RMS integration time of the signal and compare
        it to the ideal rectangular envelope.

        .. ipython:: python

            psk = sdr.PSK(2, pulse_shape="rect", sps=sps)
            symbols = np.random.randint(0, psk.order, n_symbols)
            x_rect = psk.modulate(symbols).real
            sdr.rms_integration_time(x_rect, sample_rate=sample_rate)

            @savefig sdr_rms_integration_time_1.png
            plt.figure(); \
            sdr.plot.time_domain(x_rect, sample_rate=sample_rate, label="Rectangular");

        Make the same measurements with square-root raised cosine (SRRC) pulse shaping.

        .. ipython:: python

            psk = sdr.PSK(2, pulse_shape="srrc", sps=sps)
            symbols = np.random.randint(0, psk.order, n_symbols)
            x_srrc = psk.modulate(symbols).real
            sdr.rms_integration_time(x_srrc, sample_rate=sample_rate)

            @savefig sdr_rms_integration_time_2.png
            plt.figure(); \
            sdr.plot.time_domain(x_srrc, sample_rate=sample_rate, label="SRRC");

        For a given transmit energy, the RMS integration time is improved by increasing the energy at the edges of the
        signal. This can be achieved by applying a parabolic envelope to the BPSK signal. The energy of the signals
        is normalized. Notice the RMS integration time increases by 50% for the same transmit energy and duration.

        .. ipython:: python

            x_srrc_env = x_srrc * np.linspace(-1, 1, len(x_srrc))**2
            x_srrc_env *= np.sqrt(sdr.energy(x_srrc) / sdr.energy(x_srrc_env))
            sdr.rms_integration_time(x_srrc_env, sample_rate=sample_rate)

            @savefig sdr_rms_integration_time_3.png
            plt.figure(); \
            sdr.plot.time_domain(x_srrc, sample_rate=sample_rate, label="SRRC"); \
            sdr.plot.time_domain(x_srrc_env, sample_rate=sample_rate, label="SRRC + Parabolic Envelope");

    Group:
        measurement-modulation
    """
    x = verify_arraylike(x, complex=False, ndim=1)
    verify_scalar(sample_rate, float=True, positive=True)

    t = np.arange(x.size) / sample_rate

    # Calculate the centroid of the signal
    t_mean = scipy.integrate.simpson(t * np.abs(x) ** 2, x=t)
    t_mean /= scipy.integrate.simpson(np.abs(x) ** 2, x=t)
    t -= t_mean

    # Calculate the RMS integration time
    ms_integration_time = scipy.integrate.simpson(t**2 * np.abs(x) ** 2, x=t)
    ms_integration_time /= scipy.integrate.simpson(np.abs(x) ** 2, x=t)
    rms_integration_time = np.sqrt(float(ms_integration_time))

    return rms_integration_time
