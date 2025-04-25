"""
A module containing functions for simulating various signal impairments.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import linear
from .._farrow import FarrowResampler
from .._helper import (
    convert_output,
    export,
    verify_arraylike,
    verify_not_specified,
    verify_only_one_specified,
    verify_scalar,
)
from .._measurement import average_power


@export
def awgn(
    x: npt.ArrayLike,
    snr: float | None = None,
    noise: float | None = None,
    seed: int | None = None,
) -> npt.NDArray:
    r"""
    Adds additive white Gaussian noise (AWGN) to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which AWGN is added.
        snr: The desired signal-to-noise ratio (SNR) in dB. If specified, the average signal power is measured
            explicitly. It is assumed that $x[n]$ contains signal only. If the signal power is known, the
            desired noise variance can be computed and passed in `noise`. If `snr` is `None`,
            `noise` must be specified.
        noise: The noise power (variance) in linear units. If `noise` is `None`, `snr` must be specified.
        seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng`.

    Returns:
        The noisy signal $x[n] + w[n]$.

    Notes:
        The signal-to-noise ratio (SNR) is defined as

        $$
        \text{SNR} = \frac{P_{\text{signal,avg}}}{P_{\text{noise}}}
        = \frac{\frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2}{\sigma^2} ,
        $$

        where $\sigma^2$ is the noise variance. The output signal, with the specified SNR, is $y[n] = x[n] + w[n]$.

        For real signals:
        $$w \sim \mathcal{N}(0, \sigma^2)$$

        For complex signals:
        $$w \sim \mathcal{CN}(0, \sigma^2) = \mathcal{N}(0, \sigma^2 / 2) + j\mathcal{N}(0, \sigma^2 / 2)$$

    Examples:
        Create a real sinusoid and set its $S/N$ to 10 dB.

        .. ipython:: python

            x = np.sin(2 * np.pi * 5 * np.arange(100) / 100); \
            y = sdr.awgn(x, snr=10)

            @savefig sdr_awgn_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x, label="$x[n]$"); \
            sdr.plot.time_domain(y, label="$y[n]$"); \
            plt.title("Input signal $x[n]$ and noisy output signal $y[n]$ with 10 dB SNR");

        Create a QPSK reference signal and set its $E_s/N_0$ to 10 dB. When the signal has 1 sample per symbol,
        $E_s/N_0$ is equivalent to the discrete-time $S/N$.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1_000); \
            x = psk.map_symbols(s); \
            y = sdr.awgn(x, snr=10)

            @savefig sdr_awgn_2.svg
            plt.figure(); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title(f"QPSK constellations for $x[n]$ with $\infty$ dB $E_s/N_0$\nand $y[n]$ with 10 dB $E_s/N_0$");

    Group:
        simulation-impairments
    """
    x = verify_arraylike(x, complex=True)
    verify_only_one_specified(snr, noise)

    if snr is not None:
        verify_scalar(snr, float=True)
        snr_linear = linear(snr)
        signal_power = average_power(x)
        noise_power = signal_power / snr_linear
    elif noise is not None:
        verify_scalar(noise, float=True, non_negative=True)
        noise_power = noise

    rng = np.random.default_rng(seed)
    if np.iscomplexobj(x):
        w = rng.normal(0, np.sqrt(noise_power / 2), x.shape) + 1j * rng.normal(0, np.sqrt(noise_power / 2), x.shape)
    else:
        w = rng.normal(0, np.sqrt(noise_power), x.shape)

    y = x + w

    return convert_output(y)


@export
def iq_imbalance(x: npt.ArrayLike, amplitude: float, phase: float = 0.0) -> npt.NDArray[np.complex128]:
    r"""
    Applies IQ imbalance to the complex time-domain signal $x[n]$.

    Arguments:
        x: The complex time-domain signal $x[n]$ to which IQ imbalance is applied.
        amplitude: The amplitude imbalance $A$ in dB. A positive value indicates that the in-phase component is
            larger than the quadrature component.
        phase: The phase imbalance $\phi$ in degrees. A positive value indicates that the quadrature component
            leads the in-phase component.

    Returns:
        The signal $x[n]$ with IQ imbalance applied.

    Notes:
        The IQ imbalance is applied as follows.

        $$g_I = 10^{(A/2)/20} \exp\left(j \frac{-\phi}{2} \frac{\pi}{180}\right)$$
        $$g_Q = 10^{(-A/2)/20} \exp\left(j \frac{\phi}{2} \frac{\pi}{180}\right)$$
        $$y[n] = g_I \cdot x_I[n] + j \cdot g_Q \cdot x_Q[n]$$

    Examples:
        Positive amplitude imbalance horizontally stretches the constellation, while negative amplitude imbalance
        vertically stretches the constellation.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, 4, 1_000); \
            x = psk.map_symbols(s); \
            y1 = sdr.iq_imbalance(x, 5, 0); \
            y2 = sdr.iq_imbalance(x, -5, 0)

            @savefig sdr_iq_imbalance_1.svg
            plt.figure(); \
            plt.subplot(1, 2, 1); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y1, label="$y_1[n]$"); \
            plt.title("5 dB amplitude imbalance"); \
            plt.subplot(1, 2, 2); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y2, label="$y_2[n]$"); \
            plt.title("-5 dB amplitude imbalance");

        Positive phase imbalance stretches to the northwest, while negative phase imbalance stretches to the
        northeast

        .. ipython:: python

            y1 = sdr.iq_imbalance(x, 0, 20); \
            y2 = sdr.iq_imbalance(x, 0, -20)

            @savefig sdr_iq_imbalance_2.svg
            plt.figure(); \
            plt.subplot(1, 2, 1); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y1, label="$y_1[n]$"); \
            plt.title("20 deg phase imbalance"); \
            plt.subplot(1, 2, 2); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y2, label="$y_2[n]$"); \
            plt.title("-20 deg phase imbalance");

    Group:
        simulation-impairments
    """
    x = verify_arraylike(x, complex=True, imaginary=True)
    verify_scalar(amplitude, float=True)
    verify_scalar(phase, float=True)

    phase = np.deg2rad(phase)

    # TODO: Should the phase be negative for I?
    gain_i = linear(0.5 * amplitude, type="voltage") * np.exp(1j * -0.5 * phase)
    gain_q = linear(-0.5 * amplitude, type="voltage") * np.exp(1j * 0.5 * phase)

    y = gain_i * x.real + 1j * gain_q * x.imag

    return convert_output(y)


@export
def sample_rate_offset(
    x: npt.ArrayLike,
    offset: npt.ArrayLike,
    offset_rate: npt.ArrayLike = 0.0,
    sample_rate: float = 1.0,
) -> npt.NDArray:
    r"""
    Applies a sample rate offset to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which the sample rate offset is applied.
        offset: The sample rate offset $\Delta f_s = f_{s,\text{new}} - f_{s}$ in samples/s.
        offset_rate: The sample rate offset rate $\Delta^2 f_s / \Delta t$ in samples/s^2.
        sample_rate: The sample rate $f_s$ in samples/s.

    Returns:
        The signal $x[n]$ with sample rate offset applied.

    Notes:
        The sample rate offset is applied using a Farrow resampler. The resampling rate is calculated as follows.

        $$
        \text{rate} = \frac{f_s + \Delta f_s + \frac{\Delta f_s}{f_s}}{f_s}
        $$

    Examples:
        Create a QPSK reference signal.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1_000); \
            x = psk.map_symbols(s)

        Add 10 ppm of sample rate offset.

        .. ipython:: python

            y = sdr.sample_rate_offset(x, 10e-6)

            @savefig sdr_sample_rate_offset_1.svg
            plt.figure(); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title("10 ppm sample rate offset");

        Add 100 ppm of sample rate offset.

        .. ipython:: python

            y = sdr.sample_rate_offset(x, 100e-6)

            @savefig sdr_sample_rate_offset_2.svg
            plt.figure(); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title("100 ppm sample rate offset");

    Group:
        simulation-impairments
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    offset = verify_arraylike(offset, float=True)
    offset_rate = verify_arraylike(offset_rate, float=True)
    verify_scalar(sample_rate, float=True, positive=True)

    # TODO: Is this correct....
    rate = (sample_rate + offset + offset_rate / sample_rate) / sample_rate
    farrow = FarrowResampler(3)
    y = farrow(x, rate)

    return convert_output(y)


@export
def frequency_offset(
    x: npt.ArrayLike,
    offset: npt.ArrayLike,
    offset_rate: npt.ArrayLike = 0.0,
    phase: npt.ArrayLike = 0.0,
    sample_rate: float = 1.0,
) -> npt.NDArray:
    r"""
    Applies a frequency and phase offset to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which the frequency offset is applied.
        offset: The frequency offset $\Delta f = f_{\text{new}} - f$ in Hz.
        offset_rate: The frequency offset rate $\Delta^2 f / \Delta t$ in Hz/s. For example, a frequency
            offset varying from 1 kHz to 2 kHz over 1 ms, the offset rate is 1 kHz / 1 ms or 1 MHz/s.
        phase: The phase offset $\phi$ in degrees.
        sample_rate: The sample rate $f_s$ in samples/s.

    Returns:
        The signal $x[n]$ with frequency offset applied.

    Examples:
        Create a reference signal with a constant frequency of 1 cycle per 100 samples.

        .. ipython:: python

            x = sdr.sinusoid(100, freq=1 / 100)

        Add a frequency offset of 1 cycle per 100 samples (the length of the signal). Notice that the signal now
        rotates through 2 cycles instead of 1.

        .. ipython:: python

            freq = 1 / 100
            y = sdr.frequency_offset(x, freq)

            @savefig sdr_frequency_offset_1.svg
            plt.figure(); \
            sdr.plot.time_domain(np.unwrap(np.angle(x)) / (2 * np.pi), label="$x[n]$"); \
            sdr.plot.time_domain(np.unwrap(np.angle(y)) / (2 * np.pi), label="$y[n]$"); \
            plt.ylabel("Absolute phase (cycles)"); \
            plt.title("Constant frequency offset (linear phase)");

        Add a frequency rate of change of 2 cycles per 100^2 samples. Notice that the signal now rotates through
        4 cycles instead of 2.

        .. ipython:: python

            freq_rate = 2 / 100**2
            y = sdr.frequency_offset(x, freq, freq_rate)

            @savefig sdr_frequency_offset_2.svg
            plt.figure(); \
            sdr.plot.time_domain(np.unwrap(np.angle(x)) / (2 * np.pi), label="$x[n]$"); \
            sdr.plot.time_domain(np.unwrap(np.angle(y)) / (2 * np.pi), label="$y[n]$"); \
            plt.ylabel("Absolute phase (cycles)"); \
            plt.title("Linear frequency offset (quadratic phase)");

    Group:
        simulation-impairments
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    offset = verify_arraylike(offset, float=True)
    offset_rate = verify_arraylike(offset_rate, float=True)
    phase = verify_arraylike(phase, float=True)
    verify_scalar(sample_rate, float=True, positive=True)

    t = np.arange(x.size) / sample_rate  # Time vector in seconds
    f = offset + offset_rate * t  # Frequency vector in Hz
    lo = np.exp(1j * (2 * np.pi * f * t + np.deg2rad(phase)))  # Local oscillator
    y = x * lo  # Apply frequency offset

    return convert_output(y)


@export
def clock_error(
    x: npt.ArrayLike,
    error: npt.ArrayLike,
    error_rate: float = 0.0,
    center_freq: float | None = None,
    sample_rate: float | None = None,
) -> npt.NDArray:
    r"""
    Applies a clock error to the time-domain signal $x[n]$.

    This clock error could be caused by transmitter clock error, receiver clock error, or Doppler effects.

    Arguments:
        x: The time-domain signal $x[n]$ to which the clock error is applied.

            .. warning::

                The signal must be a real passband signal or a complex baseband signal (with 0 Hz baseband frequency).

                If the signal is a real passband signal, time will be compressed resulting in a carrier frequency
                change. If the signal is a complex baseband signal, time will similarly be compressed. However,
                the zero-IF baseband signal will not observe a frequency shift, since it was always mixed to baseband.
                Therefore, there is a subsequent frequency shift corresponding to the expected frequency shift at
                passband.

                If a complex low-IF signal is provided, the IF frequency will be shifted during time compression.
                This can become noticeable at high clock errors, e.g. 1,000 ppm or more. It is not advised to use
                this function with complex low-IF signals.

        error: The fractional clock error $\epsilon$, which is unitless, with 0 representing no clock error.
            For example, 1e-6 represents 1 ppm of clock error.

            The fractional clock error can be calculated from frequency offset $\Delta f = f_{c,\text{new}} - f_c$ and
            carrier frequency $f_c$ as $\epsilon = \Delta f / f_c$. For example, a 1 kHz frequency error applied to a
            signal with a 1 GHz carrier frequency is 1e-6 or 1 ppm.

            The fractional clock error can also be calculated from sample rate offset $\Delta f_s = f_s - f_{s,\text{new}}$
            and sample rate $f_s$ as $\epsilon = \Delta f_s / f_s$. For example, a -10 S/s sample rate error applied
            to a signal with a 10 MS/s sample rate is -1e-6 or -1 ppm.

            The fractional clock error can also be calculated from relative velocity $\Delta v$ and speed of light
            $c$ as $\epsilon = \Delta v / c$. For example, a 60 mph (or 26.82 m/s) relative velocity between the
            transmitter and receiver is 8.946e-8 or 8.9 ppb.

        error_rate: The clock error $\Delta \epsilon / \Delta t$ in 1/s.
        center_freq: The center frequency $f_c$ of the complex baseband signal in Hz. 0 Hz baseband frequency must
            correspond to the signal's carrier frequency. If $x[n]$ is complex, this must be provided.
        sample_rate: The sample rate $f_s$ in samples/s. If $x[n]$ is complex, this must be provided.

    Returns:
        The signal $x[n]$ with clock error applied.

    Examples:
        This example demonstrates the effect of clock error on a real passband signal. The signal has a carrier
        frequency of 100 kHz. A frequency offset of 20 kHz is desired, corresponding to a clock error or 0.2.
        The clock error is added to the transmitter, and then removed at the receiver. Notice that the transmitted
        signal is compressed in time and shifted in frequency. Also notice that the corrected received signal
        matches the original.

        .. ipython:: python

            sample_rate = 2e6; \
            freq = 100e3; \
            duration = 1000e-6; \
            x = sdr.sinusoid(duration, freq, sample_rate=sample_rate, complex=False)

            freq_offset = 20e3; \
            error = freq_offset / freq; \
            print("Clock error:", error); \
            y = sdr.clock_error(x, error)

            error = -error / (1 + error); \
            print("Clock error:", error); \
            z = sdr.clock_error(y, error)

            @savefig sdr_clock_error_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x - 0, sample_rate=sample_rate, label="No clock error"); \
            sdr.plot.time_domain(y - 3, sample_rate=sample_rate, label="Added Tx clock error"); \
            sdr.plot.time_domain(z - 6, sample_rate=sample_rate, label="Removed Tx clock error"); \
            plt.legend(loc="lower left"); \
            plt.title("Real passband signals with and without clock error");

            @savefig sdr_clock_error_2.svg
            plt.figure(); \
            sdr.plot.dtft(x, sample_rate=sample_rate, label="No clock error"); \
            sdr.plot.dtft(y, sample_rate=sample_rate, label="Added Tx clock error"); \
            sdr.plot.dtft(z, sample_rate=sample_rate, label="Removed Tx clock error"); \
            plt.axvline(freq, color="k", linestyle="--"); \
            plt.axvline(freq + freq_offset, color="k", linestyle="--"); \
            plt.xlim(80e3, 140e3);

        This example demonstrates the effect of clock error on a complex baseband signal. The signal has a carrier
        frequency of 1 MHz and sample rate of 2 MS/s. A frequency offset of 100 kHz is desired, corresponding to a
        clock error of 0.1. The clock error is added to the transmitter, and then removed at the receiver. Notice that
        the transmitted signal is compressed in time and shifted in frequency. Also notice that the corrected received
        signal matches the original.

        .. ipython:: python

            sample_rate = 2e6; \
            center_freq = 1e6; \
            duration = 1000e-6; \
            x = sdr.sinusoid(duration, 0, sample_rate=sample_rate)

            freq_offset = 100e3; \
            error = freq_offset / center_freq; \
            print("Clock error:", error); \
            y = sdr.clock_error(x, error, 0, center_freq, sample_rate=sample_rate)

            error = -error / (1 + error); \
            print("Clock error:", error); \
            z = sdr.clock_error(y, error, 0, center_freq, sample_rate=sample_rate)

            @savefig sdr_clock_error_3.svg
            plt.figure(); \
            sdr.plot.time_domain(x - 0 - 0j, sample_rate=sample_rate, label="No clock error"); \
            sdr.plot.time_domain(y - 3 - 3j, sample_rate=sample_rate, label="Added Tx clock error"); \
            sdr.plot.time_domain(z - 6 - 6j, sample_rate=sample_rate, label="Removed Tx clock error"); \
            plt.legend(loc="lower left"); \
            plt.title("Complex baseband signals with and without clock error");

            @savefig sdr_clock_error_4.svg
            plt.figure(); \
            sdr.plot.dtft(x, sample_rate=sample_rate, label="No clock error"); \
            sdr.plot.dtft(y, sample_rate=sample_rate, label="Added Tx clock error"); \
            sdr.plot.dtft(z, sample_rate=sample_rate, label="Removed Tx clock error"); \
            plt.axvline(0, color="k", linestyle="--"); \
            plt.axvline(freq_offset, color="k", linestyle="--"); \
            plt.xlim(-20e3, 120e3);

    Group:
        simulation-impairments
    """
    x = verify_arraylike(x, complex=True, ndim=1)
    error = verify_arraylike(error, float=True)
    verify_scalar(error_rate, float=True)

    # Apply time compression using resampling
    alpha = 1 + error
    # y = sample_rate_offset(x, 1 / alpha, 0)  # TODO: This doesn't work...
    farrow = FarrowResampler(3)
    y = farrow(x, 1 / alpha)

    if np.issubdtype(x.dtype, np.floating):
        verify_not_specified(center_freq)

        # The carrier frequency was already shifted by the time compression
        z = y
    else:
        verify_scalar(center_freq, float=True, positive=True)
        verify_scalar(sample_rate, float=True, positive=True)

        # Apply frequency shift that would be observed at passband
        freq_offset = error * center_freq  # Hz
        freq_offset_rate = error_rate * center_freq  # Hz/s
        z = frequency_offset(y, freq_offset, freq_offset_rate, sample_rate=sample_rate)

    return convert_output(z)
