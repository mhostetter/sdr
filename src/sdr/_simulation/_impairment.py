"""
A module containing functions for simulating various signal impairments.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import linear
from .._farrow import FarrowResampler
from .._helper import convert_output, export, verify_arraylike, verify_only_one_specified, verify_scalar
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

            @savefig sdr_awgn_1.png
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

            @savefig sdr_awgn_2.png
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

            @savefig sdr_iq_imbalance_1.png
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

            @savefig sdr_iq_imbalance_2.png
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
        offset: The sample rate offset $\Delta f_s = f_{s,\text{new}} - f_{s,\text{old}}$ in samples/s.
        offset_rate: The sample rate offset rate $\Delta f_s / \Delta t$ in samples/s^2.
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

            @savefig sdr_sample_rate_offset_1.png
            plt.figure(); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title("10 ppm sample rate offset");

        Add 100 ppm of sample rate offset.

        .. ipython:: python

            y = sdr.sample_rate_offset(x, 100e-6)

            @savefig sdr_sample_rate_offset_2.png
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

    rate = (sample_rate + offset + offset_rate / sample_rate) / sample_rate
    farrow = FarrowResampler()
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
        offset: The frequency offset $\Delta f_c = f_{c,\text{new}} - f_{c,\text{old}}$ in Hz.
        offset_rate: The frequency offset rate $\Delta f_c / \Delta t$ in Hz/s.
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

            @savefig sdr_frequency_offset_1.png
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

            @savefig sdr_frequency_offset_2.png
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
