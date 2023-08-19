"""
A module containing functions for simulating various signal impairments.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import linear
from .._farrow import FarrowResampler
from .._helper import export
from .._measurement import average_power


@export
def awgn(
    x: npt.ArrayLike,
    snr: float | None = None,
    noise: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    r"""
    Adds additive white Gaussian noise (AWGN) to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which AWGN is added.
        snr: The desired signal-to-noise ratio (SNR) in dB. If specified, the average signal power is measured
            explicitly. It is assumed that $x[n]$ contains signal only. If the signal power is known, the
            desired noise variance can be computed and passed in `noise`. If `snr` is `None`,
            `noise` must be specified.
        noise: The noise power (variance) in linear units. If `noise` is `None`, `snr` must be specified.
        seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.

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
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="$x[n]$"); \
            sdr.plot.time_domain(y, label="$y[n]$"); \
            plt.title("Input signal $x[n]$ and noisy output signal $y[n]$ with 10 dB SNR"); \
            plt.tight_layout()

        Create a QPSK reference signal and set its $E_s/N_0$ to 10 dB. When the signal has 1 sample per symbol,
        $E_s/N_0$ is equivalent to the discrete-time $S/N$.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1_000); \
            x = psk.modulate(s); \
            y = sdr.awgn(x, snr=10)

            @savefig sdr_awgn_2.png
            plt.figure(figsize=(10, 5)); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title(f"QPSK constellations for $x[n]$ with $\infty$ dB $E_s/N_0$\nand $y[n]$ with 10 dB $E_s/N_0$"); \
            plt.tight_layout()

    Group:
        simulation-impairments
    """
    x = np.asarray(x)
    if snr is not None:
        snr_linear = linear(snr)
        signal_power = average_power(x)
        noise_power = signal_power / snr_linear
    elif noise is not None:
        noise_power = noise
    else:
        raise ValueError("Either 'snr' or 'noise' must be specified.")

    rng = np.random.default_rng(seed)
    if np.iscomplexobj(x):
        w = rng.normal(0, np.sqrt(noise_power / 2), x.shape) + 1j * rng.normal(0, np.sqrt(noise_power / 2), x.shape)
    else:
        w = rng.normal(0, np.sqrt(noise_power), x.shape)

    return x + w


@export
def iq_imbalance(x: npt.ArrayLike, amplitude: float, phase: float = 0) -> np.ndarray:
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
        $$y[n] = g_I x_I[n] + j g_Q x_Q[n]$$

    Examples:
        Positive amplitude imbalance horizontally stretches the constellation, while negative amplitude imbalance
        vertically stretches the constellation.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, 4, 1_000); \
            x = psk.modulate(s); \
            y1 = sdr.iq_imbalance(x, 5, 0); \
            y2 = sdr.iq_imbalance(x, -5, 0)

            @savefig sdr_iq_imbalance_1.png
            plt.figure(figsize=(10, 5)); \
            plt.subplot(1, 2, 1); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y1, label="$y_1[n]$"); \
            plt.legend(); \
            plt.title("5 dB amplitude imbalance"); \
            plt.subplot(1, 2, 2); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y2, label="$y_2[n]$"); \
            plt.legend(); \
            plt.title("-5 dB amplitude imbalance");

        Positive phase imbalance stretches to the northwest, while negative phase imbalance stretches to the
        northeast

        .. ipython:: python

            y1 = sdr.iq_imbalance(x, 0, 20); \
            y2 = sdr.iq_imbalance(x, 0, -20)

            @savefig sdr_iq_imbalance_2.png
            plt.figure(figsize=(10, 5)); \
            plt.subplot(1, 2, 1); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y1, label="$y_1[n]$"); \
            plt.legend(); \
            plt.title("20 deg phase imbalance"); \
            plt.subplot(1, 2, 2); \
            sdr.plot.constellation(x, label="$x[n]$"); \
            sdr.plot.constellation(y2, label="$y_2[n]$"); \
            plt.legend(); \
            plt.title("-20 deg phase imbalance");

    Group:
        simulation-impairments
    """
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        raise ValueError("Argument 'x' must be complex.")

    phase = np.deg2rad(phase)

    # TODO: Should the phase be negative for I?
    gain_i = linear(0.5 * amplitude, type="voltage") * np.exp(1j * -0.5 * phase)
    gain_q = linear(-0.5 * amplitude, type="voltage") * np.exp(1j * 0.5 * phase)

    y = gain_i * x.real + 1j * gain_q * x.imag

    return y


@export
def sample_rate_offset(x: npt.ArrayLike, ppm: float) -> np.ndarray:
    r"""
    Applies a sample rate offset to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which the sample rate offset is applied.
        ppm: The sample rate offset $f_{s,\text{new}} / f_s$ in parts per million (ppm).

    Returns:
        The signal $x[n]$ with sample rate offset applied.

    Examples:
        Create a QPSK reference signal.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1_000); \
            x = psk.modulate(s)

        Add 10 ppm of sample rate offset.

        .. ipython:: python

            ppm = 10; \
            y = sdr.sample_rate_offset(x, ppm)

            @savefig sdr_sample_rate_offset_1.png
            plt.figure(figsize=(10, 5)); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title(f"{ppm} ppm sample rate offset"); \
            plt.tight_layout()

        Add 100 ppm of sample rate offset.

        .. ipython:: python

            ppm = 100; \
            y = sdr.sample_rate_offset(x, ppm)

            @savefig sdr_sample_rate_offset_2.png
            plt.figure(figsize=(10, 5)); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title(f"{ppm} ppm sample rate offset"); \
            plt.tight_layout()

    Group:
        simulation-impairments
    """
    x = np.asarray(x)
    if not x.ndim == 1:
        raise ValueError(f"Argument 'x' must be 1D, not {x.ndim}D.")

    rate = 1 + ppm * 1e-6

    # TODO: Add ppm_rate
    # if ppm_rate:
    #     rate += ppm_rate * 1e-6 * np.arange(x.size)

    farrow = FarrowResampler()
    y = farrow(x, rate)

    return y


@export
def frequency_offset(
    x: npt.ArrayLike,
    freq: npt.ArrayLike,
    freq_rate: npt.ArrayLike = 0,
    phase: npt.ArrayLike = 0,
    sample_rate: float = 1,
) -> np.ndarray:
    r"""
    Applies a frequency and phase offset to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which the frequency offset is applied.
        freq: The frequency offset $f$ in Hz (or in cycles/sample if `sample_rate=1`).
        freq_rate: The frequency offset rate $f_{\text{rate}}$ in Hz/s (or in cycles/sample^2 if `sample_rate=1`).
        phase: The phase offset $\phi$ in degrees.
        sample_rate: The sample rate $f_s$ in samples/s.

    Returns:
        The signal $x[n]$ with frequency offset applied.

    Examples:
        Create a QPSK reference signal.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1_000); \
            x = psk.modulate(s)

        Add a frequency offset of 1 cycle per 10,000 symbols.

        .. ipython:: python

            freq = 1e-4; \
            y = sdr.frequency_offset(x, freq)

            @savefig sdr_frequency_offset_1.png
            plt.figure(figsize=(10, 5)); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title(f"{freq} cycles/sample frequency offset"); \
            plt.tight_layout()

        Add a frequency offset of -1 cycle per 20,000 symbols and a phase offset of -45 degrees.

        .. ipython:: python

            freq = -5e-5; \
            phase = -45; \
            y = sdr.frequency_offset(x, freq, phase=phase)

            @savefig sdr_frequency_offset_2.png
            plt.figure(figsize=(10, 5)); \
            sdr.plot.constellation(x, label="$x[n]$", zorder=2); \
            sdr.plot.constellation(y, label="$y[n]$", zorder=1); \
            plt.title(f"{freq} cycles/sample frequency and {phase} deg offset"); \
            plt.tight_layout()

    Group:
        simulation-impairments
    """
    x = np.asarray(x)
    if not x.ndim == 1:
        raise ValueError(f"Argument 'x' must be 1D, not {x.ndim}D.")

    freq = np.asarray(freq)
    if not (freq.ndim == 0 or freq.shape == x.shape):
        raise ValueError(f"Argument 'freq' must be scalar or have shape {x.shape}, not {freq.shape}.")

    freq_rate = np.asarray(freq_rate)
    if not (freq_rate.ndim == 0 or freq_rate.shape == x.shape):
        raise ValueError(f"Argument 'freq_rate' must be scalar or have shape {x.shape}, not {freq_rate.shape}.")

    phase = np.asarray(phase)
    if not (phase.ndim == 0 or phase.shape == x.shape):
        raise ValueError(f"Argument 'phase' must be scalar or have shape {x.shape}, not {phase.shape}.")

    t = np.arange(x.size) / sample_rate  # Time vector in seconds
    f = freq + freq_rate * t  # Frequency vector in Hz
    lo = np.exp(1j * (2 * np.pi * f * t + np.deg2rad(phase)))  # Local oscillator
    y = x * lo  # Apply frequency offset

    return y
