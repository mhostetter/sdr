"""
A module containing functions for signal manipulation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal

from ._helper import export


@export
def mix(x: npt.ArrayLike, freq: float = 0, phase: float = 0, sample_rate: float = 1) -> np.ndarray:
    r"""
    Mixes the time-domain signal $x[n]$ with a complex exponential.

    $$y[n] = x[n] \cdot \exp \left[ j \left( \frac{2 \pi f}{f_s} n + \phi \right) \right]$$

    Arguments:
        x: The time-domain signal $x[n]$.
        freq: The frequency $f$ of the complex exponential in Hz (or 1/samples if `sample_rate=1`).
            The frequency must satisfy $-f_s/2 \le f \le f_s/2$.
        phase: The phase $\phi$ of the complex exponential in degrees.
        sample_rate: The sample rate $f_s$ of the signal.

    Returns:
        The mixed signal $y[n]$.

    Examples:
        Create a complex exponential with a frequency of 10 Hz and phase of 45 degrees.

        .. ipython:: python

            sample_rate = 1e3; \
            N = 100; \
            x = np.exp(1j * (2 * np.pi * 10 * np.arange(N) / sample_rate + np.pi/4))

            @savefig sdr_mix_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, sample_rate=sample_rate); \
            plt.title(r"Complex exponential with $f=10$ Hz and $\phi=45$ degrees"); \
            plt.tight_layout();

        Mix the signal to baseband by removing the frequency rotation and the phase offset.

        .. ipython:: python

            y = sdr.mix(x, freq=-10, phase=-45, sample_rate=sample_rate)

            @savefig sdr_mix_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y, sample_rate=sample_rate); \
            plt.title(r"Baseband signal with $f=0$ Hz and $\phi=0$ degrees"); \
            plt.tight_layout();

    Group:
        dsp-signal-manipulation
    """
    x = np.asarray(x)

    if not isinstance(freq, (int, float)):
        raise TypeError(f"Argument 'freq' must be a number, not {type(freq)}.")
    if not isinstance(phase, (int, float)):
        raise TypeError(f"Argument 'phase' must be a number, not {type(phase)}.")
    if not isinstance(sample_rate, (int, float)):
        raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")

    if not -sample_rate / 2 <= freq <= sample_rate / 2:
        raise ValueError(f"Argument 'freq' must be in the range [{-sample_rate/2}, {sample_rate/2}], not {freq}.")
    if not sample_rate > 0:
        raise ValueError(f"Argument 'sample_rate' must be positive, not {sample_rate}.")

    t = np.arange(len(x)) / sample_rate  # Time vector in seconds
    y = x * np.exp(1j * (2 * np.pi * freq * t + np.deg2rad(phase)))

    return y


@export
def to_complex_bb(x_r: npt.ArrayLike) -> np.ndarray:
    r"""
    Converts the real passband signal $x_r[n]$ centered at $f_{s,r}/4$ with sample rate $f_{s,r}$ to a
    complex baseband signal $x_c[n]$ centered at $0$ with sample rate $f_{s,c} = f_{s,r}/2$.

    Arguments:
        x_r: The real passband signal $x_r[n]$ centered at $f_{s,r}/4$ with sample rate $f_{s,r}$.
            If the length is odd, one zero is appended to the end.

    Returns:
        The complex baseband signal $x_c[n]$ centered at $0$ with sample rate $f_{s,c} = f_{s,r}/2$.

    Examples:
        Create a real passband signal with frequency components at 100, 250, and 300 Hz, at a sample rate of 1 ksps.
        Notice the spectrum is complex-conjugate symmetric.

        .. ipython:: python

            sample_rate = 1e3; \
            x_r = ( \
                0.1 * np.sin(2 * np.pi * 100 / sample_rate * np.arange(1000)) \
                + 1.0 * np.sin(2 * np.pi * 250 / sample_rate * np.arange(1000)) \
                + 0.5 * np.sin(2 * np.pi * 300 / sample_rate * np.arange(1000)) \
            ); \
            x_r = sdr.awgn(x_r, snr=30)

            @savefig sdr_to_complex_bb_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x_r[0:100], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_r[n]$"); \
            plt.tight_layout();

            @savefig sdr_to_complex_bb_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.periodogram(x_r, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_r[n]$"); \
            plt.tight_layout();

        Convert the real passband signal to a complex baseband signal with sample rate 500 sps and center of 0 Hz.
        Notice the spectrum is no longer complex-conjugate symmetric.
        The real sinusoids are now complex exponentials at -150, 0, and 50 Hz.

        .. ipython:: python

            x_c = sdr.to_complex_bb(x_r); \
            sample_rate /= 2

            @savefig sdr_to_complex_bb_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x_c[0:50], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_c[n]$"); \
            plt.tight_layout();

            @savefig sdr_to_complex_bb_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.periodogram(x_c, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_c[n]$"); \
            plt.tight_layout();

    Group:
        dsp-signal-manipulation
    """
    x_r = np.asarray(x_r)

    if not x_r.ndim == 1:
        raise ValueError(f"Argument 'x_r' must be a 1D array, not {x_r.ndim}D.")
    if not np.isrealobj(x_r):
        raise ValueError("Argument 'x_r' must be real, not complex.")

    if x_r.size % 2 == 1:
        # Append one zero for the decimate by 2
        x_r = np.append(x_r, 0)

    # Compute the analytic signal. The analytic signal has the negative frequencies zeroed out.
    x_a = scipy.signal.hilbert(x_r)

    # Mix the complex signal to baseband. What was fs/4 is now 0.
    x_c = mix(x_a, freq=-0.25)

    # Decimate by 2 to achieve fs/2 sample rate. This can be done without anti-alias filtering since the
    # spectrum outside +/- fs/4 is zero.
    x_c = x_c[::2]

    return x_c


@export
def to_real_pb(x_c: npt.ArrayLike) -> np.ndarray:
    r"""
    Converts the complex baseband signal $x_c[n]$ centered at $0$ with sample rate $f_{s,c}$ to a
    real passband signal $x_r[n]$ centered at $f_{s,r}/4$ with sample rate $f_{s,r} = 2f_{s,c}$.

    Arguments:
        x_c: The complex baseband signal $x_c[n]$ centered at $0$ with sample rate $f_{s,c}$.

    Returns:
        The real passband signal $x_r[n]$ centered at $f_{s,r}/4$ with sample rate $f_{s,r} = 2f_{s,c}$.

    Examples:
        Create a complex baseband signal with frequency components at -150, 0, and 50 Hz, at a sample rate of 50 sps.
        Notice the spectrum is asymmetric.

        .. ipython:: python

            sample_rate = 500; \
            x_c = ( \
                0.1 * np.exp(1j * 2 * np.pi * -150 / sample_rate * np.arange(1000)) \
                + 1.0 * np.exp(1j * 2 * np.pi * 0 / sample_rate * np.arange(1000)) \
                + 0.5 * np.exp(1j * 2 * np.pi * 50 / sample_rate * np.arange(1000)) \
            ); \
            x_c = sdr.awgn(x_c, snr=30)

            @savefig sdr_to_real_pb_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x_c[0:50], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_c[n]$"); \
            plt.tight_layout();

            @savefig sdr_to_real_pb_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.periodogram(x_c, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_c[n]$"); \
            plt.tight_layout();

        Convert the complex baseband signal to a real passband signal with sample rate 1 ksps and center of 250 Hz.
        Notice the spectrum is now complex-conjugate symmetric.
        The complex exponentials are now real sinusoids at 100, 250, and 300 Hz.

        .. ipython:: python

            x_r = sdr.to_real_pb(x_c); \
            sample_rate *= 2

            @savefig sdr_to_real_pb_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x_r[0:100], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_r[n]$"); \
            plt.tight_layout();

            @savefig sdr_to_real_pb_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.periodogram(x_r, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_r[n]$"); \
            plt.tight_layout();

    Group:
        dsp-signal-manipulation
    """
    x_c = np.asarray(x_c)

    if not x_c.ndim == 1:
        raise ValueError(f"Argument 'x_c' must be a 1D array, not {x_c.ndim}D.")
    if not np.iscomplexobj(x_c):
        raise ValueError("Argument 'x_c' must be complex, not real.")

    # Upsample by 2 to achieve 2*fs sample rate
    x_c = scipy.signal.resample_poly(x_c, 2, 1)

    # Mix the complex baseband signal to passband. What was 0 is now fs/4.
    x_c = mix(x_c, freq=0.25)

    # Only preserve the real part, which is complex-conjugate symmetric about 0 Hz.
    x_r = x_c.real

    return x_r
