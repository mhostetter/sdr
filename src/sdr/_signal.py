"""
A module containing functions for signal manipulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal

from ._helper import convert_output, export, verify_arraylike, verify_bool, verify_scalar


@export
def sinusoid(
    duration: float,
    freq: float = 0.0,
    freq_rate: float = 0.0,
    phase: float = 0.0,
    sample_rate: float = 1.0,
    complex: bool = True,
) -> npt.NDArray:
    r"""
    Generates a complex exponential or real sinusoid.

    Arguments:
        duration: The duration of the signal in seconds (or samples if `sample_rate=1`).
        freq: The frequency $f$ of the sinusoid in Hz (or 1/samples if `sample_rate=1`).
            The frequency must satisfy $-f_s/2 \le f \le f_s/2$.
        freq_rate: The frequency rate $\frac{df}{dt}$ of the sinusoid in Hz/s (or 1/samples$^2$ if `sample_rate=1`).
        phase: The phase $\phi$ of the sinusoid in degrees.
        sample_rate: The sample rate $f_s$ of the signal.
        complex: Indicates whether to generate a complex exponential or real sinusoid.

            - `True`: $\exp \left[ j \left( 2 \pi f t + \phi \right) \right]$
            - `False`: $\cos \left( 2 \pi f t + \phi \right)$

    Returns:
        The sinusoid $x[n]$.

    Examples:
        Create a complex exponential with a frequency of 10 Hz and phase of 45 degrees.

        .. ipython:: python

            sample_rate = 1e3; \
            N = 100; \
            lo = sdr.sinusoid(N / sample_rate, freq=10, phase=45, sample_rate=sample_rate)

            @savefig sdr_sinusoid_1.svg
            plt.figure(); \
            sdr.plot.time_domain(lo, sample_rate=sample_rate); \
            plt.title(r"Complex exponential with $f=10$ Hz and $\phi=45$ degrees");

        Create a real sinusoid with a frequency of 10 Hz and phase of 45 degrees.

        .. ipython:: python

            lo = sdr.sinusoid(N / sample_rate, freq=10, phase=45, sample_rate=sample_rate, complex=False)

            @savefig sdr_sinusoid_2.svg
            plt.figure(); \
            sdr.plot.time_domain(lo, sample_rate=sample_rate); \
            plt.title(r"Real sinusoid with $f=10$ Hz and $\phi=45$ degrees");

    Group:
        dsp-signal-manipulation
    """
    verify_scalar(freq, float=True, inclusive_min=-sample_rate / 2, inclusive_max=sample_rate / 2)
    verify_scalar(phase, float=True)
    verify_scalar(sample_rate, float=True, positive=True)
    verify_bool(complex)

    n = int(duration * sample_rate)  # Number of samples
    t = np.arange(n) / sample_rate  # Time vector in seconds
    if complex:
        lo = np.exp(1j * (2 * np.pi * (freq + freq_rate * t) * t + np.deg2rad(phase)))
    else:
        lo = np.cos(2 * np.pi * (freq + freq_rate * t) * t + np.deg2rad(phase))

    return convert_output(lo)


@export
def mix(
    x: npt.ArrayLike,
    freq: float = 0.0,
    phase: float = 0.0,
    sample_rate: float = 1.0,
    complex: bool = True,
) -> npt.NDArray:
    r"""
    Mixes a time-domain signal with a complex exponential or real sinusoid.

    Arguments:
        x: The time-domain signal $x[n]$.
        freq: The frequency $f$ of the sinusoid in Hz (or 1/samples if `sample_rate=1`).
            The frequency must satisfy $-f_s/2 \le f \le f_s/2$.
        phase: The phase $\phi$ of the sinusoid in degrees.
        sample_rate: The sample rate $f_s$ of the signal.
        complex: Indicates whether to mix by a complex exponential or real sinusoid.

            - `True`: $y[n] = x[n] \cdot \exp \left[ j \left( \frac{2 \pi f}{f_s} n + \phi \right) \right]$
            - `False`: $y[n] = x[n] \cdot \cos \left( \frac{2 \pi f}{f_s} n + \phi \right)$

    Returns:
        The mixed signal $y[n]$.

    Examples:
        Create a complex exponential with a frequency of 10 Hz and phase of 45 degrees.

        .. ipython:: python

            sample_rate = 1e3; \
            x = sdr.sinusoid(100 / sample_rate, freq=10, phase=45, sample_rate=sample_rate)

            @savefig sdr_mix_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=sample_rate); \
            plt.title(r"Complex exponential with $f=10$ Hz and $\phi=45$ degrees");

        Mix the signal to baseband by removing the frequency rotation and the phase offset.

        .. ipython:: python

            y = sdr.mix(x, freq=-10, phase=-45, sample_rate=sample_rate)

            @savefig sdr_mix_2.svg
            plt.figure(); \
            sdr.plot.time_domain(y, sample_rate=sample_rate); \
            plt.title(r"Baseband signal with $f=0$ Hz and $\phi=0$ degrees");

    Group:
        dsp-signal-manipulation
    """
    x = verify_arraylike(x, ndim=1)
    verify_scalar(freq, float=True, inclusive_min=-sample_rate / 2, inclusive_max=sample_rate / 2)
    verify_scalar(phase, float=True)
    verify_scalar(sample_rate, float=True, positive=True)
    verify_bool(complex)

    lo = sinusoid(x.size / sample_rate, freq=freq, phase=phase, sample_rate=sample_rate, complex=complex)
    y = x * lo

    return convert_output(y)


@export
def to_complex_baseband(x_r: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    r"""
    Converts a real passband signal to a complex baseband signal.

    The real passband signal $x_r[n]$ is centered at $f_{s,r}/4$ with sample rate $f_{s,r}$.
    The complex baseband signal $x_c[n]$ is centered at $0$ with sample rate $f_{s,c} = f_{s,r}/2$.

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
                0.1 * sdr.sinusoid(1_000 / sample_rate, freq=100, sample_rate=sample_rate, complex=False) \
                + 1.0 * sdr.sinusoid(1_000 / sample_rate, freq=250, sample_rate=sample_rate, complex=False) \
                + 0.5 * sdr.sinusoid(1_000 / sample_rate, freq=300, sample_rate=sample_rate, complex=False) \
            ); \
            x_r = sdr.awgn(x_r, snr=30)

            @savefig sdr_to_complex_baseband_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x_r[0:100], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_r[n]$");

            @savefig sdr_to_complex_baseband_2.svg
            plt.figure(); \
            sdr.plot.periodogram(x_r, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_r[n]$");

        Convert the real passband signal to a complex baseband signal with sample rate 500 sps and center of 0 Hz.
        Notice the spectrum is no longer complex-conjugate symmetric.
        The real sinusoids are now complex exponentials at -150, 0, and 50 Hz.

        .. ipython:: python

            x_c = sdr.to_complex_baseband(x_r); \
            sample_rate /= 2

            @savefig sdr_to_complex_baseband_3.svg
            plt.figure(); \
            sdr.plot.time_domain(x_c[0:50], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_c[n]$");

            @savefig sdr_to_complex_baseband_4.svg
            plt.figure(); \
            sdr.plot.periodogram(x_c, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_c[n]$");

    Group:
        dsp-signal-manipulation
    """
    x_r = verify_arraylike(x_r, float=True, real=True, ndim=1)

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

    return convert_output(x_c)


@export
def to_real_passband(x_c: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""
    Converts a complex baseband signal to a real passband signal.

    The complex baseband signal $x_c[n]$ is centered at $0$ with sample rate $f_{s,c}$.
    The real passband signal $x_r[n]$ is centered at $f_{s,r}/4$ with sample rate $f_{s,r} = 2f_{s,c}$.

    Arguments:
        x_c: The complex baseband signal $x_c[n]$ centered at $0$ with sample rate $f_{s,c}$.

    Returns:
        The real passband signal $x_r[n]$ centered at $f_{s,r}/4$ with sample rate $f_{s,r} = 2f_{s,c}$.

    Examples:
        Create a complex baseband signal with frequency components at -150, 0, and 50 Hz, at a sample rate of 500 sps.
        Notice the spectrum is asymmetric.

        .. ipython:: python

            sample_rate = 500; \
            x_c = ( \
                0.1 * sdr.sinusoid(1_000 / sample_rate, freq=-150, sample_rate=sample_rate) \
                + 1.0 * sdr.sinusoid(1_000 / sample_rate, freq=0, sample_rate=sample_rate) \
                + 0.5 * sdr.sinusoid(1_000 / sample_rate, freq=50, sample_rate=sample_rate) \
            ); \
            x_c = sdr.awgn(x_c, snr=30)

            @savefig sdr_to_real_passband_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x_c[0:50], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_c[n]$");

            @savefig sdr_to_real_passband_2.svg
            plt.figure(); \
            sdr.plot.periodogram(x_c, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_c[n]$");

        Convert the complex baseband signal to a real passband signal with sample rate 1 ksps and center of 250 Hz.
        Notice the spectrum is now complex-conjugate symmetric.
        The complex exponentials are now real sinusoids at 100, 250, and 300 Hz.

        .. ipython:: python

            x_r = sdr.to_real_passband(x_c); \
            sample_rate *= 2

            @savefig sdr_to_real_passband_3.svg
            plt.figure(); \
            sdr.plot.time_domain(x_r[0:100], sample_rate=sample_rate); \
            plt.title("Time-domain signal $x_r[n]$");

            @savefig sdr_to_real_passband_4.svg
            plt.figure(); \
            sdr.plot.periodogram(x_r, fft=2048, sample_rate=sample_rate); \
            plt.title("Periodogram of $x_r[n]$");

    Group:
        dsp-signal-manipulation
    """
    x_c = verify_arraylike(x_c, complex=True, imaginary=True, ndim=1)

    # Upsample by 2 to achieve 2*fs sample rate
    x_c = scipy.signal.resample_poly(x_c, 2, 1)

    # Mix the complex baseband signal to passband. What was 0 is now fs/4.
    x_c = mix(x_c, freq=0.25)

    # Only preserve the real part, which is complex-conjugate symmetric about 0 Hz.
    x_r = x_c.real

    return convert_output(x_r)


@export
def upsample(x: npt.ArrayLike, rate: int) -> npt.NDArray:
    r"""
    Upsamples the time-domain signal $x[n]$ by the factor $r$, by inserting $r-1$ zeros between each sample.

    Warning:
        This function does not perform any anti-aliasing filtering. The upsampled signal $y[n]$ will have
        frequency components above the Nyquist frequency of the original signal $x[n]$. For efficient
        polyphase interpolation (with anti-aliasing filtering), see :class:`sdr.Interpolator`.

    Arguments:
        x: The time-domain signal $x[n]$ with sample rate $f_s$.
        rate: The upsampling factor $r$.

    Returns:
        The upsampled signal $y[n]$ with sample rate $f_s r$.

    See Also:
        sdr.Interpolator

    Examples:
        Upsample a complex exponential by a factor of 4.

        .. ipython:: python

            sample_rate = 100; \
            x = sdr.sinusoid(20 / sample_rate, freq=15, sample_rate=sample_rate); \
            y = sdr.upsample(x, 4)

            @savefig sdr_upsample_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=sample_rate); \
            plt.title("Input signal $x[n]$");

            @savefig sdr_upsample_2.svg
            plt.figure(); \
            sdr.plot.time_domain(y, sample_rate=sample_rate*4); \
            plt.title("Upsampled signal $y[n]$");

        The spectrum of $y[n]$ has 3 additional copies of the spectrum of $x[n]$.

        .. ipython:: python
            :okwarning:

            @savefig sdr_upsample_3.svg
            plt.figure(); \
            sdr.plot.periodogram(x, fft=2048, sample_rate=sample_rate); \
            plt.xlim(-sample_rate*2, sample_rate*2); \
            plt.ylim(-100, 0); \
            plt.title("Input signal $x[n]$");

            @savefig sdr_upsample_4.svg
            plt.figure(); \
            sdr.plot.periodogram(y, fft=2048, sample_rate=sample_rate*4); \
            plt.xlim(-sample_rate*2, sample_rate*2); \
            plt.ylim(-100, 0); \
            plt.title("Upsampled signal $y[n]$");

    Group:
        dsp-signal-manipulation
    """
    x = verify_arraylike(x, ndim=1)
    verify_scalar(rate, int=True, positive=True)

    y = np.zeros(x.size * rate, dtype=x.dtype)
    y[::rate] = x

    return convert_output(y)


@export
def downsample(x: npt.ArrayLike, rate: int) -> npt.NDArray:
    r"""
    Downsamples the time-domain signal $x[n]$ by the factor $r$, by discarding $r-1$ samples every $r$ samples.

    Warning:
        This function does not perform any anti-aliasing filtering. The downsampled signal $y[n]$ will have
        spectral aliasing. For efficient polyphase decimation (with anti-aliasing filtering), see
        :class:`sdr.Decimator`.

    Arguments:
        x: The time-domain signal $x[n]$ with sample rate $f_s$.
        rate: The downsampling factor $r$.

    Returns:
        The downsampled signal $y[n]$ with sample rate $f_s / r$.

    Examples:
        Downsample a complex exponential by a factor of 4.

        .. ipython:: python

            sample_rate = 400; \
            x1 = sdr.sinusoid(200 / sample_rate, freq=0, sample_rate=sample_rate); \
            x2 = sdr.sinusoid(200 / sample_rate, freq=130, sample_rate=sample_rate); \
            x3 = sdr.sinusoid(200 / sample_rate, freq=-140, sample_rate=sample_rate); \
            x = x1 + x2 + x3
            y = sdr.downsample(x, 4)

            @savefig sdr_downsample_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=sample_rate); \
            plt.title("Input signal $x[n]$");

            @savefig sdr_downsample_2.svg
            plt.figure(); \
            sdr.plot.time_domain(y, sample_rate=sample_rate/4); \
            plt.title("Downsampled signal $y[n]$");

        The spectrum of $x[n]$ has aliased. Any spectral content above the Nyquist frequency of $f_s / 2$
        will *fold* into the spectrum of $y[n]$. The CW at 0 Hz remains at 0 Hz (unaliased).
        The CW at 130 Hz folds into 30 Hz. The CW at -140 Hz folds into -40 Hz.

        .. ipython:: python
            :okwarning:

            @savefig sdr_downsample_3.svg
            plt.figure(); \
            sdr.plot.periodogram(x, fft=2048, sample_rate=sample_rate); \
            plt.xlim(-sample_rate/2, sample_rate/2); \
            plt.ylim(-100, 0); \
            plt.title("Input signal $x[n]$");

            @savefig sdr_downsample_4.svg
            plt.figure(); \
            sdr.plot.periodogram(y, fft=2048, sample_rate=sample_rate/4); \
            plt.xlim(-sample_rate/2, sample_rate/2); \
            plt.ylim(-100, 0); \
            plt.title("Downsampled signal $y[n]$");

    Group:
        dsp-signal-manipulation
    """
    x = verify_arraylike(x, ndim=1)
    verify_scalar(rate, int=True, positive=True)

    y = x[::rate]

    return convert_output(y)
