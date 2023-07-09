"""
A module for infinite impulse response (IIR) filters.
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from typing_extensions import Self


class IIR:
    r"""
    Implements an infinite impulse response (IIR) filter.

    This class is a wrapper for the :func:`scipy.signal.lfilter` function. It supports one-time filtering
    and streamed filtering.

    Notes:
        An IIR filter is defined by its feedforward coefficients $b_i$ and feedback coefficients $a_j$.
        These coefficients define the difference equation

        $$y[n] = \frac{1}{a_0} \left( \sum_{i=0}^{M} b_i x[n-i] - \sum_{j=1}^{N} a_j y[n-j] \right) .$$

        The transfer function of the filter is

        $$H(z) = \frac{\sum_{i=0}^{M} b_i z^{-i}}{\sum_{j=0}^{N} a_j z^{-j}} .$$

    Examples:
        See the :ref:`iir-filter` example.

    Group:
        filtering
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, streaming: bool = False):
        """
        Creates an IIR filter with feedforward coefficients $b_i$ and feedback coefficients $a_j$.

        Arguments:
            b: Feedforward coefficients, $b_i$.
            a: Feedback coefficients, $a_j$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`filter()`.

        Examples:
            See the :ref:`iir-filter` example.
        """
        self._b_taps = np.asarray(b)
        self._a_taps = np.asarray(a)
        self._streaming = streaming

        self._zi: np.ndarray  # The filter state. Will be updated in reset().
        self.reset()

        # Compute the zeros and poles of the transfer function
        self._zeros, self._poles, self._gain = scipy.signal.tf2zpk(self.b_taps, self.a_taps)

    @classmethod
    def ZerosPoles(cls, zeros: np.ndarray, poles: np.ndarray, gain: float = 1.0, streaming: bool = False) -> Self:
        """
        Creates an IIR filter from its zeros, poles, and gain.

        Arguments:
            zeros: The zeros of the transfer function.
            poles: The poles of the transfer function.
            gain: The gain of the transfer function.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`filter()`.

        Examples:
            See the :ref:`iir-filter` example.
        """
        b, a = scipy.signal.zpk2tf(zeros, poles, gain)

        return cls(b, a, streaming=streaming)

    def reset(self):
        """
        *Streaming-mode only:* Resets the filter state.

        Examples:
            See the :ref:`iir-filter` example.
        """
        self._zi = scipy.signal.lfiltic(self.b_taps, self.a_taps, y=[], x=[])

    def filter(self, x: np.ndarray) -> np.ndarray:
        r"""
        Filters the input signal $x[n]$ with the IIR filter.

        Arguments:
            x: The input signal, $x[n]$.

        Returns:
            The filtered signal, $y[n]$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        x = np.atleast_1d(x)

        if not self.streaming:
            self.reset()

        y, self._zi = scipy.signal.lfilter(self.b_taps, self.a_taps, x, zi=self._zi)

        return y

    def impulse_response(self, N: int = 100) -> np.ndarray:
        r"""
        Returns the impulse response $h[n]$ of the IIR filter.

        The impulse response $h[n]$ is the filter output when the input is an impulse $\delta[n]$.

        Arguments:
            N: The number of samples to return.

        Returns:
            The impulse response of the IIR filter, $h[n]$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        # Delta impulse function
        d = np.zeros(N, dtype=np.float32)
        d[0] = 1

        zi = self._zi
        h = self.filter(d)
        self._zi = zi  # Restore the filter state

        return h

    def step_response(self, N: int = 100) -> np.ndarray:
        """
        Returns the step response $s[n]$ of the IIR filter.

        The step response $s[n]$ is the filter output when the input is a unit step $u[n]$.

        Arguments:
            N: The number of samples to return.

        Returns:
            The step response of the IIR filter, $s[n]$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        # Unit step function
        u = np.ones(N, dtype=np.float32)

        zi = self._zi
        s = self.filter(u)
        self._zi = zi  # Restore the filter state

        return s

    def frequency_response(self, sample_rate: float = 1.0, N: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(e^{j2 \pi f})$ of the IIR filter.

        Arguments:
            sample_rate: The sample rate of the filter in samples/s.
            N: The number of samples in the frequency response.

        Returns:
            - The frequencies, $f$, from $-f_s$ to $f_s$ in Hz.
            - The frequency response of the IIR filter, $H(e^{j2 \pi f})$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        w, H = scipy.signal.freqz(self.b_taps, self.a_taps, worN=N, whole=True, fs=sample_rate)

        w[w >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        w = np.fft.fftshift(w)
        H = np.fft.fftshift(H)

        return w, H

    def frequency_response_log(
        self, sample_rate: float = 1.0, N: int = 1024, decades: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(e^{j2 \pi f})$ of the IIR filter on a logarithmic frequency axis

        Arguments:
            sample_rate: The sample rate of the filter in samples/s.
            N: The number of samples in the frequency response.
            decades: The number of frequency decades to plot.

        Returns:
            - The frequencies, $f$, from $0$ to $f_s$ in Hz. The frequencies are logarithmically-spaced.
            - The frequency response of the IIR filter, $H(e^{j2 \pi f})$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, H = scipy.signal.freqz(self.b_taps, self.a_taps, worN=w, whole=False, fs=sample_rate)

        return w, H

    def plot_impulse_response(self, N: int = 100):
        """
        Plots the impulse response $h[n]$ of the IIR filter.

        Arguments:
            N: The number of samples in the impulse response.

        Examples:
            See the :ref:`iir-filter` example.
        """
        h = self.impulse_response(N)

        # plt.stem(np.arange(h.size), h.real, linefmt="b-", markerfmt="bo")
        plt.plot(np.arange(h.size), h.real, color="b", marker=".", label="Real")
        plt.plot(np.arange(h.size), h.imag, color="r", marker=".", label="Imaginary")
        plt.legend(loc="upper right")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Impulse Response, $h[n]$")

    def plot_step_response(self, N: int = 100):
        """
        Plots the step response $s[n]$ of the IIR filter.

        Arguments:
            N: The number of samples in the step response.

        Examples:
            See the :ref:`iir-filter` example.
        """
        u = self.step_response(N)

        # plt.stem(np.arange(u.size), u.real, linefmt="b-", markerfmt="bo")
        plt.plot(np.arange(u.size), u.real, color="b", marker=".", label="Real")
        plt.plot(np.arange(u.size), u.imag, color="r", marker=".", label="Imaginary")
        plt.legend(loc="lower right")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title("Step Response, $s[n]$")

    def plot_zeros_poles(self):
        """
        Plots the zeros and poles of the IIR filter.

        Examples:
            See the :ref:`iir-filter` example.
        """
        unit_circle = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
        z = self.zeros
        p = self.poles

        plt.plot(unit_circle.real, unit_circle.imag, color="k", linestyle="--", label="Unit circle")
        plt.scatter(z.real, z.imag, color="b", marker="o", facecolor="none", label="Zeros")
        plt.scatter(p.real, p.imag, color="r", marker="x", label="Poles")
        plt.axis("equal")
        plt.legend(loc="upper left")
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.title("Zeros and Poles of $H(z)$")

    def plot_frequency_response(self, sample_rate: float = 1.0, N: int = 1024, phase: bool = True):
        r"""
        Plots the frequency response $H(\omega)$ of the IIR filter.

        Arguments:
            sample_rate: The sample rate of the filter in samples/s.
            N: The number of samples in the frequency response.
            phase: Indicates whether to plot the phase of $H(\omega)$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        w, H = self.frequency_response(sample_rate, N)

        ax1 = plt.gca()
        ax1.plot(w, 10 * np.log10(np.abs(H) ** 2), color="b", label="Power")
        ax1.set_ylabel(r"Power (dB), $|H(\omega)|^2$")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(which="both", linestyle="--")
        if sample_rate == 1.0:
            ax1.set_xlabel("Normalized Frequency, $f /f_s$")
        else:
            ax1.set_xlabel("Frequency (Hz), $f$")

        if phase:
            ax2 = ax1.twinx()
            ax2.plot(w, np.rad2deg(np.angle(H)), color="r", linestyle="--", label="Phase")
            ax2.set_ylabel(r"Phase (degrees), $\angle H(\omega)$")
            ax2.tick_params(axis="y", labelcolor="r")
            ax2.set_ylim(-180, 180)

        plt.title(r"Frequency Response, $H(\omega)$")
        plt.tight_layout()

    def plot_frequency_response_log(
        self, sample_rate: float = 1.0, N: int = 1024, phase: bool = True, decades: int = 4
    ):
        r"""
        Plots the frequency response $H(\omega)$ of the IIR filter on a logarithmic frequency axis.

        Arguments:
            sample_rate: The sample rate of the filter in samples/s.
            N: The number of samples in the frequency response.
            phase: Indicates whether to plot the phase of $H(\omega)$.
            decades: The number of frequency decades to plot.

        Examples:
            See the :ref:`iir-filter` example.
        """
        w, H = self.frequency_response_log(sample_rate, N, decades)

        ax1 = plt.gca()
        ax1.semilogx(w, 10 * np.log10(np.abs(H) ** 2), color="b", label="Power")
        ax1.set_ylabel(r"Power (dB), $|H(\omega)|^2$")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.grid(which="both", linestyle="--")
        if sample_rate == 1.0:
            ax1.set_xlabel("Normalized Frequency, $f /f_s$")
        else:
            ax1.set_xlabel("Frequency (Hz), $f$")

        if phase:
            ax2 = ax1.twinx()
            ax2.semilogx(w, np.rad2deg(np.angle(H)), color="r", linestyle="--", label="Phase")
            ax2.set_ylabel(r"Phase (degrees), $\angle H(\omega)$")
            ax2.tick_params(axis="y", labelcolor="r")
            ax2.set_ylim(-180, 180)

        plt.title(r"Frequency Response, $H(\omega)$")
        plt.tight_layout()

    def plot_group_delay(self, sample_rate: float = 1.0, N: int = 1024):
        r"""
        Plots the group delay $\tau_g(\omega)$ of the IIR filter.

        Arguments:
            sample_rate: The sample rate of the filter in samples/s.
            N: The number of samples in the frequency response.

        Examples:
            See the :ref:`iir-filter` example.
        """
        w, tau_g = scipy.signal.group_delay((self.b_taps, self.a_taps), w=N, whole=True, fs=sample_rate)

        w[w >= 0.5 * sample_rate] -= sample_rate
        w = np.fft.fftshift(w)
        tau_g = np.fft.fftshift(tau_g)

        plt.plot(w, tau_g, color="b")
        if sample_rate == 1.0:
            plt.xlabel("Normalized Frequency, $f /f_s$")
        else:
            plt.xlabel("Frequency (Hz), $f$")
        plt.ylabel(r"Group Delay (samples), $\tau_g(\omega)$")
        plt.title(r"Group Delay, $\tau_g(\omega)$")
        plt.grid(which="both", linestyle="--")
        plt.tight_layout()

    def plot_all(self, sample_rate: float = 1.0, N_time: int = 100, N_freq: int = 1024):
        """
        Plots the zeros and poles, impulse response, step response, and frequency response of the IIR filter
        in a single figure.

        Arguments:
            sample_rate: The sample rate of the filter in samples/s.
            N_time: The number of samples in the impulse and step responses.
            N_freq: The number of samples in the frequency response.

        Examples:
            See the :ref:`iir-filter` example.
        """
        plt.subplot2grid((4, 3), (0, 0), 2, 1)
        self.plot_zeros_poles()
        plt.subplot2grid((4, 3), (0, 1), 1, 2)
        self.plot_impulse_response(N=N_time)
        plt.subplot2grid((4, 3), (1, 1), 1, 2)
        self.plot_step_response(N=N_time)
        plt.subplot2grid((4, 3), (2, 0), 2, 3)
        self.plot_frequency_response(sample_rate=sample_rate, N=N_freq)

    @property
    def b_taps(self) -> np.ndarray:
        """
        Returns the feedforward filter taps, $b_i$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._b_taps

    @property
    def a_taps(self) -> np.ndarray:
        """
        Returns the feedback filter taps, $a_j$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._a_taps

    @property
    def streaming(self) -> bool:
        """
        Returns whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`filter()`.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._streaming

    @property
    def order(self) -> int:
        """
        Returns the order of the IIR filter, $N - 1$.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._a_taps.size - 1

    @property
    def zeros(self) -> np.ndarray:
        """
        Returns the zeros of the IIR filter.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._zeros

    @property
    def poles(self) -> np.ndarray:
        """
        Returns the poles of the IIR filter.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._poles

    @property
    def gain(self) -> float:
        """
        Returns the gain of the IIR filter.

        Examples:
            See the :ref:`iir-filter` example.
        """
        return self._gain
