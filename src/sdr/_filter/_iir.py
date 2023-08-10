"""
A module for infinite impulse response (IIR) filters.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Self

from .._helper import export


@export
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

        $$H(z) = \frac{\sum\limits_{i=0}^{M} b_i z^{-i}}{\sum\limits_{j=0}^{N} a_j z^{-j}} .$$

    Examples:
        See the :ref:`iir-filters` example.

    Group:
        filtering
    """

    def __init__(self, b: npt.ArrayLike, a: npt.ArrayLike, streaming: bool = False):
        """
        Creates an IIR filter with feedforward coefficients $b_i$ and feedback coefficients $a_j$.

        Arguments:
            b: The feedforward coefficients $b_i$.
            a: The feedback coefficients $a_j$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs and outputs are
                preserved between calls to :meth:`~IIR.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.
        """
        self._b_taps = np.asarray(b)
        self._a_taps = np.asarray(a)
        self._streaming = streaming

        self._zi: np.ndarray  # The filter state. Will be updated in reset().
        self.reset()

        # Compute the zeros and poles of the transfer function
        self._zeros, self._poles, self._gain = scipy.signal.tf2zpk(self.b_taps, self.a_taps)

    @classmethod
    def ZerosPoles(cls, zeros: npt.ArrayLike, poles: npt.ArrayLike, gain: float = 1.0, streaming: bool = False) -> Self:
        """
        Creates an IIR filter from its zeros, poles, and gain.

        Arguments:
            zeros: The zeros of the transfer function.
            poles: The poles of the transfer function.
            gain: The gain of the transfer function.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs and outputs are
                preserved between calls to :meth:`~IIR.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.
        """
        b, a = scipy.signal.zpk2tf(zeros, poles, gain)

        return cls(b, a, streaming=streaming)

    def reset(self):
        """
        *Streaming-mode only:* Resets the filter state.

        Examples:
            See the :ref:`iir-filters` example.
        """
        self._zi = scipy.signal.lfiltic(self.b_taps, self.a_taps, y=[], x=[])

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        r"""
        Filters the input signal $x[n]$ with the IIR filter.

        Arguments:
            x: The input signal $x[n]$.

        Returns:
            The filtered signal $y[n]$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        x = np.atleast_1d(x)

        if not self.streaming:
            self.reset()

        y, self._zi = scipy.signal.lfilter(self.b_taps, self.a_taps, x, zi=self._zi)

        return y

    def impulse_response(self, N: int = 100) -> np.ndarray:
        r"""
        Returns the impulse response $h[n]$ of the IIR filter. The impulse response $h[n]$ is the
        filter output when the input is an impulse $\delta[n]$.

        Arguments:
            N: The number of samples to return.

        Returns:
            The impulse response of the IIR filter $h[n]$.

        See Also:
            sdr.plot.impulse_response

        Examples:
            See the :ref:`iir-filters` example.
        """
        # Delta impulse function
        d = np.zeros(N, dtype=np.float32)
        d[0] = 1

        zi = self._zi
        h = self(d)
        self._zi = zi  # Restore the filter state

        return h

    def step_response(self, N: int = 100) -> np.ndarray:
        """
        Returns the step response $s[n]$ of the IIR filter. The step response $s[n]$ is the
        filter output when the input is a unit step $u[n]$.

        Arguments:
            N: The number of samples to return.

        Returns:
            The step response of the IIR filter $s[n]$.

        See Also:
            sdr.plot.step_response

        Examples:
            See the :ref:`iir-filters` example.
        """
        # Unit step function
        u = np.ones(N, dtype=np.float32)

        zi = self._zi
        s = self(u)
        self._zi = zi  # Restore the filter state

        return s

    def frequency_response(self, sample_rate: float = 1.0, N: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(f)$ of the IIR filter.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the frequency response.

        Returns:
            - The frequencies $f$ from $-f_s/2$ to $f_s/2$ in Hz.
            - The frequency response of the IIR filter $H(f)$.

        See Also:
            sdr.plot.frequency_response

        Examples:
            See the :ref:`iir-filters` example.
        """
        w, H = scipy.signal.freqz(self.b_taps, self.a_taps, worN=N, whole=True, fs=sample_rate)

        w[w >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        w = np.fft.fftshift(w)
        H = np.fft.fftshift(H)

        return w, H

    def frequency_response_log(
        self, sample_rate: float = 1.0, N: int = 1024, decades: int = 4
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(f)$ of the IIR filter on a logarithmic frequency axis.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the frequency response.
            decades: The number of frequency decades to plot.

        Returns:
            - The frequencies $f$ from $0$ to $f_s/2$ in Hz. The frequencies are logarithmically-spaced.
            - The frequency response of the IIR filter $H(f)$.

        See Also:
            sdr.plot.frequency_response

        Examples:
            See the :ref:`iir-filters` example.
        """
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, H = scipy.signal.freqz(self.b_taps, self.a_taps, worN=w, whole=False, fs=sample_rate)

        return w, H

    @property
    def b_taps(self) -> np.ndarray:
        """
        The feedforward taps $b_i$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._b_taps

    @property
    def a_taps(self) -> np.ndarray:
        """
        The feedback taps $a_j$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._a_taps

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~IIR.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._streaming

    @property
    def order(self) -> int:
        """
        The order of the IIR filter, $N$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._a_taps.size - 1

    @property
    def zeros(self) -> np.ndarray:
        """
        The zeros of the IIR filter.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._zeros

    @property
    def poles(self) -> np.ndarray:
        """
        The poles of the IIR filter.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._poles

    @property
    def gain(self) -> float:
        """
        The gain of the IIR filter.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._gain
