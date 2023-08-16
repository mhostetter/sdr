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
        dsp-filtering
    """

    def __init__(self, b: npt.ArrayLike, a: npt.ArrayLike, streaming: bool = False):
        """
        Creates an IIR filter with feedforward coefficients $b_i$ and feedback coefficients $a_j$.

        Arguments:
            b: The feedforward coefficients $b_i$ for $i = 0,...,M$.
            a: The feedback coefficients $a_j$ for $j = 0,...,N$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs and outputs are
                preserved between calls to :meth:`~IIR.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.
        """
        self._b_taps = np.asarray(b)
        self._a_taps = np.asarray(a)
        self._streaming = streaming

        self._state: np.ndarray  # The filter state. Will be updated in reset().
        self.reset()

        # Compute the zeros and poles of the transfer function
        self._zeros, self._poles, self._gain = scipy.signal.tf2zpk(self.b_taps, self.a_taps)

    ##############################################################################
    # Alternate constructors
    ##############################################################################

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

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        r"""
        Filters the input signal $x[n]$ with the IIR filter.

        Arguments:
            x: The input signal $x[n]$ with length $L$.

        Returns:
            The filtered signal $y[n]$ with length $L$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        if not self.streaming:
            self.reset()

        y, self._state = scipy.signal.lfilter(self.b_taps, self.a_taps, x, zi=self._state)

        return y

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.

        Examples:
            .. ipython:: python

                zero = 0.6
                pole = 0.8 * np.exp(1j * np.pi / 8)
                iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])
                iir
        """
        return f"sdr.{type(self).__name__}({self.b_taps.tolist()}, {self.a_taps.tolist()}, streaming={self.streaming})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.

        Examples:
            .. ipython:: python

                zero = 0.6
                pole = 0.8 * np.exp(1j * np.pi / 8)
                iir = sdr.IIR.ZerosPoles([zero], [pole, pole.conj()])
                print(iir)
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  b_taps: {self.b_taps.shape} shape"
        string += f"\n    {self.b_taps.tolist()}"
        string += f"\n  a_taps: {self.a_taps.shape} shape"
        string += f"\n    {self.a_taps.tolist()}"
        string += f"\n  zeros: {self.zeros.shape} shape"
        string += f"\n    {self.zeros.tolist()}"
        string += f"\n  poles: {self.poles.shape} shape"
        string += f"\n    {self.poles.tolist()}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        """
        Resets the filter state. Only useful when using streaming mode.

        Examples:
            See the :ref:`iir-filters` example.

        Group:
            Streaming mode only
        """
        self._state = scipy.signal.lfiltic(self.b_taps, self.a_taps, y=[], x=[])

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~IIR.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.

        Group:
            Streaming mode only
        """
        return self._streaming

    @property
    def state(self) -> np.ndarray:
        """
        The filter state.

        Examples:
            See the :ref:`iir-filters` example.

        Group:
            Streaming mode only
        """
        return self._state

    ##############################################################################
    # Methods
    ##############################################################################

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
        d = np.zeros(N, dtype=float)
        d[0] = 1

        zi = self._state
        h = self(d)
        self._state = zi  # Restore the filter state

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
        u = np.ones(N, dtype=float)

        state = self._state
        s = self(u)
        self._state = state  # Restore the filter state

        return s

    def frequency_response(self, sample_rate: float = 1.0, N: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(\omega)$ of the IIR filter.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the frequency response.

        Returns:
            - The frequencies $f$ from $-f_s/2$ to $f_s/2$ in Hz.
            - The frequency response of the IIR filter $H(\omega)$.

        See Also:
            sdr.plot.magnitude_response, sdr.plot.phase_response

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
        Returns the frequency response $H(\omega)$ of the IIR filter on a logarithmic frequency axis.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the frequency response.
            decades: The number of frequency decades to plot.

        Returns:
            - The frequencies $f$ from $0$ to $f_s/2$ in Hz. The frequencies are logarithmically-spaced.
            - The frequency response of the IIR filter $H(\omega)$.

        See Also:
            sdr.plot.magnitude_response, sdr.plot.phase_response

        Examples:
            See the :ref:`iir-filters` example.
        """
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, H = scipy.signal.freqz(self.b_taps, self.a_taps, worN=w, whole=False, fs=sample_rate)

        return w, H

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def b_taps(self) -> np.ndarray:
        """
        The feedforward taps $b_i$ for $i = 0,...,M$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._b_taps

    @property
    def a_taps(self) -> np.ndarray:
        """
        The feedback taps $a_j$ for $j = 0,...,N$.

        Examples:
            See the :ref:`iir-filters` example.
        """
        return self._a_taps

    @property
    def order(self) -> int:
        """
        The order of the IIR filter $N$.

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
