"""
A module for finite impulse response (FIR) filters.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export


@export
class FIR:
    r"""
    Implements a finite impulse response (FIR) filter.

    This class is a wrapper for the :func:`scipy.signal.convolve` function. It supports one-time filtering
    and streamed filtering.

    Notes:
        A FIR filter is defined by its feedforward coefficients $h_i$.

        $$y[n] = \sum_{i=0}^{N} h_i x[n-i] .$$

        The transfer function of the filter is

        $$H(z) = \sum\limits_{i=0}^{N} h_i z^{-i} .$$

    Examples:
        See the :ref:`fir-filters` example.

    Group:
        dsp-filtering
    """

    def __init__(self, h: npt.ArrayLike, streaming: bool = False):
        """
        Creates a FIR filter with feedforward coefficients $h_i$.

        Arguments:
            h: The feedforward coefficients $h_i$ for $i = 0,...N$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FIR.__call__()`.

        Examples:
            See the :ref:`fir-filters` example.
        """
        self._taps = np.asarray(h)
        self._streaming = streaming

        self._state: np.ndarray  # The filter state. Will be updated in reset().
        self.reset()

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["full", "valid", "same"] = "full") -> np.ndarray:
        r"""
        Filters the input signal $x[n]$ with the FIR filter.

        Arguments:
            x: The input signal $x[n]$ with length $L$.
            mode: The non-streaming convolution mode.

                - `"same"`: The output signal $y[n]$ has length $L$. Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $L + N$,
                  where $N$ is the order of the filter. Output sample :obj:`~sdr.FIR.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L$ output samples are produced per call. A final call to :meth:`~sdr.FIR.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$. The output length is dictated by the `mode` argument.

        Examples:
            See the :ref:`fir-filters` example.
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        if self.streaming:
            # Prepend previous inputs from last __call__() call
            x_pad = np.concatenate((self._state, x))
            y = scipy.signal.convolve(x_pad, self.taps, mode="valid")
            self._state = x_pad[-(self.taps.size - 1) :]
        else:
            y = scipy.signal.convolve(x, self.taps, mode=mode)

        return y

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.

        Examples:
            .. ipython:: python

                h = sdr.root_raised_cosine(0.5, 6, 5)
                fir = sdr.FIR(h)
                fir
        """
        return f"sdr.{type(self).__name__}({self.taps.tolist()}, streaming={self.streaming})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.

        Examples:
            .. ipython:: python

                h = sdr.root_raised_cosine(0.5, 6, 5)
                fir = sdr.FIR(h)
                print(fir)
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  taps: {self.taps.shape} shape"
        string += f"\n    {self.taps.tolist()}"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    def __len__(self) -> int:
        """
        Returns the filter length $N + 1$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.taps.size

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        """
        Resets the filter state. Only useful when using streaming mode.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        self._state = np.zeros(self.taps.size - 1, dtype=self.taps.dtype)

    def flush(self) -> np.ndarray:
        """
        Flushes the filter state by passing zeros through the filter. Only useful when using streaming mode.

        Returns:
            The remaining filtered signal $y[n]$.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        x = np.zeros_like(self.state)
        y = self(x)
        return y

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FIR.__call__()`.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        return self._streaming

    @property
    def state(self) -> np.ndarray:
        """
        The filter state consisting of the previous $N$ inputs.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        return self._state

    ##############################################################################
    # Methods
    ##############################################################################

    def impulse_response(self, N: int | None = None) -> np.ndarray:
        r"""
        Returns the impulse response $h[n]$ of the FIR filter. The impulse response $h[n]$ is the
        filter output when the input is an impulse $\delta[n]$.

        Arguments:
            N: The number of samples to return. The default is the filter length.

        Returns:
            The impulse response of the IIR filter $h[n]$.

        See Also:
            sdr.plot.impulse_response

        Examples:
            See the :ref:`fir-filters` example.
        """
        if N is None:
            N = self.taps.size

        if not N >= self.taps.size:
            raise ValueError("Argument 'N' must be greater than or equal to the filter length.")

        # Delta impulse function
        d = np.zeros(N - self.taps.size + 1, dtype=float)
        d[0] = 1

        h = scipy.signal.convolve(d, self.taps, mode="full")

        return h

    def step_response(self, N: int | None = None) -> np.ndarray:
        """
        Returns the step response $s[n]$ of the FIR filter. The step response $s[n]$ is the
        filter output when the input is a unit step $u[n]$.

        Arguments:
            N: The number of samples to return. The default is the filter length.

        Returns:
            The step response of the FIR filter $s[n]$.

        See Also:
            sdr.plot.step_response

        Examples:
            See the :ref:`fir-filters` example.
        """
        if N is None:
            N = self.taps.size

        if not N >= self.taps.size:
            raise ValueError("Argument 'N' must be greater than or equal to the filter length.")

        # Unit step function
        u = np.ones(N - self.taps.size + 1, dtype=float)

        s = scipy.signal.convolve(u, self.taps, mode="full")

        return s

    def frequency_response(self, sample_rate: float = 1.0, N: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(\omega)$ of the FIR filter.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the frequency response.

        Returns:
            - The frequencies $f$ from $-f_s/2$ to $f_s/2$ in Hz.
            - The frequency response of the FIR filter $H(\omega)$.

        See Also:
            sdr.plot.magnitude_response, sdr.plot.phase_response

        Examples:
            See the :ref:`fir-filters` example.
        """
        w, H = scipy.signal.freqz(self.taps, 1, worN=N, whole=True, fs=sample_rate)

        w[w >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        w = np.fft.fftshift(w)
        H = np.fft.fftshift(H)

        return w, H

    def frequency_response_log(
        self, sample_rate: float = 1.0, N: int = 1024, decades: int = 4
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(\omega)$ of the FIR filter on a logarithmic frequency axis.

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
            See the :ref:`fir-filters` example.
        """
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, H = scipy.signal.freqz(self.taps, 1, worN=w, whole=False, fs=sample_rate)

        return w, H

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def taps(self) -> np.ndarray:
        """
        The feedforward taps $h_i$ for $i = 0,...,N$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self._taps

    @property
    def order(self) -> int:
        """
        The order of the FIR filter $N$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.taps.size - 1

    @property
    def delay(self) -> int:
        r"""
        The delay of the FIR filter $d = \lfloor \frac{N + 1}{2} \rfloor$ in samples.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.taps.size // 2
