"""
A module for finite impulse response (FIR) filters.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from ._helper import export


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
        filtering
    """

    def __init__(self, h: npt.ArrayLike, streaming: bool = False):
        """
        Creates a FIR filter with feedforward coefficients $h_i$.

        Arguments:
            h: The feedforward coefficients $h_i$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FIR.filter()`.

        Examples:
            See the :ref:`fir-filters` example.
        """
        self._taps = np.asarray(h)
        self._streaming = streaming

        self._x_prev: np.ndarray  # The filter state. Will be updated in reset().
        self.reset()

    def reset(self):
        """
        *Streaming-mode only:* Resets the filter state.

        Examples:
            See the :ref:`fir-filters` example.
        """
        self._x_prev = np.zeros(self.taps.size - 1, dtype=np.float32)

    def filter(self, x: npt.ArrayLike, mode: Literal["full", "valid", "same"] = "full") -> np.ndarray:
        r"""
        Filters the input signal $x[n]$ with the FIR filter.

        Arguments:
            x: The input signal $x[n]$.
            mode: The convolution mode. See :func:`scipy.signal.convolve` for details.

        Returns:
            The filtered signal $y[n]$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        x = np.atleast_1d(x)

        if self.streaming:
            # Prepend previous inputs from last filter() call
            x_pad = np.concatenate((self._x_prev, x))
            y = scipy.signal.convolve(x_pad, self.taps, mode="valid")
            self._x_prev = x_pad[-(self.taps.size - 1) :]
        else:
            y = scipy.signal.convolve(x, self.taps, mode=mode)

        return y

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
        d = np.zeros(N - self.taps.size + 1, dtype=np.float32)
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
        u = np.ones(N - self.taps.size + 1, dtype=np.float32)

        s = scipy.signal.convolve(u, self.taps, mode="full")

        return s

    def frequency_response(self, sample_rate: float = 1.0, N: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns the frequency response $H(f)$ of the FIR filter.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the frequency response.

        Returns:
            - The frequencies $f$ from $-f_s/2$ to $f_s/2$ in Hz.
            - The frequency response of the FIR filter $H(f)$.

        See Also:
            sdr.plot.frequency_response

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
        Returns the frequency response $H(f)$ of the FIR filter on a logarithmic frequency axis.

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
            See the :ref:`fir-filters` example.
        """
        w = np.logspace(np.log10(sample_rate / 2 / 10**decades), np.log10(sample_rate / 2), N)
        w, H = scipy.signal.freqz(self.taps, 1, worN=w, whole=False, fs=sample_rate)

        return w, H

    @property
    def taps(self) -> np.ndarray:
        """
        The feedforward taps $h_i$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self._taps

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FIR.filter()`.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self._streaming

    @property
    def order(self) -> int:
        """
        The order of the FIR filter, $N$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.taps.size - 1

    @property
    def delay(self) -> int:
        """
        The delay of the FIR filter in samples.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.order // 2


@export
class FIRInterpolator:
    r"""
    Implements a polyphase finite impulse response (FIR) interpolating filter.

    Notes:
        The polyphase interpolating filter is equivalent to first upsampling the input signal $x[n]$ by $r$
        (by inserting $r-1$ zeros between each sample) and then filtering the upsampled signal with the
        prototype FIR filter with feedforward coefficients $h_{i}$.

        Instead, the polyphase interpolating filter first decomposes the prototype FIR filter into $r$ polyphase
        filters with feedforward coefficients $h_{i, j}$. The polyphase filters are then applied to the
        input signal $x[n]$ in parallel. The output of the polyphase filters are then commutated to produce
        the output signal $y[n]$. This prevents the need to multiply with zeros in the upsampled input,
        as is needed in the first case.

        .. code-block:: text
           :caption: Polyphase 2x Interpolating FIR Filter Block Diagram

                                  +------------------------+
                              +-->| h[0], h[2], h[4], h[6] |--> ..., y[2], y[0]
                              |   +------------------------+
            ..., x[1], x[0] --+
                              |   +------------------------+
                              +-->| h[1], h[3], h[5], 0    |--> ..., y[3], y[1]
                                  +------------------------+

            Input Hold                                          Output Commutator
                                                                (top-to-bottom)

        The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

        $$h_{i, j} = h_{i + j r} .$$

    Group:
        filtering
    """

    def __init__(self, taps: npt.ArrayLike, rate: int, streaming: bool = False):
        r"""
        Creates a polyphase FIR interpolating filter with feedforward coefficients $h_i$.

        Arguments:
            taps: The feedforward coefficients $h_i$.
            rate: The interpolation rate $r$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FIRInterpolator.filter()`.
        """
        self._taps = np.asarray(taps)

        if not isinstance(rate, int):
            raise TypeError("Argument 'rate' must be an integer.")
        if not rate >= 1:
            raise ValueError(f"Argument 'rate' must be at least 1, not {rate}.")
        self._rate = rate

        if not isinstance(streaming, bool):
            raise TypeError("Argument 'streaming' must be a boolean.")
        self._streaming = streaming

        N = math.ceil(self.taps.size / rate) * rate
        self._polyphase_taps = np.pad(self.taps, (0, N - self.taps.size), mode="constant").reshape(-1, rate).T

        self._x_prev: np.ndarray  # The filter state. Will be updated in reset().
        self.reset()

    def reset(self):
        """
        *Streaming-mode only:* Resets the filter state.
        """
        self._x_prev = np.zeros(self.polyphase_taps.shape[1] - 1)

    def filter(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Filters and interpolates the input signal $x[n]$ with the FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s r$.
        """
        x = np.atleast_1d(x)

        if self.streaming:
            # Prepend previous inputs from last filter() call
            x_pad = np.concatenate((self._x_prev, x))
            xx = np.tile(x_pad, (self.rate, 1))
            taps = np.tile(self.polyphase_taps, self.rate)
            yy = scipy.signal.convolve(xx, taps, mode="valid")
            self._x_prev = x_pad[-(self.polyphase_taps.shape[1] - 1) :]
        else:
            yy = np.zeros((self.rate, x.size + self.polyphase_taps.shape[1] - 1), dtype=np.complex64)
            for i in range(self.rate):
                yy[i] = scipy.signal.convolve(x, self.polyphase_taps[i], mode="full")
            # xx = np.tile(x, (self.rate, 1))
            # taps = np.tile(self.polyphase_taps, self.rate)
            # yy = scipy.signal.convolve(xx, taps, mode="full")

        y = yy.T.flatten()

        return y

    @property
    def taps(self) -> np.ndarray:
        """
        The prototype feedforward taps $h_i$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j r} .$$

        Examples:
            .. ipython:: python

                fir = sdr.FIRInterpolator(np.arange(10), 3)
                fir.taps
                fir.polyphase_taps
        """
        return self._taps

    @property
    def polyphase_taps(self) -> np.ndarray:
        """
        The polyphase feedforward taps $h_{i, j}$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j r} .$$

        Examples:
            .. ipython:: python

                fir = sdr.FIRInterpolator(np.arange(10), 3)
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps

    @property
    def rate(self) -> int:
        """
        The interpolation rate $r$.
        """
        return self._rate

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FIRInterpolator.filter()`.
        """
        return self._streaming
