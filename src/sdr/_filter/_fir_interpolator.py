"""
A module for interpolating finite impulse response (FIR) filters.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export


@export
def multirate_fir(
    P: int,
    Q: int = 1,
    half_length: int = 12,
    A_stop: float = 80,
) -> np.ndarray:
    r"""
    Computes the multirate FIR filter that achieves rational resampling by $P/Q$.

    Note:
        This filter can be used with :class:`sdr.FIRInterpolator` or :class:`sdr.FIRDecimator`.

    Arguments:
        P: The interpolation rate $P$.
        Q: The decimation rate $Q$.
        half_length: The half-length of the polyphase filters.
        A_stop: The stopband attenuation $A_{\text{stop}}$ in dB.

    Returns:
        The multirate FIR filter $h[n]$.

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters
        - https://www.mathworks.com/help/dsp/ref/designmultiratefir.html

    Examples:
        Design a multirate FIR filter for rational resampling by 2/3.

        .. ipython:: python

            h = sdr.multirate_fir(2, 3)

            @savefig sdr_multirate_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h);

            @savefig sdr_multirate_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.frequency_response(h);

    Group:
        dsp-multirate-filtering
    """
    if not isinstance(P, int):
        raise TypeError(f"Argument 'P' must be an integer, not {P}.")
    if not P >= 1:
        raise ValueError(f"Argument 'P' must be at least 1, not {P}.")

    if not isinstance(Q, int):
        raise TypeError(f"Argument 'Q' must be an integer, not {Q}.")
    if not Q >= 1:
        raise ValueError(f"Argument 'Q' must be at least 1, not {Q}.")

    B = P if P > 1 else Q  # The number of polyphase branches
    R = max(P, Q)  # Inverse of the filter's fractional bandwidth

    # Compute the filter order, which is length - 1
    N = 2 * half_length * B

    # Compute ideal lowpass filter
    n = np.arange(N + 1)
    h = P / R * np.sinc((n - N // 2) / R)

    # Compute Kaiser window
    if A_stop >= 50:
        beta = 0.1102 * (A_stop - 8.71)
    elif A_stop > 21:
        beta = 0.5842 * (A_stop - 21) ** 0.4 + 0.07886 * (A_stop - 21)
    else:
        beta = 0
    w = scipy.signal.kaiser(N + 1, beta)

    # Compute windowed filter
    h = h * w

    if not (Q > P > 1 and (half_length * P) % Q != 0):
        # The first and last elements are zero. Remove the last zero so that the filter is evenly
        # partitioned across the polyphase branches. The filter now has length 2 * half_length * B and
        # each polyphase branch has length 2 * half_length.
        h = h[:-1]

    # If the above condition is not true, the first and last elements are non-zero. The filter length
    # is 2 * half_length * B + 1 and each polyphase branch has length 2 * half_length + 1. The final
    # column in the polyphase matrix will be padded with zeros.

    return h


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
        dsp-multirate-filtering
    """

    def __init__(self, taps: npt.ArrayLike, rate: int, streaming: bool = False):
        r"""
        Creates a polyphase FIR interpolating filter with feedforward coefficients $h_i$.

        Arguments:
            taps: The feedforward coefficients $h_i$.
            rate: The interpolation rate $r$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FIRInterpolator.__call__()`.
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

    def __call__(self, x: npt.ArrayLike, mode: Literal["full", "valid", "same"] = "full") -> np.ndarray:
        """
        Filters and interpolates the input signal $x[n]$ with the FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$.
            mode: The non-streaming convolution mode. See :func:`scipy.signal.convolve` for details.
                In streaming mode, $N$ inputs always produce $N r$ outputs.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s r$.
        """
        x = np.atleast_1d(x)
        dtype = np.result_type(x, self.polyphase_taps)

        if self.streaming:
            # Prepend previous inputs from last __call__() call
            x_pad = np.concatenate((self._x_prev, x))
            yy = np.zeros((self.rate, x.size), dtype=dtype)
            for i in range(self.rate):
                yy[i] = scipy.signal.convolve(x_pad, self.polyphase_taps[i], mode="valid")

            self._x_prev = x_pad[-(self.polyphase_taps.shape[1] - 1) :]

            # xx = np.tile(x_pad, (self.rate, 1))
            # taps = np.tile(self.polyphase_taps, self.rate)
            # yy = scipy.signal.convolve(xx, taps, mode="valid")

            # Commutate the outputs of the polyphase filters
            y = yy.T.flatten()
        else:
            yy = np.zeros((self.rate, x.size + self.polyphase_taps.shape[1] - 1), dtype=dtype)
            for i in range(self.rate):
                yy[i] = scipy.signal.convolve(x, self.polyphase_taps[i], mode="full")

            # Commutate the outputs of the polyphase filters
            y = yy.T.flatten()

            if mode == "full":
                size = x.size * self.rate + self.taps.size - 1
                y = y[:size]
            elif mode == "same":
                size = max(x.size * self.rate, self.taps.size)
                offset = (min(x.size * self.rate, self.taps.size) - 1) // 2
                y = y[offset : offset + size]
            else:
                size = (max(x.size * self.rate, self.taps.size) - min(x.size * self.rate, self.taps.size)) + 1
                offset = min(x.size * self.rate, self.taps.size) - 1
                y = y[offset : offset + size]

            # xx = np.tile(x, (self.rate, 1))
            # taps = np.tile(self.polyphase_taps, self.rate)
            # yy = scipy.signal.convolve(xx, taps, mode="full")

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

        In streaming mode, the filter state is preserved between calls to :meth:`~FIRInterpolator.__call__()`.
        """
        return self._streaming
